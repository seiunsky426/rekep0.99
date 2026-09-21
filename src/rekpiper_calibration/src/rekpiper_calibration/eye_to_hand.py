"""Fixed-camera eye-to-hand solve for the dual ArUco18 schema-v2 dataset."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml

from .checkerboard_calibration import (
    matrix_to_quaternion_xyzw, validate_rigid_transform)
from .robot_world_handeye import (
    average_transforms, check_pose_diversity, invert_transform,
    transform_deviation)


CAMERAS = ("rs1", "rs3")


class EyeToHandError(ValueError):
    pass


def _marker_object_points(marker_side_m):
    half = float(marker_side_m) / 2.0
    return np.asarray([
        [-half, half, 0.0], [half, half, 0.0],
        [half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float64)


def _transform(parameters):
    values = np.asarray(parameters, dtype=float).reshape(6)
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(values[:3]).as_matrix()
    result[:3, 3] = values[3:]
    return result


def _parameters(matrix):
    value = validate_rigid_transform(matrix, "transform")
    return np.r_[Rotation.from_matrix(value[:3, :3]).as_rotvec(),
                 value[:3, 3]]


def _closure_residual(parameters, base_T_link6, camera_T_marker,
                      rotation_scale_m=0.10):
    link6_T_marker = _transform(parameters[:6])
    base_T_camera = _transform(parameters[6:])
    residuals = []
    for arm, observed in zip(base_T_link6, camera_T_marker):
        predicted = invert_transform(base_T_camera) @ arm @ link6_T_marker
        delta = invert_transform(predicted) @ observed
        residuals.extend(delta[:3, 3])
        residuals.extend(
            Rotation.from_matrix(delta[:3, :3]).as_rotvec()
            * float(rotation_scale_m))
    return np.asarray(residuals, dtype=float)


def eye_to_hand_closure_metrics(base_T_link6, camera_T_marker,
                                base_T_camera, link6_T_marker):
    residuals = []
    for arm, observed in zip(base_T_link6, camera_T_marker):
        predicted = invert_transform(base_T_camera) @ arm @ link6_T_marker
        residuals.append(transform_deviation(predicted, observed))
    translation = np.asarray(
        [item["translation_m"] for item in residuals], dtype=float)
    rotation = np.asarray(
        [item["rotation_deg"] for item in residuals], dtype=float)
    return {
        "per_pose": residuals,
        "translation_median_m": float(np.median(translation)),
        "translation_p95_m": float(np.percentile(translation, 95)),
        "translation_maximum_m": float(np.max(translation)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p95_deg": float(np.percentile(rotation, 95)),
        "rotation_maximum_deg": float(np.max(rotation)),
    }


def solve_fixed_camera_eye_to_hand(base_T_link6, camera_T_marker):
    """Solve ``base_T_camera * camera_T_marker = base_T_link6 * link6_T_marker``."""
    arms = [validate_rigid_transform(value, "base_T_link6")
            for value in base_T_link6]
    observations = [validate_rigid_transform(value, "camera_T_marker")
                    for value in camera_T_marker]
    if len(arms) != len(observations) or len(arms) < 6:
        raise EyeToHandError("matching eye-to-hand observations (at least 6) required")
    # Identity tool-to-marker gives a deterministic camera initializer from
    # base_T_camera = base_T_link6 * inv(camera_T_marker).
    camera_initial = average_transforms([
        arm @ invert_transform(observed)
        for arm, observed in zip(arms, observations)])
    initial = np.r_[np.zeros(6), _parameters(camera_initial)]
    result = least_squares(
        _closure_residual, initial, args=(arms, observations),
        method="trf", loss="soft_l1", f_scale=0.002,
        max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if not result.success or not np.all(np.isfinite(result.x)):
        raise EyeToHandError("fixed-camera eye-to-hand optimization failed")
    link6_T_marker = _transform(result.x[:6])
    base_T_camera = _transform(result.x[6:])
    return {
        "base_T_camera": base_T_camera,
        "link6_T_marker": link6_T_marker,
        "closure": eye_to_hand_closure_metrics(
            arms, observations, base_T_camera, link6_T_marker),
        "optimizer": {
            "success": bool(result.success),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "function_evaluations": int(result.nfev),
        },
    }


def bootstrap_eye_to_hand(base_T_link6, camera_T_marker, samples=100,
                          seed=180100):
    reference = solve_fixed_camera_eye_to_hand(base_T_link6, camera_T_marker)
    rng = np.random.default_rng(seed)
    camera_errors = []
    marker_errors = []
    count = len(base_T_link6)
    for _index in range(int(samples)):
        selected = rng.integers(0, count, size=count)
        try:
            result = solve_fixed_camera_eye_to_hand(
                [base_T_link6[index] for index in selected],
                [camera_T_marker[index] for index in selected])
        except EyeToHandError:
            continue
        camera_errors.append(transform_deviation(
            reference["base_T_camera"], result["base_T_camera"]))
        marker_errors.append(transform_deviation(
            reference["link6_T_marker"], result["link6_T_marker"]))
    if len(camera_errors) < max(10, int(samples) // 2):
        raise EyeToHandError("insufficient successful eye-to-hand bootstrap solves")

    def spread(values):
        return {
            "translation_std_m": float(np.std(
                [item["translation_m"] for item in values], ddof=1)),
            "rotation_std_deg": float(np.std(
                [item["rotation_deg"] for item in values], ddof=1)),
        }
    return {
        "requested_samples": int(samples),
        "successful_samples": len(camera_errors),
        "base_T_camera": spread(camera_errors),
        "link6_T_marker": spread(marker_errors),
    }


def aruco_camera_pose(corners_px, camera_info, marker_side_m, depth_plane=None):
    """Recover camera_T_marker, selecting IPPE ambiguity with saved RGB-D evidence."""
    corners = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    object_points = _marker_object_points(marker_side_m)
    camera = np.asarray(camera_info["K"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(camera_info.get("D", []), dtype=np.float64)
    output = cv2.solvePnPGeneric(
        object_points, corners, camera, distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not output[0]:
        raise EyeToHandError("ArUco IPPE pose solve failed")
    candidates = []
    depth_centroid = depth_normal = None
    if depth_plane:
        depth_centroid = np.asarray(
            depth_plane["centroid_camera_m"], dtype=float).reshape(3)
        depth_normal = np.asarray(
            depth_plane["normal_camera"], dtype=float).reshape(3)
        depth_normal /= np.linalg.norm(depth_normal)
    for rotation_vector, translation in zip(output[1], output[2]):
        matrix = np.eye(4)
        matrix[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
        matrix[:3, 3] = np.asarray(translation).reshape(3)
        if matrix[2, 3] <= 0.0:
            continue
        projected = cv2.projectPoints(
            object_points, rotation_vector, translation,
            camera, distortion)[0].reshape(4, 2)
        reprojection = float(np.sqrt(np.mean(np.square(projected - corners))))
        depth_error = 0.0
        normal_error = 0.0
        if depth_centroid is not None:
            depth_error = float(np.linalg.norm(
                matrix[:3, 3] - depth_centroid))
            normal_error = float(np.arccos(np.clip(abs(
                np.dot(matrix[:3, 2], depth_normal)), 0.0, 1.0)))
        # Depth is used to choose the geometrically consistent IPPE branch;
        # reprojection remains the primary image-space metric.
        score = reprojection + 1000.0 * depth_error + 10.0 * normal_error
        candidates.append((score, matrix, reprojection,
                           depth_error, np.degrees(normal_error)))
    if not candidates:
        raise EyeToHandError("ArUco pose has no positive-depth solution")
    candidates.sort(key=lambda item: item[0])
    selected = candidates[0]
    return {
        "camera_T_marker": validate_rigid_transform(
            selected[1], "camera_T_marker"),
        "reprojection_rmse_px": selected[2],
        "depth_centroid_error_m": selected[3],
        "depth_normal_error_deg": selected[4],
    }


def _sample_observation(dataset, sample, camera_name):
    camera = dataset["cameras"][camera_name]
    observed = sample["cameras"][camera_name]
    pose = aruco_camera_pose(
        observed["corners_px"], camera["camera_info"],
        dataset["marker"]["marker_side_m"],
        observed.get("depth_plane"))
    return pose


def _validation_projection(dataset, sample, camera_name, solved):
    camera = dataset["cameras"][camera_name]
    observed = sample["cameras"][camera_name]
    predicted = (invert_transform(solved["base_T_camera"])
                 @ np.asarray(sample["base_T_link6"], dtype=float)
                 @ solved["link6_T_marker"])
    rotation_vector = cv2.Rodrigues(predicted[:3, :3])[0]
    camera_matrix = np.asarray(
        camera["camera_info"]["K"], dtype=float).reshape(3, 3)
    distortion = np.asarray(camera["camera_info"].get("D", []), dtype=float)
    projected = cv2.projectPoints(
        _marker_object_points(dataset["marker"]["marker_side_m"]),
        rotation_vector, predicted[:3, 3], camera_matrix,
        distortion)[0].reshape(4, 2)
    corners = np.asarray(observed["corners_px"], dtype=float).reshape(4, 2)
    result = {
        "reprojection_rmse_px": float(np.sqrt(
            np.mean(np.square(projected - corners))))
    }
    plane = observed.get("depth_plane")
    if plane:
        centroid = np.asarray(plane["centroid_camera_m"], dtype=float)
        normal = np.asarray(plane["normal_camera"], dtype=float)
        normal /= np.linalg.norm(normal)
        predicted_normal = predicted[:3, 2]
        result["rgbd_plane_centroid_error_m"] = float(np.linalg.norm(
            predicted[:3, 3] - centroid))
        result["rgbd_plane_normal_error_deg"] = float(np.degrees(np.arccos(
            np.clip(abs(np.dot(predicted_normal, normal)), 0.0, 1.0))))
        points = np.asarray(plane.get("points_camera_m", []), dtype=float)
        if points.ndim == 2 and points.shape[1:] == (3,) and len(points):
            distances = ((points - predicted[:3, 3])
                         @ predicted_normal)
            result["rgbd_plane_rmse_m"] = float(np.sqrt(
                np.mean(np.square(distances))))
    return result


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _require_distinct(samples, split):
    matrices = [np.asarray(item["base_T_link6"], dtype=float) for item in samples]
    for index, first in enumerate(matrices):
        for second in matrices[index + 1:]:
            if np.allclose(first, second, atol=1e-9, rtol=0.0):
                raise EyeToHandError("{} split contains duplicate robot poses".format(split))


def solve_dual_dataset(dataset_path):
    path = Path(dataset_path).expanduser().resolve()
    try:
        dataset = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EyeToHandError("cannot load calibration dataset") from exc
    if (not isinstance(dataset, dict) or dataset.get("schema_version") != 2
            or not bool(dataset.get("complete", False))):
        raise EyeToHandError("complete hybrid RGB-D schema-v2 dataset required")
    if set(dataset.get("cameras", {})) != set(CAMERAS):
        raise EyeToHandError("dataset must contain rs1 and rs3")
    diversity = dataset.get("pose_diversity")
    required_diversity = (
        "optimization_translation_axis_span_m",
        "optimization_pair_rotation_deg",
        "validation_translation_axis_span_m",
        "validation_pair_rotation_deg",
    )
    if (not isinstance(diversity, dict)
            or any(key not in diversity for key in required_diversity)):
        raise EyeToHandError("dataset pose-diversity requirements are missing")
    samples = dataset.get("samples", [])
    results = {}
    per_camera_validation = {}
    diversity_metrics = {}
    for name in CAMERAS:
        optimization = [item for item in samples
                        if item.get("split") == "optimization"
                        and name in item.get("cameras", {})]
        validation = [item for item in samples
                      if item.get("split") == "validation"
                      and name in item.get("cameras", {})]
        if len(optimization) < 18 or len(validation) < 6:
            raise EyeToHandError(
                "{} requires at least 18 optimization and 6 validation poses".format(name))
        _require_distinct(optimization, "optimization")
        _require_distinct(validation, "validation")
        for first in optimization:
            for second in validation:
                if np.allclose(first["base_T_link6"], second["base_T_link6"],
                               atol=1e-9, rtol=0.0):
                    raise EyeToHandError(
                        "optimization and validation reuse a robot pose")
        optimization_diversity = check_pose_diversity(
            [item["base_T_link6"] for item in optimization],
            diversity["optimization_translation_axis_span_m"],
            diversity["optimization_pair_rotation_deg"], 18)
        validation_diversity = check_pose_diversity(
            [item["base_T_link6"] for item in validation],
            diversity["validation_translation_axis_span_m"],
            diversity["validation_pair_rotation_deg"], 6)
        diversity_metrics[name] = {
            "optimization": optimization_diversity,
            "validation": validation_diversity,
        }
        observations = [_sample_observation(dataset, item, name)
                        for item in optimization]
        solved = solve_fixed_camera_eye_to_hand(
            [item["base_T_link6"] for item in optimization],
            [item["camera_T_marker"] for item in observations])
        uncertainty = bootstrap_eye_to_hand(
            [item["base_T_link6"] for item in optimization],
            [item["camera_T_marker"] for item in observations])
        validation_observations = [_sample_observation(dataset, item, name)
                                   for item in validation]
        validation_closure = eye_to_hand_closure_metrics(
            [item["base_T_link6"] for item in validation],
            [item["camera_T_marker"] for item in validation_observations],
            solved["base_T_camera"], solved["link6_T_marker"])
        link_T_optical = np.asarray(
            dataset["cameras"][name]["link_T_optical"], dtype=float)
        base_T_link = solved["base_T_camera"] @ invert_transform(
            link_T_optical, "link_T_optical")
        validation_projections = [
            _validation_projection(dataset, item, name, solved)
            for item in validation]
        reprojection = [item["reprojection_rmse_px"]
                        for item in validation_projections]
        per_camera_validation[name] = {
            "sample_ids": [int(item["id"]) for item in validation],
            "camera_T_marker": [item["camera_T_marker"]
                                for item in validation_observations],
            "reprojection_rmse_px": reprojection,
            "rgbd_plane": validation_projections,
            "closure": validation_closure,
        }
        results[name] = {
            "base_T_camera": solved["base_T_camera"],
            "base_T_link": base_T_link,
            "link6_T_marker": solved["link6_T_marker"],
            "optimization_closure": solved["closure"],
            "validation_closure": validation_closure,
            "validation_reprojection_rmse_px": float(np.sqrt(
                np.mean(np.square(reprojection)))),
            "validation_rgbd_plane": validation_projections,
            "optimizer": solved["optimizer"],
            "bootstrap_uncertainty": uncertainty,
        }
    paired_distances = []
    validation_by_id = {
        name: dict(zip(per_camera_validation[name]["sample_ids"],
                       per_camera_validation[name]["camera_T_marker"]))
        for name in CAMERAS
    }
    shared = sorted(set(validation_by_id["rs1"]) & set(validation_by_id["rs3"]))
    if len(shared) < 6:
        raise EyeToHandError("six paired dual-camera validation poses required")
    for sample_id in shared:
        marker_points = [
            results[name]["base_T_camera"] @
            np.r_[validation_by_id[name][sample_id][:3, 3], 1.0]
            for name in CAMERAS]
        paired_distances.append(float(np.linalg.norm(
            marker_points[0][:3] - marker_points[1][:3])))
    report = {
        "schema_version": 1,
        "status": "PENDING_REVIEW",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(path),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "camera_serials": {
            name: str(dataset["cameras"][name]["serial"]).lstrip("_")
            for name in CAMERAS},
        "results": results,
        "validation": {
            "paired_sample_ids": shared,
            "dual_camera_distances_m": paired_distances,
            "dual_camera_3d_median_m": float(np.median(paired_distances)),
            "dual_camera_3d_p95_m": float(np.percentile(paired_distances, 95)),
            "reprojection_rmse_px": float(max(
                results[name]["validation_reprojection_rmse_px"]
                for name in CAMERAS)),
            "checks": {
                "minimum_optimization_poses_per_camera": 18,
                "minimum_validation_poses_per_camera": 6,
                "duplicates_absent": True,
                "splits_disjoint": True,
                "pose_diversity_passed": True,
                "pose_diversity_metrics": diversity_metrics,
            },
        },
    }
    return _plain(report)


def approve_dual_report(report, operator):
    if report.get("schema_version") != 1 or report.get("status") != "PENDING_REVIEW":
        raise EyeToHandError("pending dual-camera report required")
    validation = report.get("validation", {})
    failures = []
    if float(validation.get("reprojection_rmse_px", np.inf)) > 0.8:
        failures.append("reprojection_rmse_px")
    if float(validation.get("dual_camera_3d_median_m", np.inf)) > 0.005:
        failures.append("dual_camera_3d_median_m")
    if float(validation.get("dual_camera_3d_p95_m", np.inf)) > 0.008:
        failures.append("dual_camera_3d_p95_m")
    if len(validation.get("paired_sample_ids", [])) < 6:
        failures.append("validation_pose_count")
    checks = validation.get("checks", {})
    if (int(checks.get("minimum_optimization_poses_per_camera", 0)) < 18
            or int(checks.get("minimum_validation_poses_per_camera", 0)) < 6
            or not bool(checks.get("duplicates_absent", False))
            or not bool(checks.get("splits_disjoint", False))
            or not bool(checks.get("pose_diversity_passed", False))):
        failures.append("quantity_or_pose_diversity")
    if failures:
        raise EyeToHandError("camera report failed: " + ",".join(failures))
    dataset_path = Path(str(report.get("dataset_path", ""))).expanduser()
    try:
        current_dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise EyeToHandError("calibration dataset is unavailable at approval") from exc
    if current_dataset_hash != report.get("dataset_sha256"):
        raise EyeToHandError("calibration dataset changed after solving")
    serials = report.get("camera_serials", {})
    if (not isinstance(serials, dict)
            or any(not str(serials.get(name, "")).lstrip("_") for name in CAMERAS)):
        raise EyeToHandError("camera serials are missing from calibration report")
    name = str(operator).strip()
    if not name:
        raise EyeToHandError("operator name is required")
    digest = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    extrinsics = {}
    for camera_name in CAMERAS:
        result = report["results"][camera_name]
        matrix = np.asarray(result["base_T_link"], dtype=float)
        quaternion = matrix_to_quaternion_xyzw(matrix)
        extrinsics[camera_name] = {
            "schema_version": 1,
            "status": "ACCEPTED",
            "publish_tf_allowed": True,
            "precision_operation_allowed": True,
            "base_absolute_accuracy": "VERIFIED",
            "logical_name": camera_name,
            "serial": str(serials[camera_name]).lstrip("_"),
            "role": "fixed_eye_to_hand",
            "parent_frame": "base_link",
            "child_frame": camera_name + "_link",
            "base_T_link": matrix.tolist(),
            "translation_m": dict(zip(("x", "y", "z"), matrix[:3, 3].tolist())),
            "rotation_xyzw": dict(zip(("x", "y", "z", "w"), quaternion.tolist())),
            "validation_report_sha256": digest,
            "dataset_sha256": report["dataset_sha256"],
            "approval": {
                "operator": name,
                "approved_utc": datetime.now(timezone.utc).isoformat(),
            },
        }
    return extrinsics
