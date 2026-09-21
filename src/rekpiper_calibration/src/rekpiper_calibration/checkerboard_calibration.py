"""Pure geometry for known-target dual-camera checkerboard calibration.

All matrices follow ``A_T_B``: points expressed in frame B are transformed
into frame A.  OpenCV PnP returns ``camera_optical_T_checkerboard``.
"""

from dataclasses import dataclass
from itertools import combinations
import math

import cv2
import numpy as np


class CalibrationFailure(RuntimeError):
    """A fail-closed target detection or geometry error."""


@dataclass(frozen=True)
class TargetDefinition:
    pattern_cols: int
    pattern_rows: int
    square_size_m: float
    marker_dictionary: str
    marker_id: int
    marker_size_m: float
    marker_min_x_m: float
    marker_min_y_m: float

    @classmethod
    def from_dict(cls, values):
        marker = values.get("orientation_marker", {})
        result = cls(
            pattern_cols=int(values.get("pattern_cols", 0)),
            pattern_rows=int(values.get("pattern_rows", 0)),
            square_size_m=float(values.get("square_size_m", 0.0)),
            marker_dictionary=str(marker.get("dictionary", "DICT_5X5_100")),
            marker_id=int(marker.get("marker_id", 18)),
            marker_size_m=float(marker.get("marker_size_m", 0.100)),
            marker_min_x_m=float(marker.get("expected_min_x_m", 0.250)),
            marker_min_y_m=float(marker.get("expected_min_y_m", 0.250)),
        )
        result.validate()
        return result

    def validate(self):
        if self.pattern_cols < 2 or self.pattern_rows < 2:
            raise ValueError("checkerboard needs at least 2x2 inner corners")
        if not math.isfinite(self.square_size_m) or self.square_size_m <= 0.0:
            raise ValueError("square_size_m must be positive")
        if not hasattr(cv2.aruco, self.marker_dictionary):
            raise ValueError("unsupported ArUco dictionary: " + self.marker_dictionary)


@dataclass(frozen=True)
class QualityThresholds:
    minimum_board_area_ratio: float = 0.15
    minimum_median_cell_px: float = 20.0
    minimum_cell_px: float = 15.0
    image_margin_px: float = 30.0
    minimum_laplacian_variance: float = 50.0
    maximum_saturation_ratio: float = 0.20
    maximum_reprojection_rmse_px: float = 0.5
    maximum_corner_reprojection_px: float = 1.0
    maximum_ippe_best_to_second_ratio: float = 0.80

    @classmethod
    def from_dict(cls, values):
        defaults = cls()
        return cls(**{
            name: float(values.get(name, getattr(defaults, name)))
            for name in defaults.__dataclass_fields__
        })


def checkerboard_object_points(target):
    points = np.zeros((target.pattern_rows * target.pattern_cols, 3), np.float64)
    grid = np.mgrid[0:target.pattern_cols, 0:target.pattern_rows].T.reshape(-1, 2)
    points[:, :2] = grid * target.square_size_m
    return points


def transform_matrix(rotation, translation):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


def validate_rigid_transform(matrix, name="transform"):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(name + " must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(name + " rotation is not orthonormal")
    if abs(np.linalg.det(rotation) - 1.0) > 1e-6:
        raise ValueError(name + " rotation determinant is not one")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(name + " homogeneous row is invalid")
    return matrix


def base_target_from_config(values):
    rotation = np.asarray(values.get("base_T_target_rotation", []), dtype=np.float64)
    translation = np.asarray(
        values.get("base_T_target_translation_m", []), dtype=np.float64)
    return validate_rigid_transform(transform_matrix(rotation, translation), "base_T_target")


def matrix_to_quaternion_xyzw(matrix):
    """Convert a validated rotation matrix to a normalized xyzw quaternion."""
    rotation = validate_rigid_transform(matrix)[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            (rotation[2, 1] - rotation[1, 2]) / s,
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[1, 0] - rotation[0, 1]) / s,
            0.25 * s,
        ])
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.array([0.25 * s,
                                   (rotation[0, 1] + rotation[1, 0]) / s,
                                   (rotation[0, 2] + rotation[2, 0]) / s,
                                   (rotation[2, 1] - rotation[1, 2]) / s])
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.array([(rotation[0, 1] + rotation[1, 0]) / s,
                                   0.25 * s,
                                   (rotation[1, 2] + rotation[2, 1]) / s,
                                   (rotation[0, 2] - rotation[2, 0]) / s])
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.array([(rotation[0, 2] + rotation[2, 0]) / s,
                                   (rotation[1, 2] + rotation[2, 1]) / s,
                                   0.25 * s,
                                   (rotation[1, 0] - rotation[0, 1]) / s])
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion conversion failed")
    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def rotation_distance_deg(first, second):
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def compose_base_to_link(base_T_target, camera_T_target, link_T_camera):
    """Compute base_T_link without mixing link and optical camera frames."""
    base_T_target = validate_rigid_transform(base_T_target, "base_T_target")
    camera_T_target = validate_rigid_transform(camera_T_target, "camera_T_target")
    link_T_camera = validate_rigid_transform(link_T_camera, "link_T_camera")
    base_T_camera = base_T_target @ np.linalg.inv(camera_T_target)
    base_T_link = base_T_camera @ np.linalg.inv(link_T_camera)
    return validate_rigid_transform(base_T_link, "base_T_link"), base_T_camera


def _aruco_detector(target):
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, target.marker_dictionary))
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def _detect_marker(image, target):
    corners, ids, _ = _aruco_detector(target).detectMarkers(image)
    detected = [] if ids is None else ids.reshape(-1).tolist()
    matches = [i for i, marker_id in enumerate(detected) if marker_id == target.marker_id]
    if len(matches) != 1:
        raise CalibrationFailure(
            "orientation_marker_count_{}_detected_ids_{}".format(
                len(matches), detected))
    marker_corners = np.asarray(corners[matches[0]], dtype=np.float64).reshape(4, 2)
    return marker_corners, detected


def canonicalize_corners(corners, marker_center, target):
    """Resolve the checkerboard's remaining 180-degree ambiguity with ID 18."""
    image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    board_xy = checkerboard_object_points(target)[:, :2].astype(np.float64)
    valid = []
    tested = []
    for reversed_order in (False, True):
        candidate = image_points[::-1].copy() if reversed_order else image_points.copy()
        homography, _ = cv2.findHomography(candidate, board_xy, method=0)
        if homography is None:
            tested.append((reversed_order, None))
            continue
        board_marker = cv2.perspectiveTransform(
            np.asarray(marker_center, dtype=np.float64).reshape(1, 1, 2), homography
        ).reshape(2)
        tested.append((reversed_order, board_marker.copy()))
        if (board_marker[0] > target.marker_min_x_m
                and board_marker[1] > target.marker_min_y_m):
            valid.append((candidate, board_marker, reversed_order))
    if len(valid) != 1:
        details = []
        for reversed_order, board_marker in tested:
            label = "reversed" if reversed_order else "normal"
            if board_marker is None:
                details.append(label + "=homography_failed")
            else:
                details.append("{}=({:.4f},{:.4f})m".format(
                    label, float(board_marker[0]), float(board_marker[1])))
        raise CalibrationFailure(
            "checkerboard_orientation_ambiguous:{};expected_x>{:.4f},y>{:.4f}".format(
                ",".join(details), target.marker_min_x_m,
                target.marker_min_y_m))
    return valid[0]


def _cell_lengths(corners, target):
    grid = np.asarray(corners, dtype=np.float64).reshape(
        target.pattern_rows, target.pattern_cols, 2)
    horizontal = np.linalg.norm(grid[:, 1:] - grid[:, :-1], axis=2).reshape(-1)
    vertical = np.linalg.norm(grid[1:] - grid[:-1], axis=2).reshape(-1)
    return np.concatenate((horizontal, vertical))


def image_quality(image, corners, target):
    height, width = image.shape[:2]
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    hull = cv2.convexHull(points.astype(np.float32))
    area_ratio = float(cv2.contourArea(hull) / float(width * height))
    lengths = _cell_lengths(points, target)
    border = np.min(np.column_stack((points[:, 0], points[:, 1],
                                     width - 1 - points[:, 0],
                                     height - 1 - points[:, 1])))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # A checkerboard is intentionally dominated by clipped black/white pixels;
    # using grayscale clipping would reject every good target.  HSV chroma
    # saturation instead catches strong colored glare without penalizing the
    # printed pattern.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1] >= 250))
    return {
        "board_area_ratio": area_ratio,
        "minimum_cell_px": float(np.min(lengths)),
        "median_cell_px": float(np.median(lengths)),
        "minimum_border_px": float(border),
        "laplacian_variance": laplacian,
        "saturation_ratio": saturation,
    }


def _quality_reasons(metrics, thresholds):
    checks = (
        (metrics["board_area_ratio"] < thresholds.minimum_board_area_ratio,
         "board_area_too_small"),
        (metrics["median_cell_px"] < thresholds.minimum_median_cell_px,
         "median_cell_too_small"),
        (metrics["minimum_cell_px"] < thresholds.minimum_cell_px,
         "cell_too_small"),
        (metrics["minimum_border_px"] < thresholds.image_margin_px,
         "checkerboard_near_image_edge"),
        (metrics["laplacian_variance"] < thresholds.minimum_laplacian_variance,
         "image_blurred"),
        (metrics["saturation_ratio"] > thresholds.maximum_saturation_ratio,
         "image_saturated"),
    )
    return [reason for failed, reason in checks if failed]


def _pose_distance(first, second):
    translation = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    rotation = rotation_distance_deg(first, second)
    return translation, rotation


def solve_checkerboard_pose(image, camera_matrix, distortion, target, thresholds):
    """Return one accepted camera_optical_T_checkerboard observation."""
    if image is None or image.ndim != 3:
        raise CalibrationFailure("invalid_bgr_image")
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(camera_matrix)) or camera_matrix[0, 0] <= 0.0:
        raise CalibrationFailure("invalid_camera_info")

    marker_corners, detected_ids = _detect_marker(image, target)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
             | cv2.CALIB_CB_ACCURACY)
    found, corners = cv2.findChessboardCornersSB(
        gray, (target.pattern_cols, target.pattern_rows), flags=flags)
    if not found or corners is None or len(corners) != target.pattern_cols * target.pattern_rows:
        raise CalibrationFailure("checkerboard_not_detected")
    corners, marker_board_xy, reversed_order = canonicalize_corners(
        corners, np.mean(marker_corners, axis=0), target)

    metrics = image_quality(image, corners, target)
    metrics.update({
        "detected_ids": detected_ids,
        "orientation_reversed": bool(reversed_order),
        "marker_board_x_m": float(marker_board_xy[0]),
        "marker_board_y_m": float(marker_board_xy[1]),
    })
    reasons = _quality_reasons(metrics, thresholds)
    if reasons:
        raise CalibrationFailure(
            "{}:area={:.4f},min_cell={:.2f}px,median_cell={:.2f}px,"
            "border={:.2f}px,blur={:.1f},saturation={:.4f}".format(
                ",".join(reasons), metrics["board_area_ratio"],
                metrics["minimum_cell_px"], metrics["median_cell_px"],
                metrics["minimum_border_px"], metrics["laplacian_variance"],
                metrics["saturation_ratio"]))

    object_points = checkerboard_object_points(target)
    output = cv2.solvePnPGeneric(
        object_points, corners, camera_matrix, distortion,
        flags=cv2.SOLVEPNP_IPPE)
    if not output or not output[0]:
        raise CalibrationFailure("ippe_failed")
    rvecs, tvecs = output[1], output[2]
    candidates = []
    for rvec, tvec in zip(rvecs, tvecs):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        refined = cv2.solvePnPRefineLM(
            object_points, corners, camera_matrix, distortion, rvec, tvec)
        if refined is not None:
            rvec, tvec = refined
        rotation, _ = cv2.Rodrigues(rvec)
        matrix = transform_matrix(rotation, tvec)
        camera_points = (rotation @ object_points.T + tvec).T
        if (not np.all(np.isfinite(matrix)) or np.any(camera_points[:, 2] <= 0.0)
                or abs(np.linalg.det(rotation) - 1.0) > 1e-6):
            continue
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion)
        errors = np.linalg.norm(projected.reshape(-1, 2) - corners, axis=1)
        candidates.append({
            "matrix": validate_rigid_transform(matrix, "camera_T_target"),
            "rvec": rvec,
            "tvec": tvec,
            "rmse": float(np.sqrt(np.mean(errors * errors))),
            "max_error": float(np.max(errors)),
            "errors": errors,
        })
    if not candidates:
        raise CalibrationFailure("no_positive_depth_ippe_solution")
    candidates.sort(key=lambda value: value["rmse"])
    best = candidates[0]
    if best["rmse"] > thresholds.maximum_reprojection_rmse_px:
        raise CalibrationFailure(
            "reprojection_rmse_exceeded:rmse={:.4f}px,limit={:.4f}px,"
            "max_corner={:.4f}px".format(
                best["rmse"], thresholds.maximum_reprojection_rmse_px,
                best["max_error"]))
    if best["max_error"] > thresholds.maximum_corner_reprojection_px:
        worst_index = int(np.argmax(best["errors"]))
        worst_row, worst_col = divmod(worst_index, target.pattern_cols)
        worst_pixel = np.asarray(corners[worst_index]).reshape(2)
        raise CalibrationFailure(
            "corner_reprojection_exceeded:max_corner={:.4f}px,limit={:.4f}px,"
            "rmse={:.4f}px,worst_corner=(row={},col={}),pixel=({:.2f},{:.2f})".format(
                best["max_error"], thresholds.maximum_corner_reprojection_px,
                best["rmse"], worst_row, worst_col,
                float(worst_pixel[0]), float(worst_pixel[1])))
    if len(candidates) > 1:
        translation_delta, rotation_delta = _pose_distance(
            best["matrix"], candidates[1]["matrix"])
        duplicate = translation_delta < 1e-5 and rotation_delta < 1e-3
        ratio = best["rmse"] / max(candidates[1]["rmse"], 1e-12)
        metrics["ippe_best_to_second_ratio"] = float(ratio)
        if not duplicate and ratio > thresholds.maximum_ippe_best_to_second_ratio:
            raise CalibrationFailure(
                "ippe_pose_ambiguous:ratio={:.4f},limit={:.4f},"
                "translation_delta={:.6f}m,rotation_delta={:.4f}deg".format(
                    ratio, thresholds.maximum_ippe_best_to_second_ratio,
                    translation_delta, rotation_delta))

    metrics.update({
        "reprojection_rmse_px": best["rmse"],
        "maximum_reprojection_px": best["max_error"],
        "distance_m": float(np.linalg.norm(best["tvec"])),
    })
    annotated = image.copy()
    cv2.drawChessboardCorners(
        annotated, (target.pattern_cols, target.pattern_rows),
        corners.reshape(-1, 1, 2).astype(np.float32), True)
    cv2.aruco.drawDetectedMarkers(
        annotated, [marker_corners.reshape(1, 4, 2).astype(np.float32)],
        np.asarray([[target.marker_id]], dtype=np.int32))
    cv2.drawFrameAxes(annotated, camera_matrix, distortion,
                      best["rvec"], best["tvec"], target.square_size_m * 3.0)
    return {
        "camera_T_target": best["matrix"],
        "corners": corners,
        "object_points": object_points,
        # Keep the residual of every inner corner.  A single PnP pose is
        # fitted jointly from every point in ``object_points``; callers that
        # persist a calibration can therefore audit all 11 x 8 corners rather
        # than treating the board origin (p00) as a separate measurement.
        "corner_reprojection_errors_px": best["errors"],
        "metrics": metrics,
        "annotated": annotated,
    }


def depth_geometry_metrics(depth_raw, camera_matrix, distortion, depth_scale,
                           corners, object_points, base_T_camera, base_T_target):
    """Validate the RGB-only pose with independent aligned-depth geometry."""
    depth = np.asarray(depth_raw)
    if depth.ndim != 2 or depth.shape[0] <= 0 or depth.shape[1] <= 0:
        raise CalibrationFailure("invalid_aligned_depth")
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(corners.astype(np.int32)), 255)
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        raise CalibrationFailure("insufficient_board_depth_area")
    selection = np.arange(0, len(xs), max(1, len(xs) // 5000))
    pixels = np.column_stack((xs[selection], ys[selection])).astype(np.float64)
    raw = depth[ys[selection], xs[selection]].astype(np.float64)
    z = raw * float(depth_scale)
    valid = np.isfinite(z) & (z >= 0.10) & (z <= 3.0)
    if np.count_nonzero(valid) < 100:
        raise CalibrationFailure("insufficient_valid_board_depth")
    pixels, z = pixels[valid], z[valid]
    normalized = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2), camera_matrix, distortion).reshape(-1, 2)
    points_camera = np.column_stack((normalized[:, 0] * z,
                                     normalized[:, 1] * z, z, np.ones_like(z)))
    points_base = (base_T_camera @ points_camera.T).T[:, :3]
    plane_origin = base_T_target[:3, 3]
    plane_normal = base_T_target[:3, 2]
    distances = np.abs((points_base - plane_origin) @ plane_normal)
    cutoff = np.percentile(distances, 90.0)
    inliers = distances <= cutoff
    plane_rmse = float(np.sqrt(np.mean(distances[inliers] ** 2)))

    corner_errors = []
    height, width = depth.shape
    for pixel, expected_target in zip(corners, object_points):
        u, v = int(round(pixel[0])), int(round(pixel[1]))
        x0, x1 = max(0, u - 2), min(width, u + 3)
        y0, y1 = max(0, v - 2), min(height, v + 3)
        values = depth[y0:y1, x0:x1].astype(np.float64) * float(depth_scale)
        values = values[np.isfinite(values) & (values >= 0.10) & (values <= 3.0)]
        if values.size == 0:
            continue
        corner_z = float(np.median(values))
        norm = cv2.undistortPoints(
            np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2),
            camera_matrix, distortion).reshape(2)
        camera_point = np.array([norm[0] * corner_z, norm[1] * corner_z,
                                 corner_z, 1.0])
        measured_base = (base_T_camera @ camera_point)[:3]
        expected_base = (base_T_target @ np.r_[expected_target, 1.0])[:3]
        corner_errors.append(float(np.linalg.norm(measured_base - expected_base)))
    if len(corner_errors) < max(8, len(corners) // 4):
        raise CalibrationFailure("insufficient_corner_depth")
    return {
        "depth_plane_rmse_m": plane_rmse,
        "known_corner_max_error_m": float(np.max(corner_errors)),
        "known_corner_p95_error_m": float(np.percentile(corner_errors, 95.0)),
        "known_corner_median_error_m": float(np.median(corner_errors)),
        "valid_corner_depth_count": len(corner_errors),
        "valid_plane_depth_count": int(np.count_nonzero(inliers)),
    }


def repeatability_metrics(candidates, camera_name):
    translations, rotations = [], []
    for first, second in combinations(candidates, 2):
        first_matrix = np.asarray(first["cameras"][camera_name]["base_T_link"])
        second_matrix = np.asarray(second["cameras"][camera_name]["base_T_link"])
        translation, rotation = _pose_distance(first_matrix, second_matrix)
        translations.append(translation)
        rotations.append(rotation)
    return {
        "maximum_translation_m": max(translations) if translations else 0.0,
        "maximum_rotation_deg": max(rotations) if rotations else 0.0,
    }
