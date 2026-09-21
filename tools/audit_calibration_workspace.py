#!/usr/bin/env python3
"""Audit preview landmarks against a user-reported base-frame workspace."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import yaml

from rekpiper_calibration.robot_world_handeye import average_transforms
from rekpiper_calibration.trajectory_preview import playback_samples, validate_preview
from rekpiper_calibration.workspace_bounds import (
    board_corners_in_marker, point_envelope, workspace_limits)
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan_path, workspace_path = Path(args.plan), Path(args.workspace)
    plan = yaml.safe_load(plan_path.read_text())
    workspace = yaml.safe_load(workspace_path.read_text())
    lower, upper = workspace_limits(workspace)
    urdf = Path(__file__).resolve().parents[1] / "src/piper_description/urdf/piper_description.urdf"
    xml = urdf.read_text()
    if hashlib.sha256(xml.encode()).hexdigest() != plan["urdf_sha256"]:
        raise ValueError("preview URDF hash mismatch")
    link6_solver = PiperURDFIKSolver.from_urdf_xml(xml, "base_link", "link6", plan["joint_names"])
    joints = validate_preview(plan, link6_solver)
    solver = PiperURDFIKSolver.from_urdf_xml(xml, "base_link", "rekep_tcp", plan["joint_names"])
    marker_transforms = []
    marker_side = []
    for source in plan["sources"]:
        path = Path(source["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("calibration source hash mismatch")
        result = yaml.safe_load(path.read_text())
        marker_transforms.append(result["calibration"]["tool_to_target_mean"]["matrix_4x4"])
        marker_side.append(float(result["target"]["marker_size_m"]))
    if len(marker_transforms) != 2 or not np.allclose(marker_side, 0.1):
        raise ValueError("two archived 100 mm marker transforms required")
    marker = average_transforms(marker_transforms)
    corners = np.array([[-.05, .05, 0], [.05, .05, 0], [.05, -.05, 0], [-.05, -.05, 0], [0, 0, 0]])
    plate = board_corners_in_marker(workspace["board_and_mount"])
    plate_link6 = plate @ marker[:3, :3].T + marker[:3, 3]
    reported_radius = float(workspace["board_and_mount"]["user_link6_to_farthest_board_point_m"])
    predicted_radius = float(np.max(np.linalg.norm(plate_link6, axis=1)))

    def landmarks(q):
        transforms = solver.link_transforms(q)
        board = transforms["link6"] @ marker
        return np.vstack((transforms["link6"][:3, 3], transforms["rekep_tcp"][:3, 3],
                          corners @ board[:3, :3].T + board[:3, 3],
                          plate @ board[:3, :3].T + board[:3, 3]))

    path_points, violations = [], []
    for time_s, target_index, q in playback_samples(joints):
        points = landmarks(q)
        result = point_envelope(points, lower, upper)
        path_points.append(points)
        if not result["sampled_points_within_bounds"]:
            violations.append({"time_s": time_s, "target_pose_id": target_index+1,
                               "outside_landmark_indices": result["outside_point_indices"]})
    report = {
        "schema_version": 1, "status": "PARTIAL_GEOMETRY_CHECK_ONLY",
        "hardware_execution_allowed": False,
        "plan": str(plan_path.resolve()), "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "workspace": str(workspace_path.resolve()),
        "workspace_sha256": hashlib.sha256(workspace_path.read_bytes()).hexdigest(),
        "landmarks": ["link6_origin", "rekep_tcp_origin", "marker_corner_0", "marker_corner_1",
                      "marker_corner_2", "marker_corner_3", "marker_center"] +
                      ["board_corner_{}".format(i) for i in range(8)],
        "marker_geometry": "100 mm square; mean transform from two external calibration candidates",
        "link6_T_marker_candidate_mean": marker.tolist(),
        "board_corners_marker_m": plate.tolist(),
        "geometry_consistency": {
            "selected_basis": workspace["board_and_mount"].get("selected_transform_basis", "unresolved"),
            "reported_farthest_board_distance_m": reported_radius,
            "candidate_marker_center_distance_m": float(np.linalg.norm(marker[:3, 3])),
            "candidate_farthest_board_distance_m": predicted_radius,
            "candidate_exceeds_reported_envelope": bool(predicted_radius > reported_radius),
            "status": ("USER_SELECTED_OLD_CALIBRATION_CANDIDATE"
                       if workspace["board_and_mount"].get("selected_transform_basis") == "old_calibration_mean"
                       else "CONFLICT_REQUIRES_FRAME_OR_DISTANCE_RECHECK"
                       if predicted_radius > reported_radius else "UNVERIFIED"),
        },
        "point_checks": {
            "target_poses": point_envelope(np.vstack([landmarks(q) for q in joints]), lower, upper),
            "sampled_path": point_envelope(np.vstack(path_points), lower, upper),
        },
        "trajectory_sample_count": len(path_points), "violating_samples": violations,
        "unchecked": ["physical_base_axis_confirmation", "board_mount_transform", "mount_shape", "full_arm_geometry",
                      "self_collision", "objects_and_cables", "continuous_swept_volume",
                      "dual_camera_visibility", "live_start_connection", "controller_execution"],
        "inspected_images": workspace.get("inspected_images", []),
    }
    with Path(args.output).open("x", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, sort_keys=False, allow_unicode=True)
    print(yaml.safe_dump(report["point_checks"]["sampled_path"], sort_keys=False))
    print("PARTIAL_GEOMETRY_CHECK_ONLY:", args.output)


if __name__ == "__main__":
    main()
