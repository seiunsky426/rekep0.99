#!/usr/bin/env python3

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from rekpiper_calibration.eye_to_hand import (
    EyeToHandError, approve_dual_report, aruco_camera_pose,
    _validation_projection, eye_to_hand_closure_metrics,
    solve_fixed_camera_eye_to_hand)
from rekpiper_calibration.robot_world_handeye import transform_deviation


def transform(rotation_vector, translation):
    value = np.eye(4)
    value[:3, :3] = cv2.Rodrigues(
        np.asarray(rotation_vector, dtype=float).reshape(3, 1))[0]
    value[:3, 3] = translation
    return value


class EyeToHandTest(unittest.TestCase):
    def test_recovers_fixed_camera_ax_equals_yb(self):
        base_T_camera = transform([0.25, -0.12, 0.08], [0.35, -0.42, 0.72])
        link6_T_marker = transform([-0.10, 0.20, 0.05], [0.02, 0.01, 0.11])
        arms = [
            transform([0.10, 0.20, 0.10], [0.10, -0.18, 0.18]),
            transform([-0.20, 0.18, 0.35], [0.24, -0.04, 0.28]),
            transform([0.30, -0.16, 0.18], [0.39, 0.12, 0.22]),
            transform([-0.24, -0.30, 0.10], [0.18, 0.20, 0.36]),
            transform([0.12, 0.35, -0.25], [0.48, -0.10, 0.14]),
            transform([-0.32, 0.06, -0.28], [0.31, 0.17, 0.31]),
            transform([0.22, -0.24, 0.30], [0.20, -0.24, 0.42]),
            transform([-0.18, 0.31, -0.16], [0.44, 0.20, 0.26]),
        ]
        observations = [
            np.linalg.inv(base_T_camera) @ arm @ link6_T_marker
            for arm in arms]
        result = solve_fixed_camera_eye_to_hand(arms, observations)
        self.assertLess(transform_deviation(
            base_T_camera, result["base_T_camera"])["translation_m"], 1e-6)
        self.assertLess(transform_deviation(
            link6_T_marker, result["link6_T_marker"])["rotation_deg"], 1e-4)
        self.assertLess(result["closure"]["translation_maximum_m"], 1e-6)

    def test_aruco_pose_uses_metric_depth_to_select_branch(self):
        camera = np.asarray([[600.0, 0.0, 320.0],
                             [0.0, 600.0, 240.0],
                             [0.0, 0.0, 1.0]])
        side = 0.1
        half = side / 2.0
        points = np.asarray([[-half, half, 0], [half, half, 0],
                             [half, -half, 0], [-half, -half, 0]], dtype=float)
        rotation = np.asarray([0.12, -0.08, 0.03])
        translation = np.asarray([0.02, -0.01, 0.55])
        corners = cv2.projectPoints(
            points, rotation, translation, camera, np.zeros(5))[0]
        matrix = transform(rotation, translation)
        result = aruco_camera_pose(
            corners, {"K": camera.reshape(-1), "D": [0] * 5}, side,
            {"centroid_camera_m": translation,
             "normal_camera": matrix[:3, 2]})
        self.assertLess(result["reprojection_rmse_px"], 1e-4)
        self.assertLess(transform_deviation(
            matrix, result["camera_T_marker"])["translation_m"], 1e-5)

    def test_approval_is_thresholded_and_explicit(self):
        matrix = np.eye(4).tolist()
        with TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.yaml"
            dataset_path.write_text("schema_version: 2\n", encoding="utf-8")
            report = {
            "schema_version": 1,
            "status": "PENDING_REVIEW",
            "dataset_path": str(dataset_path),
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "camera_serials": {"rs1": "111", "rs3": "222"},
            "validation": {
                "reprojection_rmse_px": 0.5,
                "dual_camera_3d_median_m": 0.004,
                "dual_camera_3d_p95_m": 0.007,
                "paired_sample_ids": list(range(6)),
                "checks": {
                    "minimum_optimization_poses_per_camera": 18,
                    "minimum_validation_poses_per_camera": 6,
                    "duplicates_absent": True,
                    "splits_disjoint": True,
                    "pose_diversity_passed": True,
                },
            },
            "results": {
                "rs1": {"base_T_link": matrix},
                "rs3": {"base_T_link": matrix},
            },
        }
            accepted = approve_dual_report(report, "operator")
            self.assertTrue(accepted["rs1"]["precision_operation_allowed"])
            self.assertEqual(accepted["rs1"]["serial"], "111")
            report["validation"]["dual_camera_3d_p95_m"] = 0.009
            with self.assertRaises(EyeToHandError):
                approve_dual_report(report, "operator")

    def test_holdout_reprojection_uses_calibration_not_per_pose_pnp(self):
        camera = np.asarray([[600.0, 0.0, 320.0],
                             [0.0, 600.0, 240.0],
                             [0.0, 0.0, 1.0]])
        base_T_camera = transform([0.1, -0.1, 0.05], [0.2, -0.3, 0.5])
        arm = transform([0.2, 0.1, -0.1], [0.3, 0.1, 0.4])
        link6_T_marker = transform([0.0, 0.1, 0.0], [0.0, 0.0, 0.15])
        predicted = np.linalg.inv(base_T_camera) @ arm @ link6_T_marker
        half = 0.05
        points = np.asarray([[-half, half, 0], [half, half, 0],
                             [half, -half, 0], [-half, -half, 0]], dtype=float)
        corners = cv2.projectPoints(
            points, cv2.Rodrigues(predicted[:3, :3])[0], predicted[:3, 3],
            camera, np.zeros(5))[0].reshape(4, 2)
        dataset = {
            "marker": {"marker_side_m": 0.1},
            "cameras": {"rs1": {"camera_info": {
                "K": camera.reshape(-1), "D": [0.0] * 5}}},
        }
        sample = {"base_T_link6": arm, "cameras": {"rs1": {
            "corners_px": corners}}}
        solved = {"base_T_camera": base_T_camera,
                  "link6_T_marker": link6_T_marker}
        exact = _validation_projection(dataset, sample, "rs1", solved)
        self.assertLess(exact["reprojection_rmse_px"], 1e-8)
        shifted = dict(solved)
        shifted["base_T_camera"] = base_T_camera.copy()
        shifted["base_T_camera"][0, 3] += 0.02
        wrong = _validation_projection(dataset, sample, "rs1", shifted)
        self.assertGreater(wrong["reprojection_rmse_px"], 1.0)


if __name__ == "__main__":
    unittest.main()
