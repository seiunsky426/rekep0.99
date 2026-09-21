import unittest

import cv2
import numpy as np

from rekpiper_calibration.checkerboard_calibration import (
    CalibrationFailure, QualityThresholds, TargetDefinition,
    base_target_from_config, canonicalize_corners,
    checkerboard_object_points, compose_base_to_link,
    depth_geometry_metrics, matrix_to_quaternion_xyzw,
    repeatability_metrics, solve_checkerboard_pose,
    validate_rigid_transform)


class CheckerboardCalibrationTest(unittest.TestCase):
    @staticmethod
    def target():
        return TargetDefinition(
            11, 8, 0.025, "DICT_5X5_100", 18, 0.100, 0.250, 0.175)

    @staticmethod
    def synthetic_image():
        image = np.full((480, 640, 3), 255, np.uint8)
        origin_x, origin_y, cell = 100, 110, 30
        for row in range(9):
            for col in range(12):
                if (row + col) % 2 == 0:
                    cv2.rectangle(
                        image,
                        (origin_x + col * cell, origin_y + row * cell),
                        (origin_x + (col + 1) * cell,
                         origin_y + (row + 1) * cell),
                        (0, 0, 0), -1)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        marker = cv2.aruco.generateImageMarker(dictionary, 18, 70)
        image[390:460, 500:570] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        return image

    def test_fixed_target_geometry(self):
        values = {
            "base_T_target_translation_m": [0.4675, 0.125, -0.009],
            "base_T_target_rotation": [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
        }
        matrix = base_target_from_config(values)
        points = checkerboard_object_points(self.target())
        self.assertEqual(points.shape, (88, 3))
        self.assertTrue(np.allclose(points[-1], [0.250, 0.175, 0.0]))
        opposite_base = (matrix @ np.r_[points[-1], 1.0])[:3]
        self.assertTrue(np.allclose(opposite_base, [0.2925, -0.125, -0.009]))
        centerline_last_inner = (matrix @ np.array([0.125, 0.175, 0.0, 1.0]))[:3]
        self.assertTrue(np.allclose(centerline_last_inner, [0.2925, 0.0, -0.009]))
        centerline_next_outer = (matrix @ np.array([0.125, 0.200, 0.0, 1.0]))[:3]
        self.assertTrue(np.allclose(centerline_next_outer, [0.2675, 0.0, -0.009]))
        quaternion = matrix_to_quaternion_xyzw(matrix)
        self.assertTrue(np.allclose(
            np.abs(quaternion), [2 ** -0.5, 2 ** -0.5, 0.0, 0.0]))

    def test_synthetic_single_frame_pose_and_marker_orientation(self):
        camera_matrix = np.array([[600.0, 0.0, 320.0],
                                  [0.0, 600.0, 240.0],
                                  [0.0, 0.0, 1.0]])
        thresholds = QualityThresholds(maximum_ippe_best_to_second_ratio=1.0)
        result = solve_checkerboard_pose(
            self.synthetic_image(), camera_matrix, np.zeros(5),
            self.target(), thresholds)
        self.assertLess(result["metrics"]["reprojection_rmse_px"], 0.1)
        self.assertGreater(result["metrics"]["marker_board_x_m"], 0.250)
        self.assertGreater(result["metrics"]["marker_board_y_m"], 0.175)
        self.assertAlmostEqual(result["camera_T_target"][2, 3], 0.5, places=3)

    def test_aruco_anchor_corrects_180_degree_corner_order(self):
        normal = np.array([
            [130.0 + 30.0 * col, 140.0 + 30.0 * row]
            for row in range(8) for col in range(11)
        ], dtype=np.float32)
        marker_corners = np.array(
            [[500.0, 390.0], [570.0, 390.0], [570.0, 460.0], [500.0, 460.0]],
            dtype=np.float32)
        canonical, _, reversed_order = canonicalize_corners(
            normal[::-1], marker_corners.mean(axis=0), self.target())
        self.assertTrue(reversed_order)
        self.assertTrue(np.allclose(canonical, normal))

    def test_missing_orientation_marker_rejects_frame(self):
        image = self.synthetic_image()
        image[385:465, 495:575] = 255
        camera_matrix = np.array([[600.0, 0.0, 320.0],
                                  [0.0, 600.0, 240.0],
                                  [0.0, 0.0, 1.0]])
        with self.assertRaises(CalibrationFailure):
            solve_checkerboard_pose(
                image, camera_matrix, np.zeros(5), self.target(),
                QualityThresholds(maximum_ippe_best_to_second_ratio=1.0))

    def test_non_rigid_transform_is_rejected(self):
        reflection = np.eye(4)
        reflection[0, 0] = -1.0
        with self.assertRaises(ValueError):
            validate_rigid_transform(reflection, "reflection")

    def test_frame_composition(self):
        base_T_target = np.eye(4)
        camera_T_target = np.eye(4)
        camera_T_target[:3, 3] = [0.1, -0.2, 0.5]
        link_T_camera = np.eye(4)
        link_T_camera[:3, 3] = [0.0, 0.0, 0.05]
        base_T_link, base_T_camera = compose_base_to_link(
            base_T_target, camera_T_target, link_T_camera)
        self.assertTrue(np.allclose(base_T_camera[:3, 3], [-0.1, 0.2, -0.5]))
        self.assertTrue(np.allclose(base_T_link[:3, 3], [-0.1, 0.2, -0.55]))

    def test_aligned_depth_independent_geometry(self):
        camera_matrix = np.array([[600.0, 0.0, 320.0],
                                  [0.0, 600.0, 240.0],
                                  [0.0, 0.0, 1.0]])
        result = solve_checkerboard_pose(
            self.synthetic_image(), camera_matrix, np.zeros(5), self.target(),
            QualityThresholds(maximum_ippe_best_to_second_ratio=1.0))
        base_T_target = np.eye(4)
        base_T_camera = np.linalg.inv(result["camera_T_target"])
        depth = np.full((480, 640), 500, dtype=np.uint16)
        metrics = depth_geometry_metrics(
            depth, camera_matrix, np.zeros(5), 0.001,
            result["corners"], result["object_points"],
            base_T_camera, base_T_target)
        self.assertLess(metrics["depth_plane_rmse_m"], 0.001)
        self.assertLess(metrics["known_corner_p95_error_m"], 0.001)
        self.assertLess(metrics["known_corner_max_error_m"], 0.001)

    def test_repeatability_uses_maximum_pairwise_distance(self):
        candidates = []
        for offset in (0.0, 0.002, 0.004):
            matrix = np.eye(4)
            matrix[0, 3] = offset
            candidates.append({"cameras": {"rs1": {"base_T_link": matrix}}})
        metrics = repeatability_metrics(candidates, "rs1")
        self.assertAlmostEqual(metrics["maximum_translation_m"], 0.004)
        self.assertAlmostEqual(metrics["maximum_rotation_deg"], 0.0)


if __name__ == "__main__":
    unittest.main()
