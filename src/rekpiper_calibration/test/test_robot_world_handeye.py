import unittest

import numpy as np

from rekpiper_calibration.robot_world_handeye import (
    RobotWorldHandeyeError, check_pose_diversity, closure_metrics,
    invert_transform, select_solution, transform_deviation)


def transform(rotation_vector, translation):
    import cv2
    rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=float).reshape(3, 1))
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = np.asarray(translation, dtype=float)
    return value


class RobotWorldHandeyeTest(unittest.TestCase):
    def setUp(self):
        self.base_T_board = transform([0.2, -0.1, 0.3], [0.42, -0.18, 0.06])
        self.link6_T_camera = transform([-0.15, 0.22, -0.12], [0.03, 0.01, 0.08])
        self.base_T_link6 = [
            transform([0.10, 0.20, 0.10], [0.10, -0.18, 0.18]),
            transform([-0.20, 0.18, 0.35], [0.24, -0.04, 0.28]),
            transform([0.30, -0.16, 0.18], [0.39, 0.12, 0.22]),
            transform([-0.24, -0.30, 0.10], [0.18, 0.20, 0.36]),
            transform([0.12, 0.35, -0.25], [0.48, -0.10, 0.14]),
            transform([-0.32, 0.06, -0.28], [0.31, 0.17, 0.31]),
        ]
        self.camera_T_board = [
            invert_transform(self.link6_T_camera) @ invert_transform(base_T_link) @
            self.base_T_board for base_T_link in self.base_T_link6]

    def test_recovers_fixed_board_and_camera_mount(self):
        result = select_solution(self.base_T_link6, self.camera_T_board)
        board = transform_deviation(self.base_T_board, result["base_T_board"])
        camera = transform_deviation(self.link6_T_camera, result["link6_T_camera"])
        self.assertLess(board["translation_m"], 1e-7)
        self.assertLess(board["rotation_deg"], 1e-5)
        self.assertLess(camera["translation_m"], 1e-7)
        self.assertLess(camera["rotation_deg"], 1e-5)
        self.assertLess(result["closure"]["translation_maximum_m"], 1e-7)

    def test_diversity_gate_rejects_undersampled_session(self):
        with self.assertRaises(RobotWorldHandeyeError):
            check_pose_diversity(self.base_T_link6[:3], [0.1, 0.1, 0.1], 20.0, 4)

    def test_holdout_closure_is_reported(self):
        result = select_solution(self.base_T_link6[:4], self.camera_T_board[:4])
        metrics = closure_metrics(self.base_T_link6[4:], self.camera_T_board[4:],
                                  result["base_T_board"], result["link6_T_camera"])
        self.assertLess(metrics["translation_maximum_m"], 1e-7)


if __name__ == "__main__":
    unittest.main()
