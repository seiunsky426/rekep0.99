#!/usr/bin/env python3

import unittest

import numpy as np

from rekpiper_execution.grasp_approach import (
    GraspApproachError,
    validate_path_constraints,
)


class GraspApproachTest(unittest.TestCase):
    def test_official_constraint_callables_receive_each_interpolated_pose(self):
        calls = []

        def path_constraint(end_effector, keypoints):
            calls.append((end_effector.copy(), keypoints.copy()))
            return end_effector[2] - 0.30

        poses = np.asarray([
            [0.40, 0.00, 0.20, 0.0, 0.0, 0.0, 1.0],
            [0.41, 0.00, 0.25, 0.0, 0.0, 0.0, 1.0],
        ])
        keypoints = np.asarray([[0.42, 0.00, 0.20]])
        validate_path_constraints(poses, keypoints, [path_constraint], 1e-3)
        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(calls[1][0], poses[1, :3])
        np.testing.assert_allclose(calls[1][1], keypoints)

    def test_non_callable_or_violated_constraint_fails_closed(self):
        poses = np.asarray([[0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]])
        keypoints = np.asarray([[0.4, 0.0, 0.2]])
        with self.assertRaises(GraspApproachError):
            validate_path_constraints(poses, keypoints, [0.0], 1e-3)
        with self.assertRaises(GraspApproachError):
            validate_path_constraints(
                poses, keypoints, [lambda _ee, _kp: 0.1], 1e-3)


if __name__ == "__main__":
    unittest.main()
