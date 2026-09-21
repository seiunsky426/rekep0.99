#!/usr/bin/env python3

import unittest

import numpy as np

from rekpiper_execution.trajectory import (
    short_horizon_joint_prefix,
    JOINT_NAMES,
    normalize_feedback_to_joint_limits,
    smooth_pchip_trajectory,
    TrajectoryValidationError,
    validate_rekep_sdf_clearance,
    validate_trajectory,
    POSITION_MAX,
    POSITION_MIN,
)


class TrajectoryValidationTest(unittest.TestCase):
    def test_short_horizon_prefix_is_exactly_one_hundred_ms(self):
        current = np.zeros(6)
        path = np.asarray([current, np.full(6, 0.02), np.full(6, 0.08)])
        positions, times = short_horizon_joint_prefix(
            path, current, maximum_velocity_rad_s=0.30)
        np.testing.assert_allclose(times, [0.0, 0.05, 0.10])
        self.assertLessEqual(np.max(np.abs(positions[-1] - current)), 0.03)

    def test_pchip_smoothing_has_zero_end_velocity_and_limits(self):
        positions = np.asarray([
            [0.0, 0.4, -0.4, 0.0, 0.0, 0.0],
            [0.08, 0.45, -0.45, 0.04, -0.02, 0.0],
            [0.16, 0.55, -0.50, 0.02, -0.04, 0.03],
        ])
        result = smooth_pchip_trajectory(
            JOINT_NAMES, positions, positions[0],
            sample_rate_hz=50.0,
            maximum_velocity_rad_s=0.05,
            maximum_acceleration_rad_s2=0.10,
            maximum_jerk_rad_s3=0.50)
        np.testing.assert_allclose(result.positions[0], positions[0])
        np.testing.assert_allclose(result.positions[-1], positions[-1])
        dt0 = result.times[1] - result.times[0]
        dt1 = result.times[-1] - result.times[-2]
        self.assertLess(
            np.max(np.abs((result.positions[1] - result.positions[0]) / dt0)),
            0.002)
        self.assertLess(
            np.max(np.abs((result.positions[-1] - result.positions[-2]) / dt1)),
            0.002)
        self.assertLessEqual(result.maximum_velocity_rad_s, 0.05 + 1e-9)
        self.assertLessEqual(result.maximum_acceleration_rad_s2, 0.10 + 1e-9)
        self.assertLessEqual(result.maximum_jerk_rad_s3, 0.50 + 1e-9)
        self.assertAlmostEqual(result.sample_rate_hz, 50.0, places=1)

    def test_small_encoder_zero_offsets_snap_only_to_legal_boundary(self):
        corrected = normalize_feedback_to_joint_limits(
            [0.02, -0.00178, 0.00488, 0.027, 0.405, 0.068], 0.01)
        self.assertEqual(corrected[1], 0.0)
        self.assertEqual(corrected[2], 0.0)
        with self.assertRaisesRegex(TrajectoryValidationError, "joint feedback"):
            normalize_feedback_to_joint_limits(
                [0.02, -0.02, 0.0, 0.027, 0.405, 0.068], 0.01)

    def test_all_six_feedback_boundaries_share_the_same_tolerance(self):
        middle = 0.5 * (POSITION_MIN + POSITION_MAX)
        for index in range(6):
            below = middle.copy()
            below[index] = POSITION_MIN[index] - 0.01
            corrected = normalize_feedback_to_joint_limits(below, 0.01)
            self.assertEqual(corrected[index], POSITION_MIN[index])
            above = middle.copy()
            above[index] = POSITION_MAX[index] + 0.01
            corrected = normalize_feedback_to_joint_limits(above, 0.01)
            self.assertEqual(corrected[index], POSITION_MAX[index])
            outside = middle.copy()
            outside[index] = POSITION_MAX[index] + 0.0101
            with self.assertRaises(TrajectoryValidationError):
                normalize_feedback_to_joint_limits(outside, 0.01)

    def test_planned_trajectory_limits_remain_strict(self):
        positions = np.vstack([POSITION_MIN, POSITION_MIN])
        positions[0, 0] -= 0.001
        with self.assertRaisesRegex(TrajectoryValidationError, "joint limits"):
            validate_trajectory(
                JOINT_NAMES, positions, [0.0, 1.0], POSITION_MIN)

    def test_accepts_slow_canonical_path(self):
        positions = np.asarray([
            [0.0, 0.5, -0.5, 0.0, 0.0, 0.0],
            [0.1, 0.5, -0.5, 0.0, 0.0, 0.0],
            [0.2, 0.5, -0.5, 0.0, 0.0, 0.0],
        ])
        result = validate_trajectory(
            JOINT_NAMES, positions, [0.0, 1.0, 2.0], positions[0])
        self.assertEqual(result.positions.shape, (3, 6))

    def test_rejects_noncanonical_joint_names(self):
        with self.assertRaisesRegex(
                TrajectoryValidationError, "canonical order"):
            validate_trajectory(
                list(reversed(JOINT_NAMES)),
                np.zeros((2, 6)), [0.0, 1.0], np.zeros(6))

    def test_rejects_start_jump(self):
        positions = np.zeros((2, 6))
        positions[0, 0] = 0.2
        with self.assertRaisesRegex(
                TrajectoryValidationError, "start differs"):
            validate_trajectory(
                JOINT_NAMES, positions, [0.0, 2.0], np.zeros(6))

    def test_rejects_velocity(self):
        positions = np.zeros((2, 6))
        positions[1, 0] = 0.2
        with self.assertRaisesRegex(
                TrajectoryValidationError, "velocity"):
            validate_trajectory(
                JOINT_NAMES, positions, [0.0, 0.1], np.zeros(6),
                maximum_velocity_rad_s=0.5)

    def test_rekep_sdf_negative_free_space(self):
        self.assertAlmostEqual(
            validate_rekep_sdf_clearance(
                [-0.08, -0.03, -0.05], [True, True, True], 0.02),
            0.03)
        with self.assertRaises(TrajectoryValidationError):
            validate_rekep_sdf_clearance(
                [-0.08, -0.01], [True, True], 0.02)
        with self.assertRaises(TrajectoryValidationError):
            validate_rekep_sdf_clearance(
                [-0.08, -0.03], [True, False], 0.02)


if __name__ == "__main__":
    unittest.main()
