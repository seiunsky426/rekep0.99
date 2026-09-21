#!/usr/bin/env python3

import unittest

import numpy as np

from rekpiper_perception.keypoint_tracking import (
    OBSERVED,
    PROPAGATED,
    STALE_OCCLUDED,
    LandmarkTrack,
    Observation,
    anchor_points_to_gripper,
    attached_points_in_base,
    capture_stamp_is_fresh,
    multicamera_global_observation,
    multicamera_reference_descriptor,
    reference_frames_match_snapshot,
    robust_feature_observation,
)


class KeypointTrackingTest(unittest.TestCase):
    def test_reference_and_capture_timestamps_are_sensor_bound(self):
        snapshot = 10_000_000_000
        self.assertTrue(reference_frames_match_snapshot(
            snapshot, [9_900_000_000, 10_100_000_000], 0.15))
        self.assertFalse(reference_frames_match_snapshot(
            snapshot, [9_800_000_000, 10_100_000_000], 0.15))
        self.assertTrue(capture_stamp_is_fresh(
            10_100_000_000, 10_000_000_000, 0.15))
        self.assertFalse(capture_stamp_is_fresh(
            9_900_000_000, 10_000_000_000, 0.15))

    def test_multicamera_reference_and_one_global_top100(self):
        first_features = np.tile([1.0, 0.0], (6, 10, 1))
        second_features = np.tile([0.8, 0.2], (6, 10, 1))
        first_points = np.zeros((6, 10, 3), dtype=float)
        second_points = np.zeros((6, 10, 3), dtype=float)
        first_points[..., 0] = np.linspace(0.0, 0.009, 10)
        second_points[..., 0] = np.linspace(0.010, 0.019, 10)
        reference = multicamera_reference_descriptor(
            [first_features, second_features],
            [first_points, second_points], [0.0095, 0.0, 0.0], 0.020)
        expected = np.asarray([0.9, 0.1])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(reference, expected, atol=1e-7)
        observed = multicamera_global_observation(
            reference, [first_features, second_features],
            [first_points, second_points], 0.60, 100, 2.0)
        self.assertIsNotNone(observed)
        self.assertGreater(observed.matches, 60)
        self.assertLessEqual(observed.matches, 100)
        self.assertGreaterEqual(observed.point[0], 0.0)
        self.assertLessEqual(observed.point[0], 0.019)

    def test_feature_match_rejects_spatial_outliers(self):
        reference = np.array([1.0, 0.0, 0.0])
        features = np.tile(reference, (7, 1))
        points = np.array([
            [0.100, 0.200, 0.300],
            [0.101, 0.199, 0.300],
            [0.099, 0.200, 0.301],
            [0.100, 0.201, 0.300],
            [0.100, 0.200, 0.299],
            [2.000, 2.000, 2.000],
            [-2.000, -2.000, -2.000],
        ])
        result = robust_feature_observation(reference, features, points)
        self.assertIsNotNone(result)
        self.assertLess(np.linalg.norm(result.point - [0.1, 0.2, 0.3]), 0.005)
        self.assertGreaterEqual(result.matches, 5)

    def test_feature_match_returns_none_below_similarity_threshold(self):
        result = robust_feature_observation(
            np.array([1.0, 0.0]),
            np.array([[0.0, 1.0], [0.1, 0.9]]),
            np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
        )
        self.assertIsNone(result)

    def test_smoothing_and_stale_state(self):
        track = LandmarkTrack(
            keypoint_id=1,
            name="K1",
            group_id=1,
            reference_point=np.zeros(3),
            reference_feature=np.array([1.0, 0.0]),
            reference_pixel_rc=(10, 10),
            history_size=3,
        )
        for value in (0.0, 0.03, 0.06):
            track.update(
                Observation(np.array([value, 0.0, 0.0]), np.eye(3) * 1e-6, 0.9, 4),
                OBSERVED,
            )
        self.assertAlmostEqual(track.last_point[0], 0.03)
        track.update(
            Observation(np.array([0.09, 0.0, 0.0]), np.eye(3) * 1e-6, 0.5, 3),
            PROPAGATED,
        )
        self.assertAlmostEqual(track.last_point[0], 0.06)
        track.mark_stale(1e-5)
        self.assertEqual(track.state, STALE_OCCLUDED)
        self.assertEqual(track.stale_frames, 1)
        self.assertGreater(track.last_covariance[0, 0], 1e-6)
        track.update(
            Observation(np.array([0.12, 0.0, 0.0]), np.eye(3) * 1e-6, 0.9, 5),
            OBSERVED,
        )
        self.assertEqual(track.state, OBSERVED)
        self.assertEqual(track.stale_frames, 0)

    def test_only_explicitly_anchored_group_follows_gripper(self):
        base_from_gripper = np.eye(4)
        base_from_gripper[:3, 3] = [0.3, -0.1, 0.2]
        points = np.asarray([[0.31, -0.08, 0.22],
                             [0.29, -0.12, 0.20]])
        local = anchor_points_to_gripper(points, base_from_gripper)
        moved = base_from_gripper.copy()
        moved[:3, 3] += [0.02, 0.0, 0.0]
        predicted = attached_points_in_base(local, moved)
        np.testing.assert_allclose(predicted, points + [0.02, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
