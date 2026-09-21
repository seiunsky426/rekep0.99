#!/usr/bin/env python3

import unittest

import numpy as np

from rekpiper_mapping.frame_preparation import (
    FILTER_DEPTH_RANGE, FILTER_DYNAMIC, FILTER_RAW_INVALID, FILTER_RETAINED,
    FILTER_ROBOT, normalize_camera_sources, normalize_sdf_query,
    prepare_static_depth)


class FramePreparationTest(unittest.TestCase):
    def setUp(self):
        self.depth = np.full((5, 7), 1000, dtype=np.uint16)
        self.k = [100.0, 0.0, 3.0, 0.0, 100.0, 2.0, 0.0, 0.0, 1.0]
        self.pose = np.eye(4, dtype=np.float32)

    def test_exclusion_mask_is_dilated_and_zeroed(self):
        mask = np.zeros(self.depth.shape, dtype=np.uint8)
        mask[2, 3] = 255
        result = prepare_static_depth(
            self.depth, 0.001, self.k, self.pose,
            [-1.0, -1.0, 0.1], [1.0, 1.0, 2.0], [mask], mask_dilation_px=1,
        )
        self.assertEqual(result.input_valid_pixels, 35)
        self.assertEqual(result.explicitly_excluded_pixels, 9)
        self.assertEqual(result.retained_pixels, 26)
        self.assertTrue(np.all(result.depth_m[1:4, 2:5] == 0.0))

    def test_workspace_bounds_remove_outside_points(self):
        pose = self.pose.copy()
        pose[0, 3] = 3.0
        result = prepare_static_depth(
            self.depth, 0.001, self.k, pose,
            [-1.0, -1.0, 0.1], [1.0, 1.0, 2.0], mask_dilation_px=0,
        )
        self.assertEqual(result.retained_pixels, 0)
        self.assertTrue(np.all(result.depth_m == 0.0))

    def test_filter_reasons_are_exclusive_and_robot_wins_overlap(self):
        depth = np.array([[0, 50, 1000, 1000, 1000]], dtype=np.uint16)
        robot = np.zeros(depth.shape, dtype=np.uint8)
        dynamic = np.zeros(depth.shape, dtype=np.uint8)
        robot[0, 2] = 255
        dynamic[0, 2:4] = 255
        result = prepare_static_depth(
            depth, 0.001, [100.0, 0.0, 2.0, 0.0, 100.0, 0.0, 0.0, 0.0, 1.0],
            self.pose, [-2.0, -2.0, 0.0], [2.0, 2.0, 2.0],
            [robot, dynamic], mask_dilation_px=0,
            exclusion_labels=["robot", "dynamic"])
        self.assertEqual(result.filter_reasons.tolist(), [[
            FILTER_RAW_INVALID, FILTER_DEPTH_RANGE, FILTER_ROBOT,
            FILTER_DYNAMIC, FILTER_RETAINED]])
        self.assertEqual(sum(result.reason_counts.values()), depth.size)

    def test_explicit_mask_has_priority_over_missing_depth(self):
        depth = np.array([[0, 1000]], dtype=np.uint16)
        robot = np.array([[1, 0]], dtype=np.uint8)
        result = prepare_static_depth(
            depth, 0.001,
            [100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 1.0],
            self.pose, [-2.0, -2.0, 0.0], [2.0, 2.0, 2.0],
            [robot], mask_dilation_px=0, exclusion_labels=["robot"])
        self.assertEqual(int(result.filter_reasons[0, 0]), FILTER_ROBOT)
        self.assertEqual(int(result.filter_reasons[0, 1]), FILTER_RETAINED)

    def test_query_validation_and_default_radii(self):
        xyz, radii = normalize_sdf_query([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], [])
        self.assertEqual(xyz.shape, (2, 3))
        self.assertTrue(np.array_equal(radii, [0.0, 0.0]))
        with self.assertRaises(ValueError):
            normalize_sdf_query([[np.nan, 0.0, 0.0]], [])

    def test_invalid_transform_is_rejected(self):
        pose = self.pose.copy()
        pose[3, 3] = 0.0
        with self.assertRaises(ValueError):
            prepare_static_depth(
                self.depth, 0.001, self.k, pose,
                [-1.0, -1.0, 0.1], [1.0, 1.0, 2.0], mask_dilation_px=0,
            )

    def test_camera_sources_support_two_unique_named_streams(self):
        sources = normalize_camera_sources([
            {"name": "rs1", "camera_frame": "rs1_optical",
             "depth_topic": "/rs1/depth", "camera_info_topic": "/rs1/info"},
            {"name": "rs3", "camera_frame": "rs3_optical",
             "depth_topic": "/rs3/depth", "camera_info_topic": "/rs3/info"},
        ], {})
        self.assertEqual([source["name"] for source in sources], ["rs1", "rs3"])

    def test_camera_sources_reject_duplicate_or_incomplete_entries(self):
        duplicate = [
            {"name": "rs", "camera_frame": "a", "depth_topic": "/a",
             "camera_info_topic": "/ai"},
            {"name": "rs", "camera_frame": "b", "depth_topic": "/b",
             "camera_info_topic": "/bi"},
        ]
        with self.assertRaises(ValueError):
            normalize_camera_sources(duplicate, {})
        with self.assertRaises(ValueError):
            normalize_camera_sources([{"name": "rs1"}], {})


if __name__ == "__main__":
    unittest.main()
