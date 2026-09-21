#!/usr/bin/env python3

import unittest

import numpy as np

from rekpiper_camera.cloud_fusion import (
    closest_timestamp_pair,
    voxel_fuse_base_clouds,
)


class CloudFusionTest(unittest.TestCase):
    def test_closest_timestamp_pair_ignores_arrival_order(self):
        pair = closest_timestamp_pair(
            np.array([10.000, 10.100, 10.200]),
            np.array([9.790, 10.091, 10.180]),
        )
        self.assertEqual(pair[:2], (1, 1))
        self.assertAlmostEqual(pair[2], 0.009)

    def test_voxel_average_and_source_colours(self):
        points, colors, stats = voxel_fuse_base_clouds({
            "rs1": np.array([[0.001, 0.001, 0.001], [0.031, 0.0, 0.0]]),
            "rs3": np.array([[0.002, 0.002, 0.002], [0.061, 0.0, 0.0]]),
        }, voxel_size_m=0.01)
        self.assertEqual(len(points), 3)
        self.assertEqual(stats["overlap_voxels"], 1)
        self.assertEqual(stats["rs1_only_voxels"], 1)
        self.assertEqual(stats["rs3_only_voxels"], 1)
        np.testing.assert_allclose(points[0], [0.0015, 0.0015, 0.0015])
        np.testing.assert_array_equal(colors[0], [255, 255, 255])

    def test_non_finite_points_are_dropped(self):
        points, _colors, stats = voxel_fuse_base_clouds({
            "rs1": np.array([[np.nan, 0.0, 0.0]]),
            "rs3": np.array([[0.01, 0.0, 0.0]]),
        })
        self.assertEqual(len(points), 1)
        self.assertEqual(stats["input_points"], {"rs1": 0, "rs3": 1})


if __name__ == "__main__":
    unittest.main()
