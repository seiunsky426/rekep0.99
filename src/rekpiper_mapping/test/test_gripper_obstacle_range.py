#!/usr/bin/env python3

import unittest

import numpy as np

from rekpiper_mapping.gripper_obstacle_range import project_tsdf_surface_range


class GripperObstacleRangeTest(unittest.TestCase):
    def test_projection_uses_true_xyz_distance(self):
        vertices = np.array([
            [0.03, 0.00, 0.04],
            [0.04, 0.00, 0.04],
            [0.03, 0.01, 0.04],
        ], dtype=np.float32)
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        image, distances, minimum = project_tsdf_surface_range(
            vertices, triangles, [0.0, 0.0, 0.0],
            np.ones((101, 101), dtype=bool), half_extent_m=0.10)
        self.assertAlmostEqual(minimum, 0.05, places=5)
        self.assertTrue(np.any(np.isfinite(distances)))
        self.assertTrue(np.any(image[..., 2] > 200))

    def test_same_xy_changes_when_surface_height_changes(self):
        xy = np.array([[0.03, 0.00], [0.04, 0.00], [0.03, 0.01]], dtype=np.float32)
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        low = np.column_stack((xy, np.full(3, 0.04, np.float32)))
        high = np.column_stack((xy, np.full(3, 0.08, np.float32)))
        observed = np.ones((81, 81), dtype=bool)
        _, _, low_minimum = project_tsdf_surface_range(
            low, triangles, [0.0, 0.0, 0.0], observed, half_extent_m=0.10)
        _, _, high_minimum = project_tsdf_surface_range(
            high, triangles, [0.0, 0.0, 0.0], observed, half_extent_m=0.10)
        self.assertGreater(high_minimum, low_minimum)

    def test_empty_mesh_has_no_fabricated_surface_distance(self):
        observed = np.zeros((51, 51), dtype=bool)
        image, distances, minimum = project_tsdf_surface_range(
            np.empty((0, 3), np.float32), np.empty((0, 3), np.int32),
            [0.0, 0.0, 0.0], observed, half_extent_m=0.10)
        self.assertIsNone(minimum)
        self.assertFalse(np.any(np.isfinite(distances)))
        # Only the truthful gripper cross may be non-black.
        outside_cross = image.copy()
        center = image.shape[0] // 2
        outside_cross[center - 8:center + 9, center - 8:center + 9] = 0
        self.assertFalse(np.any(outside_cross))


if __name__ == "__main__":
    unittest.main()
