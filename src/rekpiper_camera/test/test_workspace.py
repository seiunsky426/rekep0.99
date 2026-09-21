#!/usr/bin/env python3

import unittest

import numpy as np

from rekpiper_camera.workspace import (
    estimate_table_workspace,
    filter_xyz_bounds,
    normalized_polygon_mask,
)


class WorkspaceTest(unittest.TestCase):
    def test_normalized_polygon_mask(self):
        mask = normalized_polygon_mask(
            10, 20, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
        )
        self.assertEqual((10, 20), mask.shape)
        self.assertTrue(mask[0, 0])
        self.assertTrue(mask[0, -1])
        self.assertTrue(mask[-1, -1])
        self.assertFalse(mask[-1, 0])

    def test_filter_xyz_bounds_preserves_shape_and_masks(self):
        xyz = np.array(
            [[[0.0, 0.0, -0.1], [0.5, 0.0, 0.2], [2.0, 0.0, 0.2]]],
            dtype=np.float32,
        )
        filtered, valid = filter_xyz_bounds(
            xyz, [-0.1, -0.2, -0.2], [1.0, 0.2, 0.5]
        )
        np.testing.assert_array_equal(valid, [[True, True, False]])
        np.testing.assert_array_equal(filtered[0, :2], xyz[0, :2])
        self.assertTrue(np.all(np.isnan(filtered[0, 2])))

    def test_workspace_estimation_applies_requested_margins(self):
        height, width = 20, 20
        yy, xx = np.indices((height, width), dtype=np.float32)
        cloud = np.stack(
            (xx / 19.0, yy / 19.0 - 0.5, np.full_like(xx, -0.07)),
            axis=-1,
        )
        result = estimate_table_workspace(
            {"rs1": cloud, "rs3": cloud.copy()},
            {
                "rs1": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "rs3": [[0, 0], [1, 0], [1, 1], [0, 1]],
            },
            expected_table_z_m=-0.07,
            table_band_half_width_m=0.01,
            xy_quantiles=[0.0, 1.0],
            xy_margin_m=0.10,
            below_table_allowance_m=0.10,
            max_height_above_table_m=0.60,
        )
        np.testing.assert_allclose(result["bounds_min"], [-0.1, -0.6, -0.17])
        np.testing.assert_allclose(result["bounds_max"], [1.1, 0.6, 0.53])


if __name__ == "__main__":
    unittest.main()
