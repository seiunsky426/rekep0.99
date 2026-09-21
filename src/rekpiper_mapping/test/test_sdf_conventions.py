#!/usr/bin/env python3

import unittest
from types import SimpleNamespace

import numpy as np

from rekpiper_mapping.sdf_conventions import (
    inclusive_grid_axes,
    checked_grid_response,
    sanitize_nvblox_sdf,
    validate_sdf_grid_geometry,
)


class SDFConventionTest(unittest.TestCase):
    def test_diagnostic_grid_cannot_become_a_planning_grid(self):
        response = SimpleNamespace(header=SimpleNamespace(frame_id="base_link"),
            map_generation_uuid="generation-a", map_valid=False,
            distances_m=[-.1, -100.], observed=[True, False],
            status="acceptance_map_not_safe_for_motion")
        with self.assertRaises(ValueError):
            checked_grid_response(response, 2, "base_link")
        distance, observed, valid = checked_grid_response(response, 2, "base_link", True)
        self.assertFalse(valid)
        self.assertGreater(distance[1], 0.)
        self.assertEqual(observed.tolist(), [1, 0])
        response.map_valid = True
        self.assertFalse(checked_grid_response(response, 2, "base_link", True)[2])

    def test_diagnostic_grid_still_rejects_stale_or_mismatched_data(self):
        response = SimpleNamespace(header=SimpleNamespace(frame_id="base_link"),
            map_generation_uuid="generation-a", map_valid=False,
            distances_m=[-.1], observed=[True], status="map_stale")
        with self.assertRaises(ValueError):
            checked_grid_response(response, 1, "base_link", True)
        response.status = "acceptance_map_not_safe_for_motion"
        response.header.frame_id = "wrong_frame"
        with self.assertRaises(ValueError):
            checked_grid_response(response, 1, "base_link", True)
        response.header.frame_id = "base_link"
        response.map_generation_uuid = ""
        with self.assertRaises(ValueError):
            checked_grid_response(response, 1, "base_link", True)

    def test_endpoint_inclusive_axes_match_solver_linspace(self):
        axes = inclusive_grid_axes(
            [-0.20, -0.60, -0.07], [0.80, 0.60, 0.80], 0.015)
        self.assertEqual(tuple(len(axis) for axis in axes), (68, 81, 59))
        np.testing.assert_allclose(
            [axis[0] for axis in axes], [-0.20, -0.60, -0.07])
        np.testing.assert_allclose(
            [axis[-1] for axis in axes], [0.80, 0.60, 0.80])

    def test_observed_sign_is_not_flipped(self):
        raw = np.array([-0.20, 0.0, 0.05], dtype=np.float32)
        result = sanitize_nvblox_sdf(raw)
        np.testing.assert_array_equal(result.distances_m, raw)
        self.assertTrue(result.observed.all())

    def test_float32_bounds_preserve_endpoint_inclusive_shape(self):
        lower = np.array([-0.20, -0.60, -0.07], dtype=np.float32)
        upper = np.array([0.80, 0.60, 0.80], dtype=np.float32)
        for resolution in (0.015, float(np.float32(0.015))):
            axes = inclusive_grid_axes(lower, upper, resolution)
            self.assertEqual(tuple(len(axis) for axis in axes), (68, 81, 59))
            np.testing.assert_array_equal([axis[0] for axis in axes], lower)
            np.testing.assert_array_equal([axis[-1] for axis in axes], upper)

    def test_unknown_and_nonfinite_become_occupied(self):
        raw = np.array([-100.0, -120.0, np.nan, np.inf, -0.5], dtype=np.float32)
        result = sanitize_nvblox_sdf(raw, unknown_occupied_distance_m=0.75)
        np.testing.assert_array_equal(
            result.observed, [False, False, False, False, True]
        )
        np.testing.assert_allclose(result.distances_m[:4], 0.75)
        self.assertAlmostEqual(float(result.distances_m[4]), -0.5)

    def test_grid_geometry(self):
        grid = np.zeros((11, 11, 11), dtype=np.float32)
        shape = validate_sdf_grid_geometry(
            grid,
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            0.1,
        )
        self.assertEqual(shape, (11, 11, 11))

    def test_inconsistent_grid_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_sdf_grid_geometry(
                np.zeros((4, 4, 4)),
                np.zeros(3),
                np.ones(3),
                0.1,
            )


if __name__ == "__main__":
    unittest.main()
