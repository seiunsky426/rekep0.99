import unittest

import numpy as np

from rekpiper_calibration.workspace_bounds import (
    board_corners_in_marker, point_envelope, workspace_limits)


class WorkspaceBoundsTest(unittest.TestCase):
    def test_drawing_offset_is_not_marker_center(self):
        corners = board_corners_in_marker({"full_dimensions_m": [.120, .150, .003],
                                          "center_in_marker_m": [0, -.015, -.0015]})
        self.assertEqual(corners.shape, (8, 3))
        np.testing.assert_allclose(corners.min(axis=0), [-.060, -.090, -.003])
        np.testing.assert_allclose(corners.max(axis=0), [.060, .060, 0], atol=1e-12)

    def test_board_corner_outside_even_when_center_is_inside(self):
        lower, upper = [0, -.4, .05], [.7, .4, .6]
        center = [.35, 0, .58]
        self.assertTrue(point_envelope([center], lower, upper)["sampled_points_within_bounds"])
        result = point_envelope([center, [.35, 0, .63]], lower, upper)
        self.assertFalse(result["sampled_points_within_bounds"])
        self.assertEqual(result["outside_point_indices"], [1])

    def test_margins_and_boundary_points(self):
        result = point_envelope([[0, -.4, .05], [.6, .3, .5]], [0, -.4, .05], [.7, .4, .6])
        self.assertTrue(result["sampled_points_within_bounds"])
        np.testing.assert_allclose(result["lower_margin_xyz_m"], [0, 0, 0])
        np.testing.assert_allclose(result["upper_margin_xyz_m"], [.1, .1, .1])

    def test_rejects_invalid_geometry_and_frame(self):
        for points in ([], [[0, 0, float("nan")]], [[0, 0]], [[0, float("inf"), 0]]):
            with self.assertRaises(ValueError):
                point_envelope(points, [0, -.4, .05], [.7, .4, .6])
        for frame, upper in (("camera", [.7, .4, .6]), ("base_link", [0, -.4, .05])):
            with self.assertRaises(ValueError):
                workspace_limits({"frame": frame, "lower_m": [0, -.4, .05], "upper_m": upper})


if __name__ == "__main__":
    unittest.main()
