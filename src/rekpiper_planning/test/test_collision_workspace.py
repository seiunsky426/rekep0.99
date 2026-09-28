import unittest

import numpy as np

from rekpiper_planning.collision_workspace import workspace_mask, named_region_mask


class WorkspaceScopeTest(unittest.TestCase):
    def setUp(self):
        self.workspace = dict(x_min_m=0., x_max_m=.9, y_min_m=-.4, y_max_m=.4,
                              table_plane_normal=[0., 0., 1.], table_plane_offset_m=0.,
                              below_table_tolerance_m=.015)

    def test_outside_arm_is_excluded_but_crossing_geometry_is_checked(self):
        points = np.array([[-.3, 0, .2], [-.005, 0, .2], [.9, .405, .2], [.4, 0, -.02]])
        np.testing.assert_array_equal(workspace_mask(points, self.workspace), [False]*4)
        np.testing.assert_array_equal(workspace_mask(points, self.workspace, .01), [False, True, True, True])

    def test_tilted_table_and_original_working_bounds(self):
        self.workspace['table_plane_normal'] = [0., .6, .8]
        points = np.array([[.4, .2, -.1], [.4, -.2, .1], [.91, 0., .2]])
        np.testing.assert_array_equal(workspace_mask(points, self.workspace), [True, False, False])

    def test_support_region_does_not_remove_target_or_all_dark_geometry(self):
        region = dict(name='base_support', min_m=[0., -.06, 0.], max_m=[.08, .06, .2])
        points = np.array([[.01, 0, .15], [.44, -.16, .02], [.08, .2, .15]])
        np.testing.assert_array_equal(named_region_mask(points, [region]), [True, False, False])
        np.testing.assert_array_equal(named_region_mask(points, []), [False]*3)

    def test_rejects_unbounded_exclusion(self):
        with self.assertRaises(ValueError):
            named_region_mask(np.zeros((1, 3)), [dict(name='bad', min_m=[0., 0., 0.], max_m=[1., 1., float('inf')])])


if __name__ == '__main__':
    unittest.main()
