import unittest
import numpy as np
from rekpiper_planning.table_surface import (fit_table_plane, table_pixel_mask,
    project_table_depth, table_prism, inside_table_footprint)
from rekpiper_planning.offline_observation import rgbd_points, build_depth_collision_grid


class TableSurfaceTest(unittest.TestCase):
    def model(self):
        return dict(table_plane_normal=[0., 0., 1.], table_plane_offset_m=0.,
                    table_footprint_xy_m=[[-.4, -.4], [.4, -.4], [.4, .4], [-.4, .4]],
                    thickness_m=.1)

    def frame(self):
        transform = np.diag([1., -1., -1., 1.]); transform[2, 3] = 1.
        return dict(depth=np.full((61, 61), 1.004, np.float32), depth_encoding='32FC1',
                    rgb=np.full((61, 61, 3), 180, np.uint8),
                    K=np.array([[100., 0, 30], [0, 100., 30], [0, 0, 1.]]),
                    base_from_camera=transform, rgb_stamp_s=1., depth_stamp_s=1.)

    def test_plane_robust_to_outliers_preserves_tilt(self):
        rng = np.random.RandomState(12)
        xy = rng.uniform(-.35, .35, (12000, 2))
        points = np.column_stack((xy, .04*xy[:, 0]-.02*xy[:, 1]-.03))
        points[:, 2] += rng.normal(0, .002, len(points))
        points[:2000, 2] += .08
        model = fit_table_plane(points)
        n = np.array(model['table_plane_normal'])
        expected = np.array([-.04, .02, 1.]); expected /= np.linalg.norm(expected)
        self.assertLess(np.linalg.norm(n-expected), .003)
        self.assertAlmostEqual(model['table_plane_offset_m'], .03/np.linalg.norm([-.04, .02, 1.]), places=3)

    def test_colored_object_contact_edge_and_invalid_depth_unchanged(self):
        f = self.frame(); f['depth'][25:36, 25:36] = .98
        f['rgb'][25:36, 25:36] = [10, 150, 220]
        f['depth'][0, 0] = 0
        protected = np.zeros((61, 61), bool); protected[25:36, 25:36] = True
        mask = table_pixel_mask(f, self.model(), protected)
        self.assertFalse(mask[22:39, 22:39].any())
        self.assertFalse(mask[0, 0]); self.assertTrue(mask[10, 10])
        got = project_table_depth(f, self.model(), mask)
        np.testing.assert_array_equal(got['depth'][~mask], f['depth'][~mask])
        points, _, (v, u) = rgbd_points(got)
        np.testing.assert_allclose(points[mask[v, u], 2], 0, atol=1e-7)

    def test_white_raised_object_not_flattened(self):
        f = self.frame(); f['depth'][25:36, 25:36] = .97
        mask = table_pixel_mask(f, self.model())
        self.assertFalse(mask[22:39, 22:39].any())
        self.assertTrue(mask[10, 10])

    def test_mesh_and_depth_use_same_tilted_plane(self):
        model = self.model(); n = np.array([-.05, -.03, 1.]); n /= np.linalg.norm(n)
        model.update(table_plane_normal=n, table_plane_offset_m=.03)
        vertices, faces = table_prism(model)
        np.testing.assert_allclose(vertices[:4] @ n+.03, 0, atol=1e-12)
        np.testing.assert_allclose(vertices[4:] @ n+.03, -.1, atol=1e-12)
        self.assertEqual(faces.shape, (12, 3))
        f = self.frame(); mask = np.ones((61, 61), bool)
        corrected = project_table_depth(f, model, mask)
        points, _, _ = rgbd_points(corrected)
        np.testing.assert_allclose(points @ n+.03, 0, atol=1e-7)

    def test_sdf_plane_surface_known_solid_but_no_unknown_free_invention(self):
        f = self.frame(); f['depth'][:] = 1.
        grid = build_depth_collision_grid({'rs1': f}, [-.5, -.5, -.2], [.5, .5, .3], .01,
            table_model=self.model(), table_masks={'rs1': np.ones((61, 61), bool)})
        # center: 5 cm above = free, 5 cm below = table, 15 cm below = unknown.
        self.assertLess(grid['distances_m'][50, 50, 25], 0)
        self.assertGreater(grid['distances_m'][50, 50, 15], 0)
        self.assertTrue(grid['observed'][50, 50, 15])
        self.assertFalse(grid['observed'][50, 50, 5])
        self.assertFalse(grid['observed'][0, 0, 25])
        np.testing.assert_array_equal(grid['table_plane_normal'], self.model()['table_plane_normal'])


if __name__ == '__main__':
    unittest.main()
