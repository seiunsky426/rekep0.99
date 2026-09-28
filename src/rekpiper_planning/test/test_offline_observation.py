import unittest
import numpy as np
from rekpiper_planning.offline_observation import build_depth_collision_grid, rgbd_points


class OfflineObservationTest(unittest.TestCase):
    def frames(self):
        frame = dict(depth=np.full((5, 5), 1000, dtype=np.uint16),
            depth_encoding='16UC1', rgb=np.full((5, 5, 3), [12, 34, 56], dtype=np.uint8),
            K=np.array([10., 0, 2, 0, 10., 2, 0, 0, 1]), base_from_camera=np.eye(4),
            rgb_stamp_s=1., depth_stamp_s=1.)
        return {'rs1': frame, 'rs3': dict(frame)}

    def test_rgb_and_units_preserved(self):
        points, rgb, pixels = rgbd_points(self.frames()['rs1'])
        np.testing.assert_allclose(points[:, 2], 1.)
        np.testing.assert_array_equal(rgb, np.tile([12, 34, 56], (25, 1)))
        self.assertEqual(len(pixels[0]), 25)

    def test_free_surface_and_occluded_space_are_distinct(self):
        grid = build_depth_collision_grid(self.frames(), [-.1, -.1, .5], [.1, .1, 1.5], .1)
        self.assertLess(grid['distances_m'][1, 1, 0], 0)
        self.assertGreater(grid['distances_m'][1, 1, 5], 0)
        self.assertFalse(grid['observed'][1, 1, 10])
        self.assertGreater(grid['distances_m'][1, 1, 10], 0)

    def test_hypothetical_removal_does_not_clear_hidden_space(self):
        masks = {n: np.ones((5, 5), dtype=bool) for n in ('rs1', 'rs3')}
        grid = build_depth_collision_grid(self.frames(), [-.1, -.1, .5], [.1, .1, 1.5], .1, masks)
        self.assertTrue(grid['assumed_vacated'][1, 1, 5])
        self.assertFalse(grid['observed'][1, 1, 10])
        masks['rs3'][:] = False
        grid = build_depth_collision_grid(self.frames(), [-.1, -.1, .5], [.1, .1, 1.5], .1, masks)
        self.assertTrue(grid['occupied'][1, 1, 5])

    def test_time_mismatch_rejected(self):
        frames = self.frames(); frames['rs3']['rgb_stamp_s'] = 1.1
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            build_depth_collision_grid(frames, [0, 0, .5], [.1, .1, 1.5], .1)

    def test_rs1_only_preserves_unknown_space(self):
        frames = {'rs1': self.frames()['rs1']}
        grid = build_depth_collision_grid(frames, [-.1, -.1, .5], [.1, .1, 1.5], .1)
        self.assertLess(grid['distances_m'][1, 1, 0], 0)
        self.assertTrue(grid['occupied'][1, 1, 5])
        self.assertFalse(grid['observed'][1, 1, 10])
        self.assertGreater(grid['distances_m'][1, 1, 10], 0)

    def test_rs3_cannot_replace_planning_camera(self):
        with self.assertRaisesRegex(ValueError, 'rs1_camera_frame_required'):
            build_depth_collision_grid({'rs3': self.frames()['rs3']},
                                       [-.1, -.1, .5], [.1, .1, 1.5], .1)


if __name__ == '__main__':
    unittest.main()
