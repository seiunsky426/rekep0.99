import unittest
import numpy as np
from rekpiper_camera.depth_quality import depth_support_mask, free_space_conflicts


class DepthQualityTest(unittest.TestCase):
    def frame(self):
        return dict(depth=np.ones((11, 11)), K=np.array([[20., 0, 5], [0, 20., 5], [0, 0, 1.]]),
                    base_from_camera=np.eye(4))

    def test_flat_surface_and_invalid_holes(self):
        depth=np.full((15, 15), .7); depth[4, 4]=0; depth[10, 10]=np.nan
        mask=depth_support_mask(depth)
        self.assertFalse(mask[4, 4]); self.assertFalse(mask[10, 10])
        self.assertEqual(mask.sum(), depth.size-2)

    def test_isolated_flyer_rejected_without_flattening_object_interior(self):
        depth=np.full((15, 15), .8); depth[5:10, 5:10]=.7; depth[2, 2]=.74
        before=depth.copy(); keep=depth_support_mask(depth, minimum_neighbors=3)
        self.assertFalse(keep[2, 2]); self.assertTrue(keep[7, 7])
        self.assertFalse(keep[5, 7]); self.assertTrue(keep[12, 12])
        np.testing.assert_array_equal(depth, before)

    def test_other_view_free_space_conflict_only(self):
        points=np.array([[0, 0, .95], [0, 0, .99], [0, 0, 1.1], [1., 0, .8], [0, 0, -.2]])
        np.testing.assert_array_equal(free_space_conflicts(points, self.frame(), .02),
                                      [True, False, False, False, False])

    def test_unknown_or_sparse_other_view_cannot_delete_points(self):
        f=self.frame(); f['depth'][:]=0
        self.assertFalse(free_space_conflicts([[0, 0, .8]], f, .02)[0])
        f['depth'][5, 5]=1.
        self.assertFalse(free_space_conflicts([[0, 0, .8]], f, .02)[0])

    def test_nearby_foreground_blocks_false_free_space_claim(self):
        f=self.frame(); f['depth'][5, 6]=.7
        self.assertFalse(free_space_conflicts([[0, 0, .8]], f, .02)[0])

    def test_reference_camera_transform_used(self):
        f=self.frame(); f['base_from_camera'][0, 3]=.5
        self.assertTrue(free_space_conflicts([[.5, 0, .8]], f, .02)[0])
        self.assertFalse(free_space_conflicts([[0, 0, .8]], f, .02)[0])


if __name__ == '__main__':
    unittest.main()
