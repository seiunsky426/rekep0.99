import unittest
import numpy as np
from rekpiper_grasp.target_geometry import target_region_mask


class TargetGeometryTest(unittest.TestCase):
    def test_adjacent_object_not_in_target_region(self):
        rng=np.random.RandomState(0)
        target=rng.uniform(0,.02,(150,3))
        other=target+[.04,0,0]
        mask=target_region_mask(np.vstack([target,other]),target,other)
        self.assertTrue(mask[:150].all()); self.assertFalse(mask[150:].any())
    def test_ambiguous_ownership_rejected(self):
        points=np.zeros((150,3))
        with self.assertRaises(ValueError): target_region_mask(points,points,points)


if __name__=='__main__': unittest.main()
