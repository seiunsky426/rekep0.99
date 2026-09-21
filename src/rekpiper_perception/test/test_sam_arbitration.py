#!/usr/bin/env python3

import unittest

import numpy as np

from rekpiper_perception.sam_automatic import (
    arbitrate_quality_ordered_masks, masks_to_label_map)


class SAMArbitrationTest(unittest.TestCase):
    def test_high_quality_mask_owns_overlap_and_groups_are_contiguous(self):
        high = np.zeros((8, 8), dtype=bool)
        low = np.zeros((8, 8), dtype=bool)
        empty = np.zeros((8, 8), dtype=bool)
        high[1:6, 1:6] = True
        low[4:8, 4:8] = True
        empty[2:4, 2:4] = True
        result = arbitrate_quality_ordered_masks([high, low, empty])
        self.assertEqual(len(result), 2)
        self.assertFalse(np.any(result[0] & result[1]))
        labels = masks_to_label_map(result)
        self.assertEqual(labels[4, 4], 1)
        self.assertEqual(set(np.unique(labels)), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
