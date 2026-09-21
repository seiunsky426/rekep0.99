#!/usr/bin/env python3

from types import SimpleNamespace
import unittest

import numpy as np

from rekpiper_perception.tracking_geometry import organized_xyz


class TrackingGeometryTest(unittest.TestCase):
    def test_decodes_only_canonical_organized_xyz(self):
        values = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
        fields = [SimpleNamespace(name=name, offset=offset)
                  for name, offset in (("x", 0), ("y", 4), ("z", 8))]
        message = SimpleNamespace(
            fields=fields, point_step=12, row_step=36,
            width=3, height=2, data=values.tobytes())
        np.testing.assert_array_equal(organized_xyz(message), values)

    def test_rejects_unorganized_cloud(self):
        message = SimpleNamespace(
            fields=[], point_step=12, row_step=12,
            width=1, height=1, data=b"\0" * 12)
        with self.assertRaises(ValueError):
            organized_xyz(message)


if __name__ == "__main__":
    unittest.main()
