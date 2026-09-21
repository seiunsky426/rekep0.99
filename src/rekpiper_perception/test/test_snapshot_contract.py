#!/usr/bin/env python3

from types import SimpleNamespace
import unittest

from rekpiper_perception.snapshot_contract import validate_snapshot_layout


class SnapshotContractTest(unittest.TestCase):
    def test_many_keypoints_may_bind_one_unique_instance(self):
        keypoints = [
            SimpleNamespace(id=0, name="K0", rigid_group_id=1),
            SimpleNamespace(id=1, name="K1", rigid_group_id=1),
            SimpleNamespace(id=2, name="K2", rigid_group_id=2),
        ]
        instances = [SimpleNamespace(rigid_group_id=1),
                     SimpleNamespace(rigid_group_id=2)]
        self.assertEqual(
            validate_snapshot_layout(keypoints, instances),
            (True, "immutable_snapshot_locked"))

    def test_missing_or_duplicate_instance_identity_is_rejected(self):
        keypoints = [SimpleNamespace(id=0, name="K0", rigid_group_id=2)]
        self.assertFalse(validate_snapshot_layout(
            keypoints, [SimpleNamespace(rigid_group_id=1)])[0])
        self.assertFalse(validate_snapshot_layout(
            keypoints, [SimpleNamespace(rigid_group_id=2),
                        SimpleNamespace(rigid_group_id=2)])[0])


if __name__ == "__main__":
    unittest.main()
