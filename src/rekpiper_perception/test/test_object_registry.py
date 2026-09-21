#!/usr/bin/env python3

import unittest

from rekpiper_perception.object_registry import (
    CAMERA_LOST, CAMERA_TRACKED, CAMERA_TRUSTED_NOT_VISIBLE, ObjectRegistry)


class ObjectRegistryTest(unittest.TestCase):
    def test_uuid_is_never_reused_and_revision_changes(self):
        identities = iter(("one", "two"))
        registry = ObjectRegistry(uuid_factory=lambda: next(identities))
        first = registry.register("rs1", "cup", 1)
        registry.retire(first.object_uuid, now_s=2.0)
        second = registry.register("rs3", "box", 2)
        self.assertEqual(first.object_uuid, "one")
        self.assertEqual(second.object_uuid, "two")
        self.assertEqual(registry.exclusion_signature(), ("two",))
        self.assertEqual(registry.revision, 3)

    def test_all_cameras_must_be_current_and_safe(self):
        registry = ObjectRegistry(uuid_factory=lambda: "one")
        record = registry.register("rs1", "cup", 1)
        registry.update_camera(record.object_uuid, "rs1", CAMERA_TRACKED, 0.9, 10.0)
        registry.update_camera(
            record.object_uuid, "rs3", CAMERA_TRUSTED_NOT_VISIBLE, 1.0, 10.0)
        self.assertTrue(registry.all_excluded_objects_safe(10.1, 0.2))
        self.assertFalse(registry.all_excluded_objects_safe(10.3, 0.2))
        registry.update_camera(record.object_uuid, "rs3", CAMERA_LOST, 0.0, 10.4)
        self.assertFalse(registry.all_excluded_objects_safe(10.4, 0.2))

    def test_nonexcluded_tracking_does_not_change_exclusion_generation(self):
        registry = ObjectRegistry(uuid_factory=lambda: "bottle")
        record = registry.register(
            "rs1", "bottle", 1, excluded_from_static_map=False)
        self.assertEqual(registry.exclusion_revision, 0)
        self.assertEqual(registry.exclusion_signature(), ())

        self.assertTrue(registry.set_excluded_from_static_map(
            record.object_uuid, True))
        self.assertEqual(registry.exclusion_revision, 1)
        self.assertEqual(registry.exclusion_signature(), ("bottle",))

        self.assertFalse(registry.set_excluded_from_static_map(
            record.object_uuid, True))
        self.assertEqual(registry.exclusion_revision, 1)

        self.assertTrue(registry.set_excluded_from_static_map(
            record.object_uuid, False))
        self.assertEqual(registry.exclusion_revision, 2)
        self.assertEqual(registry.exclusion_signature(), ())
        registry.retire(record.object_uuid)
        self.assertEqual(registry.exclusion_revision, 2)

    def test_rigid_group_identity_is_positive_and_unique(self):
        identities = iter(("one", "two", "three"))
        registry = ObjectRegistry(uuid_factory=lambda: next(identities))
        first = registry.register("rs1", "cup", 7, False)
        self.assertEqual(first.rigid_group_id, 7)
        with self.assertRaisesRegex(ValueError, "positive"):
            registry.register("rs1", "invalid", 0, False)
        with self.assertRaisesRegex(ValueError, "already"):
            registry.register("rs3", "duplicate", 7, False)


if __name__ == "__main__":
    unittest.main()
