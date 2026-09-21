#!/usr/bin/env python3

from types import SimpleNamespace
import unittest

from rekpiper_execution.lifecycle_contract import (
    LifecycleBindingError, grasp_batch_matches_binding,
    validate_grasp_candidate_binding)


def candidate(**changes):
    values = dict(
        candidate_id="K2-001", planning_safe=True,
        planning_authorized=False, rigid_group_id=7,
        object_uuid="object-uuid", session_id="program-session",
        program_sha256="program-hash", snapshot_id="snapshot-id",
        map_generation_uuid="map-generation", stage_index=3,
        grasp_attempt=2)
    values.update(changes)
    return SimpleNamespace(**values)


class LifecycleContractTest(unittest.TestCase):
    def _validate(self, value):
        return validate_grasp_candidate_binding(
            value, 7, "object-uuid", "program-session", "program-hash",
            "snapshot-id", "map-generation", 3, 2)

    def test_exact_binding_is_accepted(self):
        self.assertTrue(self._validate(candidate()))

    def test_every_generation_mismatch_is_rejected(self):
        mutations = {
            "rigid_group_id": 8,
            "object_uuid": "wrong-object",
            "session_id": "wrong-session",
            "program_sha256": "wrong-program",
            "snapshot_id": "wrong-snapshot",
            "map_generation_uuid": "wrong-map",
            "stage_index": 4,
            "grasp_attempt": 3,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(LifecycleBindingError, field):
                    self._validate(candidate(**{field: value}))

    def test_unaudited_or_already_authorized_candidate_is_rejected(self):
        for change in ({"planning_safe": False},
                       {"planning_authorized": True},
                       {"candidate_id": ""}):
            with self.assertRaises(LifecycleBindingError):
                self._validate(candidate(**change))

    def test_empty_inference_batch_still_has_an_immutable_binding(self):
        batch = SimpleNamespace(
            planning_authorized=False, session_id="program-session",
            program_sha256="program-hash", snapshot_id="snapshot-id",
            map_generation_uuid="map-generation", stage_index=3,
            grasp_attempt=2, candidates=[], status="no_safe_anygrasp_candidates")
        self.assertTrue(grasp_batch_matches_binding(
            batch, "program-session", "program-hash", "snapshot-id",
            "map-generation", 3, 2))
        batch.grasp_attempt = 1
        self.assertFalse(grasp_batch_matches_binding(
            batch, "program-session", "program-hash", "snapshot-id",
            "map-generation", 3, 2))


if __name__ == "__main__":
    unittest.main()
