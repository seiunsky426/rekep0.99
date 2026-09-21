#!/usr/bin/env python3

import unittest

from rekpiper_mapping.safe_map_state import SafeMapState, SafeMapSupervisor


class SafeMapStateTest(unittest.TestCase):
    def test_full_rebuild_sequence(self):
        supervisor = SafeMapSupervisor(
            warmup_frames_per_camera=2, build_frames_per_camera=3,
            minimum_esdf_updates=2)
        generation = supervisor.begin_rebuild("hash", generation_uuid="generation")
        self.assertEqual(generation, "generation")
        supervisor.cleared()
        for _ in range(2):
            supervisor.record_valid_frame("rs1")
            supervisor.record_valid_frame("rs3")
        self.assertEqual(supervisor.state, SafeMapState.BUILDING)
        for _ in range(3):
            supervisor.record_valid_frame("rs1")
            supervisor.record_valid_frame("rs3")
        self.assertEqual(supervisor.state, SafeMapState.VALIDATING)
        supervisor.record_esdf_update()
        supervisor.record_esdf_update()
        self.assertEqual(supervisor.state, SafeMapState.READY)
        self.assertTrue(supervisor.planning_safe)

    def test_pause_and_dirty_withdraw_safety(self):
        supervisor = SafeMapSupervisor()
        supervisor.state = SafeMapState.READY
        supervisor.pause("mask_lost")
        self.assertFalse(supervisor.planning_safe)
        supervisor.mark_dirty("selection_changed")
        self.assertEqual(supervisor.state, SafeMapState.DIRTY)
        self.assertEqual(supervisor.generation_uuid, "")


if __name__ == "__main__":
    unittest.main()
