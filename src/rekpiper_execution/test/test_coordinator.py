#!/usr/bin/env python3

import unittest

import numpy as np

from rekpiper_execution.coordinator import ReKepCoordinatorCore, pose_reached


class CoordinatorTest(unittest.TestCase):
    def test_official_reverse_backtrack_and_stage_one_fallback(self):
        core = ReKepCoordinatorCore(4, stage=4)
        good = lambda _ee, _k: 0.05
        bad = lambda _ee, _k: 0.11
        sets = [[], [good], [bad], [bad]]
        selected = core.select_backtrack_stage(
            sets, np.zeros(3), np.zeros((1, 3)), 0.10)
        self.assertEqual(selected, 2)
        core.enter_stage(4)
        selected = core.select_backtrack_stage(
            [[], [bad], [bad], [bad]], np.zeros(3), np.zeros((1, 3)), 0.10)
        self.assertEqual(selected, 1)

    def test_stage_entry_resets_grasp_attempt(self):
        core = ReKepCoordinatorCore(3)
        self.assertEqual(core.record_grasp_failure(), 1)
        self.assertEqual(core.record_grasp_failure(), 2)
        core.enter_stage(2)
        self.assertEqual(core.grasp_attempt, 0)

    def test_pose_reached_requires_position_and_orientation_three_ticks(self):
        target = np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
        self.assertTrue(pose_reached(target, target))
        translated = target.copy()
        translated[0] += 0.011
        self.assertFalse(pose_reached(translated, target))
        rotated = target.copy()
        angle = 0.11
        rotated[5:7] = [np.sin(angle / 2.0), np.cos(angle / 2.0)]
        self.assertFalse(pose_reached(rotated, target))


if __name__ == "__main__":
    unittest.main()
