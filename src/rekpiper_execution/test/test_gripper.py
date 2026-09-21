#!/usr/bin/env python3

import unittest

from rekpiper_execution.gripper import (
    GripperValueError, piper_command_from_total_opening,
    piper_feedback_to_total_opening, symmetric_gripper_joints)


class GripperTest(unittest.TestCase):
    def test_total_opening_maps_to_symmetric_finger_travel(self):
        self.assertEqual(symmetric_gripper_joints(0.070), (0.035, -0.035))
        self.assertEqual(symmetric_gripper_joints(0.0), (0.0, -0.0))

    def test_feedback_and_command_share_70mm_physical_limit(self):
        self.assertEqual(piper_feedback_to_total_opening(0.070), 0.070)
        self.assertEqual(piper_command_from_total_opening(0.035), 0.035)
        with self.assertRaises(GripperValueError):
            piper_command_from_total_opening(0.071)

    def test_small_negative_feedback_bias_clamps_only_on_feedback_path(self):
        self.assertEqual(piper_feedback_to_total_opening(-0.0002), 0.0)
        with self.assertRaises(GripperValueError):
            piper_command_from_total_opening(-0.0002)


if __name__ == "__main__":
    unittest.main()
