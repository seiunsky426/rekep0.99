#!/usr/bin/env python3

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / \
    "piper_joint_state_readonly_node.py"


def load_module():
    spec = importlib.util.spec_from_file_location("piper_readonly_test", str(SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PiperJointStateReadonlyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_feedback_unit_conversion(self):
        state = SimpleNamespace(**{
            "joint_{}".format(index): index * 1000 for index in range(1, 7)})
        values = self.module.feedback_values(SimpleNamespace(joint_state=state))
        self.assertAlmostEqual(values[0], math.pi / 180.0)
        self.assertAlmostEqual(values[-1], 6.0 * math.pi / 180.0)

    def test_gripper_total_opening_conversion(self):
        feedback = SimpleNamespace(
            gripper_state=SimpleNamespace(grippers_angle=35000))
        self.assertAlmostEqual(self.module.gripper_opening(feedback), 0.035)

    def test_passive_arm_status_conversion(self):
        error = SimpleNamespace(**{
            "joint_{}_angle_limit".format(index): index == 2
            for index in range(1, 7)})
        for index in range(1, 7):
            setattr(error, "communication_status_joint_{}".format(index),
                    index == 5)
        state = SimpleNamespace(
            ctrl_mode=2, arm_status=1, mode_feed=1, teach_status=2,
            motion_status=0, trajectory_num=3, err_code=18,
            err_status=error)
        message = self.module.status_message(
            SimpleNamespace(arm_status=state))
        self.assertEqual(message.ctrl_mode, 2)
        self.assertEqual(message.arm_status, 1)
        self.assertTrue(message.joint_2_angle_limit)
        self.assertTrue(message.communication_status_joint_5)
        self.assertFalse(message.communication_status_joint_4)

    def test_sdk_connect_explicitly_disables_initialization(self):
        calls = []

        class FakePiper:
            def __init__(self, can_name):
                calls.append(("construct", can_name))

            def ConnectPort(self, **kwargs):
                calls.append(("connect", kwargs))

            def GetArmJointMsgs(self):
                state = SimpleNamespace(**{
                    "joint_{}".format(index): 0 for index in range(1, 7)})
                return SimpleNamespace(
                    time_stamp=100.0, Hz=200.0, joint_state=state)

            def DisconnectPort(self):
                calls.append(("disconnect",))

        params = {
            "~can_port": "can0", "~publish_rate_hz": 100.0,
            "~feedback_timeout_s": 0.2, "~startup_timeout_s": 2.0,
            "~fail_on_can_tx": True,
        }
        with mock.patch.object(
                self.module.rospy, "get_param",
                side_effect=lambda name, default=None: params.get(name, default)), \
                mock.patch.object(self.module, "require_can_ready"), \
                mock.patch.object(self.module, "can_tx_packets", return_value=7), \
                mock.patch.object(self.module.rospy, "Publisher"), \
                mock.patch.object(self.module.rospy, "on_shutdown"), \
                mock.patch.object(self.module.rospy, "loginfo"), \
                mock.patch.object(
                    self.module.PiperJointStateReadonly,
                    "_wait_for_feedback"), \
                mock.patch.object(self.module, "C_PiperInterface_V2", FakePiper):
            node = self.module.PiperJointStateReadonly()
            node.close()
        self.assertIn(("connect", {"piper_init": False}), calls)
        forbidden = {"MotionCtrl_1", "MotionCtrl_2", "JointCtrl",
                     "GripperCtrl", "EnableArm", "EnablePiper"}
        self.assertFalse(any(call[0] in forbidden for call in calls))

    def test_tx_counter_change_is_rejected(self):
        node = self.module.PiperJointStateReadonly.__new__(
            self.module.PiperJointStateReadonly)
        node._can_port = "can0"
        node._fail_on_can_tx = True
        node._tx_baseline = 10
        with mock.patch.object(self.module, "can_tx_packets", return_value=11):
            with self.assertRaises(self.module.ReadonlyTelemetryError):
                node._assert_zero_tx()

    def test_fresh_feedback_does_not_depend_on_sdk_hz_window(self):
        state = SimpleNamespace(**{
            "joint_{}".format(index): index for index in range(1, 7)})
        node = self.module.PiperJointStateReadonly.__new__(
            self.module.PiperJointStateReadonly)
        node._feedback_timeout_s = 0.2
        node._piper = SimpleNamespace(GetArmJointMsgs=lambda: SimpleNamespace(
            time_stamp=100.0, Hz=0.0, joint_state=state),
            GetArmGripperMsgs=lambda: SimpleNamespace(
                time_stamp=100.04, Hz=0.0,
                gripper_state=SimpleNamespace(grippers_angle=70000)))
        with mock.patch.object(
                self.module.rospy.Time, "now",
                return_value=SimpleNamespace(to_sec=lambda: 100.05)):
            stamp, positions = node._feedback()
        self.assertEqual(stamp, 100.0)
        self.assertEqual(len(positions), 7)
        self.assertAlmostEqual(positions[-1], 0.070)


if __name__ == "__main__":
    unittest.main()
