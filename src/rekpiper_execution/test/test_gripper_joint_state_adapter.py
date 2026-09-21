#!/usr/bin/env python3

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

from sensor_msgs.msg import JointState


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / \
    "gripper_joint_state_adapter_node.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "gripper_joint_state_adapter_test", str(SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GripperJointStateAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_callback_restores_arm_order_and_splits_total_opening(self):
        published = []
        node = self.module.GripperJointStateAdapter.__new__(
            self.module.GripperJointStateAdapter)
        node._maximum_total_opening = 0.070
        node._negative_tolerance = 0.001
        node._pub = SimpleNamespace(publish=published.append)

        message = JointState()
        message.name = ["joint4", "gripper", "joint2", "joint6",
                        "joint1", "joint5", "joint3"]
        message.position = [4.0, 0.040, 2.0, 6.0, 1.0, 5.0, 3.0]
        message.velocity = [0.4, 0.02, 0.2, 0.6, 0.1, 0.5, 0.3]
        node._callback(message)

        self.assertEqual(len(published), 1)
        output = published[0]
        self.assertEqual(
            output.name,
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
             "joint7", "joint8"])
        self.assertEqual(output.position, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                           0.020, -0.020])
        self.assertEqual(output.velocity, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                           0.01, -0.01])


if __name__ == "__main__":
    unittest.main()
