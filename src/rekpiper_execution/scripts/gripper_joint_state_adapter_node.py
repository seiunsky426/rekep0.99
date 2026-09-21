#!/usr/bin/env python3
"""Publish Piper arm joints plus one active symmetric gripper coordinate."""

import rospy
from sensor_msgs.msg import JointState

from rekpiper_execution.gripper import GripperValueError, symmetric_gripper_joints


class GripperJointStateAdapter:
    def __init__(self):
        self._input = rospy.get_param("~input_topic", "/joint_states_single")
        self._output = rospy.get_param("~output_topic", "/joint_states")
        self._maximum_total_opening = float(rospy.get_param(
            "~maximum_opening_m", 0.070))
        self._negative_tolerance = float(
            rospy.get_param("~negative_opening_tolerance_m", 0.001))
        self._pub = rospy.Publisher(self._output, JointState, queue_size=2)
        rospy.Subscriber(self._input, JointState, self._callback, queue_size=10)

    def _callback(self, message):
        values = dict(zip(message.name, message.position))
        required = ["joint{}".format(index) for index in range(1, 7)]
        if any(name not in values for name in required) or "gripper" not in values:
            rospy.logwarn_throttle(
                2.0, "Piper state is missing arm joints or gripper opening")
            return
        try:
            joint7, joint8 = symmetric_gripper_joints(
                values["gripper"], self._maximum_total_opening,
                self._negative_tolerance)
        except GripperValueError as exc:
            rospy.logwarn_throttle(2.0, "Rejected gripper state: %s", exc)
            return
        output = JointState()
        output.header = message.header
        output.name = required + ["joint7", "joint8"]
        output.position = [float(values[name]) for name in required] + [joint7, joint8]
        velocity = dict(zip(message.name, message.velocity))
        effort = dict(zip(message.name, message.effort))
        if message.velocity:
            output.velocity = [
                float(velocity.get(name, 0.0)) for name in required
            ] + [0.5 * float(velocity.get("gripper", 0.0)),
                 -0.5 * float(velocity.get("gripper", 0.0))]
        if message.effort:
            output.effort = [
                float(effort.get(name, 0.0)) for name in required
            ] + [float(effort.get("gripper", 0.0)),
                 float(effort.get("gripper", 0.0))]
        self._pub.publish(output)


if __name__ == "__main__":
    rospy.init_node("rekep_gripper_joint_state_adapter")
    GripperJointStateAdapter()
    rospy.spin()
