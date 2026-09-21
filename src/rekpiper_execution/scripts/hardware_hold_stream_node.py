#!/usr/bin/env python3
"""Authorize one immutable, long-duration startup-pose hold trajectory."""

from copy import deepcopy
import os
import threading

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from rekpiper_acceptance import (
    AcceptanceError, assert_release_unchanged, validate_release_bundle)
from rekpiper_execution.trajectory import JOINT_NAMES, normalize_feedback_to_joint_limits
from rekpiper_msgs.msg import SafeMappingStatus


class HardwareHoldStream:
    def __init__(self):
        if os.environ.get("REKPIPER_ACCEPTANCE_BUNDLE_TYPE") != \
                "hardware_acceptance_bundle":
            raise rospy.ROSInitException(
                "hardware hold requires hardware_acceptance_bundle")
        try:
            self._release = validate_release_bundle(
                str(rospy.get_param("~release_bundle", "")),
                str(rospy.get_param("~acceptance_public_key", "")),
                str(rospy.get_param("~minimum_release_counter", "")),
                str(rospy.get_param("~robot_id", "piper-rekpiper")))
        except AcceptanceError as exc:
            raise rospy.ROSInitException("hardware hold bundle rejected: " + str(exc))
        self._lock = threading.RLock()
        self._reference = None
        self._map = None
        self._started = False
        self._armed_pub = rospy.Publisher(
            rospy.get_param(
                "~armed_topic", "/rekpiper/execution/hardware_hold_armed"),
            Bool, queue_size=1, latch=True)
        self._armed_pub.publish(Bool(data=False))
        rospy.Subscriber("/joint_states_single", JointState,
                         self._joint, queue_size=20)
        rospy.Subscriber("/rekpiper/mapping/safe_status", SafeMappingStatus,
                         self._map_status, queue_size=5)
        self._client = actionlib.SimpleActionClient(
            "/manipulator_controller/follow_joint_trajectory",
            FollowJointTrajectoryAction)
        rospy.Timer(rospy.Duration(0.2), self._try_start, oneshot=False)
        rospy.on_shutdown(lambda: self._armed_pub.publish(Bool(data=False)))

    def _joint(self, message):
        values = dict(zip(message.name, message.position))
        if any(name not in values for name in JOINT_NAMES):
            return
        try:
            joints = normalize_feedback_to_joint_limits(
                [values[name] for name in JOINT_NAMES], 0.01)
        except ValueError:
            return
        with self._lock:
            if self._reference is None:
                self._reference = joints

    def _map_status(self, message):
        with self._lock:
            self._map = deepcopy(message)

    def _try_start(self, _event):
        with self._lock:
            if self._started or self._reference is None or self._map is None:
                return
            if (self._map.state != SafeMappingStatus.READY
                    or not self._map.map_query_allowed
                    or not self._map.planning_safe
                    or not self._map.map_generation_uuid):
                return
            reference = self._reference.copy()
            generation = str(self._map.map_generation_uuid)
        try:
            assert_release_unchanged(self._release)
            rospy.wait_for_service("/enable_status_srv", timeout=0.1)
            if not rospy.ServiceProxy("/enable_status_srv", Trigger)().success:
                return
            if not self._client.wait_for_server(rospy.Duration(0.1)):
                return
        except (AcceptanceError, rospy.ROSException, rospy.ServiceException):
            return
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.stamp = rospy.Time.now()
        goal.trajectory.header.frame_id = generation
        goal.trajectory.joint_names = list(JOINT_NAMES)
        for seconds in (0.0, 660.0):
            point = JointTrajectoryPoint()
            point.positions = reference.tolist()
            point.time_from_start = rospy.Duration(seconds)
            goal.trajectory.points.append(point)
        with self._lock:
            self._started = True
        self._armed_pub.publish(Bool(data=True))
        self._client.send_goal(goal, done_cb=self._done)
        rospy.logwarn(
            "HOLD-ONLY acceptance stream started; gripper and task motion are disabled")

    def _done(self, state, result):
        self._armed_pub.publish(Bool(data=False))
        rospy.logerr("hardware hold stream ended: state=%s result=%s", state, result)
        rospy.signal_shutdown("hardware hold stream ended")


if __name__ == "__main__":
    rospy.init_node("rekpiper_hardware_hold_stream")
    HardwareHoldStream()
    rospy.spin()
