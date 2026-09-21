#!/usr/bin/env python3
"""Publish Piper joint feedback without exposing or sending CAN commands.

This node is intentionally limited to stopped-pose calibration.  In
particular, ``piper_init=False`` suppresses the SDK's initialization queries,
and a global SocketCAN TX counter guard terminates the node if any process
transmits while the calibration receiver is active.
"""

import math
from pathlib import Path
import re
import subprocess

import rospy
from piper_msgs.msg import PiperStatusMsg
from sensor_msgs.msg import JointState

from piper_sdk import C_PiperInterface_V2


JOINT_NAMES = tuple("joint{}".format(index) for index in range(1, 7))
FEEDBACK_NAMES = JOINT_NAMES + ("gripper",)
MILLIDEGREE_TO_RADIAN = math.pi / 180000.0
MICROMETER_TO_METER = 1e-6


class ReadonlyTelemetryError(RuntimeError):
    """Raised when the calibration telemetry contract cannot be maintained."""


def require_can_ready(interface, expected_bitrate=1000000, run=subprocess.run):
    """Require an UP, ERROR-ACTIVE SocketCAN interface at the expected bitrate."""
    name = str(interface)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ReadonlyTelemetryError("invalid CAN interface name")
    try:
        result = run(
            ["ip", "-details", "link", "show", "dev", name],
            capture_output=True, text=True, check=False, timeout=2.0)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReadonlyTelemetryError(
            "could not inspect {}: {}".format(name, exc)) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "interface missing").strip()
        raise ReadonlyTelemetryError(
            "{} is unavailable: {}".format(name, detail))
    flags = re.search(r"<([^>]+)>", result.stdout)
    state = re.search(r"\bcan state\s+([A-Z-]+)", result.stdout)
    bitrate = re.search(r"\bbitrate\s+(\d+)", result.stdout)
    if not flags or not state or not bitrate:
        raise ReadonlyTelemetryError(
            "could not parse SocketCAN state for " + name)
    if "UP" not in set(flags.group(1).split(",")):
        raise ReadonlyTelemetryError(name + " is not UP")
    if state.group(1) != "ERROR-ACTIVE":
        raise ReadonlyTelemetryError(
            "{} state is {}, expected ERROR-ACTIVE".format(
                name, state.group(1)))
    if int(bitrate.group(1)) != int(expected_bitrate):
        raise ReadonlyTelemetryError(
            "{} bitrate is {}, expected {}".format(
                name, bitrate.group(1), expected_bitrate))


def can_tx_packets(interface, root=Path("/sys/class/net")):
    path = root / str(interface) / "statistics" / "tx_packets"
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError) as exc:
        raise ReadonlyTelemetryError(
            "cannot read {} TX counter: {}".format(interface, exc)) from exc


def feedback_values(feedback):
    state = feedback.joint_state
    raw = (
        state.joint_1, state.joint_2, state.joint_3,
        state.joint_4, state.joint_5, state.joint_6)
    return [float(value) * MILLIDEGREE_TO_RADIAN for value in raw]


def gripper_opening(feedback):
    """Convert the SDK's 0.001 mm total jaw stroke to metres."""
    return float(feedback.gripper_state.grippers_angle) * MICROMETER_TO_METER


def status_message(feedback):
    """Convert one passive 0x2A1 feedback frame to the ROS status message."""
    state = feedback.arm_status
    error = state.err_status
    message = PiperStatusMsg()
    message.ctrl_mode = int(state.ctrl_mode)
    message.arm_status = int(state.arm_status)
    message.mode_feedback = int(state.mode_feed)
    message.teach_status = int(state.teach_status)
    message.motion_status = int(state.motion_status)
    message.trajectory_num = int(state.trajectory_num)
    message.err_code = int(state.err_code)
    message.joint_1_angle_limit = bool(error.joint_1_angle_limit)
    message.joint_2_angle_limit = bool(error.joint_2_angle_limit)
    message.joint_3_angle_limit = bool(error.joint_3_angle_limit)
    message.joint_4_angle_limit = bool(error.joint_4_angle_limit)
    message.joint_5_angle_limit = bool(error.joint_5_angle_limit)
    message.joint_6_angle_limit = bool(error.joint_6_angle_limit)
    message.communication_status_joint_1 = bool(
        error.communication_status_joint_1)
    message.communication_status_joint_2 = bool(
        error.communication_status_joint_2)
    message.communication_status_joint_3 = bool(
        error.communication_status_joint_3)
    message.communication_status_joint_4 = bool(
        error.communication_status_joint_4)
    message.communication_status_joint_5 = bool(
        error.communication_status_joint_5)
    message.communication_status_joint_6 = bool(
        error.communication_status_joint_6)
    return message


class PiperJointStateReadonly:
    def __init__(self):
        self._can_port = str(rospy.get_param("~can_port", "can0"))
        self._rate_hz = float(rospy.get_param("~publish_rate_hz", 100.0))
        self._feedback_timeout_s = float(rospy.get_param(
            "~feedback_timeout_s", 0.20))
        self._startup_timeout_s = float(rospy.get_param(
            "~startup_timeout_s", 2.0))
        self._fail_on_can_tx = bool(rospy.get_param(
            "~fail_on_can_tx", True))
        if self._rate_hz <= 0.0 or self._feedback_timeout_s <= 0.0 or \
                self._startup_timeout_s <= 0.0:
            raise ReadonlyTelemetryError("telemetry timing parameters must be positive")

        require_can_ready(self._can_port)
        self._tx_baseline = can_tx_packets(self._can_port)
        self._publisher = rospy.Publisher(
            "/joint_states_single", JointState, queue_size=1)
        self._status_publisher = rospy.Publisher(
            "/arm_status", PiperStatusMsg, queue_size=1)
        self._piper = C_PiperInterface_V2(can_name=self._can_port)
        self._closed = False
        self._last_feedback_stamp = None
        self._last_status_stamp = None
        try:
            # Critical safety property: the default PiperInit sends 13 query
            # frames; calibration must not execute it.
            self._piper.ConnectPort(piper_init=False)
            rospy.on_shutdown(self.close)
            self._wait_for_feedback()
            self._assert_zero_tx()
        except Exception:
            self.close()
            raise
        rospy.loginfo(
            "Piper calibration telemetry ready on %s; no command interfaces "
            "exist and CAN TX baseline is %d",
            self._can_port, self._tx_baseline)

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            self._piper.DisconnectPort()
        except Exception as exc:  # shutdown must continue even if CAN vanished
            rospy.logwarn("Piper read-only disconnect failed: %s", exc)

    def _assert_zero_tx(self):
        if not self._fail_on_can_tx:
            return
        current = can_tx_packets(self._can_port)
        if current != self._tx_baseline:
            raise ReadonlyTelemetryError(
                "CAN TX changed from {} to {} during read-only calibration; "
                "another process or an unsafe SDK path transmitted"
                .format(self._tx_baseline, current))

    def _feedback(self):
        feedback = self._piper.GetArmJointMsgs()
        gripper = self._piper.GetArmGripperMsgs()
        stamp = float(feedback.time_stamp)
        gripper_stamp = float(gripper.time_stamp)
        now = rospy.Time.now().to_sec()
        # The SDK's rolling ``Hz`` statistic briefly returns zero when its
        # internal window rotates, even though fresh joint frames continue to
        # arrive.  The per-feedback wall-clock timestamp is the authoritative
        # liveness signal here.
        if stamp <= 0.0:
            raise ReadonlyTelemetryError("waiting for complete Piper joint feedback")
        if gripper_stamp <= 0.0:
            raise ReadonlyTelemetryError("waiting for Piper gripper feedback")
        if abs(now - stamp) > self._feedback_timeout_s:
            raise ReadonlyTelemetryError(
                "Piper joint feedback is stale by {:.3f}s".format(now - stamp))
        if abs(now - gripper_stamp) > self._feedback_timeout_s:
            raise ReadonlyTelemetryError(
                "Piper gripper feedback is stale by {:.3f}s".format(
                    now - gripper_stamp))
        values = feedback_values(feedback) + [gripper_opening(gripper)]
        if not all(math.isfinite(value) for value in values):
            raise ReadonlyTelemetryError("Piper joint feedback is non-finite")
        return stamp, values

    def _wait_for_feedback(self):
        deadline = rospy.Time.now().to_sec() + self._startup_timeout_s
        rate = rospy.Rate(min(self._rate_hz, 100.0))
        last_error = "no feedback"
        while not rospy.is_shutdown() and rospy.Time.now().to_sec() < deadline:
            try:
                self._feedback()
                self._status()
                return
            except ReadonlyTelemetryError as exc:
                last_error = str(exc)
                rate.sleep()
        raise ReadonlyTelemetryError(
            "Piper feedback did not become ready: " + last_error)

    def _status(self):
        feedback = self._piper.GetArmStatus()
        stamp = float(feedback.time_stamp)
        now = rospy.Time.now().to_sec()
        if stamp <= 0.0:
            raise ReadonlyTelemetryError("waiting for Piper arm status")
        if abs(now - stamp) > self._feedback_timeout_s:
            raise ReadonlyTelemetryError(
                "Piper arm status is stale by {:.3f}s".format(now - stamp))
        return stamp, status_message(feedback)

    def spin(self):
        rate = rospy.Rate(self._rate_hz)
        while not rospy.is_shutdown():
            try:
                self._assert_zero_tx()
                stamp, positions = self._feedback()
                if stamp != self._last_feedback_stamp:
                    message = JointState()
                    message.header.stamp = rospy.Time.from_sec(stamp)
                    message.name = list(FEEDBACK_NAMES)
                    message.position = positions
                    self._publisher.publish(message)
                    self._last_feedback_stamp = stamp
                status_stamp, status = self._status()
                if status_stamp != self._last_status_stamp:
                    self._status_publisher.publish(status)
                    self._last_status_stamp = status_stamp
            except ReadonlyTelemetryError as exc:
                rospy.logfatal("Read-only Piper telemetry stopped: %s", exc)
                rospy.signal_shutdown(str(exc))
                break
            rate.sleep()


def main():
    rospy.init_node("piper_joint_state_readonly")
    try:
        node = PiperJointStateReadonly()
        node.spin()
    except ReadonlyTelemetryError as exc:
        rospy.logfatal("Cannot start read-only Piper telemetry: %s", exc)
        raise SystemExit(1)
    finally:
        if "node" in locals():
            node.close()


if __name__ == "__main__":
    main()
