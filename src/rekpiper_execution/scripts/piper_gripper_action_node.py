#!/usr/bin/env python3
"""Feedback-closed Piper gripper action with the same one-shot arm gate."""

import math
import threading
import time

import actionlib
from piper_msgs.msg import PiperStatusMsg
from piper_msgs.srv import Gripper, GripperRequest
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
import yaml

from rekpiper_acceptance import (
    AcceptanceError, assert_release_unchanged, validate_release_bundle)
from rekpiper_execution.gripper import (
    piper_command_from_total_opening,
    piper_feedback_to_total_opening,
)
from rekpiper_execution.gripper_close import ContactDetector, stepped_close_targets
from rekpiper_msgs.msg import (
    CommandGripperAction, CommandGripperFeedback, CommandGripperResult,
)


class PiperGripperAction:
    def __init__(self):
        self._allow = bool(rospy.get_param("~allow_hardware_commands", False))
        self._lock = threading.RLock()
        self._opening = None
        self._effort = None
        self._feedback_time = self._arm_time = 0.0
        self._arm_ok = self._armed = False
        self._timeout = float(rospy.get_param("~feedback_timeout_s", 0.25))
        self._service = rospy.get_param("~gripper_service", "/gripper_srv")
        self._stop_service = rospy.get_param("~stop_service", "/stop_srv")
        try:
            self._release = validate_release_bundle(
                str(rospy.get_param("~release_bundle", "")),
                str(rospy.get_param("~acceptance_public_key", "")),
                str(rospy.get_param("~minimum_release_counter", "")),
                expected_robot_id=str(rospy.get_param(
                    "~robot_id", "piper-rekpiper")))
            self._baseline = self._release[
                "verified_artifacts"]["gripper_baseline"]["payload"]
        except (AcceptanceError, KeyError, TypeError, ValueError) as exc:
            raise rospy.ROSInitException(
                "signed gripper baseline rejected: {}".format(exc))
        self._baseline_ready = bool(self._baseline.get("calibrated", False))
        self._evidence_mode = str(self._baseline.get(
            "contact_evidence_mode", "encoder_effort_vision"))
        if self._evidence_mode not in (
                "encoder_effort_vision", "encoder_visual"):
            raise rospy.ROSInitException("invalid contact_evidence_mode")
        rospy.Subscriber("/joint_states_single", JointState,
                         self._joint_cb, queue_size=20)
        rospy.Subscriber("/arm_status", PiperStatusMsg,
                         self._arm_cb, queue_size=10)
        rospy.Subscriber("/rekpiper/execution/hardware_armed", Bool,
                         self._armed_cb, queue_size=2)
        self._server = actionlib.SimpleActionServer(
            "/rekpiper/execution/command_gripper", CommandGripperAction,
            execute_cb=self._execute, auto_start=False)
        self._server.start()

    def _joint_cb(self, value):
        try:
            index = value.name.index("gripper")
        except ValueError:
            return
        try:
            opening = piper_feedback_to_total_opening(value.position[index])
            effort = (abs(float(value.effort[index]))
                      if len(value.effort) > index else None)
            if effort is not None and not math.isfinite(effort):
                effort = None
        except ValueError:
            return
        with self._lock:
            self._opening = opening
            self._effort = effort
            self._feedback_time = time.monotonic()

    def _arm_cb(self, value):
        with self._lock:
            self._arm_ok = bool(value.ctrl_mode == 1 and value.arm_status == 0
                                and value.mode_feedback == 1 and value.err_code == 0)
            self._arm_time = time.monotonic()

    def _armed_cb(self, value):
        with self._lock:
            self._armed = bool(value.data)

    def _health(self):
        with self._lock:
            now = time.monotonic()
            if not self._allow:
                return False, "hardware_commands_disabled"
            try:
                assert_release_unchanged(self._release)
            except AcceptanceError as exc:
                self._armed = False
                return False, "signed_release_changed:" + str(exc)
            if not self._baseline_ready:
                return False, "calibrated_gripper_baseline_required"
            if not self._armed:
                return False, "execution_not_armed"
            if not self._arm_ok or now - self._arm_time > self._timeout:
                return False, "piper_status_invalid_or_stale"
            if self._opening is None or now - self._feedback_time > self._timeout:
                return False, "gripper_feedback_stale"
            if (self._evidence_mode == "encoder_effort_vision"
                    and self._effort is None):
                return False, "gripper_effort_feedback_unavailable"
            return True, "ok"

    def _result(self, success, contact, status, command_error_m=0.0):
        with self._lock:
            opening = 0.0 if self._opening is None else self._opening
            effort = 0.0 if self._effort is None else self._effort
        return CommandGripperResult(
            success=success, stable_contact=contact,
            final_opening_m=opening, final_effort=effort,
            command_error_m=float(command_error_m),
            evidence_mode=self._evidence_mode, status=status)

    def _safe_stop(self):
        if not self._allow:
            return
        try:
            rospy.wait_for_service(self._stop_service, timeout=0.2)
            rospy.ServiceProxy(self._stop_service, Trigger)()
        except Exception as exc:
            rospy.logerr("Piper stop after gripper fault failed: %s", exc)

    def _abort(self, reason):
        self._safe_stop()
        self._server.set_aborted(self._result(False, False, reason), reason)

    def _command(self, opening, effort):
        request = GripperRequest(
            gripper_angle=piper_command_from_total_opening(opening),
            gripper_effort=float(effort), gripper_code=1, set_zero=0)
        response = rospy.ServiceProxy(self._service, Gripper)(request)
        if not response.status:
            raise RuntimeError("gripper_service_rejected")

    def _feedback(self, phase):
        with self._lock:
            opening = 0.0 if self._opening is None else self._opening
            effort = 0.0 if self._effort is None else self._effort
        self._server.publish_feedback(CommandGripperFeedback(
            phase=phase, measured_opening_m=opening,
            measured_effort=effort))

    def _execute(self, goal):
        healthy, reason = self._health()
        if not healthy:
            self._abort(reason)
            return
        try:
            rospy.wait_for_service(self._service, timeout=1.0)
            if goal.command == goal.OPEN:
                target = float(goal.total_opening_m)
                if not 0.0 < target <= 0.070:
                    raise RuntimeError("open_target_out_of_range")
                self._command(target, goal.effort)
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    if self._server.is_preempt_requested():
                        raise InterruptedError()
                    healthy, reason = self._health()
                    if not healthy:
                        raise RuntimeError(reason)
                    self._feedback("OPENING")
                    with self._lock:
                        if abs(self._opening-target) <= 0.003:
                            self._server.set_succeeded(
                                self._result(True, False, "open_verified"))
                            return
                    rospy.sleep(0.02)
                raise RuntimeError("open_timeout")
            if goal.command != goal.CLOSE or not goal.object_uuid:
                raise RuntimeError("invalid_close_goal")
            width = float(goal.total_opening_m)
            if not 0.0 < width <= 0.070:
                raise RuntimeError("close_width_out_of_range")
            with self._lock:
                start = self._opening
            commands = stepped_close_targets(start, width)
            detector = ContactDetector()
            index, commanded = 0, start
            next_command = time.monotonic()
            deadline = next_command + 6.0
            decision = None
            while time.monotonic() < deadline:
                if self._server.is_preempt_requested():
                    raise InterruptedError()
                healthy, reason = self._health()
                if not healthy:
                    raise RuntimeError(reason)
                now = time.monotonic()
                if index < len(commands) and now >= next_command:
                    commanded = commands[index]
                    self._command(commanded, goal.effort)
                    index += 1
                    next_command = now + 0.15
                with self._lock:
                    actual = self._opening
                    effort = 0.0 if self._effort is None else self._effort
                decision = detector.update(
                    now, actual, commanded, width, effort=effort,
                    empty_closed_opening_p99_m=float(self._baseline[
                        "empty_closed_opening_p99_m"]),
                    minimum_opening_margin_m=float(self._baseline.get(
                        "minimum_opening_margin_m", 0.002)),
                    empty_effort_p95=float(self._baseline["empty_effort_p95"]),
                    minimum_effort_margin=float(self._baseline.get(
                        "minimum_effort_margin", 0.2)),
                    require_effort=(
                        self._evidence_mode == "encoder_effort_vision"))
                self._feedback("CLOSING_AND_VERIFYING")
                if decision.contact:
                    break
                rospy.sleep(0.02)
            if decision is None or not decision.contact:
                reason = "empty_grasp_or_unstable_contact"
                self._server.set_aborted(
                    self._result(False, False, reason), reason)
                return
            self._server.set_succeeded(
                self._result(True, True, "stable_contact",
                             decision.command_error_m))
        except InterruptedError:
            self._safe_stop()
            self._server.set_preempted(self._result(False, False, "preempted"))
        except Exception as exc:
            self._abort(str(exc))


if __name__ == "__main__":
    rospy.init_node("piper_gripper_action")
    PiperGripperAction()
    rospy.spin()
