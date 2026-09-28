#!/usr/bin/env python3
"""Restricted FollowJointTrajectory bridge for Piper's six arm joints."""

import threading
import time
import json
from collections import deque

import actionlib
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryFeedback,
    FollowJointTrajectoryResult,
)
import numpy as np
from piper_msgs.msg import PiperStatusMsg
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from rekpiper_execution.trajectory import (
    JOINT_NAMES,
    TrajectoryValidationError,
    normalize_feedback_to_joint_limits,
    validate_trajectory,
)
from rekpiper_msgs.msg import SafeMappingStatus
from rekpiper_execution.rate_monitor import distribution


class PiperTrajectoryBridge:
    def __init__(self):
        self._supervised = None
        if rospy.get_param('~execution_profile', 'autonomous') == 'supervised':
            from rekpiper_execution.supervised_authority import RosLease
            self._supervised = RosLease()
        self._allow_commands = bool(rospy.get_param(
            "~allow_hardware_commands", False))
        self._map_timeout = float(rospy.get_param("~map_timeout_s", 1.0))
        self._feedback_timeout = float(rospy.get_param(
            "~joint_feedback_timeout_s", 0.25))
        self._goal_tolerance = float(rospy.get_param(
            "~goal_tolerance_rad", 0.05))
        self._tracking_tolerance = float(rospy.get_param(
            "~tracking_tolerance_rad", 0.20))
        self._max_velocity = float(rospy.get_param(
            "~maximum_velocity_rad_s", 0.50))
        self._max_acceleration = float(rospy.get_param(
            "~maximum_acceleration_rad_s2", 1.0))
        self._max_jerk = float(rospy.get_param(
            "~maximum_jerk_rad_s3", 0.50))
        self._start_tolerance = float(rospy.get_param(
            "~start_tolerance_rad", 0.08))
        self._speed_percent = int(rospy.get_param(
            "~driver_speed_percent", 15))
        self._command_rate = float(rospy.get_param(
            "~command_rate_hz", 20.0))
        self._hold_only = bool(rospy.get_param("~hold_only", False))
        self._hold_tolerance = float(rospy.get_param(
            "~hold_tolerance_rad", 0.01))
        self._hold_reference = None
        self._feedback_boundary_tolerance = float(rospy.get_param(
            "/rekpiper/execution/joint_feedback_boundary_tolerance_rad",
            0.01))
        self._lock = threading.RLock()
        self._positions = None
        self._raw_positions = None
        self._joint_feedback_error = "joint_feedback_missing"
        self._feedback_time = 0.0
        self._map = None
        self._map_time = 0.0
        self._map_generation = ""
        self._arm_error = "arm_status_missing"
        self._arm_status_time = 0.0
        self._armed = False
        self._command_timestamps = deque(maxlen=20000)
        self._diagnostic_sequence = 0
        self._command_pub = rospy.Publisher(
            rospy.get_param("~command_topic", "/joint_ctrl_single"),
            JointState, queue_size=1)
        self._diagnostics_pub = rospy.Publisher(
            "/rekpiper/execution/trajectory_bridge_diagnostics",
            String, queue_size=1, latch=True)
        rospy.Subscriber(
            rospy.get_param("~joint_state_topic", "/joint_states_single"),
            JointState, self._joint_callback, queue_size=20)
        rospy.Subscriber(
            rospy.get_param(
                "~mapping_status_topic", "/rekpiper/mapping/safe_status"),
            SafeMappingStatus, self._map_callback, queue_size=5)
        rospy.Subscriber(
            rospy.get_param("~arm_status_topic", "/arm_status"),
            PiperStatusMsg, self._arm_callback, queue_size=10)
        rospy.Subscriber(
            rospy.get_param("~armed_topic", "/rekpiper/execution/hardware_armed"),
            Bool, self._armed_callback, queue_size=2)
        self._stop_name = rospy.get_param("~stop_service", "/stop_srv")
        self._server = actionlib.SimpleActionServer(
            rospy.get_param(
                "~action_name",
                "/manipulator_controller/follow_joint_trajectory"),
            FollowJointTrajectoryAction, execute_cb=self._execute,
            auto_start=False)
        self._server.start()
        rospy.logwarn(
            "Piper trajectory bridge started with hardware commands %s",
            "ENABLED" if self._allow_commands else "DISABLED")

    def _joint_callback(self, message):
        values = dict(zip(message.name, message.position))
        if any(name not in values for name in JOINT_NAMES):
            return
        raw = np.asarray([values[name] for name in JOINT_NAMES], dtype=float)
        try:
            normalized = normalize_feedback_to_joint_limits(
                raw, self._feedback_boundary_tolerance)
            error = ""
        except TrajectoryValidationError as exc:
            normalized = None
            error = str(exc)
        with self._lock:
            self._raw_positions = raw
            self._positions = normalized
            self._joint_feedback_error = error
            self._feedback_time = time.monotonic()
            if self._hold_only and normalized is not None \
                    and self._hold_reference is None:
                self._hold_reference = normalized.copy()
        if error:
            rospy.logerr_throttle(1.0, "Piper feedback rejected: %s", error)
        elif not np.array_equal(raw, normalized):
            self._publish_diagnostics(
                "feedback_boundary_normalized",
                raw_positions_rad=raw.tolist(),
                normalized_positions_rad=normalized.tolist())

    def _map_callback(self, message):
        with self._lock:
            self._map = message
            self._map_time = time.monotonic()
            self._map_generation = str(message.map_generation_uuid)

    def _arm_callback(self, message):
        errors = []
        # Motor-enable feedback can remain asserted after a controller stop.
        # Require the controller itself to report normal CAN/MOVE-J operation.
        if int(message.ctrl_mode) != 1:
            errors.append("ctrl_mode_{}".format(message.ctrl_mode))
        if int(message.arm_status) != 0:
            errors.append("arm_status_{}".format(message.arm_status))
        if int(message.mode_feedback) != 1:
            errors.append("mode_feedback_{}".format(message.mode_feedback))
        if int(message.err_code) != 0:
            errors.append("err_code_{}".format(message.err_code))
        limit_fields = [
            message.joint_1_angle_limit, message.joint_2_angle_limit,
            message.joint_3_angle_limit, message.joint_4_angle_limit,
            message.joint_5_angle_limit, message.joint_6_angle_limit,
        ]
        communication_fields = [
            message.communication_status_joint_1,
            message.communication_status_joint_2,
            message.communication_status_joint_3,
            message.communication_status_joint_4,
            message.communication_status_joint_5,
            message.communication_status_joint_6,
        ]
        if any(limit_fields):
            errors.append("joint_limit")
        if any(communication_fields):
            errors.append("joint_communication")
        with self._lock:
            self._arm_error = ",".join(errors)
            self._arm_status_time = time.monotonic()

    def _armed_callback(self, message):
        with self._lock:
            self._armed = bool(message.data)
        if not message.data and self._server.is_active():
            self._safe_stop()

    def _health(self, expected_generation=None):
        with self._lock:
            now = time.monotonic()
            if not self._allow_commands:
                return False, "hardware_commands_disabled"
            supervised = getattr(self, '_supervised', None)
            if supervised:
                try:
                    lease = supervised.check('trajectory')
                    if expected_generation is not None and lease['preview_id'] != expected_generation:
                        return False, 'supervised_preview_changed'
                except (ValueError, KeyError) as exc:
                    return False, str(exc)
            precision_accepted = supervised is not None or bool(rospy.get_param(
                "/rekpiper/camera/extrinsics/"
                "precision_operation_allowed", False))
            if not precision_accepted:
                return False, "precision_camera_extrinsics_not_accepted"
            if not self._armed:
                return False, "execution_not_armed"
            if self._joint_feedback_error:
                return False, "joint_feedback_invalid:" + self._joint_feedback_error
            if self._positions is None or now - self._feedback_time > self._feedback_timeout:
                return False, "joint_feedback_stale"
            if (self._hold_only
                    and (self._hold_reference is None
                         or np.max(np.abs(
                             self._positions - self._hold_reference))
                         > self._hold_tolerance)):
                return False, "hardware_hold_position_drift"
            if (self._arm_status_time <= 0.0
                    or now - self._arm_status_time > self._feedback_timeout):
                return False, "piper_arm_status_stale"
            if self._arm_error:
                return False, "piper_{}".format(self._arm_error)
            if supervised:
                return True, 'supervised_feedback_and_segment_lease_valid'
            if self._map is None or now - self._map_time > self._map_timeout:
                return False, "safe_map_stale"
            if (self._map.state != SafeMappingStatus.READY
                    or not self._map.map_query_allowed
                    or not self._map.planning_safe):
                return False, (
                    "safe_map_not_ready:state={} query_allowed={} "
                    "planning_safe={} reason={}").format(
                        self._map.state_name,
                        bool(self._map.map_query_allowed),
                        bool(self._map.planning_safe),
                        self._map.reason)
            if (expected_generation is not None
                    and self._map_generation != expected_generation):
                return False, "safe_map_generation_changed"
            return True, "ok"

    def _safe_stop(self):
        if not self._allow_commands:
            return
        try:
            rospy.wait_for_service(self._stop_name, timeout=0.2)
            rospy.ServiceProxy(self._stop_name, Trigger)()
        except Exception as exc:
            rospy.logerr("Piper stop service failed: %s", exc)

    def _result(self, code, text):
        result = FollowJointTrajectoryResult()
        result.error_code = int(code)
        result.error_string = str(text)
        return result

    def _abort(self, code, reason):
        self._publish_diagnostics("aborted", reason=reason)
        self._safe_stop()
        self._server.set_aborted(self._result(code, reason), reason)

    def _publish_command(self, positions):
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = list(JOINT_NAMES)
        message.position = list(np.asarray(positions, dtype=float))
        # The Piper driver uses the seventh velocity entry as its global
        # speed percentage even when no gripper position is commanded.
        message.velocity = [0.0] * 6 + [float(self._speed_percent)]
        self._command_pub.publish(message)
        self._command_timestamps.append(time.monotonic())

    def _command_timing(self):
        stamps = np.asarray(self._command_timestamps, dtype=float)
        gaps = np.diff(stamps)
        metrics = distribution(gaps)
        duration = float(stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
        metrics["actual_rate_hz"] = (
            float((len(stamps) - 1) / duration) if duration > 0.0 else 0.0)
        metrics["deadline_miss_count"] = int(np.count_nonzero(
            gaps > 1.0 / self._command_rate))
        metrics["maximum_gap_s"] = float(np.max(gaps)) if len(gaps) else 0.0
        metrics["command_samples"] = int(len(stamps))
        return metrics

    def _feedback(self, desired):
        message = FollowJointTrajectoryFeedback()
        message.header.stamp = rospy.Time.now()
        message.joint_names = list(JOINT_NAMES)
        message.desired.positions = list(desired)
        with self._lock:
            actual = self._positions.copy()
        message.actual.positions = actual.tolist()
        message.error.positions = (np.asarray(desired) - actual).tolist()
        self._server.publish_feedback(message)
        return actual

    def _publish_diagnostics(self, state, **values):
        self._diagnostic_sequence += 1
        payload = {
            "state": str(state),
            "sequence": self._diagnostic_sequence,
            "event_monotonic_ns": time.monotonic_ns(),
            "command_rate_hz": self._command_rate,
            "driver_speed_percent": self._speed_percent,
        }
        payload.update(values)
        self._diagnostics_pub.publish(String(
            data=json.dumps(payload, sort_keys=True)))

    def _execute(self, goal):
        generation = str(goal.trajectory.header.frame_id)
        if not generation:
            self._abort(FollowJointTrajectoryResult.INVALID_GOAL,
                        "trajectory_map_generation_missing")
            return
        healthy, reason = self._health(generation)
        if not healthy:
            self._abort(FollowJointTrajectoryResult.INVALID_GOAL, reason)
            return
        with self._lock:
            current = self._positions.copy()
        try:
            points = goal.trajectory.points
            if getattr(self, '_supervised', None):
                from rekpiper_execution.supervised_session import trajectory_digest
                lease = self._supervised.check('trajectory', trajectory_digest(
                    goal.trajectory.joint_names, [p.positions for p in points],
                    [p.time_from_start.to_sec() for p in points]))
                self._speed_percent = int(lease['driver_percent'])
                self._max_velocity = float(lease['velocity'])
            if self._hold_only:
                if self._hold_reference is None:
                    raise TrajectoryValidationError(
                        "hold reference is unavailable")
                requested = np.asarray(
                    [point.positions for point in points], dtype=float)
                if (requested.ndim != 2 or requested.shape[1] != 6
                        or not np.all(np.isfinite(requested))
                        or np.max(np.abs(
                            requested - self._hold_reference[None, :]))
                        > self._hold_tolerance):
                    raise TrajectoryValidationError(
                        "acceptance bridge only permits the startup hold pose")
            validated = validate_trajectory(
                goal.trajectory.joint_names,
                [point.positions for point in points],
                [point.time_from_start.to_sec() for point in points],
                current,
                self._max_velocity,
                self._max_acceleration,
                self._start_tolerance,
                self._max_jerk)
        except (TrajectoryValidationError, ValueError) as exc:
            self._abort(
                FollowJointTrajectoryResult.INVALID_GOAL, str(exc))
            return

        start = time.monotonic()
        self._command_timestamps.clear()
        maximum_tracking_error = 0.0
        self._publish_diagnostics(
            "executing", sample_count=int(len(validated.times)),
            duration_s=validated.duration_s,
            maximum_velocity_rad_s=validated.maximum_velocity_rad_s,
            maximum_acceleration_rad_s2=(
                validated.maximum_acceleration_rad_s2),
            maximum_jerk_rad_s3=validated.maximum_jerk_rad_s3)
        rate = rospy.Rate(self._command_rate)
        final_time = float(validated.times[-1])
        next_stream_diagnostic = start + 1.0
        while not rospy.is_shutdown():
            if self._server.is_preempt_requested():
                self._safe_stop()
                self._server.set_preempted(
                    self._result(
                        FollowJointTrajectoryResult.SUCCESSFUL,
                        "trajectory_preempted"),
                    "trajectory_preempted")
                return
            healthy, reason = self._health(generation)
            if not healthy:
                self._abort(
                    FollowJointTrajectoryResult.PATH_TOLERANCE_VIOLATED,
                    reason)
                return
            elapsed = min(time.monotonic() - start, final_time)
            desired = np.asarray([
                np.interp(elapsed, validated.times, validated.positions[:, index])
                for index in range(6)
            ])
            self._publish_command(desired)
            now = time.monotonic()
            if (now >= next_stream_diagnostic
                    and len(self._command_timestamps) >= 20):
                self._publish_diagnostics(
                    "streaming", command_timing=self._command_timing())
                next_stream_diagnostic = now + 1.0
            actual = self._feedback(desired)
            tracking_error = np.abs(desired - actual)
            maximum_tracking_error = max(
                maximum_tracking_error, float(np.max(tracking_error)))
            if float(np.max(tracking_error)) > self._tracking_tolerance:
                index = int(np.argmax(tracking_error))
                self._abort(
                    FollowJointTrajectoryResult.PATH_TOLERANCE_VIOLATED,
                    ("joint_tracking_error:{} desired={:.4f} actual={:.4f} "
                     "error={:.4f} limit={:.4f}").format(
                        JOINT_NAMES[index], desired[index], actual[index],
                        tracking_error[index], self._tracking_tolerance))
                return
            if elapsed >= final_time:
                break
            rate.sleep()

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not rospy.is_shutdown():
            with self._lock:
                error = float(np.max(np.abs(
                    validated.positions[-1] - self._positions)))
            if error <= self._goal_tolerance:
                self._publish_diagnostics(
                    "completed", duration_s=validated.duration_s,
                    sample_count=int(len(validated.times)),
                    maximum_velocity_rad_s=(
                        validated.maximum_velocity_rad_s),
                    maximum_acceleration_rad_s2=(
                        validated.maximum_acceleration_rad_s2),
                    maximum_jerk_rad_s3=(
                        validated.maximum_jerk_rad_s3),
                    maximum_tracking_error_rad=maximum_tracking_error,
                    final_error_rad=error,
                    command_timing=self._command_timing())
                self._server.set_succeeded(
                    self._result(FollowJointTrajectoryResult.SUCCESSFUL, "ok"))
                return
            rospy.sleep(0.02)
        self._abort(
            FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED,
            "goal_tolerance_not_reached")


if __name__ == "__main__":
    rospy.init_node("rekep_piper_trajectory_bridge")
    PiperTrajectoryBridge()
    rospy.spin()
