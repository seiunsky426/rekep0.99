#!/usr/bin/env python3
"""Read-only Piper motion-readiness check for the ReKep real-robot demo."""

import math
import sys

import rospy
from piper_msgs.msg import PiperStatusMsg
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from rekpiper_execution.trajectory import (
    JOINT_NAMES, TrajectoryValidationError,
    normalize_feedback_to_joint_limits)


ARM_STATUS_TOPIC = "/arm_status"
JOINT_TOPIC = "/joint_states_single"
DEFAULT_SERVICES = {
    "reset": "/reset_srv",
    "enable": "/enable_srv",
    "enable_status": "/enable_status_srv",
    "go_zero": "/go_zero_srv",
}

ARM_STATUS_LABELS = {
    0: "normal",
    1: "emergency_stop",
    2: "no_solution",
    3: "singularity",
    4: "target_angle_limit",
    5: "joint_communication_error",
    6: "joint_brake_not_released",
    7: "collision",
    8: "overspeed_during_drag",
    9: "joint_status_error",
    10: "other_error",
    11: "teach_recording",
    12: "teach_execution",
    13: "teach_pause",
    14: "main_controller_ntc_overtemperature",
    15: "release_resistance",
}


def _wait_message(topic, message_type, timeout=3.0):
    try:
        return rospy.wait_for_message(topic, message_type, timeout=timeout)
    except rospy.ROSException as exc:
        raise RuntimeError("{} unavailable: {}".format(topic, exc))


def main():
    rospy.init_node("rekep_verify_piper_ready", anonymous=True)
    failures = []
    notes = []
    services = {
        name: str(rospy.get_param("~{}_service".format(name), default))
        for name, default in DEFAULT_SERVICES.items()
    }
    boundary_tolerance = float(rospy.get_param(
        "/rekpiper/execution/joint_feedback_boundary_tolerance_rad", 0.01))

    try:
        status = _wait_message(ARM_STATUS_TOPIC, PiperStatusMsg)
        joints = _wait_message(JOINT_TOPIC, JointState)
    except RuntimeError as exc:
        rospy.logerr(str(exc))
        return 2

    expected = {
        "ctrl_mode": (int(status.ctrl_mode), 1),
        "arm_status": (int(status.arm_status), 0),
        "mode_feedback": (int(status.mode_feedback), 1),
        "err_code": (int(status.err_code), 0),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            if name == "arm_status":
                failures.append(
                    "arm_status={}({}) expected={}(normal)".format(
                        actual, ARM_STATUS_LABELS.get(actual, "unknown"), wanted))
            else:
                failures.append("{}={} expected={}".format(name, actual, wanted))

    limit_flags = [
        status.joint_1_angle_limit, status.joint_2_angle_limit,
        status.joint_3_angle_limit, status.joint_4_angle_limit,
        status.joint_5_angle_limit, status.joint_6_angle_limit,
    ]
    communication_flags = [
        status.communication_status_joint_1,
        status.communication_status_joint_2,
        status.communication_status_joint_3,
        status.communication_status_joint_4,
        status.communication_status_joint_5,
        status.communication_status_joint_6,
    ]
    if any(limit_flags):
        failures.append("joint_limit_feedback_present")
    if any(communication_flags):
        failures.append("joint_communication_error_present")

    positions = dict(zip(joints.name, joints.position))
    required_joints = list(JOINT_NAMES)
    missing = [name for name in required_joints if name not in positions]
    if missing:
        failures.append("joint_feedback_missing:{}".format(",".join(missing)))
    elif not all(math.isfinite(float(positions[name])) for name in required_joints):
        failures.append("joint_feedback_non_finite")
    else:
        raw = [float(positions[name]) for name in required_joints]
        try:
            normalized = normalize_feedback_to_joint_limits(
                raw, boundary_tolerance)
            adjusted = [name for name, before, after in zip(
                required_joints, raw, normalized) if before != after]
            notes.append("joint_feedback_boundary_adjustments={}".format(
                ",".join(adjusted) if adjusted else "none"))
        except TrajectoryValidationError as exc:
            failures.append(str(exc))

    for service in services.values():
        try:
            rospy.wait_for_service(service, timeout=0.2)
        except rospy.ROSException:
            failures.append("service_missing:{}".format(service))

    enable_status_service = services["enable_status"]
    if "service_missing:{}".format(enable_status_service) not in failures:
        try:
            enable_status = rospy.ServiceProxy(
                enable_status_service, Trigger)()
            notes.append("enable_status: {}".format(enable_status.message))
            if not enable_status.success:
                failures.append("piper_motor_enable_status_not_ready")
        except rospy.ServiceException as exc:
            failures.append("enable_status_query_failed:{}".format(exc))

    for note in notes:
        rospy.loginfo(note)
    if failures:
        rospy.logerr("PIPER NOT READY (%d): %s", len(failures), "; ".join(failures))
        if int(status.arm_status) == 1:
            rospy.logerr(
                "RECOVERY REQUIRED: controller is in emergency-stop state. "
                "With the arm supported and the workspace clear, call "
                "{}, wait for arm_status=0, then call {} and only afterwards "
                "{}.", services["reset"], services["enable"],
                services["go_zero"])
        return 1
    rospy.loginfo(
        "PIPER READY: CAN control, normal status, MOVE-J, fresh finite feedback, "
        "joint feedback inside model boundary, recovery/enable/home services present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
