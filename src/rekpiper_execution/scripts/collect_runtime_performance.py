#!/usr/bin/env python3
"""Collect a read-only continuous benchmark report from runtime diagnostics."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import rospy
import rosnode
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
import yaml

from piper_msgs.msg import PiperStatusMsg
from rekpiper_msgs.msg import (
    CommandGripperActionGoal, ReKepHorizon, SafeMappingStatus)
from rekpiper_execution.runtime_evidence import (
    RuntimeEvidenceAccumulator, RuntimeEvidenceError)
from rekpiper_acceptance import AcceptanceError, sha256_file, validate_release_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--warmup-s", type=float, default=30.0)
    parser.add_argument("--profile", choices=("software", "hardware_hold"),
                        required=True)
    parser.add_argument("--operator-present", action="store_true")
    parser.add_argument("--physical-estop-confirmed", action="store_true")
    parser.add_argument("--hold-tolerance-rad", type=float, default=0.01)
    parser.add_argument("--replay-dataset", default="")
    parser.add_argument("--program-sha256", default="")
    parser.add_argument("--release-bundle", default="")
    parser.add_argument("--public-key", default="")
    parser.add_argument("--minimum-counter", default="")
    parser.add_argument("--robot-id", default="")
    args, _unknown = parser.parse_known_args(rospy.myargv()[1:])
    if args.duration_s < 600.0:
        print(json.dumps({"success": False,
                          "error": "duration must be at least 600 seconds"}))
        return 2
    if args.warmup_s < 0.0:
        print(json.dumps({"success": False, "error": "warmup must be non-negative"}))
        return 2
    if args.profile == "hardware_hold" and not (
            args.operator_present and args.physical_estop_confirmed):
        print(json.dumps({"success": False, "error":
                          "hardware hold requires operator and physical e-stop confirmations"}))
        return 2
    replay_hash = ""
    hardware_release = None
    if args.profile == "software":
        replay_path = Path(args.replay_dataset).expanduser().resolve()
        if (not replay_path.is_file()
                or len(args.program_sha256) != 64
                or any(c not in "0123456789abcdef"
                       for c in args.program_sha256.lower())):
            print(json.dumps({"success": False, "error":
                              "software replay dataset and program hash are required"}))
            return 2
        replay_hash = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    else:
        os.environ["REKPIPER_ACCEPTANCE_BUNDLE_TYPE"] = \
            "hardware_acceptance_bundle"
        try:
            hardware_release = validate_release_bundle(
                args.release_bundle, args.public_key, args.minimum_counter,
                args.robot_id)
        except AcceptanceError as exc:
            print(json.dumps({"success": False,
                              "error": "hardware bundle rejected:" + str(exc)}))
            return 2
    rospy.init_node("rekpiper_runtime_performance_collector")
    node_names = set(rosnode.get_node_names())
    piper_present = any(name.rsplit("/", 1)[-1].startswith("piper_driver")
                        for name in node_names)
    if args.profile == "software" and piper_present:
        print(json.dumps({"success": False,
                          "error": "software replay must have no Piper driver"}))
        return 2
    if args.profile == "hardware_hold" and not piper_present:
        print(json.dumps({"success": False,
                          "error": "hardware hold requires the Piper driver"}))
        return 2
    output = Path(args.output).expanduser().resolve()
    raw_output = output.with_suffix(output.suffix + ".events.jsonl")
    if output.exists() or raw_output.exists():
        print(json.dumps({"success": False, "error": "output already exists"}))
        return 2
    process_started = time.monotonic()
    measurement_started = process_started + args.warmup_s
    accumulator = None
    errors = []
    planning_sequence = 0
    hold_reference = None
    hold_feedback_samples = 0
    hold_command_samples = 0
    maximum_hold_error = 0.0
    stopped = False
    command_times = []
    command_sequence = 0

    def fail_hardware(reason):
        nonlocal stopped
        errors.append(reason)
        if args.profile != "hardware_hold" or stopped:
            return
        stopped = True
        try:
            rospy.wait_for_service("/stop_srv", timeout=0.2)
            rospy.ServiceProxy("/stop_srv", Trigger)()
        except Exception:
            pass

    def protect(callback):
        def wrapped(message):
            nonlocal accumulator
            if time.monotonic() < measurement_started:
                return
            if accumulator is None:
                accumulator = RuntimeEvidenceAccumulator(
                    measurement_started, profile=args.profile)
            try:
                callback(message)
            except (KeyError, TypeError, ValueError, RuntimeEvidenceError) as exc:
                errors.append(str(exc))
        return wrapped

    def add_horizon(message):
        nonlocal planning_sequence
        planning_sequence += 1
        accumulator.add_horizon(
            message.planning_latency_s, message.from_scratch,
            message.valid, sequence=planning_sequence,
            event_monotonic_ns=time.monotonic_ns())

    def joint_values(message):
        values = dict(zip(message.name, message.position))
        names = ["joint{}".format(index) for index in range(1, 7)]
        if any(name not in values for name in names):
            return None
        result = np.asarray([values[name] for name in names], dtype=float)
        return result if np.all(np.isfinite(result)) else None

    def feedback(message):
        nonlocal hold_reference, hold_feedback_samples, maximum_hold_error
        if args.profile != "hardware_hold":
            return
        values = joint_values(message)
        if values is None:
            if time.monotonic() >= measurement_started:
                fail_hardware("hardware_feedback_invalid")
            return
        if hold_reference is None:
            # Freeze the first valid feedback received after collection starts.
            # Warmup is excluded from performance metrics, but it must not
            # redefine the physical hold target.
            hold_reference = values.copy()
        error = float(np.max(np.abs(values - hold_reference)))
        measured = time.monotonic() >= measurement_started
        if measured and accumulator is not None:
            accumulator.raw_events.append({
                "stream": "hardware_feedback",
                "payload": {"event_monotonic_ns": time.monotonic_ns(),
                            "positions_rad": values.tolist(),
                            "maximum_hold_error_rad": error},
            })
        maximum_hold_error = max(maximum_hold_error, error)
        if measured:
            hold_feedback_samples += 1
        if error > args.hold_tolerance_rad:
            fail_hardware("hardware_hold_position_drift")

    def command(message):
        nonlocal hold_command_samples, maximum_hold_error, command_sequence
        now = time.monotonic()
        values = joint_values(message)
        if (args.profile == "hardware_hold" and values is not None
                and hold_reference is not None):
            # Position safety monitoring begins during warmup even though the
            # warmup samples are omitted from rate and latency acceptance.
            error = float(np.max(np.abs(values - hold_reference)))
            maximum_hold_error = max(maximum_hold_error, error)
            if error > args.hold_tolerance_rad:
                fail_hardware("hardware_hold_command_changed")
        if now < measurement_started:
            return
        command_sequence += 1
        command_times.append(now)
        payload = {"sequence": command_sequence,
                   "event_monotonic_ns": time.monotonic_ns()}
        if values is not None:
            payload["positions_rad"] = values.tolist()
        if accumulator is not None:
            accumulator.raw_events.append({
                "stream": "trajectory_command", "payload": payload})
        if args.profile != "hardware_hold":
            return
        if values is None or hold_reference is None:
            fail_hardware("hardware_hold_command_before_reference")
            return
        error = float(np.max(np.abs(values - hold_reference)))
        maximum_hold_error = max(maximum_hold_error, error)
        hold_command_samples += 1
        if error > args.hold_tolerance_rad:
            fail_hardware("hardware_hold_command_changed")

    def arm_status(message):
        if (args.profile == "hardware_hold"
                and time.monotonic() >= measurement_started
                and (message.ctrl_mode != 1 or message.arm_status != 0
                     or message.mode_feedback != 1 or message.err_code != 0)):
            fail_hardware("hardware_arm_status_invalid")

    def map_status(message):
        if (args.profile == "hardware_hold"
                and time.monotonic() >= measurement_started
                and (message.state != SafeMappingStatus.READY
                     or not message.map_query_allowed
                     or not message.planning_safe)):
            fail_hardware("hardware_safe_map_invalid")

    def gripper_goal(_message):
        if args.profile == "hardware_hold":
            fail_hardware("hardware_gripper_action_forbidden")

    rospy.Subscriber(
        "/rekpiper/mapping/sdf_grid_snapshot_diagnostics", String,
        protect(lambda message: accumulator.add_sdf(message.data)), queue_size=10)
    rospy.Subscriber(
        "/rekpiper/execution/trajectory_bridge_diagnostics", String,
        protect(lambda message: accumulator.add_trajectory(message.data)),
        queue_size=100)
    rospy.Subscriber(
        "/rekpiper/planning/horizon", ReKepHorizon,
        protect(add_horizon), queue_size=100)
    rospy.Subscriber("/joint_states_single", JointState, feedback, queue_size=100)
    rospy.Subscriber("/joint_ctrl_single", JointState, command, queue_size=100)
    rospy.Subscriber("/arm_status", PiperStatusMsg, arm_status, queue_size=100)
    rospy.Subscriber("/rekpiper/mapping/safe_status", SafeMappingStatus,
                     map_status, queue_size=10)
    rospy.Subscriber("/rekpiper/execution/command_gripper/goal",
                     CommandGripperActionGoal, gripper_goal, queue_size=10)
    deadline = measurement_started + args.duration_s
    while (not rospy.is_shutdown() and time.monotonic() < deadline
           and not errors):
        rospy.sleep(0.1)
    try:
        if errors:
            raise RuntimeEvidenceError(errors[-1])
        if accumulator is None:
            raise RuntimeEvidenceError("runtime benchmark received no measured events")
        raw_lines = [json.dumps(event, sort_keys=True, separators=(",", ":"))
                     for event in accumulator.raw_events]
        raw_bytes = (("\n".join(raw_lines) + "\n").encode("utf-8"))
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        with raw_output.open("xb") as stream:
            stream.write(raw_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(str(raw_output), 0o444)
        report = accumulator.report(time.monotonic(), raw_hash)
        if len(command_times) < 2:
            raise RuntimeEvidenceError("trajectory command stream is incomplete")
        command_gaps = np.diff(np.asarray(command_times, dtype=float))
        command_duration = command_times[-1] - command_times[0]
        report["trajectory"].update({
            "actual_rate_hz": float((len(command_times) - 1) / command_duration),
            "period_p99_s": float(np.percentile(command_gaps, 99)),
            "command_samples": len(command_times),
            "maximum_gap_s": float(np.max(command_gaps)),
            "deadline_miss_count": int(np.count_nonzero(command_gaps > 0.05)),
        })
        if args.profile == "software":
            report["software_contract"] = {"piper_driver_absent": True,
                                            "can_disabled_verified": True,
                                            "replay_dataset_sha256": replay_hash,
                                            "program_sha256": args.program_sha256.lower()}
        else:
            if (hold_reference is None or hold_feedback_samples < 11400
                    or hold_command_samples < 11400):
                raise RuntimeEvidenceError(
                    "hardware hold feedback/command coverage is incomplete")
            report["hardware_hold"] = {
                "reference_joint_positions_rad": hold_reference.tolist(),
                "feedback_samples": hold_feedback_samples,
                "command_samples": hold_command_samples,
                "maximum_error_rad": maximum_hold_error,
                "tolerance_rad": args.hold_tolerance_rad,
                "gripper_motion_forbidden": True,
                "operator_present": True,
                "physical_estop_confirmed": True,
                "robot_id": hardware_release["payload"]["robot_id"],
                "piper_identity": hardware_release["payload"]["piper_identity"],
                "hardware_acceptance_bundle_sha256": sha256_file(
                    args.release_bundle),
            }
        report["raw_event_log"] = raw_output.name
        report["warmup_s"] = args.warmup_s
        with output.open("x", encoding="utf-8") as stream:
            stream.write(yaml.safe_dump(report, sort_keys=False))
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"success": True, "output": str(output),
                          "raw_event_log": str(raw_output)},
                         sort_keys=True))
        return 0
    except (OSError, RuntimeEvidenceError) as exc:
        print(json.dumps({"success": False, "error": str(exc)},
                         sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
