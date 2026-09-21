#!/usr/bin/env python3
"""Capture stationary Piper feedback and prepare a go-zero session."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState
import yaml

from rekpiper_calibration.calibration_motion import (
    DEFAULT_LIMITS, GO_ZERO_BOUNDARY_TOLERANCE_RAD, GO_ZERO_PURPOSE,
    build_go_zero_session, digest, solver_from_file)
from rekpiper_calibration.trajectory_preview import JOINT_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--velocity-deg-s", type=float, default=1.0)
    parser.add_argument("--tip-speed-mm-s", type=float, default=5.0)
    parser.add_argument("--driver-speed-percent", type=int, default=5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    urdf = root/"src/piper_description/urdf/piper_description.urdf"
    limits = dict(DEFAULT_LIMITS, velocity_deg_s=args.velocity_deg_s,
                  tip_speed_m_s=args.tip_speed_mm_s/1000,
                  driver_speed_percent=args.driver_speed_percent)
    rospy.init_node("prepare_go_zero_motion", anonymous=True,
                    disable_signals=True)
    samples = []

    def callback(message):
        stamp = message.header.stamp.to_sec()
        if time.time()-stamp > .25 or stamp-time.time() > .05:
            return
        values = dict(zip(message.name, message.position))
        if all(name in values for name in JOINT_NAMES):
            samples.append((stamp, [values[name] for name in JOINT_NAMES]))

    subscriber = rospy.Subscriber(
        "/joint_states_single", JointState, callback, queue_size=100)
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        if len(samples) >= 50 and samples[-1][0]-samples[0][0] >= 1:
            break
        time.sleep(.02)
    subscriber.unregister()
    if len(samples) < 50:
        raise ValueError("at least 50 fresh joint frames required")
    stamps, joints = zip(*samples)
    if (stamps[-1]-stamps[0] < .9 or np.any(np.diff(stamps) <= 0)
            or np.max(np.diff(stamps)) > .1
            or np.max(np.ptp(joints, axis=0)) > .002):
        raise ValueError("joint feedback must be stationary and continuous for one second")
    start = np.asarray(joints[-1], dtype=float)
    solver = solver_from_file(urdf)
    segments = build_go_zero_session(start, solver, limits)
    code_paths = [
        "src/rekpiper_calibration/src/rekpiper_calibration/calibration_motion.py",
        "src/rekpiper_calibration/scripts/calibration_motion_node.py",
        "tools/run_go_zero.py",
    ]
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = out/"plan.yaml"
    plan.write_text(yaml.safe_dump({
        "schema_version": 1, "purpose": GO_ZERO_PURPOSE,
        "target_deg": [0, 0, 0, 0, 0, 0]}, sort_keys=False))
    session = out/"session.yaml"
    data = {
        "schema_version": 1,
        "status": "OPERATOR_SUPERVISED",
        "hardware_execution_allowed": True,
        "purpose": GO_ZERO_PURPOSE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "can_port": "can0",
        "raw_start_rad": start.tolist(),
        "start_rad": start.tolist(),
        "feedback_boundary_tolerance_rad": float(
            GO_ZERO_BOUNDARY_TOLERANCE_RAD),
        "limits": limits,
        "plan_sha256": digest(plan),
        "urdf_sha256": digest(urdf),
        "code_sha256": {name: digest(root/name) for name in code_paths},
        "segments": segments,
        "site_operator_statement": {
            "objects_and_cables_outside_path": True,
            "physical_power_cut_available": True,
        },
        "pending": ["live_start_to_zero_path_review",
                    "full_arm_board_mount_path_review"],
    }
    session.write_text(yaml.safe_dump(data, sort_keys=False))
    (out/"start_feedback.json").write_text(json.dumps(samples))
    segment = segments[0]
    print(json.dumps({
        "session": str(session),
        "start_deg": np.rad2deg(start).tolist(),
        "target_deg": [0, 0, 0, 0, 0, 0],
        "duration_s": segment["duration_s"],
        "link6_minimum_xyz_m": segment["sampled_minimum_xyz_m"],
        "link6_maximum_xyz_m": segment["sampled_maximum_xyz_m"],
        "hardware_execution_allowed": True,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
