#!/usr/bin/env python3
"""Read stationary ROS feedback and prepare, but never approve or execute, a session."""

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
    DEFAULT_LIMITS, build_session, digest, solver_from_file)
from rekpiper_calibration.trajectory_preview import JOINT_NAMES
from rekpiper_execution.trajectory import normalize_feedback_to_joint_limits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--velocity-deg-s", type=float, default=3)
    p.add_argument("--tip-speed-mm-s", type=float, default=10)
    p.add_argument("--driver-speed-percent", type=int, default=5)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    urdf = root/"src/piper_description/urdf/piper_description.urdf"
    source = Path(args.plan).resolve()
    plan = yaml.safe_load(source.read_text())
    limits = dict(DEFAULT_LIMITS, velocity_deg_s=args.velocity_deg_s,
                  tip_speed_m_s=args.tip_speed_mm_s/1000,
                  driver_speed_percent=args.driver_speed_percent)
    rospy.init_node("prepare_calibration_motion", anonymous=True, disable_signals=True)
    samples = []

    def callback(msg):
        if time.time()-msg.header.stamp.to_sec() > .25 or msg.header.stamp.to_sec()-time.time() > .05:
            return
        values = dict(zip(msg.name, msg.position))
        if all(k in values for k in JOINT_NAMES):
            samples.append((msg.header.stamp.to_sec(), [values[k] for k in JOINT_NAMES]))

    subscriber = rospy.Subscriber("/joint_states_single", JointState, callback, queue_size=100)
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        if len(samples) >= 50 and samples[-1][0]-samples[0][0] >= 1:
            break
        time.sleep(.02)
    subscriber.unregister()
    if len(samples) < 50:
        raise ValueError("at least 50 fresh joint frames required")
    stamps, joints = zip(*samples)
    gaps = np.diff(stamps)
    if (stamps[-1]-stamps[0] < .9 or np.any(gaps <= 0) or np.max(gaps) > .1
            or np.max(np.ptp(joints, axis=0)) > .002):
        raise ValueError("joint feedback must be stationary and continuous for one second")
    raw_start = np.asarray(joints[-1])
    start = normalize_feedback_to_joint_limits(raw_start, 0.01)
    segments = build_session(plan, start, solver_from_file(urdf), limits)
    code_paths = [
        "src/rekpiper_calibration/src/rekpiper_calibration/calibration_motion.py",
        "src/rekpiper_calibration/scripts/calibration_motion_node.py",
        "src/rekpiper_calibration/scripts/calibration_motion_client.py",
        "src/rekpiper_calibration/src/rekpiper_calibration/trajectory_preview.py",
        "src/rekpiper_planning/src/rekpiper_planning/piper_urdf_ik.py",
        "src/rekpiper_execution/src/rekpiper_execution/trajectory.py",
    ]
    data = {"schema_version": 1, "status": "OPERATOR_SUPERVISED",
            "hardware_execution_allowed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
            "can_port": "can0", "raw_start_rad": raw_start.tolist(),
            "start_rad": start.tolist(), "feedback_boundary_tolerance_rad": 0.01, "limits": limits,
            "plan_sha256": digest(source), "urdf_sha256": digest(urdf),
            "code_sha256": {name: digest(root/name) for name in code_paths}, "segments": segments,
            "site_operator_statement": {"objects_and_cables_outside_path": True,
                                        "physical_power_cut_available": True},
            "pending": ["start_to_trial_and_first_point_collision_review", "full_arm_board_mount_path_review",
                        "enable_hold_hardware_test", "stop_hardware_test", "trial_hardware_test",
                        "interpolated_pose_dual_camera_visibility"]}
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/"plan.yaml").write_bytes(source.read_bytes())
    (out/"session.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    (out/"start_feedback.json").write_text(json.dumps(samples))
    rows = ["# 标定运动检查清单", "", "尚未批准实机执行；先核对路径、整臂/板/支架和停止条件。", "",
            "| 段 | 时长(s) | 最大单关节变化(°) |", "|---|---:|---:|"]
    for seg in segments:
        rows.append("| {} | {:.2f} | {:.3f} |".format(seg["name"], seg["duration_s"],
            np.rad2deg(np.max(np.abs(np.asarray(seg["goal_rad"])-seg["start_rad"])))) )
    rows += ["", "使能保持使用start_rad；trial最多改变单关节1°，随后point_01连接到首点。",
             "这里的末端是link6原点，Z范围0.05–0.70 m。速度为目标轨迹的限速，不是已验证的实机速度。",
             "已做关节限位、末端采样范围和时间参数计算；没有将离散末端检查当作全机器人碰撞验收。",
             "签名须绑定此session和实际完成的路径复核报告，不得签署这份未完成的清单作为通过证据。"]
    (out/"REVIEW_PENDING.md").write_text("\n".join(rows)+"\n")
    print(json.dumps({"session": str(out/"session.yaml"), "start_deg": np.rad2deg(start).tolist(),
                      "raw_start_deg": np.rad2deg(raw_start).tolist(),
                      "trial_deg": np.rad2deg(segments[0]["goal_rad"]).tolist(),
                      "motion_duration_s": sum(s["duration_s"]+s["dwell_s"] for s in segments),
                      "hardware_execution_allowed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
