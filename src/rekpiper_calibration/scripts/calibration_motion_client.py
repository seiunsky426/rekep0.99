#!/usr/bin/env python3
"""Check offline, explicitly enable hold, or send reviewed calibration segments."""

import argparse
import json
from pathlib import Path

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import rospkg
import rospy
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from rekpiper_calibration.calibration_motion import MotionSession
from rekpiper_calibration.trajectory_preview import JOINT_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["check", "enable-hold", "trial", "point", "range", "stop"])
    p.add_argument("--session", required=True)
    p.add_argument("--point", type=int)
    p.add_argument("--first", type=int)
    p.add_argument("--last", type=int)
    args = p.parse_args()
    urdf = Path(rospkg.RosPack().get_path("piper_description"))/"urdf/piper_description.urdf"
    if args.command != "stop":
        session = MotionSession(args.session, urdf)
    if args.command == "check":
        session.require_ready()
        print(json.dumps({"valid_offline": True, "operator_supervised": True,
                          "segments": len(session.data["segments"]), "limits": session.data["limits"]}))
        return
    rospy.init_node("calibration_motion_client", anonymous=True, disable_signals=False)
    if args.command in ("enable-hold", "stop"):
        name = "/calibration_motion/" + ("enable_hold" if args.command == "enable-hold" else "stop")
        try:
            rospy.wait_for_service(name, timeout=2)
        except rospy.ROSException:
            raise SystemExit("服务未启动：先完成会话检查/审核并启动 calibration_motion.launch；不要重复使能。")
        result = rospy.ServiceProxy(name, Trigger)()
        print(result)
        if not result.success:
            raise SystemExit(1)
        return
    if args.command == "trial":
        indices = [0]
    elif args.command == "point":
        if args.point is None or not 1 <= args.point <= 20:
            p.error("--point must be 1..20")
        indices = [args.point]
    else:
        if args.first is None or args.last is None or not 1 <= args.first <= args.last <= 20:
            p.error("--first/--last must be ordered within 1..20")
        indices = range(args.first, args.last+1)
    client = actionlib.SimpleActionClient("/calibration_motion/follow_joint_trajectory",
                                          FollowJointTrajectoryAction)
    if not client.wait_for_server(rospy.Duration(2)):
        raise SystemExit("标定运动服务未启动；当前预览不能直接发送给实机。")
    rospy.on_shutdown(client.cancel_all_goals)
    for index in indices:
        seg = session.data["segments"][index]
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.frame_id = seg["name"]
        goal.trajectory.joint_names = JOINT_NAMES
        for q, seconds in ((seg["start_rad"], 0), (seg["goal_rad"], seg["duration_s"])):
            point = JointTrajectoryPoint()
            point.positions = q
            point.time_from_start = rospy.Duration.from_sec(seconds)
            goal.trajectory.points.append(point)
        client.send_goal(goal)
        if not client.wait_for_result(rospy.Duration(seg["duration_s"]+8)):
            client.cancel_goal()
            raise SystemExit("执行超时，已发送取消；检查停止反馈，必要时使用实体电源开关。")
        result = client.get_result()
        print(seg["name"], result)
        if client.get_state() != 3 or result is None or result.error_code != 0:
            raise SystemExit("本段未成功，后续段未发送。")


if __name__ == "__main__":
    main()
