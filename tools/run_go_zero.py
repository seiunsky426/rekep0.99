#!/usr/bin/env python3
"""Disable, enable, move slowly to six-axis zero, and hold until release."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import rospkg
import rospy
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from rekpiper_calibration.calibration_motion import GO_ZERO_PURPOSE, MotionSession
from rekpiper_calibration.trajectory_preview import JOINT_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    urdf = Path(rospkg.RosPack().get_path("piper_description"))/"urdf/piper_description.urdf"
    session = MotionSession(args.session, urdf)
    session.require_ready()
    if session.purpose != GO_ZERO_PURPOSE or len(session.data["segments"]) != 1:
        raise SystemExit("session is not a one-segment go-zero recovery")
    rospy.init_node("go_zero_runner", anonymous=True,
                    disable_signals=True)
    owners = sorted({node for topic, nodes in rospy.get_master().getSystemState()[2][0]
                     if topic in ("/joint_states_single", "/arm_status")
                     for node in nodes})
    if owners:
        raise SystemExit("先停止已有Piper节点: "+", ".join(owners))
    command = [
        "roslaunch", "rekpiper_calibration", "calibration_motion.launch",
        "session:="+str(Path(args.session).resolve()), "execute:=true",
    ]
    environment = os.environ.copy()
    environment["ROS_HOME"] = "/tmp/rekpiper_go_zero_ros"
    Path(environment["ROS_HOME"]).mkdir(parents=True, exist_ok=True)
    launch = subprocess.Popen(command, env=environment, start_new_session=True)
    disabled = False
    try:
        for name in ("/calibration_motion/prepare_disable",
                     "/calibration_motion/enable_hold",
                     "/calibration_motion/disable"):
            rospy.wait_for_service(name, timeout=10)
        result = rospy.ServiceProxy(
            "/calibration_motion/prepare_disable", Trigger)()
        print("自动失能:", result.message)
        if not result.success:
            raise RuntimeError(result.message)
        result = rospy.ServiceProxy(
            "/calibration_motion/enable_hold", Trigger)()
        print("自动使能并保持:", result.message)
        if not result.success:
            raise RuntimeError(result.message)
        segment = session.data["segments"][0]
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.frame_id = segment["name"]
        goal.trajectory.joint_names = list(JOINT_NAMES)
        for values, seconds in ((segment["start_rad"], 0.0),
                                (segment["goal_rad"], segment["duration_s"])):
            point = JointTrajectoryPoint()
            point.positions = values
            point.time_from_start = rospy.Duration.from_sec(seconds)
            goal.trajectory.points.append(point)
        client = actionlib.SimpleActionClient(
            "/calibration_motion/follow_joint_trajectory",
            FollowJointTrajectoryAction)
        if not client.wait_for_server(rospy.Duration(5)):
            raise RuntimeError("trajectory action unavailable")
        print("开始低速回零，预计 {:.1f} 秒".format(segment["duration_s"]))
        client.send_goal(goal)
        if not client.wait_for_result(rospy.Duration(segment["duration_s"]+8)):
            client.cancel_goal()
            raise RuntimeError("go-zero timed out")
        action_result = client.get_result()
        if client.get_state() != 3 or action_result is None or action_result.error_code != 0:
            raise RuntimeError("go-zero failed: "+str(action_result))
        print("六关节零位已确认；机械臂保持使能并持续保持零位。")
        input("确认可以撤去保持力后，按回车失能并退出：")
        result = rospy.ServiceProxy("/calibration_motion/disable", Trigger)()
        disabled = result.success
        print("已按操作者指令失能:", result.message)
        return 0 if disabled else 1
    except Exception as exc:
        print("回零停止:", exc, file=sys.stderr)
        return 1
    finally:
        if not disabled:
            try:
                result = rospy.ServiceProxy(
                    "/calibration_motion/disable", Trigger)()
                print("退出失能:", result.message)
            except Exception as exc:
                print("无法确认软件失能，请使用实体断电开关:", exc,
                      file=sys.stderr)
        os.killpg(launch.pid, signal.SIGINT)
        try:
            launch.wait(timeout=5)
        except subprocess.TimeoutExpired:
            launch.terminate()
            launch.wait(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
