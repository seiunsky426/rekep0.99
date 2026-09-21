#!/usr/bin/env python3
"""Run one prepared Piper calibration path with simple operator checkpoints."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import rospkg
import rospy
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from rekpiper_calibration.calibration_motion import MotionSession
from rekpiper_calibration.trajectory_preview import JOINT_NAMES


def send_segment(client, segment):
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.header.frame_id = segment["name"]
    goal.trajectory.joint_names = list(JOINT_NAMES)
    for values, seconds in ((segment["start_rad"], 0.0),
                            (segment["goal_rad"], segment["duration_s"])):
        point = JointTrajectoryPoint()
        point.positions = values
        point.time_from_start = rospy.Duration.from_sec(seconds)
        goal.trajectory.points.append(point)
    client.send_goal(goal)
    if not client.wait_for_result(rospy.Duration(segment["duration_s"]+8.0)):
        client.cancel_goal()
        raise RuntimeError(segment["name"]+" timed out")
    result = client.get_result()
    if client.get_state() != 3 or result is None or result.error_code != 0:
        raise RuntimeError(segment["name"]+" failed: "+str(result))


def checkpoint(text):
    value = input(text+" [回车继续，q退出并失能] ").strip().lower()
    if value == "q":
        raise KeyboardInterrupt
    if value:
        raise RuntimeError("只接受回车继续或q退出")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    urdf = Path(rospkg.RosPack().get_path("piper_description"))/"urdf/piper_description.urdf"
    session = MotionSession(args.session, urdf)
    session.require_ready()
    rospy.init_node("supervised_calibration_runner", anonymous=True,
                    disable_signals=True)
    try:
        state = rospy.get_master().getSystemState()[2]
    except Exception as exc:
        raise SystemExit("无法读取ROS master: "+str(exc))
    owners = sorted({node for topic, nodes in state[0]
                     if topic in ("/joint_states_single", "/arm_status")
                     for node in nodes})
    if owners:
        raise SystemExit("先停止已有Piper节点: "+", ".join(owners))

    command = ["roslaunch", "rekpiper_calibration", "calibration_motion.launch",
               "session:="+str(Path(args.session).resolve()), "execute:=true"]
    environment = os.environ.copy()
    environment["ROS_HOME"] = "/tmp/rekpiper_supervised_calibration_ros"
    Path(environment["ROS_HOME"]).mkdir(parents=True, exist_ok=True)
    launch = subprocess.Popen(command, env=environment,
                              start_new_session=True)
    disabled = False
    try:
        for name in ("/calibration_motion/prepare_disable",
                     "/calibration_motion/enable_hold",
                     "/calibration_motion/disable"):
            rospy.wait_for_service(name, timeout=10)
        prepare = rospy.ServiceProxy("/calibration_motion/prepare_disable", Trigger)()
        print("自动失能:", prepare.message)
        if not prepare.success:
            raise RuntimeError(prepare.message)
        checkpoint("六轴已确认000000。托住机械臂并确认工作区安全，然后使能")
        enabled = rospy.ServiceProxy("/calibration_motion/enable_hold", Trigger)()
        print("使能保持:", enabled.message)
        if not enabled.success:
            raise RuntimeError(enabled.message)
        client = actionlib.SimpleActionClient(
            "/calibration_motion/follow_joint_trajectory",
            FollowJointTrajectoryAction)
        if not client.wait_for_server(rospy.Duration(5)):
            raise RuntimeError("trajectory action unavailable")
        checkpoint("当前位置保持成功。运行最大单关节1度试动")
        send_segment(client, session.data["segments"][0])
        print("trial 完成")
        for index, segment in enumerate(session.data["segments"][1:], 1):
            checkpoint("运行 P{:02d}/20，预计 {:.1f} 秒".format(
                index, segment["duration_s"]))
            send_segment(client, segment)
            print("P{:02d} 完成".format(index))
        result = rospy.ServiceProxy("/calibration_motion/disable", Trigger)()
        disabled = result.success
        print("全部完成，失能:", result.message)
        return 0 if result.success else 1
    except KeyboardInterrupt:
        print("\n操作者终止，正在失能……")
        return 130
    except Exception as exc:
        print("执行停止:", exc, file=sys.stderr)
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
