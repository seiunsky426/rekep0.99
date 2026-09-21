#!/usr/bin/env python3
"""Acquire the exclusive session lock, then and only then start the runtime."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import rosnode
import rospy

from rekpiper_mapping.session_guard import (
    CANInterfaceNotReady, ExclusiveSessionLock, SessionAlreadyRunning,
    process_snapshot, require_can_interface_ready,
    process_tree_contains_rosmaster)


def conflicting_ros_nodes(names):
    existing = set(rosnode.get_node_names())
    return sorted(existing.intersection(str(name) for name in names))


def conflicting_piper_processes(excluded_pids=()):
    excluded = set(int(pid) for pid in excluded_pids)
    conflicts = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in excluded:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "piper_ctrl_single_node.py" in command:
            conflicts.append({"pid": int(entry.name), "command": command.strip()})
    return sorted(conflicts, key=lambda value: value["pid"])


def stop_child(process, timeout_s=10.0):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=timeout_s)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2.0)


def main():
    rospy.init_node("dual_aruco_handeye_session_guard", anonymous=True)
    domain_name = str(rospy.get_param("~domain_name", "guarded_runtime"))
    runtime_launch = Path(rospy.get_param("~runtime_launch")).resolve()
    runtime_args = [str(value) for value in rospy.get_param("~runtime_args", [])]
    lock_path = rospy.get_param("~lock_file", "") or None
    conflicts_to_check = rospy.get_param("~conflicting_nodes", [])
    check_piper_processes = bool(rospy.get_param("~check_piper_processes", False))
    require_external_master = bool(rospy.get_param("~require_external_master", True))
    required_can_interface = str(rospy.get_param("~required_can_interface", ""))
    required_can_bitrate = int(rospy.get_param("~required_can_bitrate", 1000000))
    lock = ExclusiveSessionLock(lock_path)
    child = None
    try:
        lock.acquire({"domain": domain_name, "runtime_launch": str(runtime_launch)})
        if require_external_master and process_tree_contains_rosmaster(
                os.getppid(), process_snapshot()):
            raise RuntimeError(
                "this roslaunch started its own ROS master; start an independent "
                "'roscore' first and run this launch with the --wait option")
        node_conflicts = conflicting_ros_nodes(conflicts_to_check)
        process_conflicts = conflicting_piper_processes(
            (os.getpid(), os.getppid())) if check_piper_processes else []
        if node_conflicts or process_conflicts:
            raise RuntimeError(
                "refusing hardware startup; existing_nodes={} existing_piper_processes={}".format(
                    node_conflicts, process_conflicts))
        if required_can_interface:
            details = require_can_interface_ready(
                required_can_interface, required_can_bitrate)
            rospy.loginfo(
                "CAN startup gate passed: %s UP %s at %d bit/s",
                required_can_interface, details["can_state"], details["bitrate"])
        if not runtime_launch.is_file():
            raise RuntimeError("internal runtime launch file missing: " + str(runtime_launch))
        token = uuid.uuid4().hex
        command = ["roslaunch", str(runtime_launch), "guard_token:=" + token] + runtime_args
        environment = os.environ.copy()
        environment["REKEP_DUAL_HANDEYE_GUARD_TOKEN"] = token
        rospy.loginfo("Exclusive %s lock acquired; starting guarded runtime", domain_name)
        child = subprocess.Popen(
            command, env=environment, start_new_session=True,
            pass_fds=(lock.fileno,))
        while not rospy.is_shutdown() and child.poll() is None:
            time.sleep(0.2)
        if rospy.is_shutdown():
            stop_child(child)
            return 0
        return int(child.returncode or 0)
    except SessionAlreadyRunning as exc:
        rospy.logfatal(str(exc))
        return 73
    except CANInterfaceNotReady as exc:
        rospy.logfatal("%s startup refused before Piper runtime: %s", domain_name, exc)
        return 75
    except Exception as exc:
        rospy.logfatal("%s startup refused: %s", domain_name, exc)
        return 74
    finally:
        stop_child(child)
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
