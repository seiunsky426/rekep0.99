#!/usr/bin/env python3
"""Start the internal Piper graph only after signed preflight is READY."""

import json
import os
import secrets
import subprocess
import threading

import rospy
from std_msgs.msg import String

from rekpiper_acceptance import AcceptanceError, validate_release_bundle


class HardwareSupervisor:
    def __init__(self):
        self._lock = threading.RLock()
        self._process = None
        self._preflight_ready = False
        self._release_bundle = str(rospy.get_param("~release_bundle", ""))
        self._public_key = str(rospy.get_param("~acceptance_public_key", ""))
        self._minimum_counter = str(rospy.get_param(
            "~minimum_release_counter", ""))
        self._robot_id = str(rospy.get_param("~robot_id", "piper-rekpiper"))
        self._runtime_launch = str(rospy.get_param("~runtime_launch"))
        if os.environ.get("REKPIPER_ACCEPTANCE_BUNDLE_TYPE") == \
                "hardware_acceptance_bundle":
            self._runtime_launch = str(rospy.get_param(
                "~hardware_acceptance_runtime_launch"))
        self._can_port = str(rospy.get_param("~can_port"))
        self._allow = bool(rospy.get_param("~allow_hardware_commands", False))
        self._gripper = str(rospy.get_param("~gripper_baseline"))
        if not self._allow:
            raise rospy.ROSInitException("hardware supervisor permission is false")
        rospy.on_shutdown(self._shutdown)
        rospy.Subscriber("/rekpiper/preflight", String, self._preflight,
                         queue_size=1)
        rospy.Timer(rospy.Duration(0.2), self._try_start)

    def _preflight(self, message):
        try:
            status = json.loads(message.data)
        except (TypeError, ValueError):
            self._shutdown()
            return
        if status.get("mode") != "autonomous" or not bool(status.get("ready")):
            self._shutdown()
            return
        self._preflight_ready = True
        self._try_start(None)

    def _try_start(self, _event):
        if (not self._preflight_ready
                or not bool(rospy.get_param(
                    "/rekpiper/camera/extrinsics/precision_operation_allowed",
                    False))):
            return
        with self._lock:
            if self._process is not None:
                return
            try:
                release = validate_release_bundle(
                    self._release_bundle, self._public_key,
                    self._minimum_counter, self._robot_id)
                for name in ("rs1", "rs3"):
                    expected = str(release["verified_artifacts"][
                        "camera_extrinsics_" + name]["payload"]["serial"]
                    ).lstrip("_")
                    parameters = (
                        "/{}/realsense2_camera/serial_no".format(name),
                        "/{}/serial_no".format(name))
                    actual = next((str(rospy.get_param(parameter)).lstrip("_")
                                   for parameter in parameters
                                   if rospy.has_param(parameter)), "")
                    if actual != expected:
                        return
            except (AcceptanceError, KeyError, TypeError, ValueError) as exc:
                rospy.logfatal("hardware release revalidation failed: %s", exc)
                rospy.signal_shutdown("signed hardware release rejected")
                return
            command = [
                "roslaunch", self._runtime_launch,
                "guard_token:=" + secrets.token_hex(32),
                "can_port:=" + self._can_port,
                "allow_hardware_commands:=true",
                "gripper_baseline:=" + self._gripper,
                "release_bundle:=" + self._release_bundle,
                "acceptance_public_key:=" + self._public_key,
                "minimum_release_counter:=" + self._minimum_counter,
                "robot_id:=" + self._robot_id,
            ]
            self._process = subprocess.Popen(command)
            rospy.loginfo("signed preflight passed; internal hardware graph started")

    def _shutdown(self):
        with self._lock:
            process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    rospy.init_node("rekpiper_hardware_supervisor")
    HardwareSupervisor()
    rospy.spin()
