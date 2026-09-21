#!/usr/bin/env python3
"""One operator-supervised calibration session, one action per segment."""

import fcntl
import json
from pathlib import Path
import subprocess
import threading
import time

import actionlib
import can
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryResult
import numpy as np
from piper_msgs.msg import PiperStatusMsg
from piper_sdk import C_PiperInterface_V2
import rospkg
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

from rekpiper_calibration.calibration_motion import MotionSession, StepSequence, positions
from rekpiper_calibration.trajectory_preview import JOINT_NAMES


class CalibrationMotion:
    def __init__(self):
        root = Path(rospkg.RosPack().get_path("piper_description"))
        self.session = MotionSession(
            rospy.get_param("~session"), root/"urdf/piper_description.urdf")
        if rospy.get_param("~execute", False) is not True:
            raise ValueError("hardware disabled; run the client check command for offline validation")
        self.session.require_ready()  # Before constructing SDK or opening CAN.
        self.port = self.session.data["can_port"]
        if self.port != "can0":
            raise ValueError("this site session only permits can0")
        state = subprocess.check_output(["ip", "-details", "link", "show", "dev", self.port], text=True)
        if "UP" not in state.split(">")[0].split("<")[-1].split(",") \
                or "can state ERROR-ACTIVE" not in state or "bitrate 1000000 " not in state:
            raise ValueError("can0 must be UP, ERROR-ACTIVE, 1000000 bit/s")
        system = rospy.get_master().getSystemState()[2]
        if any(topic in ("/joint_states_single", "/arm_status") and nodes for topic, nodes in system[0]):
            raise ValueError("stop the existing Piper feedback/control node before starting this owner")
        self.can_lock = open("/tmp/rekpiper_calibration_can0.lock", "a")
        fcntl.flock(self.can_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.sequence = StepSequence(self.session.data["segments"])
        self.reference = np.asarray(self.session.data["start_rad"])
        self.desired = self.reference.copy()
        self.stop_reason = ""
        self.segment = None
        self.segment_start = 0.0
        self.received = {}
        self.camera = None
        self.camera_time = 0.0
        self.log = (Path(rospy.get_param("~session")).parent /
                    ("motion_events_{}.jsonl".format(time.time_ns()))).open("x", buffering=1)
        self.driver = C_PiperInterface_V2(can_name=self.port)
        self.rx = can.interface.Bus(channel=self.port, bustype="socketcan", receive_own_messages=False)
        self.notifier = can.Notifier(self.rx, [self._can_frame], timeout=.02)
        self.driver.ConnectPort(piper_init=False)
        self.commanded = False
        rospy.on_shutdown(self.close)
        self.joint_pub = rospy.Publisher("/joint_states_single", JointState, queue_size=1)
        self.arm_pub = rospy.Publisher("/arm_status", PiperStatusMsg, queue_size=1)
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        rospy.Subscriber("/dual_aruco18_capture/status", String, self._camera, queue_size=1)
        self.action = actionlib.SimpleActionServer("~follow_joint_trajectory",
            FollowJointTrajectoryAction, execute_cb=self.execute, auto_start=False)
        rospy.Service("~enable_hold", Trigger, self.enable)
        rospy.Service("~prepare_disable", Trigger, self.prepare_disable)
        rospy.Service("~reset", Trigger, self.reset_service)
        rospy.Service("~disable", Trigger, self.disable_service)
        rospy.Service("~stop", Trigger, self.stop_service)
        self.action.start()
        self.event("connected", hardware_motion_started=False)

    def event(self, name, **detail):
        self.log.write(json.dumps(dict(event=name, monotonic_s=time.monotonic(), **detail))+"\n")

    def _can_frame(self, frame):
        if not frame.is_error_frame and frame.is_rx:
            self.received[frame.arbitration_id] = time.monotonic()

    def _camera(self, message):
        try:
            value = json.loads(message.data)
        except (ValueError, TypeError):
            return
        self.camera, self.camera_time = value, time.monotonic()

    def feedback(self, require_enabled=False, allow_controller_stop=False):
        now = time.monotonic()
        ids = [0x2A1, 0x2A5, 0x2A6, 0x2A7] + list(range(0x261, 0x267))
        if any(now-self.received.get(i, 0) > .25 for i in ids):
            raise ValueError("stale or incomplete CAN status/joint/motor frames")
        j = self.driver.GetArmJointMsgs()
        q = np.deg2rad([getattr(j.joint_state, "joint_{}".format(i))/1000 for i in range(1, 7)])
        status = self.driver.GetArmStatus().arm_status
        self.raw_positions = q.copy()
        q = self.session.normalize_feedback(q)
        if ((status.arm_status != 0
             and not (allow_controller_stop and status.arm_status == 1))
                or status.err_code != 0):
            raise ValueError(
                "Piper controller fault: arm_status={} ({!s}), err_code={}, "
                "ctrl_mode={}, mode_feed={}".format(
                    int(status.arm_status), status.arm_status,
                    int(status.err_code), int(status.ctrl_mode),
                    int(status.mode_feed)))
        error = status.err_status
        for i in range(1, 7):
            if getattr(error, "joint_{}_angle_limit".format(i)) or getattr(error, "communication_status_joint_{}".format(i)):
                raise ValueError("Piper limit/communication flag")
        low = self.driver.GetArmLowSpdInfoMsgs()
        motors = [getattr(low, "motor_{}".format(i)).foc_status for i in range(1, 7)]
        fault_keys = ("voltage_too_low", "motor_overheating",
                      "driver_overcurrent", "driver_overheating",
                      "collision_status", "driver_error_status",
                      "stall_status")
        for index, motor in enumerate(motors, 1):
            active = [key for key in fault_keys if getattr(motor, key)]
            if active:
                raise ValueError("Piper motor {} fault: {}".format(
                    index, ",".join(active)))
        enabled = all(m.driver_enable_status for m in motors)
        if require_enabled and (not enabled or status.ctrl_mode != 1 or status.mode_feed != 1):
            raise ValueError("Piper not enabled in CAN/MOVE-J")
        return q, status, enabled

    def send(self, q):
        if self.stop_event.is_set():
            raise ValueError("stop latched")
        self.driver.JointCtrl(*[int(round(v)) for v in np.rad2deg(q)*1000])
        self.commanded = True

    def _motor_enable_states(self):
        low = self.driver.GetArmLowSpdInfoMsgs()
        return [bool(getattr(low, "motor_{}".format(i))
                     .foc_status.driver_enable_status) for i in range(1, 7)]

    def _disable_motors(self, timeout_s=5.0):
        deadline = time.monotonic()+timeout_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self.driver.DisableArm(7)
            time.sleep(.10)
            states = self._motor_enable_states()
            if not any(states):
                return states
        raise ValueError("pre-enable disable did not reach motor feedback 000000")

    def stop(self, reason):
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.stop_reason = str(reason)
        with self.lock:
            self.sequence.stop()
            self.driver.MotionCtrl_1(0x01, 0, 0)
            self.event("stop_requested", reason=reason)
            self.status_pub.publish(String(data=json.dumps({"state": "STOPPED", "reason": reason,
                "completed_segments": self.sequence.cursor, "restart_allowed": False})))

    def reset_service(self, _request):
        try:
            with self.lock:
                self.session.require_ready()
                if self.sequence.state != "CONNECTED_DISABLED":
                    raise ValueError("reset is only allowed before enable")
            changed = self._reset_controller_if_stopped()
            return TriggerResponse(True, "controller reset confirmed; motors remain disabled"
                                   if changed else "controller already normal")
        except Exception as exc:
            return TriggerResponse(False, str(exc))

    def _reset_controller_if_stopped(self):
        _, status, _ = self.feedback(allow_controller_stop=True)
        if status.arm_status == 0:
            return False
        if status.arm_status != 1:
            raise ValueError("only controller-stop state can be reset here")
        deadline = time.monotonic()+3
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self.driver.MotionCtrl_1(0x02, 0, 0)
            time.sleep(.10)
            _, status, _ = self.feedback(allow_controller_stop=True)
            if status.arm_status == 0:
                self.event("controller_reset_confirmed")
                return True
        raise ValueError("controller reset acknowledgement timed out")

    def prepare_disable(self, _request):
        try:
            with self.lock:
                self.session.require_ready()
                if self.sequence.state != "CONNECTED_DISABLED":
                    raise ValueError("prepare-disable allowed once per session")
                q, _, _ = self.feedback(allow_controller_stop=True)
                if np.max(np.abs(q-self.reference)) > np.deg2rad(.2):
                    raise ValueError("arm moved since session snapshot; prepare a new session")
            self._disable_motors()
            self._reset_controller_if_stopped()
            with self.lock:
                q, status, enabled = self.feedback()
                if enabled:
                    raise ValueError("motor feedback changed after confirmed disable")
                if np.max(np.abs(q-self.reference)) > np.deg2rad(.5):
                    raise ValueError("arm moved more than 0.5 degrees while disabled")
                self.reference = q.copy()
                self.sequence.state = "PREPARED_DISABLED"
                self.event("pre_enable_disabled", motor_enable="000000",
                           reference_rad=self.reference.tolist())
                self.status_pub.publish(String(data=json.dumps({
                    "state": "PREPARED_DISABLED", "motor_enable": "000000",
                    "waiting_for_operator_enable": True})))
            return TriggerResponse(True, "motor_enable=000000; press Enter to enable")
        except Exception as exc:
            return TriggerResponse(False, str(exc))

    def disable_service(self, _request):
        try:
            self.stop_event.set()
            self.sequence.stop()
            self._disable_motors()
            self.event("motors_disabled")
            self.status_pub.publish(String(data=json.dumps({
                "state": "DISABLED", "motor_enable": "000000",
                "restart_allowed": False})))
            return TriggerResponse(True, "all six motor drives disabled; restart required")
        except Exception as exc:
            return TriggerResponse(False, str(exc))

    def stop_service(self, _request):
        self.stop("operator_stop")
        deadline = time.monotonic()+1
        while time.monotonic() < deadline:
            if (time.monotonic()-self.received.get(0x2A1, 0) < .25
                    and self.driver.GetArmStatus().arm_status.arm_status == 1):
                return TriggerResponse(True, "controller stop confirmed; session cannot resume")
            time.sleep(.02)
        return TriggerResponse(False, "stop sent but not confirmed; use the physical power switch")

    def enable(self, _request):
        try:
            with self.lock:
                self.session.require_ready()
                if self.sequence.state != "PREPARED_DISABLED":
                    raise ValueError("run prepare-disable before enable")
                q, _, _ = self.feedback()
                if self._motor_enable_states() != [False]*6:
                    raise ValueError("motors are no longer confirmed disabled")
                if np.max(np.abs(q-self.reference)) > np.deg2rad(.2):
                    raise ValueError("arm moved after prepare-disable")
                self.sequence.state = "ENABLING"
            # Enable first. Do not select a motion mode or send joint targets
            # until all six drives have acknowledged enable continuously.
            stable = 0
            deadline = time.monotonic()+6
            while time.monotonic() < deadline and not rospy.is_shutdown():
                self.driver.EnablePiper()
                time.sleep(.10)
                with self.lock:
                    if self.stop_event.is_set():
                        raise ValueError("stop during enable")
                    self.session.require_ready()
                    q, status, enabled = self.feedback()
                    if np.max(np.abs(q-self.reference)) > np.deg2rad(.5):
                        raise ValueError("position changed during enable")
                    stable = stable+1 if enabled else 0
                    if stable >= 10:
                        break
            else:
                raise ValueError("six-motor enable acknowledgement timed out")

            # Only after the drive acknowledgement, select CAN/MOVE-J.  The
            # installed Piper firmware completes this transition after the
            # first paired JointCtrl frame, so pair it with the measured
            # starting position rather than a new target.
            speed = int(self.session.data["limits"]["driver_speed_percent"])
            stable = 0
            deadline = time.monotonic()+3
            while time.monotonic() < deadline and not rospy.is_shutdown():
                self.driver.MotionCtrl_2(0x01, 0x01, speed, 0)
                self.send(self.reference)
                time.sleep(.10)
                with self.lock:
                    self.session.require_ready()
                    q, status, enabled = self.feedback()
                    if np.max(np.abs(q-self.reference)) > np.deg2rad(.5):
                        raise ValueError("position changed during mode selection")
                    ready = (enabled and int(status.ctrl_mode) == 1
                             and int(status.mode_feed) == 1)
                    stable = stable+1 if ready else 0
                    if stable >= 5:
                        break
            else:
                raise ValueError(
                    "CAN/MOVE-J acknowledgement timed out: motors={}, "
                    "ctrl_mode={}, mode_feed={}".format(
                        "".join("1" if value else "0" for value
                                in self._motor_enable_states()),
                        int(status.ctrl_mode), int(status.mode_feed)))

            # Keep the operator's actual starting position; never send zero
            # angles during enable.
            start = time.monotonic()
            while time.monotonic()-start < 2:
                with self.lock:
                    self.session.require_ready()
                    q, _, _ = self.feedback(require_enabled=True)
                    if np.max(np.abs(q-self.reference)) > np.deg2rad(.5):
                        raise ValueError("enable hold drift")
                    self.send(self.reference)
                time.sleep(.02)
            with self.lock:
                if self.stop_event.is_set():
                    raise ValueError("stop during hold")
                self.sequence.state = "CONNECTED_DISABLED"
                self.sequence.enable()
                self.desired = self.reference.copy()
                self.event("hold_verified", actual_rad=q.tolist())
            return TriggerResponse(True, "enabled; startup position held for two seconds; trial is next")
        except Exception as exc:
            self.stop(str(exc))
            return TriggerResponse(False, str(exc))

    def execute(self, goal):
        result = FollowJointTrajectoryResult()
        try:
            with self.lock:
                self.session.require_ready()
                trajectory = goal.trajectory
                if list(trajectory.joint_names) != JOINT_NAMES or len(trajectory.points) != 2:
                    raise ValueError("exactly two reviewed endpoints in joint1..joint6 order required")
                q, _, _ = self.feedback(True)
                if self.sequence.cursor >= len(self.sequence.segments):
                    raise ValueError("session complete")
                segment = self.sequence.segments[self.sequence.cursor]
                for point, expected, seconds in zip(trajectory.points,
                        [segment["start_rad"], segment["goal_rad"]], [0, segment["duration_s"]]):
                    if (len(point.positions) != 6 or not np.allclose(point.positions, expected, atol=1e-10, rtol=0)
                            or abs(point.time_from_start.to_sec()-seconds) > 1e-8
                            or point.velocities or point.accelerations or point.effort):
                        raise ValueError("trajectory differs from prepared endpoints/timing")
                self.segment = self.sequence.begin(trajectory.header.frame_id, q)
                self.segment_start = time.monotonic()
                self.event("segment_started", name=segment["name"])
            while not rospy.is_shutdown():
                if self.action.is_preempt_requested():
                    raise ValueError("action cancelled")
                with self.lock:
                    if self.sequence.state == "STOPPED":
                        raise ValueError("stop latched")
                    elapsed = time.monotonic()-self.segment_start
                    if elapsed >= segment["duration_s"]+segment["dwell_s"]:
                        q, _, _ = self.feedback(True)
                        if np.max(np.abs(q-segment["goal_rad"])) > np.deg2rad(.2):
                            raise ValueError("target not reached within 0.2 degrees")
                        if (self.session.requires_camera
                                and (time.monotonic()-self.camera_time > .3 or not self.camera
                                or self.camera.get("state") != "READY_TO_CAPTURE"
                                or set(self.camera.get("active_cameras", [])) != {"rs1", "rs3"})):
                            raise ValueError("target reached but dual-camera capture is not ready")
                        self.desired = np.asarray(segment["goal_rad"])
                        self.sequence.finish()
                        self.event("segment_completed", name=segment["name"], actual_rad=q.tolist())
                        break
                time.sleep(.02)
            if rospy.is_shutdown():
                raise ValueError("ROS shutdown")
            result.error_code = result.SUCCESSFUL
            self.action.set_succeeded(result)
        except Exception as exc:
            self.stop(str(exc))
            result.error_code, result.error_string = result.INVALID_GOAL, str(exc)
            self.action.set_aborted(result)

    def spin(self):
        last_tick = time.monotonic()
        last_guard = 0.0
        startup = last_tick
        while not rospy.is_shutdown():
            tick = time.monotonic()
            try:
                with self.lock:
                    active = self.sequence.state in ("HOLDING", "MOVING")
                    if active and tick-last_tick > .10:
                        raise ValueError("control loop missed 100 ms deadline")
                    if tick-last_guard >= .1:
                        self.session.require_ready()
                        last_guard = tick
                    q, status, _ = self.feedback(
                        active, allow_controller_stop=not active)
                    if active:
                        if self.sequence.state == "MOVING":
                            self.desired = positions(self.segment, tick-self.segment_start)
                        if np.max(np.abs(q-self.desired)) > np.deg2rad(1):
                            raise ValueError("tracking error exceeds one degree")
                        xyz = self.session.solver.forward(q)[:3, 3]
                        if np.any(xyz < [0, -.4, .05]) or np.any(xyz > [.7, .4, .7]):
                            raise ValueError("actual link6 left workspace")
                        self.send(self.desired)
                    msg = JointState()
                    msg.header.stamp = rospy.Time.now()
                    msg.name, msg.position = JOINT_NAMES, self.raw_positions.tolist()
                    gripper = self.driver.GetArmGripperMsgs()
                    if abs(time.time()-gripper.time_stamp) < .25:
                        msg.name = JOINT_NAMES+["gripper"]
                        msg.position.append(gripper.gripper_state.grippers_angle*1e-6)
                    self.joint_pub.publish(msg)
                    arm = PiperStatusMsg()
                    for name in ("ctrl_mode", "arm_status", "teach_status", "motion_status", "trajectory_num", "err_code"):
                        setattr(arm, name, int(getattr(status, name)))
                    arm.mode_feedback = int(status.mode_feed)
                    self.arm_pub.publish(arm)
                    self.status_pub.publish(String(data=json.dumps({"state": self.sequence.state,
                        "completed_segments": self.sequence.cursor, "next": self.sequence.segments[
                        self.sequence.cursor]["name"] if self.sequence.cursor < len(
                            self.sequence.segments) else "complete"})))
            except Exception as exc:
                if self.sequence.state != "STOPPED" and tick-startup > 2:
                    self.stop(str(exc))
                    rospy.logerr("calibration motion stopped: %s", exc)
            last_tick = tick
            time.sleep(max(0, .02-(time.monotonic()-tick)))

    def close(self):
        try:
            if self.commanded:
                self.stop("shutdown")
        finally:
            self.driver.DisconnectPort()
            self.notifier.stop()
            self.rx.shutdown()
            self.log.close()


if __name__ == "__main__":
    rospy.init_node("calibration_motion")
    CalibrationMotion().spin()
