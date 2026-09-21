#!/usr/bin/env python3
"""No-CAN 20 Hz sink for fixed software replay performance evidence."""

from collections import deque
import json
import threading
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, String

from rekpiper_execution.rate_monitor import distribution
from rekpiper_execution.trajectory import JOINT_NAMES
from rekpiper_msgs.msg import ReKepHorizon


class SoftwareTrajectorySink:
    def __init__(self):
        self._rate = float(rospy.get_param("~command_rate_hz", 20.0))
        self._lock = threading.RLock()
        self._trajectory = None
        self._received = 0.0
        self._stamps = deque(maxlen=20000)
        self._sequence = 0
        self._commands = rospy.Publisher(
            "/joint_ctrl_single", JointState, queue_size=1)
        self._diagnostics = rospy.Publisher(
            "/rekpiper/execution/trajectory_bridge_diagnostics",
            String, queue_size=1, latch=True)
        rospy.Subscriber("/rekpiper/planning/horizon", ReKepHorizon,
                         self._horizon, queue_size=10)
        rospy.Timer(rospy.Duration(1.0 / self._rate), self._tick)

    def _horizon(self, message):
        trajectory = message.authorized_prefix
        positions = np.asarray(
            [point.positions for point in trajectory.points], dtype=float)
        times = np.asarray(
            [point.time_from_start.to_sec() for point in trajectory.points],
            dtype=float)
        if (not message.valid or tuple(trajectory.joint_names) != JOINT_NAMES
                or positions.ndim != 2 or positions.shape[1] != 6
                or len(positions) < 2 or not np.all(np.isfinite(positions))
                or times.shape != (len(positions),)
                or np.any(np.diff(times) <= 0.0)):
            self._publish("invalid", reason="invalid_replay_horizon")
            return
        with self._lock:
            self._trajectory = (positions, times)
            self._received = time.monotonic()

    def _tick(self, _event):
        with self._lock:
            record = self._trajectory
            elapsed = time.monotonic() - self._received
        if record is None or elapsed > 0.25:
            self._publish("unavailable", reason="replay_horizon_stale")
            return
        positions, times = record
        desired = [np.interp(min(elapsed, times[-1]), times, positions[:, index])
                   for index in range(6)]
        self._commands.publish(JointState(
            header=Header(stamp=rospy.Time.now()),
            name=list(JOINT_NAMES), position=desired))
        self._stamps.append(time.monotonic())
        if len(self._stamps) % 20 == 0:
            gaps = np.diff(np.asarray(self._stamps, dtype=float))
            timing = distribution(gaps)
            duration = self._stamps[-1] - self._stamps[0]
            timing.update({
                "actual_rate_hz": ((len(self._stamps) - 1) / duration
                                   if duration > 0.0 else 0.0),
                "maximum_gap_s": float(np.max(gaps)) if len(gaps) else 0.0,
                "command_samples": len(self._stamps),
            })
            self._publish("streaming", command_timing=timing)

    def _publish(self, state, **values):
        self._sequence += 1
        payload = {"state": state, "sequence": self._sequence,
                   "event_monotonic_ns": time.monotonic_ns()}
        payload.update(values)
        self._diagnostics.publish(String(data=json.dumps(payload, sort_keys=True)))


if __name__ == "__main__":
    rospy.init_node("rekpiper_software_trajectory_sink")
    SoftwareTrajectorySink()
    rospy.spin()
