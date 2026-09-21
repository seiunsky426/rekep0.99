#!/usr/bin/env python3
"""Fuse fresh workspace clouds for grasp perception and source-colored RViz."""

import json
import threading
import time
from collections import deque

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String

from rekpiper_camera.cloud_fusion import (
    closest_timestamp_pair,
    voxel_fuse_base_clouds,
)


class DualWorkspaceCloudFusionNode:
    def __init__(self):
        self._target_frame = rospy.get_param("~target_frame", "base_link")
        self._maximum_delta_s = float(rospy.get_param(
            "~maximum_stamp_delta_s", 0.025))
        self._period_s = 1.0 / float(rospy.get_param(
            "~publish_rate_hz", 5.0))
        self._voxel_size_m = float(rospy.get_param(
            "~voxel_size_m", 0.003))
        self._pair_queue_size = int(rospy.get_param("~pair_queue_size", 16))
        self._maximum_age_s = float(rospy.get_param("~maximum_cloud_age_s", 0.5))
        if (not self._target_frame or self._maximum_delta_s <= 0.0
                or self._period_s <= 0.0 or self._pair_queue_size <= 0
                or not np.isfinite(self._maximum_age_s)
                or self._maximum_age_s <= 0.0):
            raise rospy.ROSInitException("dual fusion parameters are invalid")
        # Validate the static fusion configuration before accepting data.
        voxel_fuse_base_clouds({
            "rs1": np.empty((0, 3), dtype=np.float32),
            "rs3": np.empty((0, 3), dtype=np.float32),
        }, self._voxel_size_m)
        self._lock = threading.Lock()
        # Serialize callbacks through publication, including voxel reduction.
        self._processing_lock = threading.Lock()
        # Projection of each 640x480 cloud is relatively expensive.  ROS
        # callbacks can therefore arrive late and out of order even when the
        # hardware depth stamps are synchronized.  Keep a short history and
        # pair by header stamp, rather than comparing only the latest arrival.
        self._pending = {
            "rs1": deque(maxlen=self._pair_queue_size),
            "rs3": deque(maxlen=self._pair_queue_size),
        }
        self._last_pair = None
        self._consumed_stamps = {"rs1": 0, "rs3": 0}
        self._last_publish_s = 0.0
        self._publisher = rospy.Publisher(
            rospy.get_param("~output_topic", "/rekpiper/camera/fused/points_base"),
            PointCloud2, queue_size=1)
        self._status_publisher = rospy.Publisher(
            "/rekpiper/camera/fused/status", String, queue_size=1, latch=True)
        for camera in ("rs1", "rs3"):
            rospy.Subscriber(
                rospy.get_param(
                    "~{}_input_topic".format(camera),
                    "/rekpiper/camera/{}/points_recognition".format(camera)),
                PointCloud2, self._callback, callback_args=camera, queue_size=1,
                buff_size=32 * 1024 * 1024)
        self._publish_status("WAITING_FOR_BOTH_CLOUDS")

    def _xyz(self, message):
        if message.header.frame_id != self._target_frame:
            raise ValueError("cloud frame must be {}".format(self._target_frame))
        if (message.point_step != 12 or message.row_step != message.width * 12
                or len(message.fields) != 3 or message.is_bigendian
                or [field.name for field in message.fields] != ["x", "y", "z"]
                or [field.offset for field in message.fields] != [0, 4, 8]
                or any(field.datatype != PointField.FLOAT32
                       for field in message.fields)):
            raise ValueError("expected contiguous XYZ recognition cloud")
        return np.frombuffer(message.data, dtype=np.float32).reshape(-1, 3).copy()

    @staticmethod
    def _cloud_message(header, points, rgb):
        record = np.empty(len(points), dtype=np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4"),
        ]))
        record["x"], record["y"], record["z"] = points.T
        packed = ((rgb[:, 0].astype(np.uint32) << 16)
                  | (rgb[:, 1].astype(np.uint32) << 8)
                  | rgb[:, 2].astype(np.uint32))
        record["rgb"] = packed.view(np.float32)
        output = PointCloud2()
        output.header = header
        output.height = 1
        output.width = len(record)
        output.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]
        output.is_bigendian = False
        output.point_step = 16
        output.row_step = len(record) * output.point_step
        output.is_dense = True
        output.data = record.tobytes()
        return output

    def _callback(self, message, camera):
        with self._processing_lock:
            self._process(message, camera)

    def _fresh(self, stamp_s):
        age_s = rospy.Time.now().to_sec() - stamp_s
        return (np.isfinite(stamp_s) and stamp_s > 0.0
                and -self._maximum_delta_s <= age_s <= self._maximum_age_s)

    def _process(self, message, camera):
        stamp_s = message.header.stamp.to_sec()
        if not self._fresh(stamp_s):
            self._publish_status("STALE_OR_INVALID_INPUT", camera=camera)
            return
        try:
            points = self._xyz(message)
        except ValueError as exc:
            rospy.logwarn_throttle(2.0, "%s fusion input rejected: %s", camera, exc)
            return
        with self._lock:
            stamp_ns = message.header.stamp.to_nsec()
            if (stamp_ns <= self._consumed_stamps[camera]
                    or any(item[0].stamp.to_nsec() == stamp_ns
                           for item in self._pending[camera])):
                return
            self._pending[camera].append((message.header, stamp_s, points))
            for queue in self._pending.values():
                recent = sorted((item for item in queue if self._fresh(item[1])),
                                key=lambda item: item[1])
                queue.clear()
                queue.extend(recent)
            if not self._pending["rs1"] or not self._pending["rs3"]:
                state, payload = "WAITING_FOR_BOTH_CLOUDS", {}
            else:
                pair = closest_timestamp_pair(
                    [item[1] for item in self._pending["rs1"]],
                    [item[1] for item in self._pending["rs3"]],
                )
                rs1_index, rs3_index, delta_s = pair
                rs1 = self._pending["rs1"][rs1_index]
                rs3 = self._pending["rs3"][rs3_index]
                pair = (rs1[0].stamp.to_nsec(), rs3[0].stamp.to_nsec())
                now_s = time.monotonic()
                if delta_s > self._maximum_delta_s:
                    # The oldest stamp cannot form a better future pair once
                    # both queues are populated, so discard it and retain the
                    # recent history for a later matching arrival.
                    if self._pending["rs1"][0][1] <= self._pending["rs3"][0][1]:
                        self._pending["rs1"].popleft()
                    else:
                        self._pending["rs3"].popleft()
                    state, payload = "WAITING_FOR_SYNCHRONIZED_CLOUDS", {
                        "stamp_delta_s": delta_s,
                    }
                elif pair == self._last_pair or now_s - self._last_publish_s < self._period_s:
                    # This pair was already used (or intentionally skipped by
                    # output throttling); consume it so it cannot indefinitely
                    # remain the closest entry in the two queues.
                    for queue, selected_index in (
                            (self._pending["rs1"], rs1_index),
                            (self._pending["rs3"], rs3_index)):
                        for _ in range(selected_index + 1):
                            queue.popleft()
                    self._consumed_stamps.update(rs1=pair[0], rs3=pair[1])
                    return
                else:
                    self._last_pair = pair
                    self._consumed_stamps.update(rs1=pair[0], rs3=pair[1])
                    self._last_publish_s = now_s
                    # Consume this pair (and older samples) so it cannot be
                    # fused again on the next callback.
                    for queue, selected_index in (
                            (self._pending["rs1"], rs1_index),
                            (self._pending["rs3"], rs3_index)):
                        for _ in range(selected_index + 1):
                            queue.popleft()
                    state, payload = None, (rs1, rs3, delta_s)
        if state is not None:
            self._publish_status(state, **payload)
            return
        rs1, rs3, delta_s = payload
        try:
            points, colors, stats = voxel_fuse_base_clouds({
                "rs1": rs1[2], "rs3": rs3[2],
            }, self._voxel_size_m)
        except ValueError as exc:
            self._publish_status("FUSION_REJECTED", reason=str(exc))
            return
        if not self._fresh(rs1[1]) or not self._fresh(rs3[1]):
            self._publish_status("STALE_AFTER_FUSION")
            return
        header = rs1[0]
        header.frame_id = self._target_frame
        self._publisher.publish(self._cloud_message(header, points, colors))
        self._publish_status("READY", stamp_delta_s=delta_s,
                             source_stamps_s={"rs1": rs1[1], "rs3": rs3[1]}, **stats)

    def _publish_status(self, state, **extra):
        payload = {
            "schema_version": 1,
            "state": state,
            "frame_id": self._target_frame,
            "output_topic": self._publisher.resolved_name,
            "voxel_size_m": self._voxel_size_m,
            "maximum_stamp_delta_s": self._maximum_delta_s,
            "maximum_cloud_age_s": self._maximum_age_s,
            "pair_queue_size": self._pair_queue_size,
            "motion_command_capable": False,
        }
        payload.update(extra)
        self._status_publisher.publish(String(data=json.dumps(
            payload, sort_keys=True, separators=(",", ":"))))


if __name__ == "__main__":
    rospy.init_node("dual_workspace_cloud_fusion")
    try:
        DualWorkspaceCloudFusionNode()
        rospy.spin()
    except rospy.ROSInitException as exc:
        rospy.logfatal(str(exc))
        raise
