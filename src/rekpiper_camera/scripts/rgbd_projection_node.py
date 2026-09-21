#!/usr/bin/env python3
"""Synchronize D435 RGB-D and publish an organized cloud.

Source classification: NEW ROS adapter.
Everloom reference: real depth back-projection only. Hard-coded intrinsics,
file input, and stale transforms are intentionally not retained.
"""

import math
import threading
import time

from cv_bridge import CvBridge, CvBridgeError
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import message_filters
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header
import tf2_ros

from rekpiper_camera.projection import (
    CameraIntrinsics,
    depth_to_xyz,
    transform_points,
    transform_to_matrix,
)


class RGBDProjectionNode:
    def __init__(self):
        self._target_frame = rospy.get_param("~target_frame", "base_link")
        self._optical_frame = rospy.get_param(
            "~optical_frame", "camera_color_optical_frame"
        )
        self._depth_scale = float(rospy.get_param("~depth_scale", 0.001))
        self._min_depth_m = float(rospy.get_param("~min_depth_m", 0.10))
        self._max_depth_m = float(rospy.get_param("~max_depth_m", 2.00))
        # At 30 Hz, a 50 ms window can pair adjacent RGB/depth frames.
        self._sync_slop_s = float(rospy.get_param("~sync_slop_s", 0.015))
        self._tf_timeout_s = float(rospy.get_param("~tf_timeout_s", 0.10))
        self._use_tf = bool(rospy.get_param("~use_tf", True))
        self._correction_topic = str(rospy.get_param(
            "~extrinsic_correction_topic", "")).strip()
        self._require_correction = bool(rospy.get_param(
            "~require_extrinsic_correction", False))
        self._correction_max_age_s = float(rospy.get_param(
            "~extrinsic_correction_max_age_s", 1.0))
        queue_size = int(rospy.get_param("~sync_queue_size", 5))

        if self._depth_scale <= 0.0:
            raise rospy.ROSInitException("~depth_scale must be positive")
        if (self._sync_slop_s <= 0.0 or self._tf_timeout_s <= 0.0
                or self._correction_max_age_s <= 0.0):
            raise rospy.ROSInitException("sync and TF timeouts must be positive")

        rgb_topic = rospy.get_param("~rgb_topic", "/camera/color/image_raw")
        depth_topic = rospy.get_param(
            "~depth_topic", "/camera/aligned_depth_to_color/image_raw"
        )
        info_topic = rospy.get_param(
            "~camera_info_topic", "/camera/color/camera_info"
        )

        self._bridge = CvBridge()
        self._tf_buffer = None
        self._tf_listener = None
        if self._use_tf:
            self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        elif self._target_frame != self._optical_frame:
            raise rospy.ROSInitException(
                "without TF, target_frame must equal optical_frame"
            )
        output_topic = rospy.get_param("~points_output_topic", "~points_base")
        self._cloud_pub = rospy.Publisher(output_topic, PointCloud2, queue_size=1)
        # Keep the established XYZ-only cloud contract for geometry consumers.
        # A separate cloud carries packed RGB for RViz and human inspection.
        colored_output_topic = rospy.get_param(
            "~colored_points_output_topic", output_topic + "_rgb")
        self._colored_cloud_pub = rospy.Publisher(
            colored_output_topic, PointCloud2, queue_size=1)
        self._valid_pub = rospy.Publisher("~valid_depth", Image, queue_size=1)
        self._diagnostics_pub = rospy.Publisher(
            "/diagnostics", DiagnosticArray, queue_size=2
        )
        self._lock = threading.Lock()
        self._correction_lock = threading.Lock()
        self._extrinsic_correction = np.eye(4, dtype=float)
        self._correction_received_monotonic = None
        self._frames_ok = 0
        self._frames_dropped = 0

        rgb_sub = message_filters.Subscriber(rgb_topic, Image, queue_size=1)
        depth_sub = message_filters.Subscriber(depth_topic, Image, queue_size=1)
        info_sub = message_filters.Subscriber(info_topic, CameraInfo, queue_size=1)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub],
            queue_size=queue_size,
            slop=self._sync_slop_s,
            allow_headerless=False,
        )
        self._sync.registerCallback(self._callback)
        if self._correction_topic:
            rospy.Subscriber(
                self._correction_topic, TransformStamped,
                self._correction_callback, queue_size=1)

        rospy.loginfo(
            "ReKep RGB-D projection: %s -> %s, depth_scale=%.9f",
            self._optical_frame,
            self._target_frame,
            self._depth_scale,
        )

    def _correction_callback(self, message):
        if message.header.frame_id != self._target_frame:
            rospy.logwarn_throttle(
                2.0, "extrinsic correction ignored: expected parent %s, got %s",
                self._target_frame, message.header.frame_id)
            return
        try:
            correction = transform_to_matrix(message.transform)
        except ValueError as exc:
            rospy.logwarn_throttle(2.0, "extrinsic correction ignored: %s", exc)
            return
        with self._correction_lock:
            self._extrinsic_correction = correction
            self._correction_received_monotonic = time.monotonic()

    def _fresh_extrinsic_correction(self):
        if not self._correction_topic:
            return np.eye(4, dtype=float), "not_configured"
        with self._correction_lock:
            received = self._correction_received_monotonic
            correction = self._extrinsic_correction.copy()
        if received is None:
            return None, "not_received"
        if time.monotonic() - received > self._correction_max_age_s:
            return None, "stale"
        return correction, "ready"

    def _drop(self, reason, message):
        with self._lock:
            self._frames_dropped += 1
        rospy.logwarn_throttle(2.0, "RGB-D frame dropped (%s): %s", reason, message)
        self._publish_diagnostic(DiagnosticStatus.WARN, reason, str(message))

    def _publish_diagnostic(self, level, reason, detail):
        with self._lock:
            ok = self._frames_ok
            dropped = self._frames_dropped
        status = DiagnosticStatus(
            level=level,
            name="rekpiper_camera/rgbd_projection",
            message=reason,
            hardware_id="realsense_d435",
            values=[
                KeyValue("detail", detail),
                KeyValue("frames_ok", str(ok)),
                KeyValue("frames_dropped", str(dropped)),
            ],
        )
        self._diagnostics_pub.publish(
            DiagnosticArray(header=Header(stamp=rospy.Time.now()), status=[status])
        )

    def _callback(self, rgb_msg, depth_msg, info_msg):
        stamps = [
            rgb_msg.header.stamp.to_sec(),
            depth_msg.header.stamp.to_sec(),
            info_msg.header.stamp.to_sec(),
        ]
        if not all(math.isfinite(value) and value > 0.0 for value in stamps):
            self._drop("invalid_timestamp", stamps)
            return
        if max(stamps) - min(stamps) > self._sync_slop_s:
            self._drop("timestamp_skew", max(stamps) - min(stamps))
            return
        for name, message in (("rgb", rgb_msg), ("depth", depth_msg),
                              ("camera_info", info_msg)):
            if message.header.frame_id != self._optical_frame:
                self._drop("unexpected_" + name + "_frame", message.header.frame_id)
                return
        if depth_msg.encoding not in ("16UC1", "32FC1"):
            self._drop("unsupported_depth_encoding", depth_msg.encoding)
            return
        # This pinhole path requires rectified/undistorted pixel coordinates.
        # Do not silently apply file-calibrated distorted K to aligned depth.
        if (not np.all(np.isfinite(info_msg.D))
                or np.any(np.abs(info_msg.D) > 1e-8)):
            self._drop("distorted_camera_info", info_msg.D)
            return

        try:
            # RGB is synchronized intentionally even though cloud publication only
            # needs its geometry; decoding confirms the frame is usable downstream.
            rgb = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self._drop("cv_bridge", exc)
            return
        if rgb.shape[:2] != depth.shape[:2]:
            self._drop("unaligned_rgb_depth", (rgb.shape, depth.shape))
            return

        try:
            intrinsics = CameraIntrinsics.from_camera_matrix(
                info_msg.K, info_msg.width, info_msg.height
            )
            xyz_optical, valid = depth_to_xyz(
                depth,
                intrinsics,
                1.0 if depth_msg.encoding == "32FC1" else self._depth_scale,
                self._min_depth_m,
                self._max_depth_m,
            )
            if self._use_tf:
                stamped_tf = self._tf_buffer.lookup_transform(
                    self._target_frame,
                    self._optical_frame,
                    depth_msg.header.stamp,
                    rospy.Duration(self._tf_timeout_s),
                )
                matrix = transform_to_matrix(stamped_tf.transform)
                correction, correction_state = self._fresh_extrinsic_correction()
                if correction is None:
                    if self._require_correction:
                        self._drop("extrinsic_correction_" + correction_state,
                                   "RS3 refinement required but unavailable")
                        return
                    correction = np.eye(4, dtype=float)
                matrix = correction @ matrix
                xyz_output = transform_points(xyz_optical, matrix)
            else:
                xyz_output = xyz_optical
        except (ValueError, tf2_ros.TransformException) as exc:
            # Never fall back to latest TF or a previous frame transform.
            self._drop("projection_or_tf", exc)
            return

        header = Header(stamp=depth_msg.header.stamp, frame_id=self._target_frame)
        self._cloud_pub.publish(self._organized_cloud(header, xyz_output))
        self._colored_cloud_pub.publish(
            self._organized_rgb_cloud(header, xyz_output, rgb))
        valid_msg = self._bridge.cv2_to_imgmsg(
            (valid.astype(np.uint8) * 255), encoding="mono8"
        )
        # This is still an image-indexed mask in the optical frame even though
        # the XYZ samples themselves were transformed into target_frame.
        valid_msg.header = depth_msg.header
        self._valid_pub.publish(valid_msg)
        with self._lock:
            self._frames_ok += 1
            frames_ok = self._frames_ok
        self._publish_diagnostic(
            DiagnosticStatus.OK,
            "ok",
            "valid_pixels={}".format(int(valid.sum())),
        )
        rospy.loginfo_throttle(
            5.0,
            "RGB-D projection active: frames_ok=%d valid_depth_pixels=%d",
            frames_ok,
            int(valid.sum()),
        )

    @staticmethod
    def _organized_cloud(header, xyz):
        contiguous = np.ascontiguousarray(xyz, dtype=np.float32)
        height, width, _ = contiguous.shape
        message = PointCloud2()
        message.header = header
        message.height = height
        message.width = width
        message.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
        ]
        message.is_bigendian = False
        message.point_step = 12
        message.row_step = width * message.point_step
        message.is_dense = bool(np.all(np.isfinite(contiguous)))
        message.data = contiguous.tobytes()
        return message

    @staticmethod
    def _organized_rgb_cloud(header, xyz, bgr):
        """Make an organized PCL-compatible XYZRGB cloud for visualization.

        ``cv_bridge(..., bgr8)`` provides BGR pixels. RViz's RGB transformer
        expects the conventional packed 0xRRGGBB value stored in a FLOAT32
        field named ``rgb``.
        """
        xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        bgr = np.asarray(bgr, dtype=np.uint8)
        if xyz.ndim != 3 or xyz.shape[2] != 3 or bgr.shape != xyz.shape:
            raise ValueError("XYZ and BGR images must have matching HxWx3 shapes")
        height, width, _ = xyz.shape
        record = np.empty((height, width), dtype=np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4"),
        ]))
        record["x"], record["y"], record["z"] = (
            xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2])
        packed = ((bgr[:, :, 2].astype(np.uint32) << 16)
                  | (bgr[:, :, 1].astype(np.uint32) << 8)
                  | bgr[:, :, 0].astype(np.uint32))
        record["rgb"] = packed.view(np.float32)
        message = PointCloud2()
        message.header = header
        message.height = height
        message.width = width
        message.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]
        message.is_bigendian = False
        message.point_step = 16
        message.row_step = width * message.point_step
        message.is_dense = bool(np.all(np.isfinite(xyz)))
        message.data = record.tobytes()
        return message


if __name__ == "__main__":
    rospy.init_node("rgbd_projection")
    try:
        RGBDProjectionNode()
        rospy.spin()
    except rospy.ROSInitException as exc:
        rospy.logfatal(str(exc))
        raise
