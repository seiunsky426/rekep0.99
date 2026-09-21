#!/usr/bin/env python3
"""Collect synchronized D435 frames and report camera-only health metrics."""

import json
from pathlib import Path

from cv_bridge import CvBridge, CvBridgeError
import message_filters
import rospy
from sensor_msgs.msg import CameraInfo, Image

from rekpiper_calibration.camera_validation import CameraValidationMetrics


class CameraValidationNode:
    def __init__(self):
        self._bridge = CvBridge()
        self._target_frames = int(rospy.get_param("~target_frames", 30))
        self._output_path = Path(
            rospy.get_param("~output_path", "/tmp/rekep_d435_camera_validation.json")
        ).expanduser().resolve()
        self._expected_serial = str(rospy.get_param("~expected_serial", "")).lstrip("_")
        self._serial_param = str(rospy.get_param("~serial_param", ""))
        self._expected_frame_id = str(rospy.get_param("~expected_frame_id", ""))
        if self._expected_serial:
            if not self._serial_param or not rospy.has_param(self._serial_param):
                raise rospy.ROSInitException("configured RealSense serial parameter is missing")
            actual_serial = str(rospy.get_param(self._serial_param)).lstrip("_")
            if actual_serial != self._expected_serial:
                raise rospy.ROSInitException("live RealSense serial does not match expected serial")
        self._metrics = CameraValidationMetrics(
            depth_scale=float(rospy.get_param("~depth_scale", 0.001)),
            min_depth_m=float(rospy.get_param("~min_depth_m", 0.10)),
            max_depth_m=float(rospy.get_param("~max_depth_m", 2.00)),
            max_sync_skew_s=float(rospy.get_param("~max_sync_skew_s", 0.05)),
            min_valid_depth_fraction=float(
                rospy.get_param("~min_valid_depth_fraction", 0.20)
            ),
        )
        rgb = message_filters.Subscriber(
            rospy.get_param("~rgb_topic", "/camera/color/image_raw"), Image, queue_size=2
        )
        depth = message_filters.Subscriber(
            rospy.get_param(
                "~depth_topic", "/camera/aligned_depth_to_color/image_raw"
            ),
            Image,
            queue_size=2,
        )
        info = message_filters.Subscriber(
            rospy.get_param("~camera_info_topic", "/camera/color/camera_info"),
            CameraInfo,
            queue_size=2,
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb, depth, info], queue_size=10, slop=0.05, allow_headerless=False
        )
        self._sync.registerCallback(self._callback)

    def _callback(self, rgb_msg, depth_msg, info_msg):
        if len(self._metrics.rgb_stamps) >= self._target_frames:
            return
        try:
            rgb = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            self._metrics.add_frame(
                rgb_msg.header.stamp.to_sec(),
                depth_msg.header.stamp.to_sec(),
                rgb.shape[:2],
                depth,
                info_msg.K,
                info_msg.header.frame_id,
            )
        except (CvBridgeError, ValueError) as exc:
            rospy.logerr("Camera validation rejected frame: %s", exc)
            return
        if len(self._metrics.rgb_stamps) == self._target_frames:
            report = self._metrics.summary()
            report["serial"] = self._expected_serial
            if self._expected_frame_id and report["frame_id"] != self._expected_frame_id:
                report["passed"] = False
                report["failure_reasons"].append("frame_id_mismatch")
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            self._output_path.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            rospy.loginfo("D435 camera-only validation: %s", json.dumps(report))
            rospy.signal_shutdown("camera validation complete")


if __name__ == "__main__":
    rospy.init_node("camera_validation")
    CameraValidationNode()
    rospy.spin()
