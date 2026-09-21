#!/usr/bin/env python3
"""Exercise timestamp, unit and pixel-layout contracts without hardware."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
import tf2_ros


def load_node(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROJECTION = load_node("rgbd_projection_node")
FUSION = load_node("dual_workspace_cloud_fusion_node")


class ObservationNodesTest(unittest.TestCase):
    def setUp(self):
        for name, replacement in (
                ("get_param", lambda name, default=None: default),
                ("Publisher", lambda *a, **k: Mock(resolved_name=a[0])),
                ("Subscriber", Mock()), ("loginfo", Mock()),
                ("loginfo_throttle", Mock()), ("logwarn_throttle", Mock())):
            patcher = patch.object(rospy, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(rospy.Time, "now", return_value=rospy.Time.from_sec(100.0))
        patcher.start()
        self.addCleanup(patcher.stop)

    def projection(self):
        with patch.object(PROJECTION.tf2_ros, "TransformListener"), \
                patch.object(PROJECTION.tf2_ros, "Buffer"):
            node = PROJECTION.RGBDProjectionNode()
        node._use_tf = False
        node._target_frame = node._optical_frame
        return node

    def images(self, node, encoding="16UC1"):
        dtype, value = (np.uint16, 1000) if encoding == "16UC1" else (np.float32, 1.0)
        depth = np.full((3, 3), value, dtype=dtype)
        depth[0, 0] = 0
        rgb = node._bridge.cv2_to_imgmsg(np.zeros((3, 3, 3), dtype=np.uint8), "bgr8")
        depth = node._bridge.cv2_to_imgmsg(depth, encoding)
        info = CameraInfo(width=3, height=3, K=[2., 0., 1., 0., 2., 1., 0., 0., 1.])
        for message in (rgb, depth, info):
            message.header = Header(stamp=rospy.Time.from_sec(100), frame_id=node._optical_frame)
        return rgb, depth, info

    def test_float_and_integer_depth_keep_metric_geometry_and_nan_layout(self):
        for encoding in ("16UC1", "32FC1"):
            node = self.projection()
            node._callback(*self.images(node, encoding))
            cloud = node._cloud_pub.publish.call_args[0][0]
            self.assertEqual((cloud.height, cloud.width, cloud.point_step, cloud.row_step),
                             (3, 3, 12, 36))
            self.assertFalse(cloud.is_dense)
            xyz = np.frombuffer(cloud.data, dtype=np.float32).reshape(3, 3, 3)
            np.testing.assert_allclose(xyz[1, 1], [0, 0, 1])
            self.assertTrue(np.isnan(xyz[0, 0]).all())

    def test_skew_frames_distortion_and_encoding_reject_without_output(self):
        for fault in ("skew", "rgb", "depth", "info", "distortion", "encoding"):
            node = self.projection()
            rgb, depth, info = self.images(node)
            if fault == "skew":
                depth.header.stamp = rospy.Time.from_sec(100.0 + 1.0 / 30.0)
            elif fault in ("rgb", "depth", "info"):
                {"rgb": rgb, "depth": depth, "info": info}[fault].header.frame_id = "wrong"
            elif fault == "distortion":
                info.D = [0.1, 0, 0, 0, 0]
            else:
                depth.encoding = "mono16"
            node._callback(rgb, depth, info)
            node._cloud_pub.publish.assert_not_called()
            self.assertEqual(node._frames_dropped, 1)

    def test_missing_capture_time_tf_rejects_without_latest_fallback(self):
        node = self.projection()
        node._use_tf = True
        node._tf_buffer.lookup_transform.side_effect = tf2_ros.LookupException("missing")
        rgb, depth, info = self.images(node)
        node._callback(rgb, depth, info)
        self.assertEqual(node._tf_buffer.lookup_transform.call_count, 1)
        self.assertEqual(node._tf_buffer.lookup_transform.call_args[0][2], depth.header.stamp)
        node._cloud_pub.publish.assert_not_called()

    @staticmethod
    def cloud(stamp):
        return PROJECTION.RGBDProjectionNode._organized_cloud(
            Header(stamp=rospy.Time.from_sec(stamp), frame_id="base_link"),
            np.array([[[0.1, 0.1, 0.1], [np.nan, np.nan, np.nan]]], dtype=np.float32))

    def fusion(self):
        node = FUSION.DualWorkspaceCloudFusionNode()
        node._period_s = 0
        return node

    def test_used_pairs_cannot_return_after_new_pair(self):
        node = self.fusion()
        for stamp in (99.7, 99.8, 99.7):
            for camera in ("rs1", "rs3"):
                node._callback(self.cloud(stamp), camera)
        self.assertEqual(node._publisher.publish.call_count, 2)

    def test_stale_zero_future_and_unsynchronized_pairs_never_publish(self):
        for first, second in ((98., 98.), (0., 0.), (101., 101.), (99.7, 99.8)):
            node = self.fusion()
            node._callback(self.cloud(first), "rs1")
            node._callback(self.cloud(second), "rs3")
            node._publisher.publish.assert_not_called()

    def test_throttled_pair_is_consumed_and_late_arrival_can_pair(self):
        node = self.fusion()
        node._callback(self.cloud(99.8), "rs1")
        node._callback(self.cloud(99.75), "rs1")
        node._callback(self.cloud(99.751), "rs3")
        self.assertEqual(node._publisher.publish.call_count, 1)
        node._period_s = 1000
        node._callback(self.cloud(99.801), "rs3")
        node._period_s = 0
        for camera in ("rs1", "rs3"):
            node._callback(self.cloud(99.8 if camera == "rs1" else 99.801), camera)
        self.assertEqual(node._publisher.publish.call_count, 1)

    def test_pair_that_expires_during_fusion_is_not_published(self):
        node = self.fusion()
        fuse = FUSION.voxel_fuse_base_clouds

        def delayed_fuse(*args):
            rospy.Time.now.return_value = rospy.Time.from_sec(102.)
            return fuse(*args)

        node._callback(self.cloud(99.9), "rs1")
        with patch.object(FUSION, "voxel_fuse_base_clouds", side_effect=delayed_fuse):
            node._callback(self.cloud(99.9), "rs3")
        node._publisher.publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
