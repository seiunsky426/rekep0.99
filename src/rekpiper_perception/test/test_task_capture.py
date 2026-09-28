#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
import rospy
from std_msgs.msg import String


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "task_perception_node.py"
SPEC = importlib.util.spec_from_file_location("task_perception_node", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def stamp(ns):
    return rospy.Time(secs=ns // 1000000000, nsecs=ns % 1000000000)


class TaskCaptureTest(unittest.TestCase):
    def test_saved_segmentation_has_matching_stamp_mask_and_points(self):
        node = MODULE.TaskPerceptionNode.__new__(MODULE.TaskPerceptionNode)
        with tempfile.TemporaryDirectory() as folder:
            node._snapshot_output = str(Path(folder) / "segmentation_snapshot.npz")
            mask = np.array([[1, 0], [3, 3]], dtype=np.uint16)
            xyz = np.ones((2, 2, 3), dtype=np.float32)
            node._save_segmentation_snapshot(123456789, mask, xyz)
            with np.load(node._snapshot_output, allow_pickle=False) as data:
                self.assertEqual(int(data["stamp_ns"]), 123456789)
                np.testing.assert_array_equal(data["mask"], mask)
                np.testing.assert_array_equal(data["xyz"], xyz)

    def test_supervised_lock_does_not_clear_snapshot_on_new_frames(self):
        node = MODULE.TaskPerceptionNode.__new__(MODULE.TaskPerceptionNode)
        node._state = node.LOCKED
        node._hold_snapshot = True
        node._task_trigger_stamp_ns = 0
        node._base_frame = "base_link"
        node._sync_slop_s = .05
        node._bridge = Mock()
        node._bridge.imgmsg_to_cv2.return_value = np.zeros((2, 2, 3), np.uint8)
        node._organized_xyz = Mock(return_value=np.ones((2, 2, 3), np.float32))
        node._wait_for_task_trigger = True
        node._task_request_serial = 1
        node._scene_monitor = Mock()
        node._sam = Mock()
        rgb = Mock(); points = Mock()
        rgb.header.stamp = stamp(1234567890000000000)
        points.header.stamp = rgb.header.stamp
        points.header.frame_id = "base_link"
        with patch.object(rospy, "loginfo_throttle"):
            node._process_callback(rgb, points)
        node._scene_monitor.update_locked.assert_not_called()
        node._sam.segment.assert_not_called()
        self.assertEqual(node._state, node.LOCKED)

    def test_saved_lock_releases_models_and_next_request_can_reload(self):
        node = MODULE.TaskPerceptionNode.__new__(MODULE.TaskPerceptionNode)
        node._sam = Mock()
        node._proposer = Mock()
        node._sam_checkpoint = "sam.pth"
        node._inference_device = "cuda"
        node._sam_options = {}
        node._proposer_options = ("official", "dinov2", "weights", {})
        with patch.object(MODULE.torch.cuda, "is_available", return_value=True), \
             patch.object(MODULE.torch.cuda, "empty_cache") as empty_cache, \
             patch.object(MODULE, "SAMAutomaticSegmenter", return_value=Mock()) as sam, \
             patch.object(MODULE, "OfficialKeypointProposerAdapter", return_value=Mock()) as proposer:
            node._release_inference_models()
            self.assertIsNone(node._sam)
            self.assertIsNone(node._proposer)
            empty_cache.assert_called_once()
            node._ensure_inference_models()
            sam.assert_called_once_with("sam.pth", "cuda")
            proposer.assert_called_once_with("official", "dinov2", "weights", {})

    def test_task_uses_first_fresh_frame_without_waiting_for_scene_stability(self):
        node = MODULE.TaskPerceptionNode.__new__(MODULE.TaskPerceptionNode)
        node._callback_lock = threading.Lock()
        node._state = node.WAITING_FOR_STABLE_SCENE
        node._task_request_serial = 0
        node._task_instruction = ""
        node._locked_snapshot = None
        node._capture_stamp_ns = 0
        node._last_inference_monotonic = -np.inf
        node._min_period_s = 0.5
        node._wait_for_task_trigger = True
        node._base_frame = "base_link"
        node._sync_slop_s = 0.05
        node._task_status_pub = Mock()
        node._bridge = Mock()
        node._bridge.imgmsg_to_cv2.return_value = np.zeros((4, 5, 3), np.uint8)
        node._organized_xyz = Mock(return_value=np.ones((4, 5, 3), np.float32))
        node._sam = Mock()
        node._proposer = Mock()
        node._sam.segment.side_effect = RuntimeError("stop after capture")
        node._failure = Mock()
        node._scene_monitor = Mock()
        node._scene_monitor.update_waiting.return_value = False

        with patch.object(rospy.Time, "now", return_value=stamp(1234567890000000000)):
            node._task_request_callback(String(data="move block"))
        self.assertEqual(node._state, node.DETECTING)
        trigger_ns = node._task_trigger_stamp_ns

        rgb = Mock()
        points = Mock()
        rgb.header.stamp = stamp(trigger_ns - 1000000)
        points.header.stamp = rgb.header.stamp
        points.header.frame_id = "base_link"
        with patch.object(rospy, "loginfo_throttle"):
            node._process_callback(rgb, points)
            node._sam.segment.assert_not_called()

            rgb.header.stamp = stamp(trigger_ns + 1000000)
            points.header.stamp = rgb.header.stamp
            node._process_callback(rgb, points)
        node._sam.segment.assert_called_once()
        node._scene_monitor.update_waiting.assert_not_called()
        self.assertEqual(node._capture_stamp_ns, points.header.stamp.to_nsec())
        status = json.loads(node._task_status_pub.publish.call_args[0][0].data)
        self.assertEqual(status["detail"], "snapshot_locked_running_sam_dinov2")
        self.assertEqual(status["snapshot_stamp_ns"], node._capture_stamp_ns)


if __name__ == "__main__":
    unittest.main()
