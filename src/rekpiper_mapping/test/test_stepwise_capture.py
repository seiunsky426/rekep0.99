#!/usr/bin/env python3
"""Exercise the real capture handler with a CPU mapper and controlled frames."""
import importlib.util
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import rospy
import torch
from sensor_msgs.msg import Image

SPEC = importlib.util.spec_from_file_location('stepwise_mapper_under_test',
    Path(__file__).resolve().parents[1] / 'scripts/nvblox_mapping_node.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StepwiseCaptureTest(unittest.TestCase):
    def setUp(self):
        rospy.rostime.set_rostime_initialized(True)
        n = self.node = object.__new__(MODULE.NvbloxMappingNode)
        n._stepwise_capture = n._observation_only = True
        n._mode = 'shadow'
        n._safe_supervision_enabled = n._capture_active = False
        n._lock = threading.RLock()
        n._capture_lock = threading.Lock()
        n._input_lock = threading.Lock()
        n._visualization_state_lock = threading.Lock()
        n._pending_frames = {'rs1': 'old input'}
        n._camera_source_names = ['rs1', 'rs3']
        n._map_generation = 0
        n._instance_uuid = 'test-instance'
        n._target_frame = 'base_link'
        n._bounds_min = np.zeros(3)
        n._bounds_max = np.ones(3) * .015
        n._voxel_size_m = .015
        n._max_map_age_s = 1.
        n._unknown_occupied_distance_m = 1.
        n._snapshot_shape = (2, 2, 2)
        n._snapshot_queries = n._snapshot_closest = torch.zeros(8, 4)
        n._raw_diagnostic_mapper = None
        self.clears = []
        n._mapper = SimpleNamespace(clear=lambda _: self.clears.append(True),
            update_esdf=lambda _: None, update_hashmaps=lambda: None,
            query_sdf=lambda *_a, **_k: torch.tensor([-.03] * 7 + [-100.]))
        self.published = []
        n._snapshot_pub = SimpleNamespace(publish=self.published.append)
        n._publish_cleared_outputs = lambda: None
        n._publish_status = lambda *_a: None
        n._capture_updated = SimpleNamespace(wait=self.frames, clear=lambda: None)

    def frames(self, _timeout):
        self.assertTrue(self.clears)
        self.assertFalse(self.node._pending_frames)
        now = rospy.Time.now().to_sec()
        self.assertGreater(now, self.node._capture_after_stamp.to_sec())
        self.node._integrated_frames_by_camera = {'rs1': 5, 'rs3': 5}
        self.node._last_integrated_stamp_by_camera = {'rs1': now, 'rs3': now}

    def test_capture_rebuilds_unique_generations_and_preserves_unknown(self):
        first = self.node._capture_sdf_snapshot(None)
        second = self.node._capture_sdf_snapshot(None)
        self.assertTrue(first.success, first.message)
        self.assertTrue(second.success, second.message)
        self.assertNotEqual(first.grid.map_generation_uuid, second.grid.map_generation_uuid)
        self.assertEqual(len(self.clears), 2)
        self.assertEqual(first.grid.observed[-1], 0)
        self.assertEqual(first.grid.distances_m[-1], 1.)
        self.assertFalse(first.grid.valid)
        self.assertFalse(self.node._capture_active)
        self.assertTrue(self.node._snapshot_ready)
        self.assertEqual(self.published[0].status, 'capture_started')

    def test_missing_fresh_camera_times_out_and_invalidates_old_grid(self):
        self.assertTrue(self.node._capture_sdf_snapshot(None).success)
        self.node._capture_updated = threading.Event()
        with patch.object(MODULE, 'CAPTURE_TIMEOUT_S', .03):
            start = time.monotonic()
            result = self.node._capture_sdf_snapshot(None)
        self.assertFalse(result.success)
        self.assertIn('capture_timeout', result.message)
        self.assertLess(time.monotonic() - start, .3)
        self.assertEqual(self.published[-1].map_generation_uuid, '')
        self.assertFalse(self.node._capture_active)
        self.assertFalse(self.node._capture_lock.locked())
        self.assertFalse(self.node._snapshot_ready)

    def test_concurrent_capture_is_rejected_without_clearing(self):
        self.node._capture_lock.acquire()
        try:
            result = self.node._capture_sdf_snapshot(None)
        finally:
            self.node._capture_lock.release()
        self.assertFalse(result.success)
        self.assertEqual(result.message, 'capture_in_progress')
        self.assertFalse(self.clears)

    def test_depth_captured_before_request_cannot_enter_new_generation(self):
        n = self.node
        n._depth_period_s = 0.
        n._last_depth_monotonic = {'rs1': -np.inf}
        n._capture_active = True
        n._capture_after_stamp = rospy.Time(100)
        depth = Image()
        depth.header.stamp = rospy.Time(99)
        n._depth_callback({'name': 'rs1', 'camera_frame': 'rs1_color_optical_frame'},
                          (), depth, None)
        self.assertEqual(n._last_depth_monotonic['rs1'], -np.inf)

    def test_capture_is_disabled_in_normal_mapping(self):
        self.node._stepwise_capture = False
        result = self.node._capture_sdf_snapshot(None)
        self.assertFalse(result.success)
        self.assertFalse(self.clears)


if __name__ == '__main__':
    unittest.main()
