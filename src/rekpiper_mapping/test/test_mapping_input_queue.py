#!/usr/bin/env python3
"""Real mapper callback tests without ROS nodes, cameras or GPU inference."""
import importlib.util
from pathlib import Path
import threading
import unittest

SPEC = importlib.util.spec_from_file_location('mapping_node_under_test',
    Path(__file__).resolve().parents[1] / 'scripts/nvblox_mapping_node.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MappingInputQueueTest(unittest.TestCase):
    def setUp(self):
        self.node = object.__new__(MODULE.NvbloxMappingNode)
        self.node._input_lock = threading.Lock()
        self.node._worker_lock = threading.Lock()
        self.node._pending_frames = {}
        self.node._stepwise_capture = False
        self.node._last_depth_monotonic = {'rs1': 10., 'rs3': 5.}

    def test_latest_complete_tuple_is_bounded_and_older_camera_goes_first(self):
        calls = []
        self.node._depth_callback = lambda *args: calls.append(args)
        for frame in range(20):
            for camera in ('rs1', 'rs3'):
                self.node._queue_depth_frame({'name': camera}, ('robot',),
                                             frame, 'info', 'mask')
        self.assertEqual(len(self.node._pending_frames), 2)
        self.node._process_pending_frame(None)
        self.assertEqual(calls[0], ({'name': 'rs3'}, ('robot',), 19, 'info', 'mask'))
        self.node._process_pending_frame(None)
        self.assertEqual(calls[1][0]['name'], 'rs1')

    def test_receiving_latest_frame_does_not_wait_for_gpu_work(self):
        entered, release = threading.Event(), threading.Event()
        def busy(*_):
            entered.set()
            release.wait(2)
        self.node._depth_callback = busy
        self.node._queue_depth_frame({'name': 'rs1'}, (), 'old', 'info')
        worker = threading.Thread(target=self.node._process_pending_frame, args=(None,))
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            self.node._queue_depth_frame({'name': 'rs1'}, (), 'new', 'info')
            self.assertEqual(self.node._pending_frames['rs1'][2], 'new')
        finally:
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())


if __name__ == '__main__':
    unittest.main()
