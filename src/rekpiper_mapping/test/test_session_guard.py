#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest

from rekpiper_mapping.session_guard import (
    CANInterfaceNotReady, ExclusiveSessionLock, SessionAlreadyRunning,
    parse_can_link_details, require_can_interface_ready,
    process_tree_contains_rosmaster)


class SessionGuardTest(unittest.TestCase):
    def test_lock_rejects_second_owner_and_is_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "domain.lock"
            first = ExclusiveSessionLock(path)
            second = ExclusiveSessionLock(path)
            metadata = first.acquire({"domain": "camera"})
            self.assertEqual(metadata["domain"], "camera")
            self.assertEqual(json.loads(path.read_text())["pid"], metadata["pid"])
            with self.assertRaises(SessionAlreadyRunning):
                second.acquire()
            first.release()
            reacquired = second.acquire({"domain": "camera_restarted"})
            self.assertEqual(reacquired["domain"], "camera_restarted")
            second.release()

    def test_release_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = ExclusiveSessionLock(Path(directory) / "domain.lock")
            lock.acquire()
            lock.release()
            lock.release()

    def test_detects_master_owned_by_launch_process_tree(self):
        processes = {
            10: {"ppid": 1, "command": "python3 /opt/ros/noetic/bin/roslaunch pkg x.launch"},
            11: {"ppid": 10, "command": "python3 /opt/ros/noetic/bin/rosmaster --core"},
            12: {"ppid": 10, "command": "python3 guard_node.py"},
        }
        self.assertTrue(process_tree_contains_rosmaster(10, processes))
        self.assertFalse(process_tree_contains_rosmaster(12, processes))

    def test_can_gate_requires_up_active_and_expected_bitrate(self):
        text = """4: can0: <NOARP,UP,LOWER_UP,ECHO> mtu 16 state UNKNOWN\n
    link/can  promiscuity 0\n
    can state ERROR-ACTIVE (berr-counter tx 0 rx 0) restart-ms 0\n
          bitrate 1000000 sample-point 0.875\n"""

        class Result:
            returncode = 0
            stdout = text
            stderr = ""

        details = require_can_interface_ready(
            "can0", 1000000, run=lambda *_args, **_kwargs: Result())
        self.assertIn("UP", details["flags"])
        self.assertEqual(details["can_state"], "ERROR-ACTIVE")
        self.assertEqual(details["bitrate"], 1000000)

    def test_can_gate_rejects_down_or_wrong_state(self):
        down = """4: can0: <NOARP> mtu 16 state DOWN\n
    can state STOPPED (berr-counter tx 0 rx 0) restart-ms 0\n
          bitrate 1000000 sample-point 0.875\n"""
        self.assertEqual(parse_can_link_details(down)["can_state"], "STOPPED")

        class Result:
            returncode = 0
            stdout = down
            stderr = ""

        with self.assertRaises(CANInterfaceNotReady):
            require_can_interface_ready(
                "can0", 1000000, run=lambda *_args, **_kwargs: Result())


if __name__ == "__main__":
    unittest.main()
