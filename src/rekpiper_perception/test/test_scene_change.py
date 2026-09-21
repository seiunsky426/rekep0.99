#!/usr/bin/env python3

import unittest

import numpy as np

from rekpiper_perception.scene_change import SceneChangeMonitor


def frame(gray=80, depth=0.75, height=120, width=160):
    bgr = np.full((height, width, 3), gray, dtype=np.uint8)
    xyz = np.zeros((height, width, 3), dtype=np.float32)
    xyz[..., 2] = depth
    return bgr, xyz


class SceneChangeMonitorTest(unittest.TestCase):
    def setUp(self):
        self.monitor = SceneChangeMonitor(size=(160, 120))
        self.baseline = frame()
        self.monitor.lock_baseline(*self.baseline)

    def test_static_rgbd_does_not_trigger_refresh(self):
        for index in range(120):
            bgr, xyz = frame(gray=80 + index % 2, depth=0.75 + 0.001 * (index % 2))
            self.assertFalse(self.monitor.update_locked(bgr, xyz))

    def test_large_depth_change_requires_three_consecutive_checks(self):
        bgr, xyz = frame()
        xyz[:, :16, 2] += 0.06  # 10% changes by more than 4 cm.
        self.assertFalse(self.monitor.update_locked(bgr, xyz))
        self.assertFalse(self.monitor.update_locked(bgr, xyz))
        self.assertTrue(self.monitor.update_locked(bgr, xyz))

    def test_single_frame_obstruction_resets_change_counter(self):
        changed, changed_xyz = frame(gray=180)
        self.assertFalse(self.monitor.update_locked(changed, changed_xyz))
        self.assertFalse(self.monitor.update_locked(*self.baseline))
        self.assertFalse(self.monitor.update_locked(changed, changed_xyz))
        self.assertFalse(self.monitor.update_locked(changed, changed_xyz))
        self.assertTrue(self.monitor.update_locked(changed, changed_xyz))

    def test_large_rgb_region_can_confirm_scene_change(self):
        changed, xyz = frame()
        changed[:, :32] = 180  # 20% differs by more than 35 gray levels.
        self.assertFalse(self.monitor.update_locked(changed, xyz))
        self.assertFalse(self.monitor.update_locked(changed, xyz))
        self.assertTrue(self.monitor.update_locked(changed, xyz))

    def test_waiting_requires_five_stable_adjacent_comparisons(self):
        start = frame(gray=120, depth=0.90)
        self.monitor.begin_waiting_for_stability(*start)
        for _ in range(4):
            self.assertFalse(self.monitor.update_waiting(*start))
        self.assertTrue(self.monitor.update_waiting(*start))

    def test_motion_resets_stability_counter(self):
        still = frame(gray=120, depth=0.90)
        moving = frame(gray=150, depth=0.95)
        self.monitor.begin_waiting_for_stability(*still)
        for _ in range(3):
            self.assertFalse(self.monitor.update_waiting(*still))
        self.assertFalse(self.monitor.update_waiting(*moving))
        for _ in range(4):
            self.assertFalse(self.monitor.update_waiting(*moving))
        self.assertTrue(self.monitor.update_waiting(*moving))


if __name__ == "__main__":
    unittest.main()
