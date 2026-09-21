"""Lightweight RGB-D scene-change and stability monitoring."""

import cv2
import numpy as np


class SceneChangeMonitor:
    """Compare downsampled RGB-D frames without running perception models."""

    def __init__(
        self,
        size=(160, 120),
        depth_change_m=0.04,
        depth_change_ratio=0.05,
        rgb_change_level=35.0,
        rgb_change_ratio=0.15,
        change_frames=3,
        stable_depth_change_m=0.02,
        stable_depth_change_ratio=0.01,
        stable_rgb_mean_difference=8.0,
        stable_frames=5,
    ):
        self.size = tuple(int(value) for value in size)
        self.depth_change_m = float(depth_change_m)
        self.depth_change_ratio = float(depth_change_ratio)
        self.rgb_change_level = float(rgb_change_level)
        self.rgb_change_ratio = float(rgb_change_ratio)
        self.change_frames = int(change_frames)
        self.stable_depth_change_m = float(stable_depth_change_m)
        self.stable_depth_change_ratio = float(stable_depth_change_ratio)
        self.stable_rgb_mean_difference = float(stable_rgb_mean_difference)
        self.stable_frames = int(stable_frames)
        if min(self.size) <= 0 or self.change_frames <= 0 or self.stable_frames <= 0:
            raise ValueError("scene monitor sizes and frame counts must be positive")
        ratios = (self.depth_change_ratio, self.rgb_change_ratio,
                  self.stable_depth_change_ratio)
        if any(value < 0.0 or value > 1.0 for value in ratios):
            raise ValueError("scene monitor ratios must be in [0, 1]")
        self._baseline = None
        self._previous = None
        self._change_count = 0
        self._stable_count = 0

    def _snapshot(self, bgr, xyz):
        image = np.asarray(bgr, dtype=np.uint8)
        points = np.asarray(xyz, dtype=np.float32)
        if image.ndim != 3 or image.shape[2] != 3 or points.shape != image.shape:
            raise ValueError("RGB and organized XYZ must be matching HxWx3 arrays")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA).astype(np.float32)
        depth = cv2.resize(points[..., 2], self.size, interpolation=cv2.INTER_NEAREST)
        depth = np.asarray(depth, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0.0)
        return gray, depth, valid

    @staticmethod
    def _depth_ratio(first, second, threshold):
        _, depth_a, valid_a = first
        _, depth_b, valid_b = second
        valid = valid_a & valid_b
        count = int(valid.sum())
        if count == 0:
            return 0.0
        return float(np.count_nonzero(np.abs(depth_a[valid] - depth_b[valid]) > threshold)) / count

    @staticmethod
    def _rgb_changed_ratio(first, second, threshold):
        return float(np.mean(np.abs(first[0] - second[0]) > threshold))

    @staticmethod
    def _rgb_mean_difference(first, second):
        return float(np.mean(np.abs(first[0] - second[0])))

    def lock_baseline(self, bgr, xyz):
        self._baseline = self._snapshot(bgr, xyz)
        self._previous = None
        self._change_count = 0
        self._stable_count = 0

    def update_locked(self, bgr, xyz):
        if self._baseline is None:
            raise RuntimeError("scene baseline has not been locked")
        current = self._snapshot(bgr, xyz)
        depth_changed = self._depth_ratio(
            self._baseline, current, self.depth_change_m
        ) > self.depth_change_ratio
        rgb_changed = self._rgb_changed_ratio(
            self._baseline, current, self.rgb_change_level
        ) > self.rgb_change_ratio
        self._change_count = self._change_count + 1 if (depth_changed or rgb_changed) else 0
        return self._change_count >= self.change_frames

    def begin_waiting_for_stability(self, bgr, xyz):
        self._previous = self._snapshot(bgr, xyz)
        self._stable_count = 0
        self._change_count = 0

    def update_waiting(self, bgr, xyz):
        current = self._snapshot(bgr, xyz)
        if self._previous is None:
            self._previous = current
            return False
        depth_stable = self._depth_ratio(
            self._previous, current, self.stable_depth_change_m
        ) < self.stable_depth_change_ratio
        rgb_stable = self._rgb_mean_difference(
            self._previous, current
        ) < self.stable_rgb_mean_difference
        self._stable_count = self._stable_count + 1 if (depth_stable and rgb_stable) else 0
        self._previous = current
        return self._stable_count >= self.stable_frames
