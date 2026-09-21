"""Camera-only D435 validation metrics; no robot TF is required."""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class CameraValidationMetrics:
    depth_scale: float = 0.001
    min_depth_m: float = 0.10
    max_depth_m: float = 2.00
    max_sync_skew_s: float = 0.05
    min_valid_depth_fraction: float = 0.20
    rgb_stamps: List[float] = field(default_factory=list)
    depth_stamps: List[float] = field(default_factory=list)
    sync_skews: List[float] = field(default_factory=list)
    valid_fractions: List[float] = field(default_factory=list)
    intrinsics: List[np.ndarray] = field(default_factory=list)
    depth_samples_m: List[np.ndarray] = field(default_factory=list)
    frame_ids: List[str] = field(default_factory=list)
    shape: List[int] = field(default_factory=list)

    def add_frame(self, rgb_stamp, depth_stamp, rgb_shape, depth_raw, k, frame_id):
        depth = np.asarray(depth_raw)
        if depth.dtype != np.uint16 or depth.ndim != 2:
            raise ValueError("aligned depth must be a uint16 HxW image")
        if tuple(rgb_shape) != depth.shape:
            raise ValueError("RGB and aligned depth dimensions differ")
        matrix = np.asarray(k, dtype=float).reshape(3, 3)
        if not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError("CameraInfo.K is invalid")
        depth_m = depth.astype(np.float32) * self.depth_scale
        valid = (
            np.isfinite(depth_m)
            & (depth_m >= self.min_depth_m)
            & (depth_m <= self.max_depth_m)
        )
        self.rgb_stamps.append(float(rgb_stamp))
        self.depth_stamps.append(float(depth_stamp))
        self.sync_skews.append(abs(float(rgb_stamp) - float(depth_stamp)))
        self.valid_fractions.append(float(valid.mean()))
        self.intrinsics.append(matrix.copy())
        self.depth_samples_m.append(depth_m[valid][:: max(1, int(valid.sum()) // 2000)])
        self.frame_ids.append(str(frame_id))
        self.shape = [int(depth.shape[0]), int(depth.shape[1])]

    @staticmethod
    def _rate(stamps):
        if len(stamps) < 2:
            return 0.0
        intervals = np.diff(np.asarray(stamps, dtype=float))
        intervals = intervals[intervals > 0]
        return float(1.0 / np.mean(intervals)) if len(intervals) else 0.0

    def summary(self) -> Dict:
        if not self.rgb_stamps:
            return {"frames": 0, "passed": False, "failure_reasons": ["no_frames"]}
        intrinsics = np.asarray(self.intrinsics)
        depth_values = np.concatenate(self.depth_samples_m) if self.depth_samples_m else np.array([])
        reasons = []
        max_skew = float(np.max(self.sync_skews))
        mean_valid = float(np.mean(self.valid_fractions))
        if max_skew > self.max_sync_skew_s:
            reasons.append("timestamp_skew")
        if mean_valid < self.min_valid_depth_fraction:
            reasons.append("insufficient_valid_depth")
        if float(np.max(np.std(intrinsics, axis=0))) > 1e-6:
            reasons.append("intrinsics_changed")
        if not self.frame_ids[0] or len(set(self.frame_ids)) != 1:
            reasons.append("frame_id_invalid_or_changed")
        return {
            "frames": len(self.rgb_stamps),
            "passed": not reasons,
            "failure_reasons": reasons,
            "shape_hw": self.shape,
            "rgb_rate_hz": self._rate(self.rgb_stamps),
            "depth_rate_hz": self._rate(self.depth_stamps),
            "sync_skew_s": {
                "mean": float(np.mean(self.sync_skews)),
                "max": max_skew,
            },
            "valid_depth_fraction": {
                "mean": mean_valid,
                "min": float(np.min(self.valid_fractions)),
            },
            "depth_m_quantiles": (
                np.quantile(depth_values, [0.05, 0.50, 0.95]).tolist()
                if len(depth_values)
                else []
            ),
            "camera_intrinsics": {
                "fx": float(intrinsics[0, 0, 0]),
                "fy": float(intrinsics[0, 1, 1]),
                "cx": float(intrinsics[0, 0, 2]),
                "cy": float(intrinsics[0, 1, 2]),
                "max_std": float(np.max(np.std(intrinsics, axis=0))),
            },
            "frame_id": self.frame_ids[0],
            "robot_tf_used": False,
        }
