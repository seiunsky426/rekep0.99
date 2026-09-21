"""Pure ReKep stage coordination, independent from ROS and robot hardware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class ReKepCoordinatorCore:
    """Track official stage-entry and grasp-attempt semantics.

    ROS transports and hardware lifecycle evidence remain outside this class;
    the deterministic decisions here are replayable in software tests.
    """

    num_stages: int
    stage: int = 1
    grasp_attempt: int = 0

    def __post_init__(self):
        self.num_stages = int(self.num_stages)
        self.enter_stage(self.stage)

    def enter_stage(self, stage: int) -> int:
        value = int(stage)
        if not 1 <= value <= self.num_stages:
            raise ValueError("stage is outside the approved ReKep program")
        if value != getattr(self, "stage", None):
            self.grasp_attempt = 0
        self.stage = value
        return value

    def advance(self) -> bool:
        if self.stage >= self.num_stages:
            return False
        self.enter_stage(self.stage + 1)
        return True

    def record_grasp_failure(self) -> int:
        self.grasp_attempt += 1
        return self.grasp_attempt

    def select_backtrack_stage(
            self, path_constraints_by_stage: Sequence[Sequence[Callable]],
            end_effector: np.ndarray, keypoints: np.ndarray,
            tolerance: float = 0.10) -> int:
        """Mirror the official ``main.py`` reverse-stage search exactly."""
        if self.stage <= 1:
            return self.stage
        current = path_constraints_by_stage[self.stage - 1]
        if all(_satisfied(fn, end_effector, keypoints, tolerance)
               for fn in current):
            return self.stage
        selected = 1
        for candidate in range(self.stage - 1, 0, -1):
            constraints = path_constraints_by_stage[candidate - 1]
            if (not constraints
                    or all(_satisfied(fn, end_effector, keypoints, tolerance)
                           for fn in constraints)):
                selected = candidate
                break
        self.enter_stage(selected)
        return selected


def _satisfied(function, end_effector, keypoints, tolerance):
    value = float(function(end_effector, keypoints))
    return bool(np.isfinite(value) and value <= float(tolerance))


def pose_reached(current_pose, target_pose, position_tolerance_m=0.01,
                 rotation_tolerance_rad=0.10):
    current = np.asarray(current_pose, dtype=float)
    target = np.asarray(target_pose, dtype=float)
    if (current.shape != (7,) or target.shape != (7,)
            or not np.all(np.isfinite(current))
            or not np.all(np.isfinite(target))):
        return False
    position_error = float(np.linalg.norm(current[:3] - target[:3]))
    q1 = current[3:] / np.linalg.norm(current[3:])
    q2 = target[3:] / np.linalg.norm(target[3:])
    rotation_error = float(2.0 * np.arccos(
        np.clip(abs(float(np.dot(q1, q2))), -1.0, 1.0)))
    return (position_error <= float(position_tolerance_m)
            and rotation_error <= float(rotation_tolerance_rad))
