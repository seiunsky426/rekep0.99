"""Pure helpers for Piper's feedback-closed, position-stepped grasp."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


class GripperCloseError(ValueError):
    pass


@dataclass(frozen=True)
class ContactDecision:
    contact: bool
    window_duration_s: float
    opening_range_m: float
    command_error_m: float
    width_error_m: float
    opening_above_empty_m: float
    effort_above_empty: float
    effort_ok: bool


def stepped_close_targets(start_m, local_width_m, step_m=0.0005,
                          close_delta_m=0.003):
    """Return descending commands including exactly ``local_width-3mm``."""
    values = [float(start_m), float(local_width_m), float(step_m),
              float(close_delta_m)]
    if (not all(math.isfinite(value) for value in values)
            or start_m < 0.0 or local_width_m <= 0.0 or step_m <= 0.0
            or close_delta_m < 0.0):
        raise GripperCloseError("gripper close profile is invalid")
    target = max(0.0, float(local_width_m) - float(close_delta_m))
    if start_m <= target:
        return [target]
    result = []
    command = float(start_m)
    while command - step_m > target + 1e-12:
        command -= step_m
        result.append(command)
    result.append(target)
    return result


class ContactDetector:
    def __init__(self, window_s=0.6, maximum_motion_m=0.0003,
                 minimum_command_error_m=0.001,
                 width_tolerance_m=0.005):
        self.window_s = float(window_s)
        self.maximum_motion_m = float(maximum_motion_m)
        self.minimum_command_error_m = float(minimum_command_error_m)
        self.width_tolerance_m = float(width_tolerance_m)
        if (not all(math.isfinite(value) and value >= 0.0 for value in (
                self.window_s, self.maximum_motion_m,
                self.minimum_command_error_m, self.width_tolerance_m))
                or self.window_s <= 0.0):
            raise GripperCloseError("contact detector configuration is invalid")
        self._samples = deque()

    def update(self, stamp_s, actual_m, commanded_m, local_width_m,
               effort=0.0, empty_closed_opening_p99_m=0.0,
               minimum_opening_margin_m=0.002,
               empty_effort_p95=0.0, minimum_effort_margin=0.0,
               require_effort=False) -> ContactDecision:
        stamp = float(stamp_s)
        actual = float(actual_m)
        commanded = float(commanded_m)
        width = float(local_width_m)
        effort = float(effort)
        empty_opening = float(empty_closed_opening_p99_m)
        opening_margin = float(minimum_opening_margin_m)
        empty_effort = float(empty_effort_p95)
        effort_margin = float(minimum_effort_margin)
        if not all(math.isfinite(value) for value in (
                stamp, actual, commanded, width, effort, empty_opening,
                opening_margin, empty_effort, effort_margin)):
            raise GripperCloseError("gripper contact sample is invalid")
        self._samples.append((stamp, actual))
        if len(self._samples) > 1 and stamp <= self._samples[-2][0]:
            raise GripperCloseError("contact sample timestamps must increase")
        cutoff = stamp - self.window_s
        # Retain the one sample immediately before the window boundary.  ROS
        # feedback is not phase-locked to the timer; dropping that sample made
        # a 0.6 s window possible only for very specific sample periods.
        while (len(self._samples) > 2
               and self._samples[1][0] <= cutoff):
            self._samples.popleft()
        duration = stamp - self._samples[0][0]
        openings = [item[1] for item in self._samples]
        opening_range = max(openings) - min(openings)
        command_error = actual - commanded
        width_error = abs(actual - width)
        opening_above_empty = actual - empty_opening
        effort_above_empty = effort - empty_effort
        effort_ok = (not require_effort
                     or effort_above_empty >= effort_margin)
        contact = bool(
            duration >= self.window_s - 1e-3
            and opening_range <= self.maximum_motion_m
            and command_error >= self.minimum_command_error_m
            and width_error <= self.width_tolerance_m
            and opening_above_empty >= opening_margin
            and effort_ok)
        return ContactDecision(
            contact=contact,
            window_duration_s=duration,
            opening_range_m=opening_range,
            command_error_m=command_error,
            width_error_m=width_error,
            opening_above_empty_m=opening_above_empty,
            effort_above_empty=effort_above_empty,
            effort_ok=effort_ok)
