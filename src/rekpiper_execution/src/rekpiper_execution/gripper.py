"""Piper 70 mm total-jaw-opening conversion helpers."""

from __future__ import annotations

import math
from typing import Tuple


class GripperValueError(ValueError):
    pass


def _validated_total_opening(
        total_opening_m: float, maximum_total_opening_m: float = 0.070,
        negative_tolerance_m: float = 0.001) -> float:
    opening = float(total_opening_m)
    maximum = float(maximum_total_opening_m)
    tolerance = float(negative_tolerance_m)
    if not all(math.isfinite(value) for value in (opening, maximum, tolerance)):
        raise GripperValueError("gripper opening must be finite")
    if maximum <= 0.0 or tolerance < 0.0:
        raise GripperValueError("gripper limits are invalid")
    if opening < -tolerance or opening > maximum + 1e-9:
        raise GripperValueError(
            "total jaw opening is outside [{:.6f}, {:.6f}] m".format(
                -tolerance, maximum))
    return min(max(0.0, opening), maximum)


def symmetric_gripper_joints(
        total_opening_m: float,
        maximum_total_opening_m: float = 0.070,
        negative_tolerance_m: float = 0.001) -> Tuple[float, float]:
    """Convert Piper total jaw opening to the two ±35 mm URDF joints."""
    opening = _validated_total_opening(
        total_opening_m, maximum_total_opening_m, negative_tolerance_m)
    travel = 0.5 * opening
    return travel, -travel


def piper_feedback_to_total_opening(
        feedback_m: float, maximum_total_opening_m: float = 0.070) -> float:
    """Validate the driver's total-opening feedback without rescaling it."""
    return _validated_total_opening(feedback_m, maximum_total_opening_m)


def piper_command_from_total_opening(
        total_opening_m: float, maximum_total_opening_m: float = 0.070) -> float:
    """Validate a total opening for the driver's gripper service."""
    return _validated_total_opening(
        total_opening_m, maximum_total_opening_m, negative_tolerance_m=0.0)
