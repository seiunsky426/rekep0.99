"""Pure validation for the restricted six-axis Piper trajectory bridge."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

from rekpiper_planning.sdf import (
    SDFClearanceError,
    validate_rekep_sdf_clearance as _validate_rekep_sdf_clearance,
)


class TrajectoryValidationError(ValueError):
    pass


JOINT_NAMES = tuple("joint{}".format(index) for index in range(1, 7))
POSITION_MIN = np.asarray([-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944])
POSITION_MAX = np.asarray([2.168, 3.14, 0.0, 1.745, 1.22, 2.0944])


def normalize_feedback_to_joint_limits(
        positions: Sequence[float], tolerance_rad: float = 0.0) -> np.ndarray:
    """Snap only tiny encoder/model zero offsets to the legal model boundary.

    This does not widen a Piper motion limit: it is solely a planning-start
    correction. Larger measured violations remain a fail-closed error.
    """
    values = np.asarray(positions, dtype=float)
    tolerance = float(tolerance_rad)
    if (values.shape != (6,) or not np.all(np.isfinite(values))
            or not np.isfinite(tolerance) or tolerance < 0.0):
        raise TrajectoryValidationError("joint feedback is invalid")
    lower_error = np.maximum(POSITION_MIN - values, 0.0)
    upper_error = np.maximum(values - POSITION_MAX, 0.0)
    maximum_error = float(max(np.max(lower_error), np.max(upper_error)))
    if maximum_error > tolerance + 1e-9:
        index = int(np.argmax(np.maximum(lower_error, upper_error)))
        raise TrajectoryValidationError(
            "joint feedback {}={:.6f} exceeds Piper limit [{:.6f}, {:.6f}]"
            .format(JOINT_NAMES[index], values[index], POSITION_MIN[index],
                    POSITION_MAX[index]))
    return np.clip(values, POSITION_MIN, POSITION_MAX)


def validate_rekep_sdf_clearance(
        distances_m, observed, minimum_clearance_m: float) -> float:
    try:
        return _validate_rekep_sdf_clearance(
            distances_m, observed, minimum_clearance_m)
    except SDFClearanceError as exc:
        raise TrajectoryValidationError(str(exc)) from exc


@dataclass(frozen=True)
class ValidatedTrajectory:
    positions: np.ndarray
    times: np.ndarray
    sample_rate_hz: float = 0.0
    duration_s: float = 0.0
    maximum_velocity_rad_s: float = 0.0
    maximum_acceleration_rad_s2: float = 0.0
    maximum_jerk_rad_s3: float = 0.0


def trajectory_metrics(positions, times):
    """Return finite-difference maxima for a sampled six-joint trajectory."""
    q = np.asarray(positions, dtype=float)
    t = np.asarray(times, dtype=float)
    dt = np.diff(t)
    velocity = np.diff(q, axis=0) / dt[:, None]
    maximum_velocity = float(np.max(np.abs(velocity)))
    maximum_acceleration = 0.0
    maximum_jerk = 0.0
    if len(velocity) > 1:
        acceleration_times = 0.5 * (t[:-1] + t[1:])
        acceleration = np.diff(velocity, axis=0) / np.diff(
            acceleration_times)[:, None]
        maximum_acceleration = float(np.max(np.abs(acceleration)))
        if len(acceleration) > 1:
            jerk_times = 0.5 * (
                acceleration_times[:-1] + acceleration_times[1:])
            jerk = np.diff(acceleration, axis=0) / np.diff(
                jerk_times)[:, None]
            maximum_jerk = float(np.max(np.abs(jerk)))
    return maximum_velocity, maximum_acceleration, maximum_jerk


def short_horizon_joint_prefix(joint_path, current_joints,
                               maximum_velocity_rad_s,
                               duration_s=0.10):
    """Return a bounded three-sample prefix without any ROS/hardware side effect."""
    path = np.asarray(joint_path, dtype=float)
    current = np.asarray(current_joints, dtype=float)
    if (path.ndim != 2 or current.shape != (path.shape[1],)
            or len(path) == 0 or not np.all(np.isfinite(path))
            or not np.all(np.isfinite(current))):
        raise ValueError("short-horizon joint path is invalid")
    if np.linalg.norm(path[0] - current) > 0.08:
        raise ValueError("planned path start is discontinuous")
    target = path[min(len(path) - 1, 2)]
    delta = target - current
    scale = min(1.0, float(maximum_velocity_rad_s) * float(duration_s)
                / max(float(np.max(np.abs(delta))), 1e-9))
    endpoint = current + scale * delta
    times = np.linspace(0.0, float(duration_s), 3)
    positions = np.asarray([
        current + fraction * (endpoint - current)
        for fraction in np.linspace(0.0, 1.0, 3)])
    return positions, times


def smooth_pchip_trajectory(
        joint_names: Sequence[str],
        positions: Iterable[Sequence[float]],
        current_positions: Sequence[float],
        knot_times: Sequence[float] = None,
        sample_rate_hz: float = 50.0,
        maximum_velocity_rad_s: float = 0.05,
        maximum_acceleration_rad_s2: float = 0.10,
        maximum_jerk_rad_s3: float = 0.50,
        start_tolerance_rad: float = 0.08,
) -> ValidatedTrajectory:
    """Build a zero-end-velocity, shape-preserving cubic joint trajectory.

    PCHIP supplies monotonicity-preserving knot slopes.  A cubic Hermite curve
    uses those slopes with the first and last forced to zero.  If any velocity,
    acceleration or jerk limit is exceeded, all knot times are stretched by a
    single factor; the joint-space path geometry is never changed.
    """
    if tuple(joint_names) != JOINT_NAMES:
        raise TrajectoryValidationError(
            "trajectory must use joint1..joint6 in canonical order")
    q = np.asarray(list(positions), dtype=float)
    current = np.asarray(current_positions, dtype=float)
    limits = np.asarray([
        sample_rate_hz, maximum_velocity_rad_s,
        maximum_acceleration_rad_s2, maximum_jerk_rad_s3,
    ], dtype=float)
    if (q.ndim != 2 or q.shape[1] != 6 or len(q) < 2
            or current.shape != (6,) or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(current)) or not np.all(np.isfinite(limits))
            or np.any(limits <= 0.0)):
        raise TrajectoryValidationError("smooth trajectory inputs are invalid")
    # Consecutive duplicates make PCHIP's independent variable ill-conditioned
    # and add no path geometry. Preserve the first and final endpoint.
    keep = np.r_[True, np.max(np.abs(np.diff(q, axis=0)), axis=1) > 1e-10]
    q = q[keep]
    if len(q) == 1:
        q = np.vstack([q[0], q[0]])
    if knot_times is None:
        t = np.zeros(len(q), dtype=float)
        for index, delta in enumerate(np.diff(q, axis=0), start=1):
            t[index] = t[index - 1] + max(
                0.10,
                float(np.max(np.abs(delta))) / maximum_velocity_rad_s)
    else:
        raw_t = np.asarray(knot_times, dtype=float)
        if raw_t.shape != (len(keep),) or not np.all(np.isfinite(raw_t)):
            raise TrajectoryValidationError("smooth knot times are invalid")
        t = raw_t[keep]
        t -= t[0]
        if len(t) == 1:
            t = np.asarray([0.0, 0.10])
        if np.any(np.diff(t) <= 0.0):
            raise TrajectoryValidationError(
                "smooth knot times must be strictly increasing")
        for index, delta in enumerate(np.diff(q, axis=0), start=1):
            minimum = max(
                0.10,
                float(np.max(np.abs(delta))) / maximum_velocity_rad_s)
            if t[index] - t[index - 1] < minimum:
                t[index:] += minimum - (t[index] - t[index - 1])

    for _iteration in range(12):
        pchip = PchipInterpolator(t, q, axis=0)
        slopes = np.asarray(pchip.derivative()(t), dtype=float)
        slopes[0] = 0.0
        slopes[-1] = 0.0
        spline = CubicHermiteSpline(t, q, slopes, axis=0)
        duration = float(t[-1])
        count = max(2, int(np.ceil(duration * sample_rate_hz)) + 1)
        sampled_t = np.linspace(0.0, duration, count)
        sampled_q = np.asarray(spline(sampled_t), dtype=float)
        # Use analytical cubic derivatives to compute the stretch factor; the
        # public diagnostics below remain finite-difference measurements of
        # the actual samples sent to Piper.
        vmax = float(np.max(np.abs(spline(sampled_t, 1))))
        amax = float(np.max(np.abs(spline(sampled_t, 2))))
        jmax = float(np.max(np.abs(spline(sampled_t, 3))))
        measured_v, measured_a, measured_j = trajectory_metrics(
            sampled_q, sampled_t)
        vmax = max(vmax, measured_v)
        amax = max(amax, measured_a)
        jmax = max(jmax, measured_j)
        scale = max(
            1.0,
            vmax / maximum_velocity_rad_s,
            math.sqrt(amax / maximum_acceleration_rad_s2),
            (jmax / maximum_jerk_rad_s3) ** (1.0 / 3.0),
        )
        if scale <= 1.000001:
            break
        t *= 1.01 * scale
    else:
        raise TrajectoryValidationError(
            "smooth trajectory cannot satisfy kinematic limits")

    result = validate_trajectory(
        JOINT_NAMES, sampled_q, sampled_t, current,
        maximum_velocity_rad_s=maximum_velocity_rad_s,
        maximum_acceleration_rad_s2=maximum_acceleration_rad_s2,
        maximum_jerk_rad_s3=maximum_jerk_rad_s3,
        start_tolerance_rad=start_tolerance_rad)
    measured = trajectory_metrics(result.positions, result.times)
    return ValidatedTrajectory(
        result.positions, result.times,
        sample_rate_hz=float((len(result.times) - 1) / result.times[-1]),
        duration_s=float(result.times[-1]),
        maximum_velocity_rad_s=measured[0],
        maximum_acceleration_rad_s2=measured[1],
        maximum_jerk_rad_s3=measured[2])


def validate_trajectory(
        joint_names: Sequence[str],
        positions: Iterable[Sequence[float]],
        times: Sequence[float],
        current_positions: Sequence[float],
        maximum_velocity_rad_s: float = 0.50,
        maximum_acceleration_rad_s2: float = 1.0,
        start_tolerance_rad: float = 0.08,
        maximum_jerk_rad_s3: float = float("inf")) -> ValidatedTrajectory:
    names = tuple(joint_names)
    if names != JOINT_NAMES:
        raise TrajectoryValidationError(
            "trajectory must use joint1..joint6 in canonical order")
    q = np.asarray(list(positions), dtype=float)
    t = np.asarray(times, dtype=float)
    current = np.asarray(current_positions, dtype=float)
    if (q.ndim != 2 or q.shape[1] != 6 or len(q) < 2
            or t.shape != (len(q),) or current.shape != (6,)
            or not np.all(np.isfinite(q)) or not np.all(np.isfinite(t))
            or not np.all(np.isfinite(current))):
        raise TrajectoryValidationError("trajectory arrays are invalid")
    if t[0] < 0.0 or np.any(np.diff(t) <= 0.0):
        raise TrajectoryValidationError("trajectory times must be strictly increasing")
    if np.any(q < POSITION_MIN - 1e-9) or np.any(q > POSITION_MAX + 1e-9):
        raise TrajectoryValidationError("trajectory exceeds Piper joint limits")
    if float(np.max(np.abs(q[0] - current))) > float(start_tolerance_rad):
        raise TrajectoryValidationError("trajectory start differs from feedback")
    dt = np.diff(t)
    velocity = np.diff(q, axis=0) / dt[:, None]
    if float(np.max(np.abs(velocity))) > float(maximum_velocity_rad_s) + 1e-9:
        raise TrajectoryValidationError("trajectory velocity exceeds execution limit")
    if len(velocity) > 1:
        accel_dt = 0.5 * (dt[:-1] + dt[1:])
        acceleration = np.diff(velocity, axis=0) / accel_dt[:, None]
        if (float(np.max(np.abs(acceleration)))
                > float(maximum_acceleration_rad_s2) + 1e-9):
            raise TrajectoryValidationError(
                "trajectory acceleration exceeds execution limit")
        if len(acceleration) > 1:
            acceleration_times = 0.5 * (t[:-2] + t[1:-1])
            jerk = np.diff(acceleration, axis=0) / np.diff(
                acceleration_times)[:, None]
            if (float(np.max(np.abs(jerk)))
                    > float(maximum_jerk_rad_s3) + 1e-9):
                raise TrajectoryValidationError(
                    "trajectory jerk exceeds execution limit")
    metrics = trajectory_metrics(q, t)
    return ValidatedTrajectory(
        q, t,
        sample_rate_hz=float((len(t) - 1) / (t[-1] - t[0])),
        duration_s=float(t[-1] - t[0]),
        maximum_velocity_rad_s=metrics[0],
        maximum_acceleration_rad_s2=metrics[1],
        maximum_jerk_rad_s3=metrics[2])
