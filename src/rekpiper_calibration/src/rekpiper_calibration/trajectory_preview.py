"""Offline playback of calibration candidates; no hardware command interface."""

import numpy as np


JOINT_NAMES = ["joint{}".format(index) for index in range(1, 7)]


def validate_preview(plan, solver):
    if (plan.get("schema_version") != 1
            or plan.get("status") != "PREVIEW_ONLY"
            or plan.get("hardware_execution_allowed") is not False):
        raise ValueError("PREVIEW_ONLY plan with hardware execution disabled required")
    if plan.get("joint_names") != JOINT_NAMES:
        raise ValueError("joint order must be joint1 through joint6")
    if plan.get("base_frame") != "base_link" or plan.get("tip_frame") != "link6":
        raise ValueError("base_link -> link6 preview required")
    poses = plan.get("poses", [])
    if len(poses) != 20:
        raise ValueError("exactly 20 candidate poses required")
    joints = np.asarray([pose["positions_rad"] for pose in poses], dtype=float)
    if joints.shape != (20, 6) or not np.all(np.isfinite(joints)):
        raise ValueError("finite 20x6 joint array required")
    if np.any(joints < solver._lower) or np.any(joints > solver._upper):
        raise ValueError("candidate outside URDF joint limits")
    for index, pose in enumerate(poses):
        target = np.asarray(pose["base_T_link6"], dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("invalid base_T_link6")
        if not np.allclose(solver.forward(joints[index]), target, atol=1e-7, rtol=0):
            raise ValueError("stored candidate FK disagrees with current URDF")
        if any(np.allclose(joints[index], other, atol=1e-6, rtol=0)
               for other in joints[:index]):
            raise ValueError("duplicate candidate")
    return joints


def playback_samples(joints, transition_s=4.0, dwell_s=1.0, rate_hz=20.0):
    """Yield (time, target index, joints) for one display-only pass.

    The first pose is held immediately; the connection from a live robot to
    that pose is intentionally not inferred.  Quintic interpolation is a
    visualization, not collision checking or controller time parameterization.
    """
    if not all(np.isfinite(v) and v > 0
               for v in (transition_s, dwell_s, rate_hz)):
        raise ValueError("positive finite playback timing required")
    values = np.asarray(joints, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6 or not np.all(np.isfinite(values)):
        raise ValueError("finite Nx6 playback joints required")
    tick = 0
    for index, goal in enumerate(values):
        if index:
            count = max(1, int(np.ceil(transition_s * rate_hz)))
            for step in range(1, count + 1):
                u = step / float(count)
                blend = 10*u**3 - 15*u**4 + 6*u**5
                yield tick / rate_hz, index, values[index-1] + blend * (goal-values[index-1])
                tick += 1
        for _ in range(max(1, int(np.ceil(dwell_s * rate_hz)))):
            yield tick / rate_hz, index, goal.copy()
            tick += 1
