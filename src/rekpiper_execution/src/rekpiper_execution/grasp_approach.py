"""Pure validation helpers for the ReKep real-environment grasp approach."""

from __future__ import annotations

import numpy as np


class GraspApproachError(RuntimeError):
    pass


def validate_path_constraints(pose_vectors, keypoints, constraints, tolerance):
    """Validate every interpolated EE position with official ReKep callables."""
    poses = np.asarray(pose_vectors, dtype=float)
    points = np.asarray(keypoints, dtype=float)
    if (poses.ndim != 2 or poses.shape[1] != 7
            or points.ndim != 2 or points.shape[1] != 3
            or not np.all(np.isfinite(poses))
            or not np.all(np.isfinite(points))):
        raise GraspApproachError("grasp approach inputs are invalid")
    limit = float(tolerance)
    if not np.isfinite(limit) or limit < 0.0:
        raise GraspApproachError("constraint tolerance is invalid")
    maximum = 0.0
    for pose in poses:
        for constraint in constraints:
            if not callable(constraint):
                raise GraspApproachError("ReKep path constraint is not callable")
            value = float(constraint(pose[:3], points))
            if not np.isfinite(value) or value > limit:
                raise GraspApproachError(
                    "AnyGrasp approach violates ReKep path constraint")
            maximum = max(maximum, value)
    return maximum
