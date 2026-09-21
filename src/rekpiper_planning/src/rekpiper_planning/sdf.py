"""Shared ReKep signed-distance convention and fail-closed validation."""

from __future__ import annotations

import numpy as np


class SDFClearanceError(ValueError):
    pass


def validate_rekep_sdf_clearance(
        distances_m, observed, minimum_clearance_m: float) -> float:
    """Return free clearance for ReKep SDFs (free < 0, occupied > 0)."""
    distances = np.asarray(distances_m, dtype=float)
    visible = np.asarray(observed, dtype=bool)
    clearance = float(minimum_clearance_m)
    if (distances.ndim != 1 or distances.size == 0
            or visible.shape != distances.shape
            or not np.all(np.isfinite(distances))
            or not np.isfinite(clearance) or clearance < 0.0):
        raise SDFClearanceError("SDF clearance response is invalid")
    if not np.all(visible):
        raise SDFClearanceError("SDF path contains unobserved samples")
    minimum = float(-np.max(distances))
    if minimum + 1e-9 < clearance:
        raise SDFClearanceError(
            "SDF path clearance is below execution limit")
    return minimum
