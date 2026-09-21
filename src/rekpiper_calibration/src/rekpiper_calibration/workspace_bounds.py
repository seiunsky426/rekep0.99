"""Point-envelope diagnostics, not robot collision or motion acceptance."""

import numpy as np
from itertools import product


def board_corners_in_marker(board):
    dimensions = np.asarray(board["full_dimensions_m"], dtype=float)
    center = np.asarray(board["center_in_marker_m"], dtype=float)
    if (dimensions.shape != (3,) or center.shape != (3,)
            or not np.all(np.isfinite([dimensions, center]))
            or np.any(dimensions <= 0)):
        raise ValueError("positive board dimensions and finite marker-frame center required")
    return center + np.asarray(list(product((-1, 1), repeat=3))) * dimensions / 2


def workspace_limits(workspace):
    if workspace.get("frame") != "base_link":
        raise ValueError("workspace must be expressed in base_link")
    lower = np.asarray(workspace["lower_m"], dtype=float)
    upper = np.asarray(workspace["upper_m"], dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,)
            or not np.all(np.isfinite([lower, upper]))
            or np.any(upper <= lower)):
        raise ValueError("finite, ordered XYZ workspace limits required")
    return lower, upper


def point_envelope(points, lower, upper):
    values = np.asarray(points, dtype=float)
    lower, upper = workspace_limits({
        "frame": "base_link", "lower_m": lower, "upper_m": upper})
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) \
            or not np.all(np.isfinite(values)):
        raise ValueError("finite nonempty Nx3 point array required")
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    violations = np.any((values < lower) | (values > upper), axis=1)
    return {
        "sampled_points_within_bounds": bool(not np.any(violations)),
        "minimum_xyz_m": minimum.tolist(),
        "maximum_xyz_m": maximum.tolist(),
        "lower_margin_xyz_m": (minimum-lower).tolist(),
        "upper_margin_xyz_m": (upper-maximum).tolist(),
        "outside_point_indices": np.flatnonzero(violations).tolist(),
    }
