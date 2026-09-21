"""Make nvblox_torch distances safe for official ReKep collision costs.

Source classification: SAFETY.

The official ReKep collision cost expects:

* negative: observed free space;
* zero: surface;
* positive: obstacle interior.

The pinned nvblox_torch query kernel already follows this convention, but uses
``-100.0`` for an unobserved voxel. Passing that sentinel through would make an
unknown region look extremely safe. This adapter never permits that behavior.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class SanitizedSDF:
    distances_m: np.ndarray
    observed: np.ndarray

    @property
    def observed_fraction(self) -> float:
        if self.observed.size == 0:
            return 0.0
        return float(np.mean(self.observed))


def inclusive_grid_axes(bounds_min, bounds_max, nominal_resolution_m):
    """Return endpoint-inclusive axes shared by solver and SDF consumers."""
    lower = np.asarray(bounds_min, dtype=float)
    upper = np.asarray(bounds_max, dtype=float)
    resolution = float(nominal_resolution_m)
    if (lower.shape != (3,) or upper.shape != (3,)
            or np.any(lower >= upper) or not np.isfinite(resolution)
            or resolution <= 0.0):
        raise ValueError("invalid endpoint-inclusive grid specification")
    # Bounds pass through float32 camera/map messages. Match the planning
    # snapshot reader's tolerance at exact multiples (e.g. 0.87/0.015).
    counts = np.ceil((upper - lower) / resolution - 1e-5).astype(int) + 1
    return tuple(np.linspace(lo, hi, int(count), dtype=np.float32)
                 for lo, hi, count in zip(lower, upper, counts))


def checked_grid_response(response, count, target_frame, diagnostic_only=False):
    """Diagnostic grids retain data and provenance but never become valid."""
    if (response.header.frame_id != target_frame
            or not response.map_generation_uuid
            or len(response.distances_m) != count
            or len(response.observed) != count):
        raise ValueError("SDF query returned an incomplete or mismatched generation")
    if not response.map_valid and not (
            diagnostic_only and response.status == "acceptance_map_not_safe_for_motion"):
        raise ValueError("SDF query is not valid for this consumer: " + response.status)
    distances = np.asarray(response.distances_m, dtype=np.float32).copy()
    observed = np.asarray(response.observed, dtype=np.uint8)
    if not np.all(np.isfinite(distances)) or not np.isin(observed, [0, 1]).all():
        raise ValueError("SDF generation contains invalid values")
    # Unknown cells must remain conservative under ReKep's positive-occupied
    # convention, even if a future provider sends zero or negative sentinels.
    distances[observed == 0] = np.maximum(distances[observed == 0], 1.0)
    return distances, observed, bool(response.map_valid and not diagnostic_only)


def sanitize_nvblox_sdf(
    raw_distances_m: np.ndarray,
    unknown_cutoff_m: float = -99.0,
    unknown_occupied_distance_m: float = 1.0,
) -> SanitizedSDF:
    """Replace unobserved/non-finite nvblox values with occupied-space values.

    ``unknown_occupied_distance_m`` is deliberately positive under ReKep's
    convention. The returned observed mask must still gate final trajectory
    execution; the replacement is defense in depth for optimization.
    """
    raw = np.asarray(raw_distances_m, dtype=np.float32)
    if raw.size == 0:
        raise ValueError("SDF query cannot be empty")
    if not np.isfinite(unknown_cutoff_m) or unknown_cutoff_m >= 0.0:
        raise ValueError("unknown_cutoff_m must be finite and negative")
    if (
        not np.isfinite(unknown_occupied_distance_m)
        or unknown_occupied_distance_m <= 0.0
    ):
        raise ValueError("unknown replacement must be finite and positive")

    observed = np.isfinite(raw) & (raw > np.float32(unknown_cutoff_m))
    safe = raw.copy()
    safe[~observed] = np.float32(unknown_occupied_distance_m)
    return SanitizedSDF(distances_m=safe, observed=observed)


def validate_sdf_grid_geometry(
    distances_m: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    voxel_size_m: float,
) -> Tuple[int, int, int]:
    """Validate grid units and dimensions before creating an interpolator."""
    grid = np.asarray(distances_m)
    lower = np.asarray(bounds_min, dtype=float)
    upper = np.asarray(bounds_max, dtype=float)
    if grid.ndim != 3 or min(grid.shape) < 2:
        raise ValueError("SDF grid must be 3D with at least two samples per axis")
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError("SDF bounds must be ordered XYZ vectors")
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m must be finite and positive")
    expected = np.floor((upper - lower) / voxel_size_m).astype(int) + 1
    actual = np.asarray(grid.shape, dtype=int)
    if np.any(np.abs(actual - expected) > 1):
        raise ValueError(
            "SDF shape {} is inconsistent with bounds/voxel size {}".format(
                tuple(actual), tuple(expected)
            )
        )
    return tuple(int(value) for value in actual)
