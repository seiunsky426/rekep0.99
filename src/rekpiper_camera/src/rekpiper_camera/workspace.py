"""Geometry helpers for the measured recognition workspace."""

from typing import Dict, Iterable, Sequence, Tuple

import cv2
import numpy as np


def normalized_polygon_mask(
    height: int, width: int, vertices: Sequence[Sequence[float]]
) -> np.ndarray:
    """Rasterize normalized ``[x, y]`` polygon vertices into an image mask."""
    height = int(height)
    width = int(width)
    points = np.asarray(vertices, dtype=np.float64)
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError("polygon must contain at least three [x,y] vertices")
    if not np.all(np.isfinite(points)):
        raise ValueError("polygon vertices must be finite")
    if np.any(points < 0.0) or np.any(points > 1.0):
        raise ValueError("normalized polygon vertices must lie in [0,1]")
    pixels = np.empty(points.shape, dtype=np.int32)
    pixels[:, 0] = np.rint(points[:, 0] * (width - 1)).astype(np.int32)
    pixels[:, 1] = np.rint(points[:, 1] * (height - 1)).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pixels], 1)
    return mask.astype(bool)


def filter_xyz_bounds(
    xyz: np.ndarray,
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace points outside an axis-aligned base-frame workspace with NaN."""
    points = np.asarray(xyz, dtype=np.float32)
    lower = np.asarray(bounds_min, dtype=np.float32)
    upper = np.asarray(bounds_max, dtype=np.float32)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("xyz must have final dimension 3")
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("bounds must contain exactly three values")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("bounds must be finite")
    if np.any(lower >= upper):
        raise ValueError("each lower bound must be less than its upper bound")
    valid = np.all(np.isfinite(points), axis=-1)
    valid &= np.all(points >= lower, axis=-1)
    valid &= np.all(points <= upper, axis=-1)
    filtered = points.copy()
    filtered[~valid] = np.nan
    return filtered, valid


def estimate_table_workspace(
    clouds: Dict[str, np.ndarray],
    polygons: Dict[str, Iterable[Sequence[float]]],
    expected_table_z_m: float,
    table_band_half_width_m: float,
    xy_quantiles: Sequence[float],
    xy_margin_m: float,
    below_table_allowance_m: float,
    max_height_above_table_m: float,
) -> dict:
    """Estimate a conservative XYZ box from manually indicated table pixels."""
    expected_z = float(expected_table_z_m)
    band = float(table_band_half_width_m)
    margin = float(xy_margin_m)
    below = float(below_table_allowance_m)
    above = float(max_height_above_table_m)
    quantiles = np.asarray(xy_quantiles, dtype=np.float64)
    if not np.isfinite(expected_z):
        raise ValueError("expected table height must be finite")
    if band <= 0.0 or margin < 0.0 or below < 0.0 or above <= 0.0:
        raise ValueError("workspace distances are invalid")
    if (
        quantiles.shape != (2,)
        or not 0.0 <= quantiles[0] < quantiles[1] <= 1.0
    ):
        raise ValueError("xy_quantiles must satisfy 0 <= low < high <= 1")
    if set(clouds) != set(polygons) or not clouds:
        raise ValueError("cloud and polygon camera sets must match")

    table_points = []
    per_camera = {}
    for name in sorted(clouds):
        xyz = np.asarray(clouds[name], dtype=np.float32)
        if xyz.ndim != 3 or xyz.shape[2] != 3:
            raise ValueError("{} cloud must be organized HxWx3".format(name))
        roi = normalized_polygon_mask(xyz.shape[0], xyz.shape[1], polygons[name])
        finite = np.all(np.isfinite(xyz), axis=-1)
        in_band = np.abs(xyz[:, :, 2] - expected_z) <= band
        selected = xyz[roi & finite & in_band]
        if selected.shape[0] < 100:
            roi_z = xyz[:, :, 2][roi & finite]
            quantile_text = (
                np.quantile(
                    roi_z, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
                ).tolist()
                if roi_z.size
                else []
            )
            raise ValueError(
                "{} has too few table-band points ({}); ROI z quantiles "
                "[1,10,25,50,75,90,99]%={}".format(
                    name, selected.shape[0], quantile_text
                )
            )
        table_points.append(selected)
        per_camera[name] = {
            "roi_pixels": int(np.count_nonzero(roi)),
            "finite_roi_points": int(np.count_nonzero(roi & finite)),
            "table_band_points": int(selected.shape[0]),
            "table_z_median_m": float(np.median(selected[:, 2])),
        }

    combined = np.concatenate(table_points, axis=0)
    table_z = float(np.median(combined[:, 2]))
    x_low, x_high = np.quantile(combined[:, 0], quantiles)
    y_low, y_high = np.quantile(combined[:, 1], quantiles)
    bounds_min = [
        float(x_low - margin),
        float(y_low - margin),
        float(table_z - below),
    ]
    bounds_max = [
        float(x_high + margin),
        float(y_high + margin),
        float(table_z + above),
    ]
    return {
        "table_height_m": table_z,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "xy_margin_m": margin,
        "below_table_allowance_m": below,
        "per_camera": per_camera,
    }
