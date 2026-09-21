"""Pure voxel fusion helpers for base-frame diagnostic point clouds."""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np


_RS1_RGB = np.array([255, 80, 80], dtype=np.uint8)
_RS3_RGB = np.array([80, 220, 255], dtype=np.uint8)
_BOTH_RGB = np.array([255, 255, 255], dtype=np.uint8)


def closest_timestamp_pair(
        rs1_stamps: np.ndarray, rs3_stamps: np.ndarray,
) -> Optional[Tuple[int, int, float]]:
    """Return the closest cross-camera pair from two short timestamp queues."""
    first = np.asarray(rs1_stamps, dtype=np.float64).reshape(-1)
    second = np.asarray(rs3_stamps, dtype=np.float64).reshape(-1)
    if not len(first) or not len(second):
        return None
    if not (np.all(np.isfinite(first)) and np.all(np.isfinite(second))):
        raise ValueError("timestamps must be finite")
    difference = np.abs(first[:, None] - second[None, :])
    flat_index = int(np.argmin(difference))
    first_index, second_index = np.unravel_index(flat_index, difference.shape)
    return int(first_index), int(second_index), float(difference.flat[flat_index])


def voxel_fuse_base_clouds(
        clouds: Mapping[str, np.ndarray], voxel_size_m: float = 0.003,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Voxel-average RS1/RS3 base-frame clouds for RViz inspection.

    Output colors make registration errors visible: RS1-only voxels are red,
    RS3-only voxels cyan, and voxels supported by both cameras white.
    """
    voxel_size = float(voxel_size_m)
    if not np.isfinite(voxel_size) or not 0.001 <= voxel_size <= 0.050:
        raise ValueError("voxel_size_m must lie in [0.001, 0.050]")
    if set(clouds) != {"rs1", "rs3"}:
        raise ValueError("clouds must contain exactly rs1 and rs3")

    points_by_camera = {}
    for camera in ("rs1", "rs3"):
        points = np.asarray(clouds[camera], dtype=np.float32).reshape(-1, 3)
        points_by_camera[camera] = points[np.all(np.isfinite(points), axis=1)]
    input_counts = {
        camera: int(len(points_by_camera[camera]))
        for camera in ("rs1", "rs3")
    }
    if not any(input_counts.values()):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            {"input_points": input_counts, "fused_points": 0},
        )

    points = np.concatenate(
        (points_by_camera["rs1"], points_by_camera["rs3"]), axis=0)
    source = np.concatenate((
        np.zeros(len(points_by_camera["rs1"]), dtype=np.uint8),
        np.ones(len(points_by_camera["rs3"]), dtype=np.uint8),
    ))
    voxel_index = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxel_index, axis=0, return_inverse=True)
    count = np.bincount(inverse).astype(np.float32)
    summed = np.zeros((len(count), 3), dtype=np.float64)
    np.add.at(summed, inverse, points)
    fused = (summed / count[:, None]).astype(np.float32)

    rs3_count = np.bincount(
        inverse, weights=source, minlength=len(count)).astype(np.int64)
    rs1_count = count.astype(np.int64) - rs3_count
    colors = np.empty((len(count), 3), dtype=np.uint8)
    colors[:] = _BOTH_RGB
    colors[rs3_count == 0] = _RS1_RGB
    colors[rs1_count == 0] = _RS3_RGB
    return fused, colors, {
        "input_points": input_counts,
        "fused_points": int(len(fused)),
        "rs1_only_voxels": int(np.count_nonzero(rs3_count == 0)),
        "rs3_only_voxels": int(np.count_nonzero(rs1_count == 0)),
        "overlap_voxels": int(np.count_nonzero(
            (rs1_count > 0) & (rs3_count > 0))),
    }
