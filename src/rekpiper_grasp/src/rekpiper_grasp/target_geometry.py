"""Mask-derived target ownership, independent of color labels and bounding boxes."""
import numpy as np
from scipy.spatial import cKDTree


def target_region_mask(scene_points, target_points, other_points=None, tolerance_m=.005):
    scene, target = np.asarray(scene_points), np.asarray(target_points)
    if (scene.ndim != 2 or scene.shape[1] != 3 or target.ndim != 2
            or target.shape[1] != 3 or len(target) < 120
            or not np.all(np.isfinite(scene)) or not np.all(np.isfinite(target))):
        raise ValueError('insufficient_or_invalid_mask_derived_target_cloud')
    distance = cKDTree(target).query(scene)[0]
    mask = distance <= tolerance_m
    if other_points is not None and len(other_points):
        other_distance = cKDTree(np.asarray(other_points)).query(scene)[0]
        mask &= distance + .002 < other_distance
    if np.count_nonzero(mask) < 120:
        raise ValueError('insufficient_unambiguous_target_points')
    return mask
