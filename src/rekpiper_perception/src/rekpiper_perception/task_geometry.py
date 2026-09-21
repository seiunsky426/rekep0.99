"""Category-free organized RGB-D geometry for ReKep scene masks."""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


def capture_follows_task_trigger(rgb_stamp_ns, depth_stamp_ns, trigger_stamp_ns):
    """A new task cannot consume a synchronized pair queued before its trigger."""
    return (int(rgb_stamp_ns) > 0 and int(depth_stamp_ns) > 0
            and min(int(rgb_stamp_ns), int(depth_stamp_ns)) >= int(trigger_stamp_ns))


@dataclass(frozen=True)
class InstanceGeometryEstimate:
    mask_area_pixels: int
    visual_mask_area_pixels: int
    geometry_valid_pixels: int
    bbox_xywh: Tuple[int, int, int, int]
    surface_medoid: np.ndarray
    medoid_pixel_rc: Tuple[int, int]
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    contour_xy: np.ndarray


def erode_foreground_mask(mask, erosion_px=3):
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or int(erosion_px) < 0:
        raise ValueError("mask/erosion is invalid")
    if int(erosion_px) <= 1:
        return binary.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(erosion_px), int(erosion_px)))
    return cv2.erode(binary.astype(np.uint8), kernel).astype(bool)


def prepare_sam_visual_mask(mask, closing_px=5):
    binary = np.asarray(mask, dtype=bool)
    if (binary.ndim != 2 or int(closing_px) < 0 or int(closing_px) > 15
            or (int(closing_px) not in (0, 1) and int(closing_px) % 2 == 0)):
        raise ValueError("SAM mask cleanup configuration is invalid")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(binary)
    visual = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(closing_px) > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(closing_px), int(closing_px)))
        visual = cv2.morphologyEx(
            visual.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    return visual


def point_sample_inliers(samples, mad_scale=5.0, statistical_k=16,
                         statistical_std_scale=2.5):
    selected = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
    inliers = np.all(np.isfinite(selected), axis=1)
    inliers &= np.linalg.norm(np.nan_to_num(selected), axis=1) > 1e-8
    indices = np.flatnonzero(inliers)
    valid = selected[indices]
    if len(valid) < 4:
        return inliers
    center = np.median(valid, axis=0)
    radius = np.linalg.norm(valid - center, axis=1)
    median = float(np.median(radius))
    mad = float(np.median(np.abs(radius - median)))
    if mad > 1e-9:
        keep = radius <= median + float(mad_scale) * 1.4826 * mad
        inliers[indices[~keep]] = False
        indices, valid = indices[keep], valid[keep]
    if int(statistical_k) > 1 and len(valid) > int(statistical_k) + 1:
        from scipy.spatial import cKDTree
        distances, _ = cKDTree(valid).query(
            valid, k=min(int(statistical_k) + 1, len(valid)), workers=1)
        means = distances[:, 1:].mean(axis=1)
        keep = means <= means.mean() + float(statistical_std_scale) * means.std()
        inliers[indices[~keep]] = False
    return inliers


def filter_point_samples(samples, **kwargs):
    selected = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
    return selected[point_sample_inliers(selected, **kwargs)]


def organized_workspace_mask(points, bounds_min, bounds_max):
    cloud = np.asarray(points, dtype=np.float64)
    lower = np.asarray(bounds_min, dtype=np.float64)
    upper = np.asarray(bounds_max, dtype=np.float64)
    if (cloud.ndim != 3 or cloud.shape[2] != 3
            or lower.shape != (3,) or upper.shape != (3,)
            or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)):
        raise ValueError("organized cloud or workspace is invalid")
    return (np.all(np.isfinite(cloud), axis=2)
            & np.all(cloud >= lower, axis=2)
            & np.all(cloud <= upper, axis=2))


def workspace_mask_area_ratio(instance_mask, workspace_pixels):
    instance = np.asarray(instance_mask, dtype=bool)
    workspace = np.asarray(workspace_pixels, dtype=bool)
    if instance.ndim != 2 or instance.shape != workspace.shape:
        raise ValueError("instance and workspace masks must match")
    denominator = int(np.count_nonzero(workspace))
    if denominator == 0:
        raise ValueError("workspace contains no valid pixels")
    return float(np.count_nonzero(instance & workspace)) / denominator


def clean_organized_mask(points, mask, erosion_px=3):
    cloud = np.asarray(points, dtype=np.float64)
    binary = erode_foreground_mask(mask, erosion_px)
    if cloud.ndim != 3 or cloud.shape[2] != 3 or binary.shape != cloud.shape[:2]:
        raise ValueError("organized point cloud and mask dimensions differ")
    rows, cols = np.nonzero(binary)
    cleaned = np.zeros_like(binary)
    if len(rows):
        cleaned[rows, cols] = point_sample_inliers(cloud[rows, cols])
    return cleaned


def estimate_instance_geometry(points, mask, contour_epsilon_px=2.0,
                               max_contour_points=128, visual_mask=None):
    cloud = np.asarray(points, dtype=np.float64)
    binary = np.asarray(mask, dtype=bool)
    visual = binary if visual_mask is None else np.asarray(visual_mask, dtype=bool)
    if (cloud.ndim != 3 or cloud.shape[2] != 3
            or binary.shape != cloud.shape[:2] or visual.shape != binary.shape
            or not np.any(binary) or not np.any(visual)):
        raise ValueError("instance geometry input is invalid")
    rows, cols = np.nonzero(binary)
    samples = cloud[rows, cols]
    finite = np.all(np.isfinite(samples), axis=1)
    rows, cols, samples = rows[finite], cols[finite], samples[finite]
    if not len(samples):
        raise ValueError("instance has no finite surface samples")
    center = np.median(samples, axis=0)
    medoid_index = int(np.argmin(np.sum((samples - center) ** 2, axis=1)))
    x, y, width, height = cv2.boundingRect(visual.astype(np.uint8))
    contours, _ = cv2.findContours(
        visual.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("instance contour is unavailable")
    contour = cv2.approxPolyDP(
        max(contours, key=cv2.contourArea), float(contour_epsilon_px),
        True).reshape(-1, 2)
    if len(contour) > int(max_contour_points):
        contour = contour[np.linspace(
            0, len(contour) - 1, int(max_contour_points), dtype=int)]
    return InstanceGeometryEstimate(
        mask_area_pixels=int(len(samples)),
        visual_mask_area_pixels=int(np.count_nonzero(visual)),
        geometry_valid_pixels=int(len(samples)),
        bbox_xywh=(int(x), int(y), int(width), int(height)),
        surface_medoid=samples[medoid_index].copy(),
        medoid_pixel_rc=(int(rows[medoid_index]), int(cols[medoid_index])),
        bounds_min=np.min(samples, axis=0), bounds_max=np.max(samples, axis=0),
        contour_xy=np.asarray(contour, dtype=np.int32).reshape(-1, 2))
