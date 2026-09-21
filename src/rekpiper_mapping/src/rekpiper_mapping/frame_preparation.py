"""Safety checks and masking for depth frames entering the static nvblox map."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np


FILTER_RAW_INVALID = 0
FILTER_DEPTH_RANGE = 1
FILTER_WORKSPACE = 2
FILTER_ROBOT = 3
FILTER_DYNAMIC = 4
FILTER_RETAINED = 5

FILTER_REASON_NAMES = {
    FILTER_RAW_INVALID: "raw_invalid",
    FILTER_DEPTH_RANGE: "depth_range",
    FILTER_WORKSPACE: "workspace",
    FILTER_ROBOT: "robot",
    FILTER_DYNAMIC: "dynamic",
    FILTER_RETAINED: "retained",
}


@dataclass(frozen=True)
class PreparedDepth:
    depth_m: np.ndarray
    input_valid_pixels: int
    retained_pixels: int
    explicitly_excluded_pixels: int
    filter_reasons: np.ndarray
    reason_counts: Dict[str, int]

    @property
    def retained_fraction(self) -> float:
        denominator = max(self.depth_m.size, 1)
        return float(self.retained_pixels) / denominator


def normalize_camera_sources(
    configured,
    default_source: Mapping[str, str],
) -> List[Dict[str, str]]:
    """Validate one or more named RGB-D sources for a shared static map."""
    required = ("name", "camera_frame", "depth_topic", "camera_info_topic")
    values = [dict(default_source)] if configured is None else configured
    if not isinstance(values, list) or not values:
        raise ValueError("cameras must be a non-empty list")
    normalized = []
    names = set()
    for index, supplied in enumerate(values):
        if not isinstance(supplied, Mapping):
            raise ValueError("camera source {} must be a mapping".format(index))
        source = {str(key): str(value) for key, value in supplied.items()}
        for key in required:
            if not source.get(key, "").strip():
                raise ValueError("camera source {} missing {}".format(index, key))
        if source["name"] in names:
            raise ValueError("camera source names must be unique")
        names.add(source["name"])
        normalized.append(source)
    return normalized


def _ordered_bounds(bounds_min, bounds_max):
    lower = np.asarray(bounds_min, dtype=np.float32)
    upper = np.asarray(bounds_max, dtype=np.float32)
    if lower.shape != (3,) or upper.shape != (3,) or not np.all(np.isfinite(lower)):
        raise ValueError("workspace bounds must be finite XYZ vectors")
    if not np.all(np.isfinite(upper)) or np.any(lower >= upper):
        raise ValueError("workspace bounds must be ordered")
    return lower, upper


def _camera_matrix(values):
    matrix = np.asarray(values, dtype=np.float32).reshape(3, 3)
    if not np.all(np.isfinite(matrix)) or matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("camera intrinsics must contain positive finite focal lengths")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError("camera matrix last row must be [0, 0, 1]")
    return matrix


def prepare_static_depth(
    depth_raw: np.ndarray,
    depth_scale: float,
    camera_matrix: Sequence[float],
    target_from_camera: np.ndarray,
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    exclusion_masks: Iterable[np.ndarray] = (),
    mask_dilation_px: int = 3,
    min_depth_m: float = 0.10,
    max_depth_m: float = 2.00,
    exclusion_labels: Sequence[str] = (),
) -> PreparedDepth:
    """Return metric depth with robot/dynamic/out-of-workspace pixels set to zero."""
    depth = np.asarray(depth_raw)
    if depth.ndim != 2 or depth.size == 0:
        raise ValueError("depth must be a non-empty HxW image")
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("depth_scale must be finite and positive")
    if not (0.0 <= min_depth_m < max_depth_m):
        raise ValueError("expected 0 <= min_depth_m < max_depth_m")
    intrinsics = _camera_matrix(camera_matrix)
    transform = np.asarray(target_from_camera, dtype=np.float32)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("target_from_camera must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError("target_from_camera must be homogeneous")
    lower, upper = _ordered_bounds(bounds_min, bounds_max)

    depth_m = depth.astype(np.float32, copy=False) * np.float32(depth_scale)
    finite_positive = np.isfinite(depth_m) & (depth_m > 0.0)
    valid = finite_positive & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    input_valid_pixels = int(valid.sum())

    supplied_masks = list(exclusion_masks)
    supplied_labels = list(exclusion_labels)
    if supplied_labels and len(supplied_labels) != len(supplied_masks):
        raise ValueError("exclusion labels must match exclusion masks")
    if not supplied_labels:
        supplied_labels = ["robot"] + ["dynamic"] * max(0, len(supplied_masks) - 1)
    labelled_masks = []
    excluded = np.zeros(depth.shape, dtype=bool)
    for supplied_mask, label in zip(supplied_masks, supplied_labels):
        if label not in ("robot", "dynamic"):
            raise ValueError("exclusion label must be robot or dynamic")
        mask = np.asarray(supplied_mask)
        if mask.shape != depth.shape:
            raise ValueError("every exclusion mask must match the depth image")
        labelled_masks.append((label, mask.astype(bool)))
    if mask_dilation_px < 0:
        raise ValueError("mask_dilation_px cannot be negative")
    if mask_dilation_px:
        size = 2 * int(mask_dilation_px) + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        labelled_masks = [
            (label, cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool))
            for label, mask in labelled_masks
        ]
    for _, mask in labelled_masks:
        excluded |= mask

    height, width = depth.shape
    v, u = np.indices((height, width), dtype=np.float32)
    z = depth_m
    camera_points = np.stack(
        ((u - intrinsics[0, 2]) * z / intrinsics[0, 0],
         (v - intrinsics[1, 2]) * z / intrinsics[1, 1], z),
        axis=-1,
    )
    target_points = camera_points @ transform[:3, :3].T + transform[:3, 3]
    inside = np.all(target_points >= lower, axis=-1) & np.all(target_points <= upper, axis=-1)
    retained = valid & inside & ~excluded
    output = np.zeros(depth.shape, dtype=np.float32)
    output[retained] = depth_m[retained]

    reasons = np.full(depth.shape, FILTER_RAW_INVALID, dtype=np.uint8)
    reasons[finite_positive & ~valid] = FILTER_DEPTH_RANGE
    reasons[valid & ~inside] = FILTER_WORKSPACE
    # Explicit masks have the highest diagnostic priority. Dynamic is drawn
    # first so a robot mask wins if both masks cover the same pixel. This is
    # diagnostic precedence, independent of whether D435 supplied depth there.
    for label in ("dynamic", "robot"):
        for supplied_label, mask in labelled_masks:
            if supplied_label == label:
                reasons[mask] = (FILTER_ROBOT if label == "robot"
                                 else FILTER_DYNAMIC)
    reasons[retained] = FILTER_RETAINED
    counts = {
        name: int(np.count_nonzero(reasons == code))
        for code, name in FILTER_REASON_NAMES.items()
    }
    return PreparedDepth(
        depth_m=np.ascontiguousarray(output),
        input_valid_pixels=input_valid_pixels,
        retained_pixels=int(retained.sum()),
        explicitly_excluded_pixels=int(np.count_nonzero(valid & excluded)),
        filter_reasons=np.ascontiguousarray(reasons),
        reason_counts=counts,
    )


def normalize_sdf_query(points, radii_m, max_queries: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not 0 < len(xyz) <= int(max_queries):
        raise ValueError("SDF query points must be an Nx3 array within the configured limit")
    if not np.all(np.isfinite(xyz)):
        raise ValueError("SDF query points must be finite")
    radii = np.asarray(radii_m, dtype=np.float32)
    if radii.size == 0:
        radii = np.zeros(len(xyz), dtype=np.float32)
    if radii.shape != (len(xyz),) or not np.all(np.isfinite(radii)) or np.any(radii < 0.0):
        raise ValueError("radii_m must be empty or contain one finite nonnegative value per point")
    return np.ascontiguousarray(xyz), np.ascontiguousarray(radii)
