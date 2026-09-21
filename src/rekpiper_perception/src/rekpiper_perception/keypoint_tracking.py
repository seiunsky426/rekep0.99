"""Pure, deterministic primitives for multi-camera ReKep keypoint tracking.

The ROS node owns image conversion and DINO inference.  This module keeps the
matching, rigid propagation and state handling independent from ROS so they can
be replayed and unit tested without a camera or GPU.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


OBSERVED = "OBSERVED"
PROPAGATED = "PROPAGATED"
STALE_OCCLUDED = "STALE_OCCLUDED"
LOST = "LOST"


def reference_frames_match_snapshot(snapshot_stamp_ns, frame_stamps_ns,
                                    maximum_offset_s=0.15):
    """Require the fixed DINO reference views to belong to the snapshot.

    The snapshot only records cameras used for initial keypoint proposal; the
    tracker therefore performs this independent timestamp contract for every
    camera contributing to the fixed multi-camera descriptor.
    """
    snapshot = int(snapshot_stamp_ns)
    frames = [int(value) for value in frame_stamps_ns]
    maximum = int(float(maximum_offset_s) * 1e9)
    if snapshot <= 0 or not frames or maximum <= 0:
        return False
    return all(value > 0 and abs(value - snapshot) <= maximum
               for value in frames)


def capture_stamp_is_fresh(now_ns, capture_ns, maximum_age_s=0.15):
    """Validate age from sensor acquisition time, rejecting future stamps."""
    age = int(now_ns) - int(capture_ns)
    return 0 <= age <= int(float(maximum_age_s) * 1e9)


def nearest_reference_frames(frame_buffers, snapshot_stamp_ns, maximum_offset_s):
    """Select acquisition-time references even when SAM finishes seconds later."""
    if any(not frames for frames in frame_buffers.values()):
        return None
    selected = {camera: min(frames, key=lambda frame:
                            abs(frame[0].to_nsec() - snapshot_stamp_ns))
                for camera, frames in frame_buffers.items()}
    if not reference_frames_match_snapshot(
            snapshot_stamp_ns, [frame[0].to_nsec() for frame in selected.values()],
            maximum_offset_s):
        return None
    return selected


def tensor_reference_descriptor(feature_maps, point_maps, center, radius_m):
    """Gather only a keypoint's small reference neighborhood from GPU maps."""
    import torch
    samples = []
    for features, points in zip(feature_maps, point_maps):
        selected = np.isfinite(points).all(axis=2)
        selected &= np.linalg.norm(points - np.asarray(center), axis=2) <= radius_m
        if selected.any():
            samples.append(features[torch.as_tensor(selected, device=features.device)].float())
    if not samples:
        return None
    value = torch.cat(samples).mean(dim=0)
    return value / value.norm().clamp_min(1e-8)


def tensor_feature_observation(reference, feature_maps, point_maps,
                               similarity_threshold=0.60, top_k=100,
                               mad_multiplier=2.0, masks=None):
    """Global top-k across cameras, transferring only candidates to the CPU.

    Optional object masks constrain identity; empty/occluded masks yield no
    observation instead of matching a visually similar neighboring object.
    """
    import torch
    candidates, scores = [], []
    for index, (features, points) in enumerate(zip(feature_maps, point_maps)):
        valid = np.isfinite(points).all(axis=2)
        if masks is not None:
            valid &= masks[index]
        if not valid.any():
            continue
        selected = features[torch.as_tensor(valid, device=features.device)].float()
        similarity = (selected @ reference.float()) / selected.norm(dim=1).clamp_min(1e-8)
        value, indices = torch.topk(similarity, min(int(top_k), len(similarity)))
        keep = value >= similarity_threshold
        indices, value = indices[keep].cpu().numpy(), value[keep].cpu().numpy()
        candidates.append(points[valid][indices])
        scores.append(value)
    if not scores or not any(len(score) for score in scores):
        return None
    scores = np.concatenate(scores)
    points = np.concatenate(candidates)
    order = np.argsort(scores)[::-1][:int(top_k)]
    points, scores = points[order], scores[order]
    distances = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    middle = np.median(distances)
    mad = np.median(np.abs(distances - middle))
    keep = distances <= middle + mad_multiplier * max(1.4826 * mad, 1e-6)
    points, scores = points[keep], scores[keep]
    covariance = (np.cov(points, rowvar=False) / len(points) if len(points) > 1
                  else np.eye(3) * 1e-6)
    return Observation(np.median(points, axis=0), covariance,
                       float(np.mean(scores)), len(points))


@dataclass(frozen=True)
class Observation:
    point: np.ndarray
    covariance: np.ndarray
    confidence: float
    matches: int


@dataclass
class LandmarkTrack:
    keypoint_id: int
    name: str
    group_id: int
    reference_point: np.ndarray
    reference_feature: np.ndarray
    reference_pixel_rc: Tuple[int, int]
    history_size: int = 10
    last_point: Optional[np.ndarray] = None
    last_covariance: Optional[np.ndarray] = None
    state: str = STALE_OCCLUDED
    stale_frames: int = 0
    history: deque = field(default_factory=deque)

    def __post_init__(self):
        self.reference_point = _vector3(self.reference_point, "reference_point")
        self.reference_feature = _unit_vector(self.reference_feature)
        self.reference_pixel_rc = tuple(int(value) for value in self.reference_pixel_rc)
        self.history = deque(maxlen=int(self.history_size))

    def update(self, observation: Observation, state: str) -> None:
        if state not in (OBSERVED, PROPAGATED):
            raise ValueError("track update requires an observed or propagated state")
        self.history.append(_vector3(observation.point, "observation_point"))
        points = np.stack(tuple(self.history))
        self.last_point = np.mean(points, axis=0)
        self.last_covariance = _matrix3(observation.covariance, "observation_covariance")
        self.state = state
        self.stale_frames = 0

    def mark_stale(self, covariance_growth_m2: float) -> None:
        self.state = STALE_OCCLUDED
        self.stale_frames += 1
        if self.last_covariance is None:
            self.last_covariance = np.eye(3, dtype=np.float64) * float(covariance_growth_m2)
        else:
            self.last_covariance = self.last_covariance + np.eye(3) * float(covariance_growth_m2)


def _vector3(value: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("{} must be finite XYZ".format(name))
    return result


def _matrix3(value: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite 3x3 matrix".format(name))
    return result


def _unit_vector(value: Iterable[float]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(result))
    if result.size == 0 or not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("feature must have non-zero finite norm")
    return result / norm


def robust_feature_observation(
    reference_feature: np.ndarray,
    current_features: np.ndarray,
    current_points: np.ndarray,
    similarity_threshold: float = 0.60,
    top_k: int = 100,
    mad_multiplier: float = 2.0,
) -> Optional[Observation]:
    """Match a fixed DINO descriptor against a current 3-D feature map.

    The implementation mirrors the ReKep appendix: select high cosine matches,
    retain the top candidates, then reject spatial outliers by median deviation.
    """
    reference = _unit_vector(reference_feature)
    features = np.asarray(current_features, dtype=np.float64)
    points = np.asarray(current_points, dtype=np.float64)
    if features.ndim != 2 or points.shape != (features.shape[0], 3):
        raise ValueError("features and points must be NxD and Nx3")
    if features.shape[1] != reference.size:
        raise ValueError("reference/current feature dimensions differ")
    valid = np.all(np.isfinite(features), axis=1) & np.all(np.isfinite(points), axis=1)
    if not np.any(valid):
        return None
    features = features[valid]
    points = points[valid]
    norms = np.linalg.norm(features, axis=1)
    valid_norms = norms > 1e-12
    if not np.any(valid_norms):
        return None
    features, points, norms = features[valid_norms], points[valid_norms], norms[valid_norms]
    similarity = (features @ reference) / norms
    candidates = np.flatnonzero(similarity >= float(similarity_threshold))
    if candidates.size == 0:
        return None
    order = candidates[np.argsort(similarity[candidates])[::-1][:int(top_k)]]
    candidate_points = points[order]
    center = np.median(candidate_points, axis=0)
    distances = np.linalg.norm(candidate_points - center, axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    tolerance = median_distance + float(mad_multiplier) * max(1.4826 * mad, 1e-6)
    keep = distances <= tolerance
    inliers = candidate_points[keep]
    if len(inliers) == 0:
        return None
    point = np.median(inliers, axis=0)
    if len(inliers) == 1:
        covariance = np.eye(3, dtype=np.float64) * 1e-6
    else:
        covariance = np.cov(inliers, rowvar=False, bias=False) / float(len(inliers))
    confidence = float(np.mean(similarity[order][keep]))
    return Observation(point, covariance, confidence, int(len(inliers)))


def multicamera_reference_descriptor(feature_maps, point_maps, center,
                                     radius_m=0.020):
    """Create one fixed descriptor from all camera samples near a 3-D K."""
    if len(feature_maps) != len(point_maps) or not feature_maps:
        raise ValueError("feature_maps and point_maps must be non-empty peers")
    center = _vector3(center, "reference_center")
    samples = []
    feature_width = None
    for features, points in zip(feature_maps, point_maps):
        features = np.asarray(features, dtype=np.float64)
        points = np.asarray(points, dtype=np.float64)
        if features.ndim != 3 or points.shape != features.shape[:2] + (3,):
            raise ValueError("each feature/point map must be HxWxD and HxWx3")
        if feature_width is None:
            feature_width = features.shape[-1]
        elif feature_width != features.shape[-1]:
            raise ValueError("camera feature dimensions differ")
        valid = np.all(np.isfinite(points), axis=2)
        valid &= np.all(np.isfinite(features), axis=2)
        valid &= np.linalg.norm(points - center.reshape(1, 1, 3), axis=2) \
            <= float(radius_m)
        if np.any(valid):
            samples.append(features[valid])
    if not samples:
        return None
    return _unit_vector(np.concatenate(samples, axis=0).mean(axis=0))


def multicamera_global_observation(reference_feature, feature_maps,
                                   point_maps, similarity_threshold=0.60,
                                   top_k=100, mad_multiplier=2.0):
    """Run one global top-k search over every valid pixel in every camera."""
    if len(feature_maps) != len(point_maps) or not feature_maps:
        raise ValueError("feature_maps and point_maps must be non-empty peers")
    features = []
    points = []
    for feature_map, point_map in zip(feature_maps, point_maps):
        feature_map = np.asarray(feature_map)
        point_map = np.asarray(point_map)
        if feature_map.ndim != 3 or point_map.shape != feature_map.shape[:2] + (3,):
            raise ValueError("each feature/point map must be HxWxD and HxWx3")
        features.append(feature_map.reshape(-1, feature_map.shape[-1]))
        points.append(point_map.reshape(-1, 3))
    return robust_feature_observation(
        reference_feature, np.concatenate(features, axis=0),
        np.concatenate(points, axis=0), similarity_threshold, top_k,
        mad_multiplier)


def anchor_points_to_gripper(points_base, base_from_gripper):
    points = np.asarray(points_base, dtype=np.float64)
    transform = np.asarray(base_from_gripper, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3
            or transform.shape != (4, 4)
            or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(transform))):
        raise ValueError("attached keypoint anchor inputs are invalid")
    gripper_from_base = np.linalg.inv(transform)
    return (points @ gripper_from_base[:3, :3].T
            + gripper_from_base[:3, 3])


def attached_points_in_base(points_gripper, base_from_gripper):
    points = np.asarray(points_gripper, dtype=np.float64)
    transform = np.asarray(base_from_gripper, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3
            or transform.shape != (4, 4)
            or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(transform))):
        raise ValueError("attached keypoint propagation inputs are invalid")
    return points @ transform[:3, :3].T + transform[:3, 3]
