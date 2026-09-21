"""AnyGrasp-to-Piper pose conversion used by the ReKep grasp stage."""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .anygrasp_adapter import CameraGrasp


@dataclass
class BaseGraspCandidate:
    candidate_id: str
    interaction_region_id: str
    part_name: str
    grasp_pose: np.ndarray
    pregrasp_pose: np.ndarray
    tcp_position: np.ndarray
    approach_axis_base: np.ndarray
    closing_axis_base: np.ndarray
    predicted_width_m: float
    suggested_preopen_width_m: float
    insertion_depth_m: float
    network_score: float
    source_cameras: List[str]
    candidate_origin: str = "licensed_anygrasp"
    part_membership_ok: bool = False
    contact_membership_ok: bool = False
    contact_points_base: List[List[float]] = field(default_factory=list)
    width_ok: bool = False
    finger_collision_free: bool = False
    approach_clear: bool = False
    ik_ok: bool = False
    robot_collision_free: bool = False
    esdf_clear: bool = False
    cross_view_consistent: bool = False
    perception_valid: bool = False
    planning_safe: bool = False
    planning_authorized: bool = False
    rejection_reasons: List[str] = field(default_factory=list)


def _finite_transform(value, name):
    transform = np.asarray(value, dtype=float)
    if (transform.shape != (4, 4) or not np.all(np.isfinite(transform))
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7)):
        raise ValueError("{} must be a finite homogeneous transform".format(name))
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or np.linalg.det(rotation) < 0.999):
        raise ValueError("{} rotation is invalid".format(name))
    return transform


def anygrasp_to_piper_pose(
        grasp: CameraGrasp, base_from_camera: np.ndarray,
        piper_from_anygrasp_rotation=np.array(
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        pregrasp_distance_m=0.0, preopen_margin_m=0.010,
        physical_opening_m=0.070,
        candidate_id="grasp", interaction_region_id="source",
        part_name="unknown"):
    """Map AnyGrasp axes to Piper's TCP, with no default approach offset."""
    base_from_camera = _finite_transform(base_from_camera, "base_from_camera")
    mapping = np.asarray(piper_from_anygrasp_rotation, dtype=float)
    if (mapping.shape != (3, 3) or not np.all(np.isfinite(mapping))
            or not np.allclose(mapping.T @ mapping, np.eye(3), atol=1e-5)
            or np.linalg.det(mapping) < 0.999):
        raise ValueError("Piper/AnyGrasp rotation mapping is invalid")
    rotation_camera = np.asarray(grasp.rotation, dtype=float)
    translation_camera = np.asarray(grasp.translation, dtype=float)
    if (rotation_camera.shape != (3, 3) or translation_camera.shape != (3,)
            or not np.all(np.isfinite(rotation_camera))
            or not np.all(np.isfinite(translation_camera))):
        raise ValueError("AnyGrasp pose is invalid")
    tcp_camera = translation_camera + float(grasp.depth_m) * rotation_camera[:, 0]
    tcp_base = base_from_camera[:3, :3] @ tcp_camera + base_from_camera[:3, 3]
    rotation_base = base_from_camera[:3, :3] @ rotation_camera
    approach_base = rotation_base[:, 0]
    closing_base = rotation_base[:, 1]
    grasp_pose = np.eye(4)
    grasp_pose[:3, :3] = rotation_base @ mapping.T
    grasp_pose[:3, 3] = tcp_base
    pregrasp_pose = grasp_pose.copy()
    pregrasp_pose[:3, 3] -= float(pregrasp_distance_m) * approach_base
    width = float(grasp.width_m)
    preopen = min(float(physical_opening_m), width + float(preopen_margin_m))
    result = BaseGraspCandidate(
        candidate_id=str(candidate_id),
        interaction_region_id=str(interaction_region_id),
        part_name=str(part_name), grasp_pose=grasp_pose,
        pregrasp_pose=pregrasp_pose, tcp_position=tcp_base,
        approach_axis_base=approach_base, closing_axis_base=closing_base,
        predicted_width_m=width, suggested_preopen_width_m=preopen,
        insertion_depth_m=float(grasp.depth_m),
        network_score=float(grasp.score),
        source_cameras=[str(grasp.source_camera)],
        candidate_origin=str(grasp.candidate_origin))
    result.width_ok = bool(
        0.0 < width <= float(physical_opening_m)
        and preopen <= float(physical_opening_m))
    if not result.width_ok:
        result.rejection_reasons.append("width_limit_exceeded")
    return result
