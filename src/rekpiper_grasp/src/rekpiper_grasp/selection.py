"""Pure AnyGrasp filtering used by the ROS selector and software replay."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraspBinding:
    rigid_group_id: int
    object_uuid: str
    session_id: str
    program_sha256: str
    snapshot_id: str
    map_generation_uuid: str
    stage_index: int
    grasp_attempt: int


def inference_request_key(binding):
    """One AnyGrasp inference is legal for each immutable attempt binding."""
    return (
        str(binding.session_id), str(binding.program_sha256),
        str(binding.snapshot_id), str(binding.map_generation_uuid),
        int(binding.stage_index), int(binding.grasp_attempt),
        int(binding.rigid_group_id), str(binding.object_uuid),
    )


def horizon_requests_grasp(horizon, binding):
    """Request a target before planning; this horizon never authorizes motion."""
    return bool(
        horizon is not None and horizon.valid
        and not horizon.authorized
        and horizon.status == "grasp_target_pending"
        and horizon.session_id == binding.session_id
        and horizon.program_sha256 == binding.program_sha256
        and horizon.snapshot_id == binding.snapshot_id
        and horizon.map_generation_uuid == binding.map_generation_uuid
        and int(horizon.stage_index) == int(binding.stage_index))


def select_keypoint_candidate(candidates, anchor, binding,
                              maximum_distance_m=0.10):
    """Return the closest fully audited native AnyGrasp candidate or None."""
    anchor = np.asarray(anchor, dtype=float)
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        raise ValueError("anchor must contain three finite coordinates")
    eligible = []
    for item in candidates:
        native = str(item.candidate_origin).lower() in (
            "anygrasp", "native_anygrasp", "anygrasp_sdk",
            "licensed_anygrasp")
        safe = (item.perception_valid and item.planning_safe and item.ik_ok
                and item.robot_collision_free and item.esdf_clear
                and item.contact_membership_ok and item.finger_collision_free
                and item.approach_clear and not item.rejection_reasons)
        bound = (
            int(item.rigid_group_id) == int(binding.rigid_group_id)
            and item.object_uuid == binding.object_uuid
            and item.session_id == binding.session_id
            and item.program_sha256 == binding.program_sha256
            and item.snapshot_id == binding.snapshot_id
            and item.map_generation_uuid == binding.map_generation_uuid
            and int(item.stage_index) == int(binding.stage_index)
            and int(item.grasp_attempt) == int(binding.grasp_attempt))
        point = np.asarray([item.tcp_position.x, item.tcp_position.y,
                            item.tcp_position.z], dtype=float)
        distance = float(np.linalg.norm(point - anchor))
        if (native and safe and bound and np.isfinite(distance)
                and distance <= float(maximum_distance_m)):
            eligible.append((distance,
                             float(getattr(item,'joint_motion_cost',float('inf'))),
                             -float(item.network_score),
                             item.candidate_id, item))
    return None if not eligible else min(eligible, key=lambda value: value[:4])[4]
