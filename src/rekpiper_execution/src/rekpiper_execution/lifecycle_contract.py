"""Pure fail-closed identity contract for one ReKep grasp event."""


class LifecycleBindingError(ValueError):
    pass


def validate_grasp_candidate_binding(
        candidate, rigid_group_id, object_uuid, session_id,
        program_sha256, snapshot_id, map_generation_uuid, stage_index,
        grasp_attempt):
    """Reject a candidate if any immutable lifecycle generation differs."""
    checks = {
        "candidate_id": bool(candidate and candidate.candidate_id),
        "planning_safe": bool(candidate and candidate.planning_safe),
        "planning_authorized_false": bool(
            candidate and not candidate.planning_authorized),
        "rigid_group_id": bool(candidate and
            int(candidate.rigid_group_id) == int(rigid_group_id)),
        "object_uuid": bool(candidate and
            candidate.object_uuid == object_uuid),
        "session_id": bool(candidate and candidate.session_id == session_id),
        "program_sha256": bool(candidate and
            candidate.program_sha256 == program_sha256),
        "snapshot_id": bool(candidate and candidate.snapshot_id == snapshot_id),
        "map_generation_uuid": bool(candidate and
            candidate.map_generation_uuid == map_generation_uuid),
        "stage_index": bool(candidate and
            int(candidate.stage_index) == int(stage_index)),
        "grasp_attempt": bool(candidate and
            int(candidate.grasp_attempt) == int(grasp_attempt)),
    }
    failed = tuple(sorted(name for name, valid in checks.items() if not valid))
    if failed:
        raise LifecycleBindingError(
            "grasp_candidate_binding_mismatch:" + ",".join(failed))
    return True


def grasp_batch_matches_binding(
        batch, session_id, program_sha256, snapshot_id,
        map_generation_uuid, stage_index, grasp_attempt):
    """Bind even an empty AnyGrasp result to exactly one inference attempt."""
    return bool(
        batch is not None and not batch.planning_authorized
        and batch.session_id == session_id
        and batch.program_sha256 == program_sha256
        and batch.snapshot_id == snapshot_id
        and batch.map_generation_uuid == map_generation_uuid
        and int(batch.stage_index) == int(stage_index)
        and int(batch.grasp_attempt) == int(grasp_attempt))
