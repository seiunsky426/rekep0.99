"""Real-Piper adapters around the pinned official ReKep core."""

from .upstream import (
    EXPECTED_OFFICIAL_COMMIT, OfficialCoreModules, load_official_core,
    resolve_official_root)
from .official_program import (
    OfficialConstraintProgram, OfficialProgramError,
    build_official_python_prompt, export_stage_constraint_texts,
    compute_program_sha256,
    generate_official_program, load_official_stage_constraints,
    parse_official_program, save_official_program,
    validate_official_program_numerics)
from .realtime_planner import (
    PersistentReKepPlanner, PlanningGeneration, RealtimePlanningError,
    RealtimePlanningRequest, RealtimePlanningResult)
from .trajectory_audit import (
    TrajectoryAuditError, audit_joint_path, sample_sdf_nearest)
from .paper_real_solver import (
    PaperRealPathSolver, PaperRealSubgoalSolver, PaperRealWeights,
    endpoint_collision_mask, table_penetration_cost)
from .solver_acceptance import (
    SolverAcceptanceError, calibrate_replay_report, load_acceptance,
    load_workspace_table_height, validate_acceptance)

__all__ = [
    "EXPECTED_OFFICIAL_COMMIT", "OfficialCoreModules", "load_official_core",
    "resolve_official_root", "OfficialConstraintProgram",
    "OfficialProgramError", "build_official_python_prompt",
    "compute_program_sha256",
    "export_stage_constraint_texts", "generate_official_program",
    "load_official_stage_constraints", "parse_official_program",
    "save_official_program", "validate_official_program_numerics",
    "PersistentReKepPlanner", "PlanningGeneration", "RealtimePlanningError",
    "RealtimePlanningRequest", "RealtimePlanningResult",
    "TrajectoryAuditError", "audit_joint_path", "sample_sdf_nearest",
    "PaperRealPathSolver", "PaperRealSubgoalSolver", "PaperRealWeights",
    "endpoint_collision_mask", "table_penetration_cost",
    "SolverAcceptanceError", "calibrate_replay_report", "load_acceptance",
    "load_workspace_table_height", "validate_acceptance",
]
