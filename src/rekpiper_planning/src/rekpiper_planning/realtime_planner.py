"""Persistent short-horizon adapter around the pristine official ReKep solvers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .official_program import (
    load_official_stage_constraints,
    parse_official_program,
)
from .upstream import load_official_core
from .solver_deadline import bounded_solver_call
from .continuous_ik import densify_joint_path
from .paper_real_solver import (
    PaperRealPathSolver, PaperRealSubgoalSolver, PaperRealWeights)


class RealtimePlanningError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlanningGeneration:
    snapshot_id: str
    snapshot_layout_hash: str
    map_uuid: str
    program_sha256: str
    state_sequence: int


@dataclass(frozen=True)
class RealtimePlanningRequest:
    generation: PlanningGeneration
    program_directory: str
    stage: int
    ee_pose: np.ndarray
    joint_positions: np.ndarray
    keypoints: np.ndarray
    rigid_group_ids: np.ndarray
    held_keypoint: int
    sdf_voxels: np.ndarray
    collision_points: np.ndarray
    grasping_cost_fn: Callable[[int], float]
    is_grasp_stage: bool = False
    joint_validity_fn: Optional[Callable] = None
    joint_fallback_fn: Optional[Callable] = None
    solver_call_timeouts: Optional[Tuple[float, float]] = None
    grasp_target_pose: Optional[np.ndarray] = None


@dataclass(frozen=True)
class RealtimePlanningResult:
    stage: int
    semantic_subgoal_pose: np.ndarray
    target_pose: np.ndarray
    cartesian_path: np.ndarray
    joint_path: np.ndarray
    subgoal_values: Tuple[float, ...]
    path_max_values: Tuple[float, ...]
    planning_latency_s: float
    subgoal_from_scratch: bool
    path_from_scratch: bool
    generation: PlanningGeneration
    diagnostics: Optional[dict] = None


def _without_warmup(solver_type, config, ik_solver, reset_joints):
    """Construct an official solver while avoiding its random synthetic warmup."""
    solver = solver_type.__new__(solver_type)
    solver.config = dict(config)
    solver.ik_solver = ik_solver
    solver.reset_joint_pos = np.asarray(reset_joints, dtype=float)
    solver.last_opt_result = None
    return solver


def grasp_execution_pose(semantic_pose, transform_utils, grasp_depth_m=0.10):
    """Apply the official environment's half-depth retreat along EE +X/-X."""
    semantic = np.asarray(semantic_pose, dtype=float)
    if (semantic.shape != (7,) or not np.all(np.isfinite(semantic))
            or not np.isfinite(grasp_depth_m) or grasp_depth_m <= 0.0):
        raise RealtimePlanningError("invalid semantic grasp target")
    result = semantic.copy()
    matrix = transform_utils.pose2mat([semantic[:3], semantic[3:]])
    result[:3] += matrix[:3, :3] @ np.asarray(
        [-float(grasp_depth_m) / 2.0, 0.0, 0.0])
    return result


def grasp_target_constraints(subgoals, paths, target_pose):
    """Pin the grasp stage to the detector TCP instead of EE-on-keypoint.

    The program still chooses the grasp keypoint/object. Its single grasp
    subgoal is instantiated by the bound detector pose; other stages retain
    their original constraints. Orientation is pinned by the PathSolver goal
    and checked against measured FK before closing.
    """
    if target_pose is None:
        return subgoals, paths
    if len(subgoals) != 1 or paths:
        raise RealtimePlanningError("invalid detector grasp stage constraints")
    target = np.asarray(target_pose, dtype=float)
    if (target.shape != (7,) or not np.all(np.isfinite(target))
            or not np.isclose(np.linalg.norm(target[3:]), 1., atol=1e-5)):
        raise RealtimePlanningError("invalid detector grasp TCP pose")
    position = target[:3].copy()
    return [lambda ee, _keypoints: float(np.linalg.norm(ee - position))], paths


class PersistentReKepPlanner:
    """Keep one SubgoalSolver and PathSolver alive for every active stage.

    A stage's first call follows the official global Dual Annealing + SLSQP
    branch. Later calls reuse ``last_opt_result`` and therefore take the
    official SLSQP warm-start branch. A changed program or keypoint layout
    clears every cached solver; a new map UUID also invalidates all stages.
    """

    def __init__(self, subgoal_config: dict, path_config: dict, ik_solver,
                 reset_joint_positions: Sequence[float], modules=None,
                 maximum_position_step_m: float = 0.015,
                 maximum_rotation_step_rad: float = 0.10,
                 maximum_joint_jump_rad: float = 0.35,
                 continuity_guard_enabled: bool = True,
                 maximum_warm_target_jump_m: float = 0.20,
                 grasp_depth_m: float = 0.10,
                 official_interpolate_position_step_m: float = 0.05,
                 official_interpolate_rotation_step_rad: float = 0.34,
                 solver_profile: str = "official_exact",
                 paper_real_weights: Optional[dict] = None,
                 table_height_m: Optional[float] = None,
                 robot_collision_points_fn=None):
        self.subgoal_config = dict(subgoal_config)
        self.path_config = dict(path_config)
        self.ik_solver = ik_solver
        self.reset_joints = np.asarray(reset_joint_positions, dtype=float)
        self.modules = modules or load_official_core()
        self.maximum_position_step_m = float(maximum_position_step_m)
        self.maximum_rotation_step_rad = float(maximum_rotation_step_rad)
        self.maximum_joint_jump_rad = float(maximum_joint_jump_rad)
        self.continuity_guard_enabled = bool(continuity_guard_enabled)
        self.maximum_warm_target_jump_m = float(maximum_warm_target_jump_m)
        self.grasp_depth_m = float(grasp_depth_m)
        self.official_interpolate_position_step_m = float(
            official_interpolate_position_step_m)
        self.official_interpolate_rotation_step_rad = float(
            official_interpolate_rotation_step_rad)
        self.solver_profile = str(solver_profile)
        if self.solver_profile not in ("official_exact", "paper_real"):
            raise ValueError("solver_profile must be official_exact or paper_real")
        self.paper_real_weights = None
        self.table_height_m = table_height_m
        self.robot_collision_points_fn = robot_collision_points_fn
        if self.solver_profile == "paper_real":
            self.paper_real_weights = PaperRealWeights.from_mapping(
                paper_real_weights or {})
            if table_height_m is None or not np.isfinite(float(table_height_m)):
                raise ValueError("paper_real requires an accepted table height")
            if robot_collision_points_fn is None:
                raise ValueError("paper_real requires full-arm collision sampling")
            self.table_height_m = float(table_height_m)
        self._solvers: Dict[int, tuple] = {}
        self._binding: Optional[Tuple[str, str]] = None
        self._last_target: Dict[int, np.ndarray] = {}
        self._active_stage: Optional[int] = None

    def invalidate(self) -> None:
        self._solvers.clear()
        self._last_target.clear()
        self._binding = None
        self._active_stage = None

    def _bind(self, generation: PlanningGeneration) -> None:
        binding = (generation.program_sha256, generation.snapshot_layout_hash,
                   generation.map_uuid)
        if self._binding != binding:
            self.invalidate()
            self._binding = binding

    def _solver_pair(self, stage: int, joint_positions: np.ndarray):
        if stage not in self._solvers:
            reset = self.reset_joints
            if reset.size < joint_positions.size + 1:
                reset = np.r_[joint_positions, 0.0]
            if self.solver_profile == "paper_real":
                self._solvers[stage] = (
                    PaperRealSubgoalSolver(
                        self.subgoal_config, self.ik_solver, reset,
                        self.modules, self.paper_real_weights),
                    PaperRealPathSolver(
                        self.path_config, self.ik_solver, reset,
                        self.modules, self.paper_real_weights,
                        self.table_height_m,
                        self.robot_collision_points_fn),
                )
            else:
                self._solvers[stage] = (
                    _without_warmup(self.modules.subgoal_solver.SubgoalSolver,
                                    self.subgoal_config, self.ik_solver, reset),
                    _without_warmup(self.modules.path_solver.PathSolver,
                                    self.path_config, self.ik_solver, reset),
                )
        return self._solvers[stage]

    def _enter_stage(self, stage: int) -> None:
        """Restore official first-iteration semantics on every stage entry."""
        stage = int(stage)
        if self._active_stage == stage:
            return
        if stage in self._solvers:
            for solver in self._solvers[stage]:
                if hasattr(solver, "reset_history"):
                    solver.reset_history()
                else:
                    solver.last_opt_result = None
        self._last_target.pop(stage, None)
        self._active_stage = stage

    @staticmethod
    def _validate(request: RealtimePlanningRequest):
        ee = np.asarray(request.ee_pose, dtype=float)
        joints = np.asarray(request.joint_positions, dtype=float).reshape(-1)
        keypoints = np.asarray(request.keypoints, dtype=float)
        groups = np.asarray(request.rigid_group_ids, dtype=np.int64).reshape(-1)
        sdf = np.asarray(request.sdf_voxels, dtype=float)
        collision = np.asarray(request.collision_points, dtype=float)
        if ee.shape != (7,) or keypoints.ndim != 2 or keypoints.shape[1] != 3:
            raise RealtimePlanningError("invalid EE pose or keypoint shape")
        if len(keypoints) == 0 or groups.shape != (len(keypoints),):
            raise RealtimePlanningError("keypoint rigid-group layout is invalid")
        if joints.size < 6 or sdf.ndim != 3 or min(sdf.shape) < 2:
            raise RealtimePlanningError("joint state or SDF is invalid")
        if collision.ndim != 2 or collision.shape[1] != 3 or not len(collision):
            raise RealtimePlanningError("collision sample cloud is unavailable")
        if any(not np.all(np.isfinite(x)) for x in (ee, joints, keypoints, sdf, collision)):
            raise RealtimePlanningError("planning generation contains NaN or Inf")
        if not request.generation.map_uuid or not request.generation.snapshot_id:
            raise RealtimePlanningError("unbound snapshot or SDF generation")
        if request.is_grasp_stage != (request.grasp_target_pose is not None):
            raise RealtimePlanningError("grasp stage requires a bound detector TCP target")
        return ee, joints, keypoints, groups, sdf, collision

    def _movable_mask(self, groups: np.ndarray, held_keypoint: int) -> np.ndarray:
        movable = np.zeros(len(groups) + 1, dtype=bool)
        movable[0] = True  # the first official keypoint is the end effector
        if held_keypoint >= 0:
            if held_keypoint >= len(groups):
                raise RealtimePlanningError("held keypoint index is outside the snapshot")
            movable[1:] = groups == groups[held_keypoint]
        return movable

    def _safety_resample(self, controls: np.ndarray) -> np.ndarray:
        dense = []
        for index, (start, end) in enumerate(zip(controls[:-1], controls[1:])):
            count = self.modules.utils.get_linear_interpolation_steps(
                start, end, self.maximum_position_step_m,
                self.maximum_rotation_step_rad)
            segment = self.modules.utils.linear_interpolate_poses(start, end, count)
            dense.extend(segment if index == 0 else segment[1:])
        return np.asarray(dense, dtype=float)

    def _official_spline_path(self, controls: np.ndarray) -> np.ndarray:
        """Apply the byte-exact upstream path post-processing semantics.

        The official controller chooses the number of output poses from the
        start/end displacement and then fits one position/rotation spline
        through every optimized control point.  Piper adds a denser sampling
        pass afterwards solely for IK and collision auditing.
        """
        count = self.modules.utils.get_linear_interpolation_steps(
            controls[0], controls[-1],
            self.official_interpolate_position_step_m,
            self.official_interpolate_rotation_step_rad)
        official = self.modules.utils.spline_interpolate_poses(controls, count)
        return self._safety_resample(np.asarray(official, dtype=float))

    def _transformed(self, pose, current_ee, full_keypoints, movable):
        current = self.modules.transform_utils.pose2mat(
            [current_ee[:3], current_ee[3:]])
        target = self.modules.transform_utils.pose2mat([pose[:3], pose[3:]])
        centered = self.modules.utils.transform_keypoints(
            np.linalg.inv(current), full_keypoints, movable)
        return self.modules.utils.transform_keypoints(target, centered, movable)

    def _check_constraints(self, ee, full_keypoints, movable, semantic_target, dense,
                           subgoals, paths):
        tolerance = float(self.path_config.get("constraint_tolerance", 0.0001))
        target_points = self._transformed(
            semantic_target, ee, full_keypoints, movable)
        subgoal_values = tuple(float(fn(target_points[0], target_points[1:]))
                               for fn in subgoals)
        if any(not np.isfinite(v) or v > tolerance for v in subgoal_values):
            raise RealtimePlanningError("official subgoal result violates a constraint")
        maxima = [-np.inf] * len(paths)
        for pose in dense:
            transformed = self._transformed(pose, ee, full_keypoints, movable)
            for index, function in enumerate(paths):
                value = float(function(transformed[0], transformed[1:]))
                if not np.isfinite(value) or value > tolerance:
                    raise RealtimePlanningError(
                        "dense path violates constraint {}".format(index + 1))
                maxima[index] = max(maxima[index], value)
        return subgoal_values, tuple(maxima)

    def _joint_path(self, dense: np.ndarray, current_joints: np.ndarray) -> np.ndarray:
        matrices = self.modules.transform_utils.convert_pose_quat2mat(dense)
        if hasattr(self.ik_solver, "validate_pose_sequence"):
            report = self.ik_solver.validate_pose_sequence(matrices, current_joints)
            if not report.get("valid", False):
                self._last_ik_report = report
                raise RealtimePlanningError(
                    "Piper IK sequence failed at {}: {}".format(
                        report.get('failed_pose_index'), report.get("reason", "unknown")))
            result = np.asarray([step["joint_positions"] for step in report["steps"]],
                                dtype=float)
        else:
            result = []
            seed = current_joints
            for matrix in matrices:
                solved = self.ik_solver.solve(matrix, initial_joint_pos=seed)
                if not solved.success:
                    raise RealtimePlanningError("Piper IK sequence failed")
                seed = np.asarray(solved.cspace_position[:6], dtype=float)
                result.append(seed)
            result = np.asarray(result, dtype=float)
        if len(result) > 1 and np.max(np.abs(np.diff(result, axis=0))) > self.maximum_joint_jump_rad:
            raise RealtimePlanningError("Piper IK sequence contains a joint jump")
        return densify_joint_path(result)

    def solve(self, request: RealtimePlanningRequest) -> RealtimePlanningResult:
        started = time.monotonic()
        ee, joints, keypoints, groups, sdf, collision = self._validate(request)
        self._bind(request.generation)
        self._enter_stage(request.stage)
        root = Path(request.program_directory)
        try:
            metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            program = parse_official_program(
                (root / "program.py").read_text(encoding="utf-8"),
                str(metadata["instruction"]), len(keypoints))
        except (OSError, KeyError, ValueError) as exc:
            raise RealtimePlanningError("approved ReKep program is unavailable: {}".format(exc)) from exc
        if not 1 <= request.stage <= program.num_stages:
            raise RealtimePlanningError("requested stage is absent from the program")
        if request.is_grasp_stage != (program.grasp_keypoints[request.stage - 1] >= 0):
            raise RealtimePlanningError("grasp target does not match program stage")

        subgoals, paths = load_official_stage_constraints(
            root, request.stage, len(keypoints), request.grasping_cost_fn)
        subgoals, paths = grasp_target_constraints(
            subgoals, paths, request.grasp_target_pose)
        movable = self._movable_mask(groups, request.held_keypoint)
        full_keypoints = np.vstack([ee[:3], keypoints])
        subgoal_solver, path_solver = self._solver_pair(request.stage, joints)
        budgets = request.solver_call_timeouts or (None,None)
        if request.is_grasp_stage:
            semantic_target = np.asarray(request.grasp_target_pose, dtype=float).copy()
            subgoal_global = False
        else:
            subgoal_global = subgoal_solver.last_opt_result is None
            semantic_target, _subgoal_debug = bounded_solver_call(subgoal_solver.solve,budgets[0],
                ee, full_keypoints, movable, subgoals, paths, sdf, collision,
                False, np.r_[joints[:6], 0.0], from_scratch=subgoal_global)
            semantic_target = np.asarray(semantic_target, dtype=float)
        target = semantic_target.copy()

        previous = self._last_target.get(request.stage)
        if (self.continuity_guard_enabled and previous is not None
                and np.linalg.norm(target[:3] - previous[:3]) > self.maximum_warm_target_jump_m):
            raise RealtimePlanningError("warm-start target changed discontinuously")
        path_global = path_solver.last_opt_result is None
        path_error = None
        try:
            controls_tail, _path_debug = bounded_solver_call(path_solver.solve,budgets[1],
                ee, target, full_keypoints, movable, paths, sdf, collision,
                np.r_[joints[:6], 0.0], from_scratch=path_global)
            dense = self._official_spline_path(np.vstack([ee, controls_tail]))
        except (TimeoutError, RuntimeError) as exc:
            if request.joint_fallback_fn is None:
                raise
            path_error = str(exc)
            # This is not an accepted Cartesian shortcut: force audited RRT.
            dense = np.asarray([ee,target])
        subgoal_values, path_values = self._check_constraints(
            ee, full_keypoints, movable, semantic_target, dense, subgoals, paths)
        diagnostics = {'path_backend': 'PathSolver', 'fallback_attempts': [],
                       'subgoal_source': 'bound_anygrasp_tcp' if request.is_grasp_stage
                       else 'SubgoalSolver'}
        tolerance = float(self.path_config.get('constraint_tolerance', .0001))
        def state_valid(q):
            matrix = self.ik_solver.forward(q)
            position, quaternion = self.modules.transform_utils.mat2pose(matrix)
            pose = np.r_[position, quaternion]
            transformed = self._transformed(pose, ee, full_keypoints, movable)
            values = [float(fn(transformed[0], transformed[1:])) for fn in paths]
            if any(not np.isfinite(v) or v > tolerance for v in values):
                return False
            return request.joint_validity_fn is not None and bool(request.joint_validity_fn(q))
        try:
            if path_error is not None:
                raise RealtimePlanningError(path_error)
            joint_path = self._joint_path(dense, joints[:6])
            if request.joint_validity_fn is not None and not all(state_valid(q) for q in joint_path):
                raise RealtimePlanningError('joint_path_state_check_failed')
        except RealtimePlanningError as exc:
            diagnostics['primary_failure'] = str(exc)
            diagnostics['ik'] = getattr(self, '_last_ik_report', {})
            if request.joint_fallback_fn is None or request.joint_validity_fn is None:
                raise
            endpoint = self.modules.transform_utils.pose2mat([target[:3], target[3:]])
            solved = self.ik_solver.solve(endpoint, initial_joint_pos=joints[:6], max_iterations=100)
            if not solved.success:
                raise RealtimePlanningError('fallback_endpoint_ik_failed') from exc
            joint_path = None
            for attempt in range(3):
                try:
                    joint_path = request.joint_fallback_fn(
                        joints[:6], solved.cspace_position[:6], state_valid, 2., attempt)
                    break
                except RuntimeError as error:
                    diagnostics['fallback_attempts'].append(str(error))
            if joint_path is None:
                raise RealtimePlanningError('fallback_exhausted:'+str(diagnostics)) from exc
            diagnostics['path_backend'] = 'OMPL_RRTConnect'
        # Validate actual FK, including inserted IK points / joint-space detours.
        actual_dense = []
        for q in joint_path:
            position, quaternion = self.modules.transform_utils.mat2pose(self.ik_solver.forward(q))
            actual_dense.append(np.r_[position, quaternion])
        dense = np.asarray(actual_dense)
        if request.is_grasp_stage:
            position_error = float(np.linalg.norm(dense[-1, :3] - target[:3]))
            rotation_error = float(2. * np.arccos(np.clip(
                abs(np.dot(dense[-1, 3:], target[3:])), 0., 1.)))
            if position_error > .003 or rotation_error > .03:
                raise RealtimePlanningError('joint_path_does_not_reach_grasp_tcp')
        subgoal_values, path_values = self._check_constraints(
            ee, full_keypoints, movable, semantic_target, dense, subgoals, paths)
        self._last_target[request.stage] = np.asarray(target, dtype=float).copy()
        return RealtimePlanningResult(
            request.stage, semantic_target, np.asarray(target, dtype=float),
            dense, joint_path,
            subgoal_values, path_values, time.monotonic() - started,
            subgoal_global, path_global, request.generation, diagnostics)
