"""Paper-real ReKep objectives layered over the pristine public utilities.

The public ReKep snapshot remains byte-exact.  These solver classes reuse its
geometry and interpolation helpers while implementing the real-system costs
described in paper appendix A.8/A.9.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import dual_annealing, minimize


@dataclass(frozen=True)
class PaperRealWeights:
    subgoal_consistency: float
    path_consistency: float
    table_clearance: float

    @classmethod
    def from_mapping(cls, values):
        try:
            result = cls(
                float(values["subgoal_consistency"]),
                float(values["path_consistency"]),
                float(values["table_clearance"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("paper-real weights are incomplete") from exc
        if any(not np.isfinite(value) or value < 0.0 for value in (
                result.subgoal_consistency, result.path_consistency,
                result.table_clearance)):
            raise ValueError("paper-real weights must be finite and non-negative")
        return result


def endpoint_collision_mask(poses_homo, start_pose, end_pose,
                            exemption_radius_m=0.05):
    """Select samples outside both paper-defined endpoint neighbourhoods."""
    poses = np.asarray(poses_homo, dtype=float)
    start = np.asarray(start_pose, dtype=float).reshape(6)
    end = np.asarray(end_pose, dtype=float).reshape(6)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("path poses must be homogeneous transforms")
    radius = float(exemption_radius_m)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("endpoint exemption radius is invalid")
    positions = poses[:, :3, 3]
    return ((np.linalg.norm(positions - start[:3], axis=1) >= radius)
            & (np.linalg.norm(positions - end[:3], axis=1) >= radius))


def table_penetration_cost(poses_homo, collision_points_centered,
                           table_height_m, utilities):
    """Return summed penetration of EE-attached geometry below the table."""
    if collision_points_centered is None:
        return 0.0
    poses = np.asarray(poses_homo, dtype=float)
    points = np.asarray(collision_points_centered, dtype=float)
    height = float(table_height_m)
    if (poses.ndim != 3 or poses.shape[1:] != (4, 4)
            or points.ndim != 2 or points.shape[1] != 3
            or not np.isfinite(height)):
        raise ValueError("table-clearance inputs are invalid")
    transformed = utilities.batch_transform_points(points, poses)
    return float(np.sum(np.maximum(height - transformed[..., 2], 0.0)))


def robot_table_penetration_cost(poses_homo, ik_solver, initial_joint_pos,
                                 robot_points_fn, table_height_m):
    """Evaluate full-arm samples at every dense pose, not only controls."""
    if robot_points_fn is None:
        raise ValueError("paper-real full-arm table sampler is unavailable")
    height = float(table_height_m)
    if not np.isfinite(height):
        raise ValueError("table height is invalid")
    seed = np.asarray(initial_joint_pos, dtype=float)
    penetration = 0.0
    feasible = []
    for pose in np.asarray(poses_homo, dtype=float):
        result = ik_solver.solve(
            pose, max_iterations=100, initial_joint_pos=seed)
        feasible.append(bool(result.success))
        if not result.success:
            # The regular reachability term and the final dense IK audit also
            # reject this path.  A finite objective penalty keeps scipy able to
            # search away from the failed sample.
            penetration += 3.0
            continue
        seed = np.asarray(result.cspace_position, dtype=float)
        points = robot_points_fn(seed[:6])
        if isinstance(points, tuple):
            points = points[0]
        points = np.asarray(points, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3
                or not len(points) or not np.all(np.isfinite(points))):
            raise ValueError("full-arm table samples are invalid")
        penetration += float(np.sum(np.maximum(height - points[:, 2], 0.0)))
    return penetration, np.asarray(feasible, dtype=bool)


def _sdf(config, voxels):
    axes = [np.linspace(config["bounds_min"][index],
                        config["bounds_max"][index], voxels.shape[index])
            for index in range(3)]
    return RegularGridInterpolator(
        tuple(axes), voxels, bounds_error=False, fill_value=0)


def _acceptable_result(result):
    if result.success:
        return True
    message = str(result.message).lower()
    return any(value in message for value in (
        "maximum", "iteration", "not necessarily"))


def _constraint_check(result, debug, tolerance, include_subgoal):
    if _acceptable_result(result):
        result.success = True
    else:
        result.message = str(result.message) + "; invalid solution"
    names = (["subgoal_violation"] if include_subgoal else []) + ["path_violation"]
    for name in names:
        values = debug.get(name)
        if values is not None and any(
                not np.isfinite(value) or value > tolerance for value in values):
            result.success = False
            result.message = str(result.message) + "; {} not satisfied".format(name)
    feasible = debug.get("ik_feasible")
    if feasible is not None and not bool(np.all(feasible)):
        result.success = False
        result.message = str(result.message) + "; ik not feasible"
    return result


def _subgoal_objective(opt_vars, og_bounds, keypoints_centered,
                       movable, goal_constraints, path_constraints, sdf_func,
                       collision_points_centered, init_pose_homo, ik_solver,
                       initial_joint_pos, reset_joint_pos, is_grasp_stage,
                       previous_solution_homo, modules, weights,
                       collision_threshold_m, return_debug_dict=False):
    utils, transforms = modules.utils, modules.transform_utils
    pose = utils.unnormalize_vars(opt_vars, og_bounds)
    pose_homo = transforms.pose2mat(
        [pose[:3], transforms.euler2quat(pose[3:])])
    debug = {}
    total = 0.0
    if collision_points_centered is not None:
        value = 0.8 * utils.calculate_collision_cost(
            pose_homo[None], sdf_func, collision_points_centered,
            float(collision_threshold_m))
        debug["collision_cost"] = value
        total += value
    value = utils.consistency(
        pose_homo[None], init_pose_homo[None], rot_weight=1.5)
    debug["init_pose_cost"] = value
    total += value
    if previous_solution_homo is not None and weights.subgoal_consistency > 0.0:
        value = weights.subgoal_consistency * utils.consistency(
            pose_homo[None], previous_solution_homo[None], rot_weight=1.5)
        debug["previous_solution_cost"] = value
        total += value
    maximum_iterations = 100
    ik = ik_solver.solve(
        pose_homo, max_iterations=maximum_iterations,
        initial_joint_pos=initial_joint_pos)
    value = 20.0 * (ik.num_descents / maximum_iterations)
    debug.update(ik_feasible=ik.success, ik_pos_error=ik.position_error,
                 ik_cost=value)
    total += value
    if ik.success:
        reset = np.clip(np.linalg.norm(
            ik.cspace_position[:-1] - reset_joint_pos[:-1]), 0.0, 3.0)
    else:
        reset = 3.0
    value = 0.2 * reset
    debug["reset_reg_cost"] = value
    total += value
    if is_grasp_stage:
        preferred = np.asarray([0.0, 0.0, -1.0])
        value = 10.0 * (-np.dot(pose_homo[:3, 0], preferred) + 1.0)
        debug["grasp_cost"] = value
        total += value
    transformed = utils.transform_keypoints(
        pose_homo, keypoints_centered, movable)
    debug["subgoal_violation"] = None
    if goal_constraints:
        violations = [float(function(transformed[0], transformed[1:]))
                      for function in goal_constraints]
        value = 200.0 * np.sum(np.maximum(violations, 0.0))
        debug.update(subgoal_violation=violations,
                     subgoal_constraint_cost=value)
        total += value
    debug["path_violation"] = None
    if path_constraints:
        violations = [float(function(transformed[0], transformed[1:]))
                      for function in path_constraints]
        value = 200.0 * np.sum(np.maximum(violations, 0.0))
        debug.update(path_violation=violations, path_constraint_cost=value)
        total += value
    debug["total_cost"] = total
    return (total, debug) if return_debug_dict else total


def _path_objective(opt_vars, og_bounds, start_pose, end_pose,
                    keypoints_centered, movable, path_constraints, sdf_func,
                    collision_points_centered, table_points_centered,
                    interpolate_position_step,
                    interpolate_rotation_step, ik_solver, initial_joint_pos,
                    reset_joint_pos, previous_dense_homo, table_height_m,
                    robot_points_fn, modules, weights, collision_threshold_m,
                    endpoint_exemption_radius_m, return_debug_dict=False):
    utils, transforms = modules.utils, modules.transform_utils
    unnormalized = utils.unnormalize_vars(opt_vars, og_bounds)
    controls_euler = np.concatenate([
        start_pose[None], unnormalized.reshape(-1, 6), end_pose[None]], axis=0)
    controls_homo = transforms.convert_pose_euler2mat(controls_euler)
    controls_quat = transforms.convert_pose_mat2quat(controls_homo)
    poses_quat, count = utils.get_samples_jitted(
        controls_homo, controls_quat, interpolate_position_step,
        interpolate_rotation_step)
    poses_homo = transforms.convert_pose_quat2mat(poses_quat)
    debug = {"num_control_points": len(opt_vars) // 6, "num_poses": count}
    total = 0.0
    if collision_points_centered is not None:
        mask = endpoint_collision_mask(
            poses_homo, start_pose, end_pose,
            endpoint_exemption_radius_m)
        selected = poses_homo[mask]
        value = (0.5 * utils.calculate_collision_cost(
            selected, sdf_func, collision_points_centered,
            float(collision_threshold_m)) if len(selected) else 0.0)
        debug.update(collision_cost=value,
                     collision_checked_samples=int(np.count_nonzero(mask)))
        total += value
        table = weights.table_clearance * table_penetration_cost(
            poses_homo, table_points_centered, table_height_m, utils)
        debug["table_clearance_cost"] = table
        total += table
        robot_table, table_ik_feasible = robot_table_penetration_cost(
            poses_homo, ik_solver, initial_joint_pos, robot_points_fn,
            table_height_m)
        robot_table *= weights.table_clearance
        debug.update(robot_table_clearance_cost=robot_table,
                     table_ik_feasible=table_ik_feasible)
        total += robot_table
    position_length, rotation_length = utils.path_length(poses_homo)
    value = 4.0 * (position_length + rotation_length)
    debug["path_length_cost"] = value
    total += value
    if previous_dense_homo is not None and weights.path_consistency > 0.0:
        value = weights.path_consistency * utils.consistency(
            poses_homo, previous_dense_homo, rot_weight=0.5)
        debug["previous_solution_cost"] = value
        total += value
    maximum_iterations = 100
    ik_cost = 0.0
    reset_cost = 0.0
    feasible, errors = [], []
    for control in controls_homo:
        ik = ik_solver.solve(
            control, max_iterations=maximum_iterations,
            initial_joint_pos=initial_joint_pos)
        feasible.append(ik.success)
        errors.append(ik.position_error)
        ik_cost += 20.0 * (ik.num_descents / maximum_iterations)
        reset = (np.clip(np.linalg.norm(
            ik.cspace_position[:-1] - reset_joint_pos[:-1]), 0.0, 3.0)
                 if ik.success else 3.0)
        reset_cost += 0.2 * reset
    debug.update(ik_feasible=np.asarray(feasible),
                 ik_pos_error=np.asarray(errors), ik_cost=ik_cost,
                 reset_reg_cost=reset_cost)
    total += ik_cost + reset_cost
    debug["path_violation"] = None
    if path_constraints:
        violations = []
        # ReKep path constraints are checked on every dense sample.  The 5 cm
        # exemption applies only to scene collision, not task semantics.
        for pose in poses_homo:
            transformed = utils.transform_keypoints(
                pose, keypoints_centered, movable)
            violations.extend(float(function(
                transformed[0], transformed[1:]))
                for function in path_constraints)
        value = 200.0 * np.sum(np.maximum(violations, 0.0))
        debug.update(path_violation=violations, path_constraint_cost=value)
        total += value
    debug["total_cost"] = total
    return (total, debug, poses_homo) if return_debug_dict else total


class PaperRealSubgoalSolver:
    def __init__(self, config, ik_solver, reset_joint_pos, modules, weights,
                 collision_threshold_m=0.15):
        self.config = dict(config)
        self.ik_solver = ik_solver
        self.reset_joint_pos = np.asarray(reset_joint_pos, dtype=float)
        self.modules = modules
        self.weights = weights
        self.collision_threshold_m = float(collision_threshold_m)
        self.last_opt_result = None
        self.previous_solution_homo = None

    def reset_history(self):
        self.last_opt_result = None
        self.previous_solution_homo = None

    def solve(self, ee_pose, keypoints, movable, goal_constraints,
              path_constraints, sdf_voxels, collision_points, is_grasp_stage,
              initial_joint_pos, from_scratch=False):
        utils, transforms = self.modules.utils, self.modules.transform_utils
        if len(collision_points) > self.config["max_collision_points"]:
            collision_points = utils.farthest_point_sampling(
                collision_points, self.config["max_collision_points"])
        sdf_func = _sdf(self.config, sdf_voxels)
        ee_homo = transforms.pose2mat([ee_pose[:3], ee_pose[3:]])
        ee_euler = np.r_[ee_pose[:3], transforms.quat2euler(ee_pose[3:])]
        original_bounds = list(zip(
            np.r_[self.config["bounds_min"], [-np.pi] * 3],
            np.r_[self.config["bounds_max"], [np.pi] * 3]))
        bounds = [(-1.0, 1.0)] * 6
        if not from_scratch and self.last_opt_result is not None:
            initial = self.last_opt_result.x.copy()
        else:
            initial = utils.normalize_vars(ee_euler, original_bounds)
            from_scratch = True
        centering = np.linalg.inv(ee_homo)
        centered_collision = (
            collision_points @ centering[:3, :3].T + centering[:3, 3])
        centered_keypoints = utils.transform_keypoints(
            centering, keypoints, movable)
        args = (original_bounds, centered_keypoints, movable,
                goal_constraints, path_constraints, sdf_func,
                centered_collision, ee_homo, self.ik_solver,
                initial_joint_pos, self.reset_joint_pos, is_grasp_stage,
                self.previous_solution_homo, self.modules, self.weights,
                self.collision_threshold_m)
        started = time.monotonic()
        if from_scratch:
            result = dual_annealing(
                _subgoal_objective, bounds, args=args,
                maxfun=self.config["sampling_maxfun"], x0=initial,
                no_local_search=False,
                minimizer_kwargs={"method": "SLSQP",
                                  "bounds": bounds,
                                  "options": self.config["minimizer_options"]})
        else:
            result = minimize(
                _subgoal_objective, initial, args=args, bounds=bounds,
                method="SLSQP", options=self.config["minimizer_options"])
        _, debug = _subgoal_objective(
            result.x, *args, return_debug_dict=True)
        debug.update(solve_time=time.monotonic() - started,
                     from_scratch=from_scratch, type="paper_real_subgoal_solver")
        debug.update(optimizer_success=bool(result.success),
                     optimizer_message=str(result.message),
                     optimizer_evaluations=int(result.nfev))
        raw = utils.unnormalize_vars(result.x, original_bounds)
        solution = np.r_[raw[:3], transforms.euler2quat(raw[3:])]
        result = _constraint_check(
            result, debug, self.config["constraint_tolerance"], True)
        if result.success:
            self.last_opt_result = copy.deepcopy(result)
            self.previous_solution_homo = transforms.pose2mat(
                [solution[:3], solution[3:]])
        return solution, debug


class PaperRealPathSolver:
    def __init__(self, config, ik_solver, reset_joint_pos, modules, weights,
                 table_height_m, robot_points_fn, collision_threshold_m=0.15,
                 endpoint_exemption_radius_m=0.05):
        self.config = dict(config)
        self.ik_solver = ik_solver
        self.reset_joint_pos = np.asarray(reset_joint_pos, dtype=float)
        self.modules = modules
        self.weights = weights
        self.table_height_m = float(table_height_m)
        if robot_points_fn is None:
            raise ValueError("paper-real requires full-arm collision sampling")
        self.robot_points_fn = robot_points_fn
        self.collision_threshold_m = float(collision_threshold_m)
        self.endpoint_exemption_radius_m = float(endpoint_exemption_radius_m)
        self.last_opt_result = None
        self.previous_dense_homo = None

    def reset_history(self):
        self.last_opt_result = None
        self.previous_dense_homo = None

    def solve(self, start_pose, end_pose, keypoints, movable,
              path_constraints, sdf_voxels, collision_points,
              initial_joint_pos, from_scratch=False):
        utils, transforms = self.modules.utils, self.modules.transform_utils
        table_collision_points = np.asarray(
            collision_points, dtype=float).copy()
        if len(collision_points) > self.config["max_collision_points"]:
            collision_points = utils.farthest_point_sampling(
                collision_points, self.config["max_collision_points"])
        sdf_func = _sdf(self.config, sdf_voxels)
        count = int(np.clip(utils.get_linear_interpolation_steps(
            start_pose, end_pose, self.config["opt_pos_step_size"],
            self.config["opt_rot_step_size"]), 3, 6))
        start_euler = np.r_[start_pose[:3], transforms.quat2euler(start_pose[3:])]
        end_euler = np.r_[end_pose[:3], transforms.quat2euler(end_pose[3:])]
        one_bounds = list(zip(self.config["bounds_min"],
                              self.config["bounds_max"])) + [
                                  (-np.pi, np.pi)] * 3
        original_bounds = np.asarray(one_bounds * (count - 2), dtype=float)
        bounds = [(-1.0, 1.0)] * len(original_bounds)
        if not from_scratch and self.last_opt_result is not None:
            initial = self.last_opt_result.x.copy()
            variables = len(bounds)
            if len(initial) < variables:
                extended = np.empty(variables)
                extended[:len(initial)] = initial
                for index in range(len(initial), variables, 6):
                    extended[index:index + 6] = initial[-6:]
                initial = extended
            else:
                initial = initial[-variables:]
        else:
            interpolated = utils.linear_interpolate_poses(
                start_euler, end_euler, count)
            initial = utils.normalize_vars(
                interpolated[1:-1].reshape(-1), original_bounds)
            from_scratch = True
        initial = np.clip(initial, -1.0, 1.0)
        start_homo = transforms.pose2mat([start_pose[:3], start_pose[3:]])
        centering = np.linalg.inv(start_homo)
        centered_collision = (
            collision_points @ centering[:3, :3].T + centering[:3, 3])
        centered_table_points = (
            table_collision_points @ centering[:3, :3].T
            + centering[:3, 3])
        centered_keypoints = utils.transform_keypoints(
            centering, keypoints, movable)
        args = (original_bounds, start_euler, end_euler,
                centered_keypoints, movable, path_constraints, sdf_func,
                centered_collision, centered_table_points,
                self.config["opt_interpolate_pos_step_size"],
                self.config["opt_interpolate_rot_step_size"], self.ik_solver,
                initial_joint_pos, self.reset_joint_pos,
                self.previous_dense_homo, self.table_height_m,
                self.robot_points_fn, self.modules,
                self.weights, self.collision_threshold_m,
                self.endpoint_exemption_radius_m)
        started = time.monotonic()
        if from_scratch:
            result = dual_annealing(
                _path_objective, bounds, args=args,
                maxfun=self.config["sampling_maxfun"], x0=initial,
                no_local_search=False,
                minimizer_kwargs={"method": "SLSQP",
                                  "bounds": bounds,
                                  "options": self.config["minimizer_options"]})
        else:
            result = minimize(
                _path_objective, initial, args=args, bounds=bounds,
                method="SLSQP", options=self.config["minimizer_options"])
        _, debug, dense_homo = _path_objective(
            result.x, *args, return_debug_dict=True)
        debug.update(solve_time=time.monotonic() - started,
                     from_scratch=from_scratch, type="paper_real_path_solver")
        debug.update(optimizer_success=bool(result.success),
                     optimizer_message=str(result.message),
                     optimizer_evaluations=int(result.nfev),
                     maximum_normalized_control_change=float(
                         np.max(np.abs(result.x - initial))))
        raw = utils.unnormalize_vars(result.x, original_bounds)
        tail_euler = np.concatenate([
            raw.reshape(-1, 6), end_euler[None]], axis=0)
        tail = transforms.convert_pose_euler2quat(tail_euler)
        result = _constraint_check(
            result, debug, self.config["constraint_tolerance"], False)
        if result.success:
            self.last_opt_result = copy.deepcopy(result)
            self.previous_dense_homo = np.asarray(dense_homo, dtype=float).copy()
        return tail, debug
