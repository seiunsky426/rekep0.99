"""Piper path costs over the pinned ReKep optimizer, without patching upstream."""
from types import FunctionType, MethodType

import numpy as np


PIPER_PATH_DEFAULTS = dict(
    opt_pos_step_size=.10, opt_rot_step_size=float(np.deg2rad(15.)),
    ik_max_iterations=100, ik_failure_weight=1000.,
    ik_position_weight=20., ik_orientation_weight=20.,
    joint_limit_weight=10., joint_limit_margin_deg=10.)


def sample_spline_path(modules, controls, position_step=.05, rotation_step=.34,
                       maximum_position_step=.015, maximum_rotation_step=.10):
    """Use exactly the same spline and dense samples in optimization and audit."""
    count = modules.utils.get_linear_interpolation_steps(
        controls[0], controls[-1], position_step, rotation_step)
    spline = modules.utils.spline_interpolate_poses(controls, count)
    dense = []
    for index, (start, end) in enumerate(zip(spline[:-1], spline[1:])):
        count = modules.utils.get_linear_interpolation_steps(
            start, end, maximum_position_step, maximum_rotation_step)
        segment = modules.utils.linear_interpolate_poses(start, end, count)
        dense.extend(segment if index == 0 else segment[1:])
    return np.asarray(spline), np.asarray(dense)


def continuous_ik_cost(ik, poses, initial_joints, config, maximum_joint_jump=.35):
    """Sequential warm starts, residual costs and a soft margin inside hard limits.

    A failed sample never supplies a seed for the next sample. This objective
    guides the optimizer; the existing continuous-IK/FK-edge audit remains final.
    """
    dof = len(ik._lower)
    seed = np.asarray(initial_joints, dtype=float)[:dof].copy()
    margin = np.deg2rad(config['joint_limit_margin_deg'])
    position, rotation, margins, feasible, jumps = [], [], [], [], []
    for pose in poses:
        result = ik.solve(pose, initial_joint_pos=seed,
                          max_iterations=config['ik_max_iterations'])
        q = np.asarray(result.cspace_position[:dof], dtype=float)
        delta = float(np.max(np.abs(q-seed)))
        ok = bool(result.success and delta <= maximum_joint_jump)
        position.append(float(result.position_error))
        rotation.append(float(result.rotation_error))
        feasible.append(ok)
        jumps.append(delta)
        margins.append(np.minimum(q-ik._lower, ik._upper-q))
        if ok:
            seed = q.copy()
    position, rotation, margins = map(np.asarray, (position, rotation, margins))
    if not all(np.isfinite(v).all() for v in (position, rotation, margins, jumps)):
        raise ValueError('nonfinite_path_ik_result')
    failure_cost = config['ik_failure_weight'] * np.count_nonzero(~np.asarray(feasible))
    position_cost = config['ik_position_weight'] * np.sum(np.maximum(position/ik.position_tolerance-1., 0.)**2)
    rotation_cost = config['ik_orientation_weight'] * np.sum(np.maximum(rotation/ik.orientation_tolerance-1., 0.)**2)
    limit_cost = config['joint_limit_weight'] * np.sum(np.maximum(1.-margins/margin, 0.)**2)
    return dict(ik_cost=float(failure_cost+position_cost+rotation_cost),
                ik_failure_cost=float(failure_cost), ik_position_cost=float(position_cost),
                ik_orientation_cost=float(rotation_cost), joint_limit_cost=float(limit_cost),
                ik_feasible=feasible, ik_pos_error=position, ik_rot_error=rotation,
                ik_joint_step_rad=jumps, joint_limit_margins_rad=margins,
                ik_sample_count=len(position))


def make_piper_path_solver(config, ik, reset_joints, modules, sampler,
                           maximum_joint_jump=.35):
    """Keep ReKep search/control parameterization; replace only its objective."""
    settings = dict(PIPER_PATH_DEFAULTS, **config)
    keys = ('ik_failure_weight', 'ik_position_weight', 'ik_orientation_weight',
            'joint_limit_weight', 'joint_limit_margin_deg', 'ik_max_iterations')
    if any(not np.isfinite(settings[k]) or settings[k] <= 0 for k in keys):
        raise ValueError('invalid_piper_path_cost_settings')
    settings['ik_max_iterations'] = int(settings['ik_max_iterations'])

    def objective(opt_vars, bounds, start, end, keypoints, movable, constraints,
                  sdf, collision_points, _position_step, _rotation_step,
                  ik_solver, initial_joints, _reset_joints, return_debug_dict=False):
        raw = modules.utils.unnormalize_vars(opt_vars, bounds).reshape(-1, 6)
        controls = modules.transform_utils.convert_pose_euler2quat(np.vstack([start, raw, end]))
        _, poses = sampler(controls)
        matrices = modules.transform_utils.convert_pose_quat2mat(poses)
        debug = continuous_ik_cost(ik_solver, matrices, initial_joints, settings, maximum_joint_jump)
        collision = (0.5 * modules.utils.calculate_collision_cost(
            matrices[1:-1], sdf, collision_points, .20) if collision_points is not None else 0.)
        length, angle = modules.utils.path_length(matrices)
        path_cost = 4. * (length+angle)
        violations = []
        for pose in matrices[1:-1]:
            transformed = modules.utils.transform_keypoints(pose, keypoints, movable)
            violations.extend(float(fn(transformed[0], transformed[1:])) for fn in (constraints or []))
        constraint_cost = 200. * float(np.maximum(violations, 0.).sum())
        total = float(collision+path_cost+constraint_cost+debug['ik_cost']+debug['joint_limit_cost'])
        debug.update(num_control_points=len(raw), num_poses=len(poses),
                     collision_cost=float(collision), path_length_cost=float(path_cost),
                     path_constraint_cost=constraint_cost, path_violation=violations or None,
                     reset_reg_cost=0., total_cost=total,
                     sample_policy='final_spline_and_dense_audit_samples')
        return (total, debug) if return_debug_dict else total

    solver_type = modules.path_solver.PathSolver
    solver = solver_type.__new__(solver_type)
    solver.config, solver.ik_solver = settings, ik
    solver.reset_joint_pos = np.asarray(reset_joints, dtype=float)
    solver.last_opt_result = None
    method = solver_type.solve
    namespace = dict(method.__globals__, objective=objective)
    cloned = FunctionType(method.__code__, namespace, method.__name__, method.__defaults__, method.__closure__)
    cloned.__kwdefaults__ = method.__kwdefaults__
    solver.solve = MethodType(cloned, solver)
    return solver
