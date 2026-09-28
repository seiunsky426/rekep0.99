"""Fixed local task templates and supervised grasp planning."""
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from rekpiper_planning.continuous_ik import densify_joint_path
from rekpiper_planning.motion_backend import MotionBackend
from rekpiper_planning.piper_collision_sampling import PiperCollisionSampler
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver
from rekpiper_planning.official_program import parse_official_program, save_official_program, compute_program_sha256
from rekpiper_planning.realtime_planner import PersistentReKepPlanner, RealtimePlanningRequest, PlanningGeneration, RealtimePlanningError
from .supervised_geometry import transform_points
from .trajectory import JOINT_NAMES, smooth_pchip_trajectory
from .supervised_anygrasp import generate_candidates


def template(target, start, vertical=False, floor=None, approach_axis=None):
    source = ('num_stages = 1\ngrasp_keypoints = [-1]\nrelease_keypoints = [-1]\n'
              'def stage1_subgoal_constraint1(end_effector, keypoints):\n'
              '    return np.linalg.norm(end_effector - np.array({}))\n'.format(target[:3].tolist()))
    if approach_axis is not None:
        source += ('def stage1_path_constraint1(end_effector, keypoints):\n'
                   '    return np.linalg.norm(np.cross(end_effector - np.array({}), np.array({}))) - 0.003\n'.format(
                       start[:3].tolist(), np.asarray(approach_axis).tolist()))
    elif vertical:
        source += ('def stage1_path_constraint1(end_effector, keypoints):\n'
                   '    return np.linalg.norm(end_effector[:2] - np.array({})) - 0.003\n'.format(start[:2].tolist()))
    elif floor is not None:
        source += ('def stage1_path_constraint1(end_effector, keypoints):\n'
                   '    return {} - end_effector[2]\n'.format(float(floor)))
    return source


class ExperimentPlanner:
    def __init__(self, xml, config, system, output):
        self.xml, self.config, self.system = xml, config, system
        self.output = Path(output)
        self.ik = PiperURDFIKSolver.from_urdf_xml(xml, 'base_link', 'rekep_tcp', JOINT_NAMES,
                                               position_tolerance=.003, orientation_tolerance=.03)
        self.sampler = PiperCollisionSampler(xml, self.ik, voxel_size_m=.01,
                                            maximum_points_per_link=300)
        self.backend = MotionBackend(xml)
        self.candidate = None
        self.target = self.support = self.held_local = None
        self.grasp_matrix = None
        self.rejections = []
        self.cancelled = lambda: False

    def check_cancel(self):
        if self.cancelled():
            raise InterruptedError('planning_cancelled')

    def candidates(self, scene, target, joints):
        self.candidate = None
        self.rejections = []
        values = generate_candidates(self, scene, joints)
        self.candidate = values[0]
        return values

    def audit(self, path, scene, stage, opening=None):
        from .trajectory import POSITION_MIN, POSITION_MAX
        path = np.asarray(path, dtype=float)[:, :6]
        if (not np.isfinite(path).all() or np.any(path < POSITION_MIN) or np.any(path > POSITION_MAX)):
            raise ValueError('joint_limits_or_nonfinite_path')
        dense = densify_joint_path(path) if len(path) > 1 else path
        candidate = self.candidate
        if candidate is None:
            raise ValueError('grasp_candidate_missing')
        opening = (candidate.suggested_preopen_width_m if stage <= 2
                   else candidate.predicted_width_m) if opening is None else opening
        contacts = np.asarray(candidate.contact_points_base)
        target_tree = cKDTree(self.target)
        other_points = scene.points[target_tree.query(scene.points)[0] > .006]
        other_tree = cKDTree(other_points)
        for q in dense:
            self.check_cancel()
            if not self.backend.self_clear(q, opening):
                raise ValueError('robot_self_collision')
            points, radii, labels = self.sampler.contact_samples(q, opening)
            values, known = scene.sample(points)
            bad = (-values-radii) < self.config['minimum_clearance_m']
            matrix = self.ik.forward(q)
            contact_points = contacts
            if self.held_local is not None and stage >= 3:
                contact_points = transform_points(contacts, matrix @ np.linalg.inv(self.grasp_matrix))
            near_goal = self.ik._pose_errors(matrix, candidate.grasp_pose)
            if stage >= 3 or (stage == 2 and near_goal[0] <= .005 and near_goal[1] <= .10):
                allowed = (np.isin(labels, ['link7', 'link8'])
                           & (cKDTree(contact_points).query(points)[0] <= radii+.005)
                           & (other_tree.query(points)[0] > radii+.01))
                bad &= ~allowed
            if not known.all() or bad.any():
                raise ValueError('whole_arm_environment_or_unknown_collision')
            if self.held_local is not None and stage >= 3:
                held = transform_points(self.held_local, matrix)
                value, observed = scene.sample(held)
                distances = other_tree.query(held)[0]
                near_support = np.zeros(len(held), dtype=bool)
                if stage == 4 and self.support is not None:
                    top = float(np.quantile(self.support[:, 2], .95))
                    near_support = ((held[:, 2] >= top-.002) & (held[:, 2] <= top+.003)
                                    & (cKDTree(self.support).query(held)[0] <= .005))
                if (not observed.all() or np.any((distances < .01) & ~near_support)
                        or np.any((value >= 0) & ~near_support)):
                    raise ValueError('held_object_environment_collision')
                arm = points[~np.isin(labels, ['gripper_base', 'link7', 'link8'])]
                if len(arm) and np.any(cKDTree(arm).query(held)[0] < .02):
                    raise ValueError('held_object_robot_collision')
        return dict(valid=True, dense_waypoints=len(dense), self_collision=True,
                    environment=True, unknown_space_blocked=True)

    def sweep(self, joints, scene, stage, start, end):
        for width in np.linspace(start, end, max(2, int(abs(end-start)/.001)+1)):
            self.audit([joints], scene, stage, float(width))

    def held(self, joints):
        self.grasp_matrix = self.ik.forward(joints)
        self.held_local = transform_points(self.target, np.linalg.inv(self.grasp_matrix))

    def solve_leg(self, joints, goal, scene, directory, vertical=False, floor=None, approach_axis=None):
        self.check_cancel()
        current = self.ik.forward(joints)
        keypoints = np.array([self.target.mean(axis=0), self.support.mean(axis=0)])
        source = template(goal[:3, 3], current[:3, 3], vertical, floor, approach_axis)
        program = parse_official_program(source, 'fixed supervised pick and place', len(keypoints))
        save_official_program(program, directory, keypoints)
        sub = dict(self.system['subgoal_solver'], bounds_min=scene.lower.tolist(), bounds_max=scene.upper.tolist())
        path = dict(self.system['path_solver'], bounds_min=scene.lower.tolist(), bounds_max=scene.upper.tolist())
        path.update(self.config.get('path_solver', {}))
        planner = PersistentReKepPlanner(sub, path, self.ik, np.r_[joints, 0.], solver_profile='official_exact',
                                         path_trace_directory=str(directory/'trace'))
        pose = lambda m: np.r_[m[:3, 3], Rotation.from_matrix(m[:3, :3]).as_quat()]
        # The supervised template fixes the candidate TCP goal; the planner still
        # checks its constraint, continuous IK, and the complete path audit.
        request = RealtimePlanningRequest(
            PlanningGeneration(scene.id, 'two-measured-objects', scene.id,
                               compute_program_sha256(directory), 1), str(directory), 1,
            pose(current), joints, keypoints, np.array([1, 2]), -1,
            scene.grid['distances_m'], self.sampler.samples(joints)[0], lambda _: 0.,
            solver_call_timeouts=(None, None), fixed_target_pose=pose(goal))
        result = planner.solve(request)
        q = result.joint_path[:, :6]
        error = self.ik._pose_errors(self.ik.forward(q[-1]), goal)
        if error[0] > .003 or error[1] > .03:
            raise ValueError('fixed_template_endpoint_not_reached')
        if self.held_local is not None or vertical or approach_axis is not None:
            for values in q:
                if self.ik._pose_errors(self.ik.forward(values), goal)[1] > .10:
                    raise ValueError('fixed_orientation_path_violated')
        return q

    def plan(self, stage, joints, scene, speed):
        root = self.output/('plan-'+uuid.uuid4().hex); root.mkdir(parents=True)
        candidate = self.candidate
        (root/'candidate.json').write_text(json.dumps(dict(
            id=candidate.candidate_id, joints=np.asarray(joints).tolist(),
            opening_m=candidate.suggested_preopen_width_m,
            grasp_pose=candidate.grasp_pose.tolist(), pregrasp_pose=candidate.pregrasp_pose.tolist(),
            contact_supported=candidate.contact_membership_ok), indent=2))
        current = self.ik.forward(joints)
        goal = candidate.pregrasp_pose.copy() if stage == 1 else candidate.grasp_pose.copy()
        legs = []
        if stage >= 3:
            if self.held_local is None:
                raise ValueError('operator_confirmed_held_geometry_missing')
            goal = current.copy()
            support_center = np.median(self.support, axis=0)
            held = transform_points(self.held_local, current)
            bottom = np.min(held[:, 2]); center = np.mean(held[:, :2], axis=0)
            goal[:2, 3] += support_center[:2]-center
            support_z = float(np.quantile(self.support[:, 2], .95))
            if stage == 3:
                lift = current.copy()
                lift[2, 3] += max(self.config['lift_m'], support_z+.08-bottom)
                first = self.solve_leg(joints, lift, scene, root/'lift', vertical=True)
                self.audit(first, scene, stage)
                legs.append(first)
                joints = first[-1]
                goal[2, 3] = lift[2, 3]
            else:
                goal[2, 3] += support_z+.001-bottom
        path = self.solve_leg(joints, goal, scene, root/'main', vertical=stage == 4,
                              floor=goal[2, 3]-.003 if stage == 3 else None,
                              approach_axis=candidate.approach_axis_base if stage == 2 else None)
        legs.append(path)
        path = np.vstack([leg if i == 0 else leg[1:] for i, leg in enumerate(legs)])
        np.save(root/'proposed_joint_path.npy', path)
        audit = self.audit(path, scene, stage)
        smooth = smooth_pchip_trajectory(JOINT_NAMES, path, path[0], sample_rate_hz=50.,
            maximum_velocity_rad_s=speed.velocity, maximum_acceleration_rad_s2=speed.acceleration,
            maximum_jerk_rad_s3=speed.jerk, start_tolerance_rad=.01)
        self.audit(smooth.positions, scene, stage)
        if stage == 2:
            self.sweep(smooth.positions[-1], scene, stage, candidate.suggested_preopen_width_m,
                       max(0., candidate.predicted_width_m-.003))
        if stage == 4:
            self.supported(self.ik.forward(smooth.positions[-1]))
            self.sweep(smooth.positions[-1], scene, stage, candidate.predicted_width_m,
                       candidate.suggested_preopen_width_m)
        return smooth, dict(goal_matrix=goal.tolist(), goal_joints=smooth.positions[-1].tolist(),
                            duration_s=smooth.duration_s, candidate=candidate.candidate_id,
                            source=candidate.candidate_origin, audit=audit, rejections=self.rejections,
                            grasp_evidence=getattr(self, 'grasp_evidence', None),
                            contact_supported=candidate.contact_membership_ok,
                            preview_only=self.config.get('extrinsics_status') == 'EXPERIMENTAL_PREVIEW_ONLY',
                            planning_authorized=False,
                            grasp_pose=candidate.grasp_pose.tolist(), pregrasp_pose=candidate.pregrasp_pose.tolist(),
                            opening_m=candidate.suggested_preopen_width_m,
                            program_directory=str(root))

    def supported(self, matrix):
        from scipy.spatial import ConvexHull
        held = transform_points(self.held_local, matrix)
        top = float(np.quantile(self.support[:, 2], .95))
        support = self.support[self.support[:, 2] >= top-.003]
        hull = ConvexHull(support[:, :2])
        if (np.any(held[:, :2] @ hull.equations[:, :2].T+hull.equations[:, 2] > -.002)
                or not -.002 <= np.min(held[:, 2])-top <= .003):
            raise ValueError('object_not_supported_for_release')


def _planning_job(connection, job):
    """Spawned process: numerical planning/inference only, no hardware clients."""
    try:
        planner = ExperimentPlanner(job['xml'],job['config'],job['system'],job['output'])
        planner.target, planner.support = job['target'], job['support']
        planner.held_local, planner.grasp_matrix = job['held_local'], job['grasp_matrix']
        planner.candidate = job['candidate']
        if job['stage'] == 1:
            candidates = planner.candidates(job['scene'],job['target'],job['joints'])
        else:
            candidates = [planner.candidate]
        failures = []
        for candidate in candidates:
            planner.check_cancel()
            planner.candidate = candidate
            if hasattr(candidate, 'target_points_base'):
                planner.target = candidate.target_points_base
            try:
                initial_error = None
                if job['stage'] == 1:
                    try:
                        planner.sweep(job['joints'],job['scene'],1,job['opening'],candidate.suggested_preopen_width_m)
                    except ValueError as exc:
                        if job['config'].get('extrinsics_status') != 'EXPERIMENTAL_PREVIEW_ONLY':
                            raise
                        # Compute a diagnostic draft, but never install an accepted
                        # preview when the initial opening sweep failed.
                        initial_error = 'initial_opening_sweep: '+str(exc)
                smooth, detail = planner.plan(job['stage'],job['joints'],job['scene'],job['speed'])
                if initial_error:
                    raise ValueError(initial_error)
            except (ValueError, RealtimePlanningError) as exc:
                failures.append(dict(candidate=candidate.candidate_id, reason=str(exc)))
                (planner.output/'anygrasp_path_failures.json').write_text(json.dumps(failures, indent=2))
                continue
            detail['earlier_candidate_path_failures'] = failures
            connection.send(('ok',(smooth,detail,planner.candidate)))
            break
        else:
            raise ValueError('anygrasp_no_validated_path: {}; evidence={}'.format(
                failures, getattr(planner, 'grasp_evidence', planner.output)))
    except Exception as exc:
        connection.send(('error',type(exc).__name__+': '+str(exc)))
    finally:
        connection.close()


def plan_in_process(job, cancelled, timeout_s=600.):
    import multiprocessing
    import time
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_planning_job,args=(child,job))
    process.start(); child.close()
    deadline = time.monotonic()+timeout_s
    try:
        while time.monotonic()<deadline:
            if cancelled(): raise InterruptedError('planning_cancelled')
            if parent.poll(.05):
                kind,value = parent.recv()
                if kind != 'ok': raise RuntimeError(value)
                return value
            if not process.is_alive(): raise RuntimeError('planning_worker_exited')
        raise TimeoutError('supervised_planning_budget_exceeded')
    finally:
        if process.is_alive(): process.terminate()
        process.join(timeout=.5)
        if process.is_alive(): process.kill(); process.join(timeout=.5)
        parent.close()
