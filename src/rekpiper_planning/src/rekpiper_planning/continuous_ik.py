"""Bounded IK continuation; failures never become executable waypoints."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def interpolate_pose(start, end, fraction):
    pose = np.eye(4)
    pose[:3, 3] = (1-fraction)*start[:3, 3] + fraction*end[:3, 3]
    pose[:3, :3] = Slerp([0., 1.], Rotation.from_matrix(
        np.array([start[:3, :3], end[:3, :3]])))([fraction]).as_matrix()[0]
    return pose


def densify_joint_path(joints, maximum_l1_step_rad=.005):
    """Bound combined joint travel, including elbow motion with nearly fixed TCP."""
    joints=np.asarray(joints,dtype=float)
    if joints.ndim != 2 or not len(joints) or not np.all(np.isfinite(joints)):
        raise ValueError('invalid joint path')
    dense=[joints[0]]
    for start,end in zip(joints[:-1],joints[1:]):
        count=max(1,int(np.ceil(np.sum(np.abs(end-start))/maximum_l1_step_rad)))
        dense.extend(np.linspace(start,end,count+1)[1:])
    return np.asarray(dense)


def validate_continuous_ik(ik, poses, initial_joint_pos, max_iterations=100,
                           maximum_joint_jump_rad=.35, maximum_refinement=3,
                           maximum_seeds=8, trace=None):
    """Validate with the production policy; optionally append diagnostic attempts.

    ``selected`` means a local segment was chosen. Only ``committed`` attempts
    belong to the returned prefix: successful subdivisions of a failed original
    waypoint are rolled back. Tracing does not change seeds or acceptance.
    """
    poses = np.asarray(poses, dtype=float)
    dof = len(ik._lower)
    seed = np.asarray(initial_joint_pos, dtype=float)[:dof].copy()
    if (poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses)
            or seed.shape != (dof,) or not np.all(np.isfinite(poses))
            or not np.all(np.isfinite(seed))):
        raise ValueError('invalid continuous IK inputs')
    if (not 1 <= maximum_seeds <= 8 or not 0 <= maximum_refinement <= 3
            or max_iterations <= 0 or maximum_joint_jump_rad <= 0):
        raise ValueError('invalid continuous IK budget')
    steps, diagnostics = [], []

    def edge_valid(q0, q1, p0, p1, evidence):
        count = max(2, int(np.ceil(np.sum(np.abs(q1-q0))/.005))+1)
        if evidence is not None:
            evidence.update(samples_planned=count, samples_checked=0,
                            maximum_position_error_m=0., maximum_rotation_error_rad=0.)
        for t in np.linspace(0., 1., count):
            actual = ik.forward((1-t)*q0+t*q1)
            desired = interpolate_pose(p0, p1, t)
            pos, rot = ik._pose_errors(actual, desired)
            if evidence is not None:
                evidence['samples_checked'] += 1
                evidence['maximum_position_error_m'] = max(evidence['maximum_position_error_m'], float(pos))
                evidence['maximum_rotation_error_rad'] = max(evidence['maximum_rotation_error_rad'], float(rot))
            if pos > ik.position_tolerance or rot > ik.orientation_tolerance:
                if evidence is not None:
                    evidence['first_failed_fraction'] = float(t)
                return False
        return True

    def segment(p0, p1, q0, index, depth, interval=(0., 1.)):
        trials = [q0]
        # Deterministic local seeds; never use random/global jumps as continuity.
        for joint, offset in ((1,.05),(2,-.05),(3,.05),(3,-.05),
                              (4,.05),(4,-.05),(5,.05)):
            if joint >= dof:
                continue
            trial = q0.copy()
            trial[joint] += offset
            trials.append(np.clip(trial, ik._lower, ik._upper))
        accepted = []
        first = None
        for trial_index, trial in enumerate(trials[:maximum_seeds]):
            result = ik.solve(p1, initial_joint_pos=trial, max_iterations=max_iterations)
            q1 = np.asarray(result.cspace_position[:dof])
            delta = float(np.max(np.abs(q1-q0)))
            edge = {} if trace is not None else None
            reason = ('ik_residual_exceeded' if not result.success else
                      'joint_jump' if delta > maximum_joint_jump_rad else
                      'fk_edge_deviation' if not edge_valid(q0,q1,p0,p1,edge) else 'ok')
            entry = dict(pose_index=int(index), refinement_depth=depth,
                         success=reason == 'ok', status=reason,
                         position_error=float(result.position_error),
                         rotation_error=float(result.rotation_error),
                         maximum_joint_step_rad=delta, joint_positions=q1.tolist())
            attempt = None
            if trace is not None:
                attempt = dict(entry, attempt_id=len(trace), seed_index=trial_index,
                               interval=list(interval), target_pose=p1.tolist(),
                               previous_joints=q0.tolist(), seed_joints=trial.tolist(),
                               joint_delta_rad=(q1-q0).tolist(), ik_success=bool(result.success),
                               num_descents=getattr(result, 'num_descents', None),
                               edge_check=edge, selected=False, committed=False)
                trace.append(attempt)
            if first is None:
                first = entry
            if reason == 'ok':
                accepted.append((delta, entry, q1, attempt))
                if len(trials) and trial is trials[0]:
                    break
        if accepted:
            _, entry, q1, attempt = min(accepted, key=lambda item: item[0])
            if attempt is not None:
                attempt['selected'] = True
            return [(entry, p1)], q1
        diagnostics.append(first)
        if depth >= maximum_refinement:
            return None, q0
        mid = interpolate_pose(p0, p1, .5)
        half = sum(interval)/2
        left, qm = segment(p0, mid, q0, index, depth+1, (interval[0], half))
        if left is None:
            return None, q0
        right, q1 = segment(mid, p1, qm, index, depth+1, (half, interval[1]))
        return (None, q0) if right is None else (left+right, q1)

    previous = ik.forward(seed)
    accepted_poses = []
    for index, pose in enumerate(poses):
        trace_start = len(trace) if trace is not None else 0
        result, q = segment(previous, pose, seed, index, 0)
        if result is None:
            return dict(valid=False, reason='ik_failed', steps=steps,
                        failed_pose_index=index, diagnostics=diagnostics,
                        poses=np.asarray(accepted_poses))
        if trace is not None:
            for attempt in trace[trace_start:]:
                attempt['committed'] = attempt['selected']
        for entry, matrix in result:
            steps.append(entry)
            accepted_poses.append(matrix)
        previous, seed = pose, q
    return dict(valid=True, reason='ok', steps=steps,
                diagnostics=diagnostics, poses=np.asarray(accepted_poses))
