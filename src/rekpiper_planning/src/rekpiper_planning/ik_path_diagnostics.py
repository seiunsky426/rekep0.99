"""Offline evidence for the existing continuation checker, never a motion gate."""

from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation


COLORS = {
    'accepted': (.1, .9, .2),
    'joint_jump': (1., .85, .0),
    'ik_residual_exceeded': (.8, .1, 1.),
    'fk_edge_deviation': (1., .1, .1),
    'not_reached': (.55, .55, .55),
    'near_limit_or_singularity': (1., .45, .0),
}


def pose7(matrix):
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()].tolist()


def joint_metrics(ik, joints):
    """Raw bounded-joint margins and a scaled geometric-Jacobian diagnostic.

    Translation rows are divided by 0.5 m; rotation rows use radians. Central
    FK differences (1e-6 rad) may cross a limit for differentiation only. The
    0.001 singular-value and 2-degree margin flags are display heuristics, not
    new acceptance criteria or proof that a task cannot be followed.
    """
    q = np.asarray(joints, dtype=float)
    margin = np.minimum(q-ik._lower, ik._upper-q)
    columns = []
    for index in range(len(q)):
        delta = np.zeros(len(q)); delta[index] = 1e-6
        plus, minus = ik.forward(q+delta), ik.forward(q-delta)
        translation = (plus[:3, 3]-minus[:3, 3]) / (2e-6 * .5)
        rotation = Rotation.from_matrix(plus[:3, :3] @ minus[:3, :3].T).as_rotvec() / 2e-6
        columns.append(np.r_[translation, rotation])
    singular = np.linalg.svd(np.asarray(columns).T, compute_uv=False)
    return dict(joint_margin_rad=margin.tolist(), minimum_joint_margin_rad=float(min(margin)),
                within_joint_limits=bool(np.all(margin >= 0.)),
                scaled_jacobian_singular_values=singular.tolist(),
                scaled_sigma_min=float(singular[-1]),
                near_limit=bool(min(margin) < np.deg2rad(2.)),
                near_singularity=bool(singular[-1] < .001))


def diagnose_path(ik, matrices, initial_joints):
    """Trace production validation once; scan every original pose independently.

    After the first failed original waypoint, continuation is NOT resumed.
    Independent solves always use the archived q0 and cannot certify a path.
    """
    attempts = []
    audited = ik.validate_pose_sequence(matrices, initial_joints, trace=attempts)
    accepted_poses = audited.pop('poses')
    for attempt in attempts:
        attempt['target_pose7'] = pose7(np.asarray(attempt.pop('target_pose')))
        attempt.update(joint_metrics(ik, attempt['joint_positions']))

    rows = []
    previous_independent = np.asarray(initial_joints)
    for index, matrix in enumerate(matrices):
        calls = [a for a in attempts if a['pose_index'] == index]
        committed = [a for a in calls if a['committed']]
        # Original endpoint after successful refinements, or the first full
        # target attempt on failure. All alternatives remain in attempts.json.
        primary = committed[-1] if committed else (calls[0] if calls else None)
        status = 'accepted' if committed else ('failed' if calls else 'not_reached')
        isolated = ik.solve(matrix, initial_joint_pos=initial_joints, max_iterations=100)
        q = np.asarray(isolated.cspace_position[:len(initial_joints)])
        independent = dict(ik_success=bool(isolated.success),
                           position_error=float(isolated.position_error),
                           rotation_error=float(isolated.rotation_error),
                           num_descents=int(isolated.num_descents), joint_positions=q.tolist(),
                           delta_from_q0_rad=(q-initial_joints).tolist(),
                           delta_from_previous_independent_rad=(q-previous_independent).tolist(),
                           **joint_metrics(ik, q))
        previous_independent = q
        display = 'accepted' if committed else (primary['status'] if primary else 'not_reached')
        rows.append(dict(pose_index=index, target_pose7=pose7(matrix),
                         continuity_status=status, display_status=display,
                         attempt_ids=[a['attempt_id'] for a in calls],
                         committed_attempt_ids=[a['attempt_id'] for a in committed],
                         primary_attempt_id=primary['attempt_id'] if primary else None,
                         risk_flag=bool(primary and (primary['near_limit'] or primary['near_singularity'])),
                         independent=independent))

    failed = audited.get('failed_pose_index')
    failure_calls = [a for a in attempts if a['pose_index'] == failed]
    terminal = []
    if failure_calls:
        last = failure_calls[-1]
        terminal = [a for a in failure_calls if a['refinement_depth'] == last['refinement_depth']
                    and a['interval'] == last['interval']]
    maximum = max(attempts, key=lambda a: a['maximum_joint_step_rad'])
    return dict(
        motion_allowed=False, hardware_commands_sent=False,
        scope='same_cartesian_path_offline_ik_diagnostic',
        collision_checked=False, task_constraints_rechecked=False,
        singularity_diagnostic=dict(translation_scale_m=.5, difference_step_rad=1e-6,
                                   sigma_min_warning=.001, joint_margin_warning_deg=2., acceptance_gate=False),
        continuous_ik=audited, accepted_poses7=[pose7(p) for p in accepted_poses],
        initial_joint_metrics=joint_metrics(ik, initial_joints),
        waypoints=rows, attempts=attempts,
        summary=dict(original_waypoint_count=len(rows),
                     continuous_waypoints_accepted=sum(r['continuity_status'] == 'accepted' for r in rows),
                     not_reached_count=sum(r['continuity_status'] == 'not_reached' for r in rows),
                     independent_success_count=sum(r['independent']['ik_success'] for r in rows),
                     first_failed_pose_index=failed,
                     failure_full_target_status_counts=dict(Counter(
                         a['status'] for a in failure_calls if a['refinement_depth'] == 0)),
                     terminal_interval=terminal[0]['interval'] if terminal else None,
                     terminal_status_counts=dict(Counter(a['status'] for a in terminal)),
                     maximum_attempt_jump=dict(attempt_id=maximum['attempt_id'],
                                               pose_index=maximum['pose_index'],
                                               radians=maximum['maximum_joint_step_rad'])))
