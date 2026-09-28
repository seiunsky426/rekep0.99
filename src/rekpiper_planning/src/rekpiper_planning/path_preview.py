"""Hardware-free geometry and live-seeded IK for the PathSolver inspector."""

import numpy as np
from scipy.spatial.transform import Rotation

from .continuous_ik import validate_continuous_ik


def display_indices(count, maximum):
    if maximum < 2:
        raise ValueError('display limit must preserve both endpoints')
    return np.unique(np.linspace(0, count - 1, min(count, maximum), dtype=int))


def pose_matrices(poses):
    poses = np.asarray(poses, dtype=float)
    if (poses.ndim != 2 or poses.shape[1] != 7 or not len(poses)
            or not np.all(np.isfinite(poses))
            or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1., atol=1e-5)):
        raise ValueError('expected finite unit-quaternion poses: x y z qx qy qz qw')
    matrices = np.tile(np.eye(4), (len(poses), 1, 1))
    matrices[:, :3, 3] = poses[:, :3]
    matrices[:, :3, :3] = Rotation.from_quat(poses[:, 3:]).as_matrix()
    return matrices


def matrix_pose(matrix):
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]


def live_joint_seed(names, positions, stamp, now, received_age, ik, maximum_age=.5):
    if not 0 < stamp or not 0 <= now - stamp <= maximum_age or not 0 <= received_age <= maximum_age:
        raise ValueError('live joint feedback is missing, stale or future-dated')
    if len(names) != len(positions) or len(set(names)) != len(names):
        raise ValueError('joint feedback has duplicate names or inconsistent lengths')
    mapping = dict(zip(names, positions))
    if any(name not in mapping for name in ik.joint_names):
        raise ValueError('live joint feedback is missing required joints')
    seed = np.asarray([mapping[name] for name in ik.joint_names], dtype=float)
    if (not np.all(np.isfinite(seed)) or np.any(seed < ik._lower)
            or np.any(seed > ik._upper)):
        raise ValueError('live joints are nonfinite or outside URDF limits')
    return seed


def preview_ik(ik, poses, seed, sequence=True):
    """Full-resolution path validation; display thinning never affects IK."""
    matrices = pose_matrices(poses)
    seed = np.asarray(seed, dtype=float)
    if sequence:
        attempts = []
        report = validate_continuous_ik(ik, matrices, seed, trace=attempts)
        report['attempts'] = attempts
    else:
        result = ik.solve(matrices[-1], initial_joint_pos=seed, max_iterations=100)
        report = dict(valid=bool(result.success), reason=result.status,
                      failed_pose_index=None if result.success else 0,
                      steps=[dict(joint_positions=result.cspace_position[:len(seed)],
                                  position_error=result.position_error,
                                  rotation_error=result.rotation_error)] if result.success else [],
                      target_result=dict(position_error=result.position_error,
                                         rotation_error=result.rotation_error,
                                         candidate_joints=result.cspace_position[:len(seed)]))
    # A failed continuation may contain a valid prefix, but cannot be played as
    # a successful full path. Keep it only as diagnostic evidence.
    joints = np.asarray([seed] + [step['joint_positions'] for step in report['steps']])
    report.update(initial_joint_positions=seed, joint_names=ik.joint_names,
                  joint_positions=joints, joint_delta_rad=joints - seed,
                  joint_step_rad=np.diff(joints, axis=0),
                  motion_allowed=False, collision_checked=False,
                  preview_kind='continuous_path' if sequence else 'endpoint_only')
    return report
