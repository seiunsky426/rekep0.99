#!/usr/bin/env python3
"""Replay the red preview array itself using production continuous IK; no ROS IO."""

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

from rekpiper_planning import continuous_ik, ik_path_diagnostics, piper_urdf_ik
from rekpiper_planning.ik_path_diagnostics import COLORS, diagnose_path
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def vector_columns(name, values):
    return {name+str(i+1): v for i, v in enumerate(values)}


def export_csv(output, report):
    calls = report['attempts']
    call_rows = []
    for a in calls:
        row = {k: a[k] for k in ('attempt_id', 'pose_index', 'refinement_depth', 'seed_index',
               'ik_success', 'status', 'selected', 'committed', 'num_descents', 'position_error',
               'rotation_error', 'maximum_joint_step_rad', 'minimum_joint_margin_rad',
               'scaled_sigma_min', 'near_limit', 'near_singularity', 'within_joint_limits')}
        row.update(interval_start=a['interval'][0], interval_end=a['interval'][1])
        row.update(zip(('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'), a['target_pose7']))
        for field, prefix in [('seed_joints', 'seed_q'), ('previous_joints', 'previous_q'),
                              ('joint_positions', 'q'), ('joint_delta_rad', 'delta_q'),
                              ('joint_margin_rad', 'margin_q')]:
            row.update(vector_columns(prefix, a[field]))
        row.update({'edge_'+k: v for k, v in a['edge_check'].items()})
        call_rows.append(row)
    write_csv(output/'attempts.csv', call_rows)
    rows = []
    for w in report['waypoints']:
        row = {k: w[k] for k in ('pose_index', 'continuity_status', 'display_status', 'primary_attempt_id')}
        row.update(zip(('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'), w['target_pose7']))
        if w['primary_attempt_id'] is not None:
            row.update({'continuous_'+k: v for k, v in call_rows[w['primary_attempt_id']].items()})
        a = w['independent']
        for k in ('ik_success', 'position_error', 'rotation_error', 'num_descents',
                  'minimum_joint_margin_rad', 'scaled_sigma_min'):
            row['independent_'+k] = a[k]
        for field, prefix in [('joint_positions', 'independent_q'),
                              ('delta_from_q0_rad', 'independent_delta_from_q0_q'),
                              ('delta_from_previous_independent_rad', 'independent_adjacent_delta_q'),
                              ('joint_margin_rad', 'independent_margin_q')]:
            row.update(vector_columns(prefix, a[field]))
        rows.append(row)
    write_csv(output/'waypoints.csv', rows)


def plot_evidence(output, original, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    ax = fig.add_subplot(221, projection='3d')
    ax.plot(*original[:, :3].T, color='red', lw=1.5, label='same rejected Cartesian path')
    for status, color in COLORS.items():
        points = [w['pose_index'] for w in report['waypoints'] if w['display_status'] == status]
        if points:
            ax.scatter(*original[points, :3].T, c=[color], s=20, label=status)
    failed = report['summary']['first_failed_pose_index']
    if failed is not None:
        ax.text(*original[failed, :3], '  first failure #{}'.format(failed))
    ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)', title='Original 121-point path; gray = continuation not reached')
    ax.legend(fontsize=8)

    ax = fig.add_subplot(222)
    for index, style in [(0, 'position (mm)'), (1, 'rotation (rad)')]:
        key = 'position_error' if index == 0 else 'rotation_error'
        scale = 1000 if index == 0 else 1
        ax.plot([w['pose_index'] for w in report['waypoints']],
                [w['independent'][key]*scale for w in report['waypoints']], label=style)
    ax.axhline(10, color='C0', ls=':', label='position tolerance 10 mm')
    ax.axhline(.1, color='C1', ls=':', label='rotation tolerance 0.1 rad')
    ax.set(yscale='log', xlabel='original waypoint index', title='Independent q0-seeded IK only; does not certify continuity')
    ax.legend(fontsize=8)

    attempts = report['attempts']
    ids = [a['attempt_id'] for a in attempts]
    ax = fig.add_subplot(223)
    for j in range(6):
        ax.plot(ids, [a['joint_delta_rad'][j] for a in attempts], lw=.8, label='joint{}'.format(j+1))
    ax.axhline(.35, c='black', ls='--'); ax.axhline(-.35, c='black', ls='--')
    ax.set(xlabel='production attempt ID (includes seeds / refinements)', ylabel='q_result - q_previous (rad)',
           title='Actual production attempts; bounded joints, no angle wrapping')
    ax.legend(ncol=3, fontsize=8)

    ax = fig.add_subplot(224)
    ax.plot(ids, [a['minimum_joint_margin_rad'] for a in attempts], label='minimum limit margin (rad)')
    ax.plot(ids, [a['scaled_sigma_min'] for a in attempts], label='scaled Jacobian sigma_min')
    ax.axhline(np.deg2rad(2), c='C0', ls=':', label='2 degree warning')
    ax.axhline(.001, c='C1', ls=':', label='sigma warning 0.001')
    ax.set_yscale('symlog', linthresh=1e-6)
    ax.set(xlabel='production attempt ID',
           title='Diagnostic risk indicators; not new rejection gates')
    ax.legend(fontsize=8)
    fig.suptitle('M8 frozen red-path replay | offline only | no hardware commands', fontsize=16)
    fig.savefig(output/'diagnostics.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-directory', required=True)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args()
    source, output = Path(args.evidence_directory).resolve(), Path(args.output_directory).resolve()
    archive = source/'m8_preview_data.npz'
    state_file, preview_file = source/'state_manifest.json', source/'preview_manifest.json'
    state, preview = json.loads(state_file.read_text()), json.loads(preview_file.read_text())
    model = ROOT/state['robot_description']['path']
    if digest(archive) != preview['preview_data_sha256'] or digest(model) != state['robot_description']['sha256']:
        raise ValueError('archived preview or URDF hash mismatch')
    with np.load(archive, allow_pickle=False) as data:
        poses7 = data['candidate_path_poses7'].copy()
        q0 = np.asarray(state['initial_joints_rad'], dtype=float)
        if bool(data['motion_allowed']) or not np.array_equal(q0, data['endpoint_joints_rad'][0]):
            raise ValueError('not an offline preview with matching archived q0')
    ik = PiperURDFIKSolver.from_urdf_xml(model.read_text(), state['base_frame'],
                                      state['robot_description']['tip_frame'],
                                      ['joint{}'.format(i) for i in range(1, 7)])
    if any(j.joint_type == 'continuous' for j in ik._chain):
        raise ValueError('this Piper diagnostic expects bounded revolute joints')
    matrices = np.repeat(np.eye(4)[None], len(poses7), axis=0)
    matrices[:, :3, 3] = poses7[:, :3]
    matrices[:, :3, :3] = Rotation.from_quat(poses7[:, 3:]).as_matrix()
    if not np.allclose(ik.forward(q0), matrices[0], atol=1e-8):
        raise ValueError('red path does not start at archived FK(q0)')
    output.mkdir(parents=True, exist_ok=False)
    print('Replaying original red preview: {} poses, sha256={}'.format(len(poses7), digest(archive)), flush=True)
    report = diagnose_path(ik, matrices, q0)
    report['provenance'] = dict(
        source_files={str(p): digest(p) for p in (archive, state_file, preview_file, model)},
        source_code={str(p): digest(p) for p in map(Path, (continuous_ik.__file__,
                     ik_path_diagnostics.__file__, piper_urdf_ik.__file__, __file__))},
        path_array_sha256=hashlib.sha256(poses7.tobytes()).hexdigest(),
        original_failure_index=preview['candidate_failure_index'], initial_joints_rad=q0.tolist(),
        initial_joint_source=state['initial_joint_source'], formal_current_state_valid=state['formal_current_state_valid'],
        python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        base_frame=ik.base_frame, tip_frame=ik.tip_frame,
        position_tolerance_m=ik.position_tolerance, orientation_tolerance_rad=ik.orientation_tolerance,
        production_budgets={k: v.default for k, v in inspect.signature(continuous_ik.validate_continuous_ik).parameters.items()
                            if k.startswith('max')},
        independent_scan='same original waypoints, q0 seed per pose, max_iterations=100; no propagation')
    text = json.dumps(report, indent=2, allow_nan=False)
    (output/'report.json').write_text(text+'\n')
    export_csv(output, report)
    np.savez_compressed(output/'continuous_ik_preview.npz', candidate_path_poses7=poses7,
                        initial_joints_rad=q0, continuous_ik_report_json=text, motion_allowed=False)
    plot_evidence(output, poses7, report)
    print(json.dumps(report['summary'], indent=2), flush=True)
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    main()
