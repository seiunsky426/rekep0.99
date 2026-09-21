#!/usr/bin/env python3
"""Audit a user-declared future zero start without changing robot feedback."""
import argparse
from dataclasses import asdict
import io
import json
from pathlib import Path
import time

import numpy as np
import rospy
import yaml

from rekpiper_msgs.msg import SDFGrid, SafeMappingStatus
from rekpiper_planning.motion_backend import MotionBackend
from rekpiper_planning.piper_collision_sampling import PiperCollisionSampler
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver
from rekpiper_planning.trajectory_audit import audit_joint_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-directory', required=True)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args()
    source, output = Path(args.evidence_directory), Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    rospy.init_node('m8_zero_start_readonly_audit', anonymous=True, disable_signals=True)
    start = time.monotonic()
    report = {'motion_allowed': False, 'hardware_commands_sent': False,
              'initial_state_source': 'user_declared_future_post_zero',
              'planned_initial_joints_rad': [0.] * 6,
              'actual_start_verified': False, 'full_trajectory_qualified': False,
              'full_trajectory_generated': False, 'endpoints': []}
    backend = None
    try:
        xml = (source/'robot_description.urdf').read_text()
        capture = json.loads((source/'capture_report.json').read_text())
        report['source_snapshot_id'] = capture['snapshot_id']
        report['recorded_actual_joints'] = capture['joints']
        ik = PiperURDFIKSolver.from_urdf_xml(
            xml, 'base_link', 'rekep_tcp', ['joint'+str(i) for i in range(1, 7)])
        q0 = np.zeros(6)
        backend = MotionBackend(xml)
        report['zero_fk_base_tcp'] = ik.forward(q0).tolist()
        report['zero_joint_bounds_pass'] = bool(np.all(q0 >= ik._lower) and np.all(q0 <= ik._upper))
        report['zero_self_collision_by_planned_opening'] = {
            str(width): backend.self_clear(q0, width) for width in [0., .001, .01, .04, .07]}
        scene = yaml.safe_load((source/'scene_snapshot.yaml').read_text())
        points = {k['id']: np.array([k['position'][axis] for axis in ('x', 'y', 'z')])
                  for k in scene['keypoints']['keypoints']}
        # This particular archived snapshot was visually checked: blue K9/G5,
        # yellow K8/G1. Reject another layout instead of silently reusing IDs.
        if capture['snapshot_id'] != 'snap-1789388469225849152-c1506d059bbd':
            raise ValueError('snapshot_semantic_binding_requires_new_review')
        report['semantic_binding'] = {'blue_cube': 9, 'yellow_disk': 8,
                                     'transport_height_m': .10, 'place_height_m': .02}
        candidates = json.loads((source/'target_anygrasp_report.json').read_text())['candidates']
        for index, candidate in enumerate(candidates):
            base = np.array(candidate['grasp_pose_base']); seed = q0.copy()
            row = {'candidate_index': index,
                   'contact_distance_screen_pass': candidate['both_contacts_supported_within_5mm'],
                   'holding_relation': 'hypothetical_for_preview_only', 'poses': []}
            for label in ('pregrasp', 'grasp', 'transport', 'place'):
                target = (np.array(candidate['pregrasp_pose_base']) if label == 'pregrasp' else base.copy())
                if label in ('transport', 'place'):
                    # Carry the same observed object-to-TCP offset, not TCP=K.
                    target[:3, 3] += points[8]+[0., 0., .10 if label == 'transport' else .02]-points[9]
                result = ik.solve(target, initial_joint_pos=seed, max_iterations=100)
                row['poses'].append({'label': label, 'pose_base': target.tolist(), **asdict(result)})
                if result.success:
                    seed = result.cspace_position[:6]
            row['all_fixed_orientation_endpoints_pass'] = all(p['success'] for p in row['poses'])
            report['endpoints'].append(row)
            print('ZERO_ENDPOINTS', index, [bool(p['success']) for p in row['poses']], flush=True)
        status = rospy.wait_for_message('/rekpiper/mapping/safe_status', SafeMappingStatus, timeout=5)
        report['current_map_status'] = {'planning_safe': status.planning_safe,
                                      'map_query_allowed': status.map_query_allowed, 'reason': status.reason}
        grid = rospy.wait_for_message('/rekpiper/mapping/diagnostic_sdf_grid', SDFGrid, timeout=5)
        buffer = io.BytesIO(); grid.serialize(buffer)
        (output/'unmodified_diagnostic_grid.rosmsg').write_bytes(buffer.getvalue())
        report['map'] = {'valid': grid.valid, 'status': grid.status,
                         'stamp_s': grid.header.stamp.to_sec(),
                         'age_at_audit_s': (rospy.Time.now()-grid.header.stamp).to_sec()}
        sampler = PiperCollisionSampler(xml, ik, voxel_size_m=.025, maximum_points_per_link=160)
        try:
            report['zero_environment_audit'] = audit_joint_path([q0], sampler, grid)
        except Exception as exc:
            report['zero_environment_audit'] = {'valid': False, 'step': 'start_index_0_environment',
                                               'error': type(exc).__name__+': '+str(exc)}
        report['planning_stopped_before_path_search'] = True
        report['blocking_reason'] = 'no_valid_current_environment_for_full_path_audit'
        report['stage_status'] = {str(i): 'not_qualified_environment_gate_failed' for i in (1, 2, 3)}
    except Exception as exc:
        report['error'] = type(exc).__name__+': '+str(exc)
    finally:
        if backend is not None:
            backend.close()
        report['elapsed_s'] = time.monotonic()-start
        def encode(value):
            return value.tolist() if isinstance(value, np.ndarray) else value.item()
        (output/'report.json').write_text(json.dumps(report, indent=2, default=encode))
        print('ZERO_START_REPORT', output/'report.json', 'qualified=False', flush=True)
    return 1 if 'error' in report else 0


if __name__ == '__main__':
    raise SystemExit(main())
