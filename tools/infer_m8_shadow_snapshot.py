#!/usr/bin/env python3
"""Task-instance-bound offline SDK diagnostic; no ROS publishers or actions.

This is not a tracked-object or approved-program substitute. The report retains
those missing gates, including truncated masks and an invalid measured start.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

from rekpiper_execution.trajectory import normalize_feedback_to_joint_limits
from rekpiper_grasp.anygrasp_adapter import AnyGraspAdapter
from rekpiper_grasp.grasp_geometry import anygrasp_to_piper_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-directory', required=True)
    parser.add_argument('--target-group', required=True, type=int)
    args = parser.parse_args()
    root = Path(args.evidence_directory)
    result_path = root / 'target_anygrasp_report.json'
    if result_path.exists():
        raise FileExistsError(str(result_path))
    capture = json.loads((root / 'capture_report.json').read_text())
    if not capture['capture_success']:
        raise ValueError('capture did not pass')
    began = time.monotonic()
    finished = threading.Event()
    def heartbeat():
        while not finished.wait(5):
            print('TARGET_SDK_RUNNING elapsed_s={:.1f}'.format(time.monotonic()-began), flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    report = {'motion_allowed': False, 'hardware_commands_sent': False,
              'full_task_accepted': False, 'snapshot_id': capture['snapshot_id'],
              'target_group': args.target_group, 'candidate_scope': 'immutable_snapshot_diagnostic',
              'tracked_object_uuid': None, 'program_rebinding_required': True,
              'start_and_environment_audit': {'valid': False, 'reasons': []}}
    try:
        labels = cv2.imread(str(root / 'scene_mask.png'), cv2.IMREAD_UNCHANGED)
        mask = labels == args.target_group
        if mask.sum() < 120:
            raise ValueError('target_mask_has_fewer_than_120_pixels')
        report['mask_sha256'] = hashlib.sha256(mask.tobytes()).hexdigest()
        report['target_mask_pixels'] = int(mask.sum())
        report['target_mask_touches_image_edge'] = bool(
            mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        data = np.load(root / 'rs1_rgbd.npz', allow_pickle=False)
        z = data['depth'].astype(np.float32)
        encoding = str(data['depth_encoding'].item())
        if encoding in ('16UC1', 'mono16'):
            z *= .001
        elif encoding != '32FC1':
            raise ValueError('unknown_depth_units')
        k = data['K']; v, u = np.indices(z.shape)
        points = np.stack([(u-k[2])*z/k[0], (v-k[5])*z/k[4], z], axis=-1).reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > .15) & (points[:, 2] < 1.5)
        target_indices = np.flatnonzero(valid & mask.ravel())
        background_indices = np.flatnonzero(valid & ~mask.ravel())
        background_indices = background_indices[::max(1, int(np.ceil(len(background_indices)/30000)))]
        indices = np.r_[target_indices, background_indices]
        scene = np.ascontiguousarray(points[indices], dtype=np.float32)
        region = np.ascontiguousarray(mask.ravel()[indices], dtype=np.bool_)
        if region.sum() < 120:
            raise ValueError('insufficient_target_depth_points')
        transform = data['base_from_camera']
        target_base = points[target_indices] @ transform[:3, :3].T + transform[:3, 3]
        report['inference_points'] = len(scene)
        report['target_depth_points'] = len(target_base)
        report['target_center_base'] = np.median(target_base, axis=0).tolist()
        q = np.array([capture['joints']['joint{}'.format(i)] for i in range(1, 7)])
        reasons = report['start_and_environment_audit']['reasons']
        try:
            normalize_feedback_to_joint_limits(q, .01)
        except ValueError as exc:
            reasons.append({'step': 'trajectory_start_index_0', 'error': str(exc)})
        safe = capture.get('safe_map') or {}
        if not safe.get('planning_safe') or not safe.get('map_query_allowed'):
            reasons.append({'step': 'environment_sdf', 'error': 'formal_safe_sdf_not_ready'})
        reasons.append({'step': 'object_program_binding',
                        'error': 'live_tracked_uuid_and_new_snapshot_program_binding_missing'})
        if report['target_mask_touches_image_edge']:
            reasons.append({'step': 'target_geometry', 'error': 'target_mask_truncated_at_image_boundary'})
        sdk = Path(os.environ['REKPIPER_VENDOR_ROOT']) / 'anygrasp_sdk/grasp_detection'
        adapter = AnyGraspAdapter(sdk, sdk/'checkpoint_detection.tar', sdk/'license')
        inference_start = time.monotonic()
        print('TARGET_ANYGRASP_START group={} points={} target={}'.format(
            args.target_group, len(scene), int(region.sum())), flush=True)
        grasps = adapter.infer(scene, region, 'rs1', max_candidates=20)
        report['sdk_inference_s'] = time.monotonic()-inference_start
        tree = cKDTree(target_base)
        converted, rows = [], []
        for index, grasp in enumerate(grasps):
            candidate = anygrasp_to_piper_pose(grasp, transform,
                candidate_id='snapshot-grasp-{}'.format(index),
                interaction_region_id='{}:G{}'.format(capture['snapshot_id'], args.target_group),
                part_name='blue_cube')
            contact = np.array([candidate.tcp_position + sign*grasp.width_m*.5*candidate.closing_axis_base
                                for sign in (-1, 1)])
            distance = tree.query(contact)[0]
            rows.append({'candidate_id': candidate.candidate_id,
                         'network_score': candidate.network_score,
                         'grasp_pose_base': candidate.grasp_pose.tolist(),
                         'pregrasp_pose_base': candidate.pregrasp_pose.tolist(),
                         'predicted_width_m': candidate.predicted_width_m,
                         'suggested_preopen_width_m': candidate.suggested_preopen_width_m,
                         'contact_distance_to_observed_target_m': distance.tolist(),
                         'both_contacts_supported_within_5mm': bool(np.all(distance <= .005)),
                         'executable': False})
            converted.append(candidate)
        report['sdk_success'] = True
        report['candidate_count'] = len(rows)
        report['candidates'] = rows
        report['trajectory_generated'] = False
        report['trajectory_rejection'] = 'invalid_measured_start_and_missing_scene_gates'
        np.savez_compressed(root/'target_anygrasp_preview.npz',
            target_points_base=target_base,
            grasp_matrices=np.array([c.grasp_pose for c in converted]).reshape(-1, 4, 4),
            pregrasp_matrices=np.array([c.pregrasp_pose for c in converted]).reshape(-1, 4, 4),
            widths_m=np.array([c.predicted_width_m for c in converted]),
            actual_joints_rad=q, actual_tcp=np.array(capture['fk_base_tcp']),
            snapshot_id=capture['snapshot_id'], motion_allowed=False)
    except Exception as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        report['elapsed_s'] = time.monotonic()-began
        result_path.write_text(json.dumps(report, indent=2))
        finished.set()
        print('TARGET_SDK_RESULT', {k: v for k, v in report.items() if k != 'candidates'}, flush=True)
    return 1 if 'error' in report else 0


if __name__ == '__main__':
    raise SystemExit(main())
