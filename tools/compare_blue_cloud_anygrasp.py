#!/usr/bin/env python3
"""Frozen three-input color/AnyGrasp experiment; no hardware command clients."""
import argparse
import json
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np
import yaml

from rekpiper_grasp.blue_cloud_comparison import (
    CASES, POLICY, MODEL_POLICY, sha256, write_json, select_blue, prepare_case,
    transform_points, native_pose_record, gripper_segments, write_ply)

REPO = Path(__file__).resolve().parents[1]


def capture(root):
    import rospy
    from rekpiper_execution.supervised_io import Observations
    from rekpiper_execution.supervised_geometry import load_experiment
    config_path = REPO/'src/rekpiper_bringup/config/supervised_workbench.yaml'
    config = yaml.safe_load(config_path.read_text().replace('${REKPIPER_ROOT}', str(REPO)))
    transforms, binding = load_experiment(config)
    arm_from_link = np.linalg.inv(np.asarray(config['base_link_T_arm_base']))
    rospy.init_node('blue_cloud_comparison_capture', anonymous=True)
    for name in ('rs1', 'rs3'):
        actual = str(rospy.get_param('/'+name+'/realsense2_camera/serial_no')).lstrip('_')
        if actual != str(config['serials'][name]):
            raise ValueError(name+'_serial_mismatch')
    observations = Observations()
    observations.transforms = {n: arm_from_link @ t for n, t in transforms.items()}
    deadline, error = time.monotonic()+30, ''
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        try:
            frames = observations.capture()
            break
        except ValueError as exc:
            error = str(exc)
            rospy.sleep(.05)
    else:
        raise RuntimeError('capture_timeout:'+error)
    _, current = load_experiment(config)
    if current != binding:
        raise ValueError('extrinsics_changed_during_capture')
    root.mkdir(parents=True, exist_ok=False)
    (root/'extrinsics').mkdir()
    shutil.copyfile(config_path, root/'extrinsics/workbench.yaml')
    for name, key in (('rs1_handeye.yaml', 'rs1_extrinsics'), ('stereo_result.yaml', 'stereo_extrinsics')):
        shutil.copyfile(config[key], root/'extrinsics'/name)
    arrays = {n+'_'+k: v for n, frame in frames.items() for k, v in frame.items()}
    np.savez_compressed(root/'snapshot.npz', **arrays)
    stamps = [float(f[k]) for f in frames.values() for k in ('rgb_stamp_s', 'depth_stamp_s', 'info_stamp_s')]
    metadata = dict(frame='arm_base', inference_reference='rs1_color_optical_frame',
                    pairing='software RGB-D timestamp span <=25ms; static scene, not hardware exposure sync',
                    maximum_timestamp_span_s=max(stamps)-min(stamps),
                    snapshot_sha256=sha256(root/'snapshot.npz'), binding=binding,
                    workspace_bounds_min=config['bounds_min'], workspace_bounds_max=config['bounds_max'],
                    workspace_from_arm=config['base_link_T_arm_base'],
                    arm_from_camera={n: f['base_from_camera'].tolist() for n, f in frames.items()},
                    extrinsics_status=config['extrinsics_status'],
                    policy=POLICY, model_policy=MODEL_POLICY, hardware_commands_sent=False,
                    registration_adjustment_applied=False)
    write_json(root/'capture.json', metadata)
    for n, frame in frames.items():
        cv2.imwrite(str(root/(n+'_source.png')), cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2BGR))
    print(json.dumps(dict(output=str(root), timestamp_span_s=max(stamps)-min(stamps))), flush=True)


def prepare(root):
    meta = json.loads((root/'capture.json').read_text())
    if sha256(root/'snapshot.npz') != meta['snapshot_sha256']:
        raise ValueError('snapshot_hash_mismatch')
    selections, reports = {}, {}
    with np.load(root/'snapshot.npz', allow_pickle=False) as data:
        for name in ('rs1', 'rs3'):
            frame = {k[len(name)+1:]: data[k] for k in data.files if k.startswith(name+'_')}
            selected = select_blue(frame, np.asarray(meta['workspace_bounds_min']),
                                   np.asarray(meta['workspace_bounds_max']),
                                   np.asarray(meta['workspace_from_arm']))
            selections[name] = selected
            reports[name] = selected['report']
            overlay = cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2BGR)
            contours, _ = cv2.findContours(selected['mask'].astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
            cv2.putText(overlay, '{}: {} valid blue points'.format(name.upper(), len(selected['target'])),
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
            cv2.imwrite(str(root/(name+'_color_selection.png')), overlay)
            cv2.imwrite(str(root/(name+'_color_mask.png')), selected['mask'].astype(np.uint8)*255)
            write_ply(root/(name+'_target_raw.ply'), selected['raw'], selected['raw_rgb'])
            write_ply(root/(name+'_target_filtered.ply'), selected['target'], selected['target_rgb'])
            np.savez_compressed(root/(name+'_color_clouds.npz'), **{
                k: v for k, v in selected.items() if k != 'report'})
    for case in CASES:
        names = ('rs1', 'rs3') if case == 'fused' else (case,)
        case_root = root/case
        case_root.mkdir(exist_ok=True)
        data = prepare_case(selections, names, np.asarray(meta['arm_from_camera']['rs1']))
        np.savez_compressed(case_root/'input.npz', **data)
        failed = [n for n in names if reports[n]['status'] != 'ready']
        mask = data['region_mask']
        report = dict(case=case, source_cameras=list(names), frame='arm_base',
                      status='insufficient_blue_depth_points' if failed else 'ready',
                      failed_cameras=failed, snapshot_sha256=meta['snapshot_sha256'],
                      input_sha256=sha256(case_root/'input.npz'), policy=POLICY,
                      inference_reference='rs1_color_optical_frame',
                      target_points=int(mask.sum()), background_points=int((~mask).sum()),
                      background_voxels_before_sampling=int(data['background_voxels_before_sampling']),
                      source_bits={'1': 'rs1', '2': 'rs3', '3': 'both'},
                      target_source_counts={str(bit): int(np.count_nonzero(data['source_bits'][mask] == bit))
                                            for bit in (1, 2, 3)})
        write_json(case_root/'input.json', report)
        write_ply(case_root/'input_arm_base.ply', data['points_arm'], data['rgb'])
        write_ply(case_root/'target_arm_base.ply', data['points_arm'][mask], data['rgb'][mask])
    write_json(root/'color_selection.json', reports)
    print(json.dumps(reports), flush=True)


def infer(root):
    import torch
    from rekpiper_grasp.anygrasp_adapter import AnyGraspAdapter
    sdk = REPO/'runtime/vendor/anygrasp_sdk/grasp_detection'
    meta = json.loads((root/'capture.json').read_text())
    started = time.monotonic()
    initialization_error = None
    try:
        detector = AnyGraspAdapter(sdk, sdk/'checkpoint_detection.tar', sdk/'license',
                                  max_gripper_width_m=.070, gripper_height_m=.03)
    except Exception as exc:
        initialization_error = '{}: {}'.format(type(exc).__name__, exc)
    load_s = time.monotonic()-started
    checkpoint_hash = sha256(sdk/'checkpoint_detection.tar') if (sdk/'checkpoint_detection.tar').is_file() else None
    summary = dict(snapshot_sha256=meta['snapshot_sha256'], frame='arm_base',
                   model='AnyGrasp', model_policy=MODEL_POLICY, model_load_s=load_s,
                   checkpoint_sha256=checkpoint_hash, hardware_commands_sent=False,
                   physical_success_evaluated=False, cases=[])
    for case in CASES:
        directory = root/case
        input_report = json.loads((directory/'input.json').read_text())
        report = dict(case=case, status='error', candidates=[], model='AnyGrasp',
                      frame='arm_base', pose_convention='native_AnyGrasp_not_Piper_TCP',
                      snapshot_sha256=meta['snapshot_sha256'], input_sha256=input_report['input_sha256'],
                      model_policy=MODEL_POLICY, checkpoint_sha256=checkpoint_hash,
                      random_seed=0, model_load_s=load_s, inference_s=None,
                      hardware_commands_sent=False, ik_checked=False, collision_checked=False)
        try:
            if initialization_error:
                raise RuntimeError(initialization_error)
            if input_report['snapshot_sha256'] != meta['snapshot_sha256'] or sha256(directory/'input.npz') != input_report['input_sha256']:
                raise ValueError('input_snapshot_hash_mismatch')
            if input_report['status'] != 'ready':
                raise ValueError(input_report['status'])
            random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            with np.load(directory/'input.npz', allow_pickle=False) as data:
                points, mask = data['points_camera'], data['region_mask']
                arm_from_reference = data['arm_from_reference']
            torch.cuda.synchronize()
            started = time.monotonic()
            grasps = detector.infer(points, mask, case, dense_grasp=False,
                                    collision_detection=False, max_candidates=None)
            torch.cuda.synchronize()
            report['inference_s'] = time.monotonic()-started
            report['candidates'] = [native_pose_record(g, arm_from_reference, '{}-{:03d}'.format(case, i+1))
                                    for i, g in enumerate(grasps)]
            report['status'] = 'ok' if grasps else 'zero_candidates'
        except Exception as exc:
            report['error'] = '{}: {}'.format(type(exc).__name__, exc)
        write_json(directory/'grasps.json', report)
        row = dict(case=case, target_points=input_report['target_points'],
                   background_points=input_report['background_points'],
                   candidates=len(report['candidates']),
                   highest_score=report['candidates'][0]['score'] if report['candidates'] else None,
                   inference_s=report['inference_s'], status=report['status'], error=report.get('error'))
        summary['cases'].append(row)
        write_json(root/'summary.json', summary)
        print(json.dumps(row), flush=True)


def verify(root):
    import rosbag
    from sensor_msgs import point_cloud2
    meta = json.loads((root/'capture.json').read_text())
    if sha256(root/'snapshot.npz') != meta['snapshot_sha256']:
        raise ValueError('snapshot_changed')
    assert meta['maximum_timestamp_span_s'] <= .025
    assert sha256(root/'extrinsics/rs1_handeye.yaml') == meta['binding']['rs1']
    assert sha256(root/'extrinsics/stereo_result.yaml') == meta['binding']['stereo']
    for name in ('rs1', 'rs3'):
        assert cv2.imread(str(root/(name+'_color_selection.png'))) is not None
    recording = json.loads((root/'rviz_records.json').read_text())
    results, total = [], 0
    for case in CASES:
        input_report = json.loads((root/case/'input.json').read_text())
        report = json.loads((root/case/'grasps.json').read_text())
        assert report['snapshot_sha256'] == input_report['snapshot_sha256'] == meta['snapshot_sha256']
        assert report['input_sha256'] == sha256(root/case/'input.npz')
        assert report['model_policy'] == MODEL_POLICY
        with np.load(root/case/'input.npz', allow_pickle=False) as data:
            points, mask = data['points_arm'], data['region_mask']
            transform = data['arm_from_reference']
            np.testing.assert_allclose(transform_points(data['points_camera'], transform), points, atol=1e-6)
            assert int((~mask).sum()) <= 30000
            assert set(np.unique(data['source_bits'])) <= ({1, 2, 3} if case == 'fused' else {1 if case == 'rs1' else 2})
        for row in report['candidates']:
            pose = np.asarray(row['pose_arm_base'])
            np.testing.assert_allclose(transform @ row['pose_reference'], pose, atol=1e-9)
            np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
        for top_k in (1, 5):
            record = next(r for r in recording['records'] if (r['case'], r['top_k']) == (case, top_k))
            camera = record['actual_camera']
            expected = recording['camera']
            np.testing.assert_allclose([camera[n] for n in ('Distance', 'Pitch', 'Yaw')],
                                       [expected[n] for n in ('Distance', 'Pitch', 'Yaw')], atol=1e-6)
            np.testing.assert_allclose([camera['Focal Point'][n] for n in ('X', 'Y', 'Z')],
                                       [expected['Focal Point'][n] for n in ('X', 'Y', 'Z')], atol=1e-6)
            screenshot = root/record['screenshot']
            image = cv2.imread(str(screenshot))
            assert image is not None and image.shape[1] >= 800 and image.shape[0] >= 500
            assert sha256(screenshot) == record['screenshot_sha256']
            with rosbag.Bag(str(root/record['bag'])) as bag:
                messages = {topic.rsplit('/', 1)[-1]: msg for topic, msg, _ in bag.read_messages()}
            assert set(messages) == {'tf_static', 'background', 'target', 'markers'}
            for name, subset in (('target', mask), ('background', ~mask)):
                msg = messages[name]
                assert msg.header.frame_id == 'arm_base'
                saved = np.asarray(list(point_cloud2.read_points(msg, field_names=('x', 'y', 'z')))).reshape(-1, 3)
                np.testing.assert_allclose(saved, points[subset], atol=1e-7)
            grasp_lines = [m for m in messages['markers'].markers if m.type == 5 and m.action == 0]
            assert len(grasp_lines) == min(top_k, len(report['candidates']))
            for marker, row in zip(grasp_lines, report['candidates']):
                assert marker.header.frame_id == 'arm_base'
                actual = np.array([[p.x, p.y, p.z] for p in marker.points])
                np.testing.assert_allclose(actual, gripper_segments(row), atol=1e-9)
            results.append(dict(case=case, top_k=top_k, screenshot=str(screenshot.relative_to(root)),
                                bag=record['bag'], markers_match_saved_poses=True))
        total += len(report['candidates'])
    result = dict(valid=True, total_candidates=total, maximum_timestamp_span_s=meta['maximum_timestamp_span_s'],
                  frame='arm_base', checked_views=results, hardware_commands_sent=False)
    write_json(root/'verification.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['capture', 'prepare', 'infer', 'verify'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.step](args.output.resolve())


if __name__ == '__main__':
    main()
