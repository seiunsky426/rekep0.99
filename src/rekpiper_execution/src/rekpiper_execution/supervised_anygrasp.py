"""Frozen selected masks -> licensed directional AnyGrasp -> preview candidates."""
import json
from pathlib import Path
import time
import uuid

import numpy as np

from rekpiper_grasp.anygrasp_adapter import AnyGraspAdapter, CameraGrasp
from rekpiper_grasp.blue_cloud_comparison import prepare_case, sha256, write_json
from rekpiper_grasp.directional_grasp import load_gripper
from rekpiper_grasp.grasp_geometry import anygrasp_to_piper_pose
from .supervised_geometry import load_experiment


def frozen_input(root, config):
    """Use both confirmed masks at their original coordinates, never display shifts."""
    snapshot = root/'segmentation_snapshot.npz'
    selection = json.loads((root/'vlm_selection.json').read_text())
    snapshot_hash = sha256(snapshot)
    if selection.get('snapshot_sha256') != snapshot_hash:
        raise ValueError('anygrasp_selection_snapshot_mismatch')
    transforms, binding = load_experiment(config)
    selected = {}
    with np.load(snapshot, allow_pickle=False) as data:
        if abs(float(selection['stamp'])-int(data['stamp_ns'])*1e-9) > .001:
            raise ValueError('anygrasp_selection_stamp_mismatch')
        if abs(int(data['rs3_stamp_ns'])-int(data['stamp_ns'])) > 25_000_000:
            raise ValueError('anygrasp_frozen_pair_skew_exceeds_25ms')
        for name, prefix, group in (
                ('rs1', '', selection['selection'][0]['rigid_group_id']),
                ('rs3', 'rs3_', selection['rs3_group_id'])):
            xyz, mask = data[prefix+'xyz'], data[prefix+'mask']
            if xyz.shape != mask.shape+(3,) or int(group) <= 0:
                raise ValueError('anygrasp_frozen_mask_shape_or_group')
            valid = np.isfinite(xyz).all(axis=2)
            valid &= np.all(xyz >= config['bounds_min'], axis=2)
            valid &= np.all(xyz <= config['bounds_max'], axis=2)
            target = xyz[valid & (mask == int(group))]
            background = xyz[valid & (mask != int(group))]
            if len(target) < 120:
                raise ValueError(name+'_target_mask_has_insufficient_depth')
            selected[name] = dict(target=target, background=background,
                target_rgb=np.tile([0, 190, 255], (len(target), 1)),
                background_rgb=np.full((len(background), 3), 155))
    packed = prepare_case(selected, ('rs1', 'rs3'), transforms['rs1'])
    return packed, dict(snapshot_sha256=snapshot_hash, selection=selection,
                       extrinsics=binding, frame='base_link', reference='rs1_color_optical_frame')


def generate_candidates(planner, scene, joints):
    folder = planner.output/('anygrasp-'+uuid.uuid4().hex)
    folder.mkdir(parents=True)
    packed, binding = frozen_input(planner.output, planner.config)
    # The exact selected dual masks now bind both candidate and path target geometry.
    planner.target = packed['points_arm'][packed['region_mask']]
    np.savez_compressed(folder/'input.npz', **packed)
    gripper = load_gripper(planner.xml)
    repo = Path(__file__).resolve().parents[4]
    sdk = repo/'runtime/vendor/anygrasp_sdk/grasp_detection'
    started = time.monotonic()
    detector = AnyGraspAdapter(sdk, sdk/'checkpoint_detection.tar', sdk/'license',
                              max_gripper_width_m=gripper.maximum_opening_m,
                              gripper_height_m=gripper.finger_height_m)
    load_seconds = time.monotonic()-started
    started = time.monotonic()
    rows = detector.infer_directional(
        packed['points_camera'], packed['region_mask'], 'rs1+rs3',
        packed['arm_from_reference'], scene.table['table_plane_normal'],
        scene.table['table_plane_offset_m'], gripper, planner.cancelled)
    report = dict(binding=binding, input_sha256=sha256(folder/'input.npz'),
                  table=scene.table, model_load_s=load_seconds,
                  inference_and_geometry_s=time.monotonic()-started,
                  candidates=rows, hardware_commands_sent=False,
                  planning_authorized=False, contact_evidence_required_for_execution=True)
    # Save all model/geometry results before IK so failures remain inspectable.
    write_json(folder/'audit.json', report)
    eligible = []
    for row in sorted(rows, key=lambda r: (not r['contact_supported'], r['mode'] != 'top', -r['score'])):
        planner.check_cancel()
        if not row['geometry_pass']:
            continue
        if not row['contact_supported'] and planner.config.get('extrinsics_status') != 'EXPERIMENTAL_PREVIEW_ONLY':
            row['planning_rejection'] = 'observed_contacts_unverified'
            continue
        pose = np.asarray(row['pose_reference'])
        native = CameraGrasp(pose[:3, 3], pose[:3, :3], row['width_m'], row['depth_m'], row['score'], 'rs1+rs3')
        candidate = anygrasp_to_piper_pose(native, packed['arm_from_reference'],
            pregrasp_distance_m=.06, physical_opening_m=gripper.maximum_opening_m,
            candidate_id=row['id'], part_name='selected_blue_cube')
        candidate.contact_points_base = row['contact_points_arm_base']
        candidate.contact_membership_ok = row['contact_supported']
        candidate.target_points_base = planner.target
        pre = planner.ik.solve(candidate.pregrasp_pose, max_iterations=150, initial_joint_pos=joints)
        grasp = planner.ik.solve(candidate.grasp_pose, max_iterations=150, initial_joint_pos=pre.cspace_position[:6])
        row['endpoint_ik'] = dict(pregrasp=bool(pre.success), grasp=bool(grasp.success),
                                 pregrasp_position_error_m=pre.position_error,
                                 grasp_position_error_m=grasp.position_error,
                                 pregrasp_rotation_error_rad=pre.rotation_error,
                                 grasp_rotation_error_rad=grasp.rotation_error)
        if not pre.success or not grasp.success:
            row['planning_rejection'] = 'pregrasp_or_grasp_ik_failed'
            continue
        if not planner.backend.self_clear(pre.cspace_position[:6], candidate.suggested_preopen_width_m) or not planner.backend.self_clear(grasp.cspace_position[:6], candidate.suggested_preopen_width_m):
            row['planning_rejection'] = 'endpoint_self_collision'
            continue
        candidate.ik_ok = True
        eligible.append(candidate)
    report['eligible_for_path'] = [c.candidate_id for c in eligible]
    write_json(folder/'audit.json', report)
    planner.grasp_evidence = str(folder)
    planner.rejections = [dict(id=r['id'], reasons=r['rejection_reasons'],
                               planning_rejection=r.get('planning_rejection'))
                          for r in rows if not r['geometry_pass'] or r.get('planning_rejection')]
    if not eligible:
        raise ValueError('anygrasp_no_preview_candidate: raw={} geometry={} endpoint_ik_and_self_clear=0; evidence={}'.format(
            len(rows), sum(r['geometry_pass'] for r in rows), folder))
    return eligible
