#!/usr/bin/env python3
"""Reuse frozen three-view inputs for top/side AnyGrasp and Piper/table audits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from graspnetAPI import GraspGroup

from rekpiper_grasp.anygrasp_adapter import AnyGraspAdapter, CameraGrasp
from rekpiper_grasp.blue_cloud_comparison import sha256, write_json, native_pose_record
from rekpiper_grasp.directional_grasp import load_gripper, direction_queries, DirectionalAudit
from rekpiper_grasp.horizontal_grasp import observed_table_plane

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source, root = args.source.resolve(), args.output.resolve()
    meta = json.loads((source/'capture.json').read_text())
    assert sha256(source/'snapshot.npz') == meta['snapshot_sha256']
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source/'capture.json', root/'capture.json')
    xml_path = REPO/'src/piper_description/urdf/piper_description.urdf'
    xml = xml_path.read_text()
    (root/'piper.urdf').write_text(xml)
    gripper = load_gripper(xml)
    with np.load(source/'rs1/input.npz', allow_pickle=False) as data:
        center = np.median(data['points_arm'][data['region_mask']], axis=0)
        normal, offset, rms = observed_table_plane(data['points_arm'], data['region_mask'], center)
    table = dict(normal=normal.tolist(), offset_m=offset, rms_m=rms,
                 source='frozen_rs1_background_shared_by_all_six_cases',
                 source_input_sha256=sha256(source/'rs1/input.npz'), frame='arm_base',
                 minimum_clearance_m=.003, approach_distance_m=.06)
    write_json(root/'table.json', table)
    geometry = dict(urdf_sha256=sha256(xml_path), maximum_opening_m=gripper.maximum_opening_m,
                    finger_height_m=gripper.finger_height_m, usable_depth_m=gripper.usable_depth_m,
                    tcp_offset_m=gripper.tcp_offset_m, part_meshes=[])
    from rekpiper_planning.piper_collision_sampling import _resolve_mesh
    from urdf_parser_py.urdf import URDF
    robot = URDF.from_xml_string(xml)
    for name in ('gripper_base','link7','link8'):
        for collision in robot.link_map[name].collisions:
            if hasattr(collision.geometry, 'filename'):
                p = _resolve_mesh(collision.geometry.filename)
                geometry['part_meshes'].append(dict(link=name,path=str(p),sha256=sha256(p)))
    write_json(root/'gripper_geometry.json', geometry)
    sdk = REPO/'runtime/vendor/anygrasp_sdk/grasp_detection'
    detector = AnyGraspAdapter(sdk, sdk/'checkpoint_detection.tar', sdk/'license',
                              max_gripper_width_m=gripper.maximum_opening_m,
                              gripper_height_m=gripper.finger_height_m)
    summary = dict(source=str(source), snapshot_sha256=meta['snapshot_sha256'],table=table,
                    geometry=geometry, cases=[],hardware_commands_sent=False,
                    note='Geometry PASS checks direction, width/depth and Piper/table clearance only; contact support separate. No IK/full scene collision or execution.')
    # Retain exact inputs; a fused reference lets the existing viewer share one view.
    (root/'fused').mkdir()
    shutil.copyfile(source/'fused/input.npz', root/'fused/input.npz')
    shutil.copyfile(source/'fused/grasps.json', root/'fused/grasps.json')
    for camera in ('rs1','rs3','fused'):
        input_report = json.loads((source/camera/'input.json').read_text())
        assert input_report['snapshot_sha256'] == meta['snapshot_sha256']
        assert input_report['input_sha256'] == sha256(source/camera/'input.npz')
        with np.load(source/camera/'input.npz', allow_pickle=False) as d:
            points, mask, transform = d['points_camera'], d['region_mask'], d['arm_from_reference']
            target = d['points_arm'][mask]
        audit = DirectionalAudit(target, normal, offset, gripper)
        for mode in ('top','side'):
            name = camera+'_'+mode
            folder = root/name; folder.mkdir()
            shutil.copyfile(source/camera/'input.npz', folder/'input.npz')
            shutil.copyfile(source/camera/'input.json', folder/'input.json')
            directions, cone = direction_queries(normal,mode)
            calls, raw = [], []
            started = time.monotonic()
            for i, direction in enumerate(directions):
                random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
                torch.backends.cudnn.benchmark=False
                torch.backends.cudnn.deterministic=True
                steering = transform[:3,:3].T @ direction
                grasps = detector.infer(points,mask,camera,approach_steering=steering,
                                        approach_thresh_rad=np.deg2rad(cone), max_candidates=None,
                                        dense_grasp=False,collision_detection=False)
                calls.append(dict(direction_arm=direction.tolist(),direction_reference=steering.tolist(),
                                  cone_deg=cone,candidates=len(grasps)))
                for g in grasps:
                    raw.append(np.r_[g.score,g.width_m,gripper.finger_height_m,g.depth_m,
                                     g.rotation.ravel(),g.translation,-1.])
            torch.cuda.synchronize()
            inference_s = time.monotonic()-started
            merged = GraspGroup(np.asarray(raw).reshape(-1,17)).nms().sort_by_score() if raw else []
            rows=[]
            for i,g in enumerate(merged):
                grasp=CameraGrasp(g.translation,g.rotation_matrix,g.width,g.depth,g.score,camera)
                row=native_pose_record(grasp,transform,'{}-{:03d}'.format(name,i+1))
                row.update(audit.audit(grasp,transform,mode,row['id']))
                rows.append(row)
            geometry_pass=[r for r in rows if r['geometry_pass']]
            supported=[r for r in geometry_pass if r['contact_supported']]
            # Show supported candidates first, then geometry-only candidates;
            # never display a rejected candidate as though it passed.
            selected=sorted(geometry_pass,key=lambda r:(not r['contact_supported'],-r['score']))
            report=dict(case=name,mode=mode,status='ok' if selected else 'no_geometry_pass',
                        audit_kind='piper_table_direction',candidates=selected,
                        snapshot_sha256=meta['snapshot_sha256'],input_sha256=input_report['input_sha256'],
                        frame='arm_base',pose_convention='native_and_Piper_TCP',
                        raw_count=len(rows),contact_supported_count=len(supported),
                        collision_checked='Piper meshes vs shared measured table only',
                        hardware_commands_sent=False,planning_authorized=False)
            write_json(folder/'grasps.json',report)
            write_json(folder/'audit.json',dict(calls=calls,candidates=rows,
                                               inference_s=inference_s,table=table))
            reasons=Counter(reason for r in rows for reason in r['rejection_reasons'])
            item=dict(case=name,raw_before_merged_nms=len(raw),raw_after_merged_nms=len(rows),
                      geometry_pass=len(selected),contact_supported=len(supported),
                      rejection_counts=dict(reasons),inference_s=inference_s)
            summary['cases'].append(item)
            write_json(root/'summary.json',summary)
            print(json.dumps(item),flush=True)


if __name__=='__main__':
    main()
