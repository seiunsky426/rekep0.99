#!/usr/bin/env python3
"""Replay archived failure with the production continuation and native backend.

Never imports an execution node or sends ROS/CAN commands. Self-collision-only
RRT is explicitly a diagnostic, not an environment/constraint accepted path.
"""
import argparse
import json
from pathlib import Path
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver
from rekpiper_planning.motion_backend import MotionBackend


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence-directory',required=True)
    parser.add_argument('--output-directory',required=True)
    args=parser.parse_args()
    source=Path(args.evidence_directory); output=Path(args.output_directory)
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic(); stopped=threading.Event()
    def heartbeat():
        while not stopped.wait(5): print('M8_REPLAY_RUNNING elapsed_s={:.1f}'.format(time.monotonic()-start),flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    report={'motion_allowed':False,'hardware_commands_sent':False,'comparisons':[]}
    try:
        xml=(source/'robot_description.urdf').read_text()
        poses=np.load(source/'recompute_lowbudget_report_attempt_0_path.npz')['dense_poses7']
        matrices=np.repeat(np.eye(4)[None],len(poses),axis=0)
        matrices[:,:3,3]=poses[:,:3]
        matrices[:,:3,:3]=Rotation.from_quat(poses[:,3:]).as_matrix()
        for tolerance in [.10,.12]:
            ik=PiperURDFIKSolver.from_urdf_xml(xml,'base_link','rekep_tcp',
                ['joint'+str(i) for i in range(1,7)],orientation_tolerance=tolerance)
            seed=np.zeros(6); legacy=[]
            for index,matrix in enumerate(matrices):
                result=ik.solve(matrix,initial_joint_pos=seed,max_iterations=100)
                q=result.cspace_position[:6]
                legacy.append({'index':index,'success':bool(result.success),
                    'joint_jump_rad':float(np.max(np.abs(q-seed))),
                    'position_error':result.position_error,'rotation_error':result.rotation_error})
                seed=q
            audited=ik.validate_pose_sequence(matrices,np.zeros(6))
            item={'tolerance_rad':tolerance,'legacy_diagnostic':legacy,'continuous_ik':audited}
            report['comparisons'].append(item)
            print('CONTINUOUS_IK_RESULT',tolerance,audited['valid'],
                  'failed_index',audited.get('failed_pose_index'),flush=True)
        ik.orientation_tolerance=.10
        endpoint=ik.solve(matrices[-1],initial_joint_pos=np.zeros(6),max_iterations=100)
        report['endpoint_ik_success']=bool(endpoint.success)
        backend=MotionBackend(xml)
        if endpoint.success:
            try:
                path=backend.plan(ik._lower,ik._upper,np.zeros(6),endpoint.cspace_position[:6],
                                  lambda q:True,timeout_s=2.,seed=0)
                report['self_collision_only_rrt']={'success':True,'waypoints':len(path),
                    'maximum_joint_step_rad':float(np.max(np.abs(np.diff(path,axis=0)))),
                    'environment_checked':False,'task_constraints_checked':False}
                np.savez_compressed(output/'self_collision_only_candidate.npz',joint_path=path,
                    cartesian_matrices=np.array([ik.forward(q) for q in path]),motion_allowed=False)
            except Exception as exc:
                report['self_collision_only_rrt']={'success':False,'error':str(exc)}
        backend.close()
        assert report['comparisons'][1]['legacy_diagnostic'][2]['joint_jump_rad']>.35
        assert max(s['joint_jump_rad'] for s in report['comparisons'][1]['legacy_diagnostic'])>1.66
        report['archived_failure_reproduced']=True
    except Exception as exc:
        report['error']=type(exc).__name__+': '+str(exc)
    finally:
        report['elapsed_s']=time.monotonic()-start
        (output/'report.json').write_text(json.dumps(report,indent=2,
            default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
        stopped.set()
        print('M8_REPLAY_SAVED',output/'report.json',flush=True)
    return 1 if 'error' in report else 0


if __name__=='__main__': raise SystemExit(main())
