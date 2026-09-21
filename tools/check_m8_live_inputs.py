#!/usr/bin/env python3
"""Subscribe-only input evidence. Optionally run local, untargeted SDK smoke inference.

No publishers, services, robot actions, VLM uploads or motion authorization.
Untargeted SDK output is NOT a task-bound GraspCandidate.
"""
import argparse
import io
import json
import os
from pathlib import Path
import threading
import time
import numpy as np
import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, JointState
from rekpiper_msgs.msg import TrackedObjectCloudArray, SDFGrid


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-directory',required=True)
    parser.add_argument('--sdk-smoke-only',action='store_true')
    args=parser.parse_args(); output=Path(args.output_directory)
    output.mkdir(parents=True,exist_ok=False)
    rospy.init_node('m8_upgrade_readonly_inputs',anonymous=True,disable_signals=True)
    start=rospy.Time.now(); latest={}; counts={'rs1':0,'rs3':0}; subs=[]
    bridge=CvBridge(); finished=threading.Event()
    def heartbeat():
        while not finished.wait(5): print('LIVE_INPUT_STATUS',counts,flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    def frame(name,rgb,depth,info):
        if min(x.header.stamp for x in [rgb,depth,info]) < start: return
        counts[name]+=1; latest[name]=(rgb,depth,info)
    for name in counts:
        topics=[message_filters.Subscriber('/'+name+suffix,kind) for suffix,kind in [
            ('/color/image_raw',Image),('/aligned_depth_to_color/image_raw',Image),
            ('/color/camera_info',CameraInfo)]]
        sync=message_filters.ApproximateTimeSynchronizer(topics,10,.05)
        sync.registerCallback(lambda *msgs,n=name:frame(n,*msgs)); subs.append((topics,sync))
    for name,topic,kind in [('joints','/joint_states_single',JointState),
                           ('objects','/rekpiper/objects/tracked_clouds',TrackedObjectCloudArray),
                           ('sdf','/rekpiper/mapping/sdf_grid',SDFGrid)]:
        subs.append(rospy.Subscriber(topic,kind,lambda m,n=name:latest.__setitem__(n,m)))
    deadline=time.monotonic()+15.
    while time.monotonic()<deadline and not rospy.is_shutdown(): time.sleep(.05)
    report={'motion_allowed':False,'hardware_commands_sent':False,
            'camera_frame_counts':counts.copy(),'missing':[],
            'sdk_inference_is_task_bound':False,'full_task_accepted':False}
    for name in ['rs1','rs3','joints','objects','sdf']:
        if name not in latest: report['missing'].append(name)
    if 'joints' in latest:
        msg=latest['joints']; report['joints']={'names':msg.name,'positions_rad':msg.position,
            'age_s':(rospy.Time.now()-msg.header.stamp).to_sec()}
    if 'sdf' in latest: report['sdf_valid']=bool(latest['sdf'].valid)
    for name in ['rs1','rs3']:
        if name not in latest: continue
        rgb,depth,info=latest[name]
        np.savez_compressed(output/(name+'_rgbd.npz'),
            rgb=bridge.imgmsg_to_cv2(rgb,'rgb8'),depth=bridge.imgmsg_to_cv2(depth,'passthrough'),
            K=info.K,rgb_stamp_s=rgb.header.stamp.to_sec(),depth_stamp_s=depth.header.stamp.to_sec())
    try:
        if args.sdk_smoke_only and 'rs1' in latest:
            from rekpiper_grasp.anygrasp_adapter import AnyGraspAdapter
            rgb,depth,info=latest['rs1']
            z=bridge.imgmsg_to_cv2(depth,'passthrough').astype(np.float32)
            if depth.encoding in ('16UC1','mono16'): z*=.001
            elif depth.encoding != '32FC1': raise RuntimeError('unknown_depth_units')
            v,u=np.indices(z.shape); k=info.K
            points=np.stack([(u-k[2])*z/k[0],(v-k[5])*z/k[4],z],axis=-1).reshape(-1,3)
            points=points[np.isfinite(points).all(axis=1)&(points[:,2]>.15)&(points[:,2]<1.5)]
            points=np.ascontiguousarray(points[::max(1,int(np.ceil(len(points)/30000)))],dtype=np.float32)
            vendor=Path(os.environ['REKPIPER_VENDOR_ROOT'])/'anygrasp_sdk/grasp_detection'
            print('SDK_SMOKE_START untargeted_rs1_camera_points',len(points),flush=True)
            adapter=AnyGraspAdapter(vendor,vendor/'checkpoint_detection.tar',vendor/'license')
            began=time.monotonic()
            grasps=adapter.infer(points,np.ones(len(points),dtype=np.bool_),'rs1',max_candidates=10)
            report['sdk_smoke']={'success':True,'candidate_count':len(grasps),
                'elapsed_s':time.monotonic()-began,'scope':'untargeted_camera_cloud_dependency_smoke_only'}
            np.savez_compressed(output/'untargeted_sdk_diagnostic.npz',
                positions_camera=np.array([g.translation for g in grasps]),
                rotations_camera=np.array([g.rotation for g in grasps]),
                widths_m=np.array([g.width_m for g in grasps]),motion_allowed=False)
    except Exception as exc:
        report['sdk_smoke']={'success':False,'error':type(exc).__name__+': '+str(exc)}
    finally:
        finished.set()
        (output/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
