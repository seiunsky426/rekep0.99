#!/usr/bin/env python3
"""Control-path tests with fake feedback and no CAN, cameras, or ROS master."""
import importlib.util
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from rekpiper_execution.supervised_session import Session

path=Path(__file__).resolve().parents[1]/'scripts/supervised_workbench_node.py'
spec=importlib.util.spec_from_file_location('supervised_workbench_node',str(path))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WorkbenchTest(unittest.TestCase):
    def test_nested_roslaunch_does_not_inherit_session_namespace(self):
        work=module.Workbench.__new__(module.Workbench)
        work.children={}
        with patch.dict(os.environ,{'ROS_NAMESPACE':'/rekpiper/supervised'}), \
             patch.object(module.subprocess,'Popen',return_value=Mock()) as popen:
            work.launch('cameras',Path('/tmp/cameras.launch'))
        self.assertNotIn('ROS_NAMESPACE',popen.call_args.kwargs['env'])
        self.assertEqual(popen.call_args.args[0],['roslaunch','/tmp/cameras.launch'])

    def test_connect_reads_can_and_does_not_enable(self):
        work=module.Workbench.__new__(module.Workbench)
        work.connected=False
        work.heartbeat=time.monotonic()
        work.config={'can_port':'can0'}
        work.control_id='unit-test'
        work.launch=Mock()
        master=Mock()
        master.getSystemState.return_value=(1,'ok',[[],[],[]])
        output='<UP,LOWER_UP> can state ERROR-ACTIVE bitrate 1000000'
        with patch.object(module.rospy,'get_master',return_value=master), \
             patch.object(module.subprocess,'check_output',return_value=output), \
             patch.object(module.rospy,'wait_for_service') as ready, \
             patch.object(module.rospy,'ServiceProxy') as service:
            result=work.dispatch('connect',{})
        self.assertIn('尚未使能',result)
        self.assertTrue(work.connected)
        self.assertEqual(work.launch.call_count,1)
        ready.assert_called_once()
        service.assert_not_called()

    def test_enable_can_confirm_motors_without_authorising_out_of_limit_motion(self):
        work=module.Workbench.__new__(module.Workbench)
        work.lock=threading.RLock()
        work.session=Session()
        work.display=None
        work.connected=True
        work.enabled=False
        work.warning=''
        work.joints=work.opening=work.raw_joints=None
        work.feedback_time=work.arm_time=work.raw_feedback_time=-float('inf')
        work.joint_validation_error=''
        work.arm=SimpleNamespace(arm_status=0,teach_status=0,err_code=0,
                                 ctrl_mode=0,mode_feedback=1)
        for index in range(1,7):
            setattr(work.arm,'joint_{}_angle_limit'.format(index),False)
            setattr(work.arm,'communication_status_joint_{}'.format(index),False)
        work.arm_time=time.monotonic()
        positions=[.038,-.045,.036,-.055,.461,.173,0.]
        stamp=time.time()
        message=SimpleNamespace(name=list(module.JOINT_NAMES)+['gripper'],
            position=positions,header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda:stamp)))
        with patch.object(module.rospy.Time,'now',return_value=SimpleNamespace(to_sec=lambda:stamp)):
            work.joint_cb(message)
        self.assertIsNone(work.joints)
        self.assertEqual(len(work.raw_joints),6)
        self.assertIn('joint2',work.joint_validation_error)
        def service_proxy(name, _kind):
            if name.endswith('/enable_status_srv'):
                return lambda:SimpleNamespace(success=True,message='motors=111111')
            def enable(_value):
                work.arm.ctrl_mode=1
                return SimpleNamespace(enable_response=True)
            return enable
        with patch.object(module.rospy,'ServiceProxy',side_effect=service_proxy) as service:
            result=work.dispatch('enable',{})
        self.assertTrue(work.enabled)
        self.assertEqual(service.call_count,2)
        self.assertIn('禁止规划运动',result)
        with self.assertRaisesRegex(ValueError,'joint_feedback_outside_model_limits'):
            work.feedback(True)

    def test_teach_or_stop_state_blocks_enable_before_driver_call(self):
        work=module.Workbench.__new__(module.Workbench)
        work.lock=threading.RLock()
        work.arm=SimpleNamespace(arm_status=1,teach_status=1,err_code=0,
                                 ctrl_mode=0,mode_feedback=1)
        work.connected=True
        work.arm_time=time.monotonic()
        work.joints=np.zeros(6)
        work.opening=.02
        work.feedback_time=time.monotonic()
        with patch.object(module.rospy,'ServiceProxy') as service:
            with self.assertRaisesRegex(ValueError,'exit_drag_teach_first'):
                work.dispatch('enable',{})
        service.assert_not_called()

    def test_emergency_stop_still_blocks_enable_after_teach_exits(self):
        work=module.Workbench.__new__(module.Workbench)
        work.lock=threading.RLock()
        work.connected=True
        work.arm=SimpleNamespace(arm_status=1,teach_status=2,err_code=0)
        work.arm_time=time.monotonic()
        with patch.object(module.rospy,'ServiceProxy') as service:
            with self.assertRaisesRegex(ValueError,'piper_emergency_stop_active'):
                work.dispatch('enable',{})
        service.assert_not_called()

    def test_camera_cloud_matches_perception_layout_and_fused_cloud_keeps_rgb(self):
        from rekpiper_perception.tracking_geometry import organized_xyz

        observations=module.Observations.__new__(module.Observations)
        observations.clouds={'rs1':Mock()}
        observations.fused=Mock()
        frame={'depth':np.zeros((2,3),dtype=np.float32),'rgb_stamp_s':12.}
        points=np.array([[.1,.2,.3],[.4,.5,.6]],dtype=np.float32)
        colors=np.array([[255,0,0],[0,255,0]],dtype=np.uint8)
        pixels=(np.array([0,1]),np.array([1,2]))
        with patch('rekpiper_execution.supervised_io.rgbd_points',
                   return_value=(points,colors,pixels)):
            observations.publish({'rs1':frame})
        cloud=observations.clouds['rs1'].publish.call_args.args[0]
        self.assertEqual((cloud.point_step,cloud.row_step,cloud.width,cloud.height),
                         (12,36,3,2))
        self.assertEqual([field.name for field in cloud.fields],['x','y','z'])
        decoded=organized_xyz(cloud)
        np.testing.assert_array_equal(decoded[0,1],points[0])
        np.testing.assert_array_equal(decoded[1,2],points[1])
        self.assertTrue(np.isnan(decoded[0,0]).all())
        fused=observations.fused.publish.call_args.args[0]
        self.assertEqual((fused.point_step,[field.name for field in fused.fields]),
                         (16,['x','y','z','rgb']))

    def test_tracking_status_requires_current_keypoint_layout(self):
        work=module.Workbench.__new__(module.Workbench)
        work.lock=threading.RLock()
        work.keys=[dict(id=3,rigid_group_id=7)]
        work.key_stamp=12.
        work.tracking_stamp=0.
        work.tracking_valid=0
        def message(stamp, group):
            return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda:stamp)),
                keypoints=[SimpleNamespace(id=3,rigid_group_id=group,valid=True)])
        work.tracking_cb(message(13.,8))
        work.tracking_cb(message(11.,7))
        self.assertEqual(work.tracking_stamp,0.)
        work.tracking_cb(message(13.,7))
        self.assertEqual(work.tracking_stamp,13.)
        self.assertEqual(work.tracking_valid,1)

    def test_preview_does_not_call_motion_services(self):
        with tempfile.TemporaryDirectory() as folder:
            work=module.Workbench.__new__(module.Workbench)
            work.operation_lock=threading.Lock()
            work.cancel=threading.Event()
            work.session=Session()
            work.selection=[{'rigid_group_id':1,'position':[0,0,0]},
                            {'rigid_group_id':2,'position':[.1,0,0]}]
            work.mask=np.zeros((1,2),dtype=np.uint16)
            saved_mask=np.array([[1,2]],dtype=np.uint16)
            saved_xyz=np.zeros((1,2,3),dtype=np.float32)
            work.segmentation_snapshot={"mask":saved_mask,"xyz":saved_xyz}
            work.xml='<robot/>'
            work.config={}
            work.system={}
            work.root=Path(folder)
            work.binding=lambda **_:{'binding':'measured'}
            work.initialize_planner=lambda:None
            work.feedback=lambda *args:(np.zeros(6),.03)
            work.event=Mock()
            work.display=SimpleNamespace(ghosts=Mock())
            work.planner=SimpleNamespace(candidate=None,held_local=None,grasp_matrix=None,
                support=None,target=None,check_cancel=lambda:None)
            cloud={1:np.zeros((150,3)),2:np.tile([.1,0,0],(150,1))}
            target_points=Mock(side_effect=lambda frame,mask,group,**_:cloud[group])
            scene=SimpleNamespace(frames={'rs1':{'sample':np.zeros((1,))},
                                          'rs3':{'sample':np.zeros((1,))}},
                                  target_points=target_points)
            work.scene=lambda *args:scene
            work.plan_action=SimpleNamespace(publish_feedback=Mock(),set_succeeded=Mock(),
                set_aborted=Mock(),set_preempted=Mock(),is_preempt_requested=lambda:False)
            trajectory=SimpleNamespace(positions=np.array([[0.]*6,[.001,0,0,0,0,0]]),
                                       times=np.array([0.,1.]))
            detail={'opening_m':.03}
            with patch.object(module,'plan_in_process',return_value=(trajectory,detail,SimpleNamespace(contact_membership_ok=False))), \
                 patch.object(module.rospy,'ServiceProxy') as service:
                work.plan(SimpleNamespace(session_id=work.session.id,stage=1))
            self.assertEqual(work.session.state,'PREVIEW_READY')
            self.assertTrue(work.plan_action.set_succeeded.called)
            self.assertEqual(target_points.call_count,2)
            for call in target_points.call_args_list:
                np.testing.assert_array_equal(call.args[1],saved_mask)
                np.testing.assert_array_equal(call.kwargs['source_points'],saved_xyz)
            service.assert_not_called()

    def test_saved_segmentation_is_bound_to_exact_keypoint_stamp(self):
        from rekpiper_execution.supervised_geometry import load_segmentation_snapshot
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"segmentation_snapshot.npz"
            mask=np.array([[1,3]],dtype=np.uint16)
            xyz=np.array([[[.4,.1,.03],[.7,-.2,.02]]],dtype=np.float32)
            np.savez_compressed(str(path),stamp_ns=np.int64(123),mask=mask,xyz=xyz)
            saved=load_segmentation_snapshot(path,123)
            np.testing.assert_array_equal(saved["mask"],mask)
            np.testing.assert_array_equal(saved["xyz"],xyz)
            with self.assertRaisesRegex(ValueError,"segmentation_snapshot_mismatch"):
                load_segmentation_snapshot(path,124)
            rs3_mask=np.array([[2,2]],dtype=np.uint16)
            np.savez_compressed(str(path),stamp_ns=np.int64(123),mask=mask,xyz=xyz,
                                rs3_stamp_ns=np.int64(123),rs3_mask=rs3_mask,rs3_xyz=xyz)
            saved=load_segmentation_snapshot(path,123)
            np.testing.assert_array_equal(saved['rs3']['mask'],rs3_mask)
            np.savez_compressed(str(path),stamp_ns=np.int64(123),mask=mask,xyz=xyz,
                                rs3_stamp_ns=np.int64(100000001),rs3_mask=rs3_mask,rs3_xyz=xyz)
            with self.assertRaisesRegex(ValueError,'rs3_segmentation_snapshot_mismatch'):
                load_segmentation_snapshot(path,123)

    def test_manual_extrinsics_allow_preview_binding_but_not_execution(self):
        work=module.Workbench.__new__(module.Workbench)
        source_binding={'rs1':'file-sha','stereo':'file-sha','serials':{
            'rs1':'346522071783','rs3':'934222070377'}}
        work.config={'serials':source_binding['serials']}
        work.calibration_binding=source_binding
        work.calibration=None
        work.selection=[{'id':1},{'id':0}]
        work.constraints={'task':'fixed'}
        work.model_hash='model-sha'
        work.session=Session()
        work.key_stamp=42.
        with patch.object(module,'load_experiment',return_value=({},source_binding)),              patch.object(module.rospy,'get_param',side_effect=[
                 '346522071783','934222070377']*2):
            preview_binding=work.binding(require_calibration=False)
            with self.assertRaisesRegex(ValueError,'experiment_calibration_check_required'):
                work.binding()
        self.assertEqual(preview_binding['extrinsics'],module.digest(source_binding))
        self.assertEqual(preview_binding['calibration'],module.digest(None))
        with patch.object(module,'load_experiment',return_value=({},dict(source_binding,rs1='changed'))):
            with self.assertRaisesRegex(ValueError,'experiment_extrinsics_changed_since_camera_start'):
                work.binding(require_calibration=False)

    def test_unverified_preview_cannot_stop_or_execute_arm(self):
        work=module.Workbench.__new__(module.Workbench)
        work.config={}
        work.calibration=None
        work.execute_action=SimpleNamespace(set_aborted=Mock())
        work.stop=Mock()
        work.execute(SimpleNamespace())
        work.execute_action.set_aborted.assert_called_once()
        self.assertEqual(work.execute_action.set_aborted.call_args.args[0].status,
                         'experiment_calibration_check_required')
        work.stop.assert_not_called()

    def test_experimental_preview_never_executes_arm(self):
        work=module.Workbench.__new__(module.Workbench)
        work.config={'extrinsics_status':'EXPERIMENTAL_PREVIEW_ONLY'}
        work.calibration={'binding':'measured'}
        work.execute_action=SimpleNamespace(set_aborted=Mock())
        work.stop=Mock()
        work.execute(SimpleNamespace())
        work.execute_action.set_aborted.assert_called_once()
        self.assertEqual(work.execute_action.set_aborted.call_args.args[0].status,
                         'experimental_extrinsics_preview_only')
        work.stop.assert_not_called()

    def test_stop_revokes_preview_and_cancels_both_actions(self):
        work=module.Workbench.__new__(module.Workbench)
        work.session=Session()
        work.session.begin_plan()
        work.cancel=threading.Event()
        work.armed_pub=Mock()
        work.trajectory=SimpleNamespace(cancel_all_goals=Mock())
        work.gripper=SimpleNamespace(cancel_all_goals=Mock())
        work.event=Mock()
        work.connected=False
        work.stop('operator_stop')
        self.assertTrue(work.cancel.is_set())
        self.assertEqual(work.session.state,'IDLE')
        work.trajectory.cancel_all_goals.assert_called_once()
        work.gripper.cancel_all_goals.assert_called_once()

    def test_stop_reports_driver_rejection(self):
        work=module.Workbench.__new__(module.Workbench)
        work.session=Session();work.cancel=threading.Event()
        work.armed_pub=Mock();work.trajectory=SimpleNamespace(cancel_all_goals=Mock())
        work.gripper=SimpleNamespace(cancel_all_goals=Mock())
        work.event=Mock();work.connected=True;work.warning=''
        with patch.object(module.rospy,'wait_for_service'), \
             patch.object(module.rospy,'ServiceProxy',return_value=lambda:SimpleNamespace(
                 success=False,message='fault')):
            work.stop('operator_stop')
        self.assertIn('fault',work.warning)


if __name__=='__main__':unittest.main()
