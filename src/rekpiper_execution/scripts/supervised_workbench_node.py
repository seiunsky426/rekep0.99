#!/usr/bin/env python3
"""Operator-owned four-stage workbench. Starting this node never opens CAN."""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid

import actionlib
import cv2
from cv_bridge import CvBridge
import numpy as np
import rospy
import rospkg
from scipy.spatial import cKDTree
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from piper_msgs.msg import PiperStatusMsg
from piper_msgs.srv import Enable
import yaml

from rekpiper_msgs.msg import (Keypoint3DArray, CommandGripperAction, CommandGripperGoal,
    PlanSupervisedStageAction, PlanSupervisedStageResult, PlanSupervisedStageFeedback,
    ExecuteSupervisedStageAction, ExecuteSupervisedStageResult, ExecuteSupervisedStageFeedback)
from rekpiper_msgs.srv import SupervisedCommand, SupervisedCommandResponse
from rekpiper_execution.supervised_session import (Session, Speed, select_keypoints,
    fixed_constraints, FIXED_TASK, digest, select_instance_group)
from rekpiper_execution.supervised_geometry import (load_experiment, file_hash, point_at,
    validate_known_points, SegmentScene, transform_points, mask_near_geometry,
    load_segmentation_snapshot)
from rekpiper_execution.supervised_planning import ExperimentPlanner, plan_in_process
from rekpiper_execution.supervised_io import Observations, Display, NS
from rekpiper_execution.trajectory import JOINT_NAMES, normalize_feedback_to_joint_limits
from rekpiper_planning.offline_observation import rgbd_points


def expand(value):
    if isinstance(value, dict): return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list): return [expand(v) for v in value]
    return os.path.expandvars(os.path.expanduser(value)) if isinstance(value, str) else value


class Workbench:
    def __init__(self):
        self.lock = threading.RLock()
        self.operation_lock = threading.Lock()
        self.session = Session()
        self.control_id = uuid.uuid4().hex
        self.config = expand(yaml.safe_load(Path(rospy.get_param('~config')).read_text()))
        self.system = yaml.safe_load(Path(rospy.get_param('~system_config')).read_text())
        self.root = Path(self.config['session_root'])/self.control_id
        self.root.mkdir(parents=True)
        self.log = (self.root/'events.jsonl').open('a', buffering=1)
        self.feedback_log = (self.root/'joint_feedback.jsonl').open('a', buffering=1)
        self.feedback_log_time = 0.
        self.xml = Path(rospy.get_param('~robot_file')).read_text()
        self.model_hash = digest(self.xml)
        self.observations = Observations()
        self.planner = None
        self.display = None
        self.children = {}
        self.heartbeat = -float('inf')
        self.feedback_time = self.arm_time = -float('inf')
        self.joints = self.arm = self.opening = None
        self.raw_joints = None
        self.raw_feedback_time = -float('inf')
        self.joint_validation_error = ''
        self.keys = []; self.mask = self.candidate_image = None
        self.segmentation_snapshot = None; self.segmentation_hash = None
        self.rs3_candidate_image = None; self.rs3_image_stamp = 0.
        self.key_stamp = 0.; self.mask_stamp = self.image_stamp = 0.
        self.tracking_stamp = 0.; self.tracking_valid = 0
        self.instruction = ''
        self.selection = None
        self.constraints = None
        self.calibration_rows = []; self.calibration = None
        self.calibration_frames = None
        self.transforms = self.calibration_binding = None
        self.preview_scene = None
        self.operation = ''; self.lease_detail = {}
        self.sequence = 0
        self.cancel = threading.Event()
        self.monitor_time = -float('inf')
        self.monitor_error = 'no_execution_scene'
        self.enabled = False
        self.connected = False
        self.warning = ''
        self.camera_requested = False
        self.camera_error = ''
        self.bridge = CvBridge()
        self.lease_pub = rospy.Publisher(NS+'/lease', String, queue_size=1)
        self.status_pub = rospy.Publisher(NS+'/status', String, queue_size=1, latch=True)
        self.calibration_images = {name: rospy.Publisher(NS+'/calibration/'+name, Image, queue_size=1, latch=True)
                                   for name in ('rs1','rs3')}
        self.armed_pub = rospy.Publisher(NS+'/armed', Bool, queue_size=1, latch=True)
        self.trigger_pub = rospy.Publisher('/rekpiper/perception/task_request', String, queue_size=1)
        self.trajectory = actionlib.SimpleActionClient(NS+'/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.gripper = actionlib.SimpleActionClient(NS+'/command_gripper', CommandGripperAction)
        rospy.Subscriber(NS+'/heartbeat', String, self.heartbeat_cb, queue_size=1)
        rospy.Subscriber('/joint_states_single', JointState, self.joint_cb, queue_size=1)
        rospy.Subscriber('/arm_status', PiperStatusMsg, self.arm_cb, queue_size=1)
        rospy.Subscriber('/rekpiper/perception/keypoints', Keypoint3DArray, self.key_cb, queue_size=1)
        rospy.Subscriber('/rekpiper/tracking/keypoints', Keypoint3DArray, self.tracking_cb, queue_size=1)
        rospy.Subscriber('/rekpiper/perception/geometry_valid_mask', Image, self.mask_cb, queue_size=1)
        rospy.Subscriber('/rekpiper/perception/candidate_image', Image, self.image_cb, queue_size=1)
        rospy.Subscriber('/rekpiper/perception/rs3_candidate_image', Image, self.rs3_image_cb, queue_size=1)
        rospy.Service(NS+'/command', SupervisedCommand, self.command)
        self.plan_action = actionlib.SimpleActionServer(NS+'/plan', PlanSupervisedStageAction,
            execute_cb=self.plan, auto_start=False)
        self.execute_action = actionlib.SimpleActionServer(NS+'/execute', ExecuteSupervisedStageAction,
            execute_cb=self.execute, auto_start=False)
        self.plan_action.start(); self.execute_action.start()
        self.timer = rospy.Timer(rospy.Duration(.05), self.tick)
        self.sensor_worker = threading.Thread(target=self.sensor_loop, daemon=True); self.sensor_worker.start()
        rospy.on_shutdown(self.shutdown)
        self.event('workbench_started', opens_can=False)

    def event(self, name, **value):
        with self.lock:
            self.log.write(json.dumps(dict(event=name, wall_s=time.time(), monotonic_s=time.monotonic(),
                session_id=self.session.id, stage=self.session.stage, **value), ensure_ascii=False,
                default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v))+'\n')

    def heartbeat_cb(self, message):
        if message._connection_header.get('callerid') == NS+'/panel' and message.data == self.control_id:
            self.heartbeat = time.monotonic()

    def joint_cb(self, msg):
        try:
            values = dict(zip(msg.name, msg.position))
            if not 0 <= rospy.Time.now().to_sec()-msg.header.stamp.to_sec() <= .25:
                return
            raw_joints = np.asarray([values[n] for n in JOINT_NAMES], dtype=float)
            if not np.all(np.isfinite(raw_joints)): return
            opening = float(values['gripper'])
            if not np.isfinite(opening) or not -.001 <= opening <= .070: return
            with self.lock:
                self.raw_joints = raw_joints
                self.raw_feedback_time = time.monotonic()
            try:
                joints = normalize_feedback_to_joint_limits(raw_joints, .01)
            except ValueError as exc:
                with self.lock:
                    self.joints = self.opening = None
                    self.feedback_time = -float('inf')
                    self.joint_validation_error = str(exc)
                return
            with self.lock:
                self.joints, self.opening = joints, max(0., opening)
                self.feedback_time = time.monotonic()
                self.joint_validation_error = ''
                if self.session.state == 'EXECUTING' and self.feedback_time-self.feedback_log_time >= .05:
                    self.feedback_log.write(json.dumps(dict(stamp=msg.header.stamp.to_sec(),
                        joints=joints.tolist(),opening=opening,stage=self.session.stage))+'\n')
                    self.feedback_log_time=self.feedback_time
            if self.display: self.display.feedback(joints)
        except (KeyError, ValueError):
            return

    def arm_cb(self, msg):
        with self.lock:
            self.arm, self.arm_time = msg, time.monotonic()

    def key_cb(self, msg):
        with self.lock:
            if self.session.state == 'EXECUTING':
                return
            if msg.header.stamp.to_sec() <= self.key_stamp: return
            keys = []
            for k in msg.keypoints:
                keys.append(dict(id=k.id, position=[k.position.x,k.position.y,k.position.z],
                    rigid_group_id=k.rigid_group_id, valid=k.valid, valid_depth_pixels=k.valid_depth_pixels,
                    pixel=[k.pixel_x,k.pixel_y]))
            if not msg.all_valid or not keys: return
            snapshot_path = self.root/'segmentation_snapshot.npz'
            try:
                snapshot = load_segmentation_snapshot(snapshot_path, msg.header.stamp.to_nsec())
                snapshot_hash = file_hash(snapshot_path)
            except (OSError, KeyError, ValueError) as exc:
                rospy.logwarn('Ignoring keypoints without matching saved segmentation: %s', exc)
                return
            self.segmentation_snapshot, self.segmentation_hash = snapshot, snapshot_hash
            self.keys, self.key_stamp = keys, msg.header.stamp.to_sec()
            self.selection = None
            self.constraints = None
            self.session.invalidate('关键点已更新，请生成固定约束')
            if self.display: self.display.keypoints(keys)

    def tracking_cb(self, msg):
        with self.lock:
            stamp = msg.header.stamp.to_sec()
            layout = {(int(k.id), int(k.rigid_group_id)) for k in msg.keypoints}
            expected = {(int(k['id']), int(k['rigid_group_id'])) for k in self.keys}
            if not expected or layout != expected or stamp < self.key_stamp:
                return
            self.tracking_stamp = stamp
            self.tracking_valid = sum(bool(k.valid) for k in msg.keypoints)

    def mask_cb(self, msg):
        with self.lock:
            self.mask = self.bridge.imgmsg_to_cv2(msg, 'passthrough').copy()
            self.mask_stamp = msg.header.stamp.to_sec()

    def image_cb(self, msg):
        with self.lock:
            self.candidate_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8').copy()
            self.image_stamp = msg.header.stamp.to_sec()

    def rs3_image_cb(self, msg):
        with self.lock:
            self.rs3_candidate_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8').copy()
            self.rs3_image_stamp = msg.header.stamp.to_sec()

    def enable_feedback(self):
        # Motor enable needs real telemetry and a healthy controller, but an
        # out-of-model joint start is a motion-planning fault, not missing CAN.
        with self.lock:
            now = time.monotonic()
            if not self.connected: raise ValueError('connect_can_first')
            if self.arm is None or now-self.arm_time > .25:
                raise ValueError('controller_feedback_missing_or_stale')
            a = self.arm
            if a.teach_status not in (0, 2):
                raise ValueError('exit_drag_teach_first: teach_status={}'.format(a.teach_status))
            if a.arm_status == 1:
                raise ValueError('piper_emergency_stop_active')
            if a.arm_status != 0 or a.err_code:
                raise ValueError('piper_controller_fault: arm_status={} err_code={}'
                                 .format(a.arm_status, a.err_code))
            if any(getattr(a, 'joint_{}_angle_limit'.format(i)) or
                   getattr(a, 'communication_status_joint_{}'.format(i)) for i in range(1,7)):
                raise ValueError('piper_joint_or_communication_fault')
            if self.raw_joints is None or now-self.raw_feedback_time > .25:
                raise ValueError('joint_feedback_missing_or_stale')

    def feedback(self, require_enabled=False):
        with self.lock:
            now = time.monotonic()
            if self.arm is None or now-self.arm_time > .25:
                raise ValueError('robot_feedback_missing_or_stale')
            a = self.arm
            if a.arm_status == 1 or a.teach_status not in (0, 2):
                raise ValueError('piper_stop_or_teach_active: arm_status={} teach_status={} motors_not_enabled'
                                 .format(a.arm_status, a.teach_status))
            if self.joint_validation_error and now-self.raw_feedback_time <= .25:
                raise ValueError('joint_feedback_outside_model_limits: '+self.joint_validation_error)
            if self.joints is None or now-self.feedback_time > .25:
                raise ValueError('robot_feedback_missing_or_stale')
            if a.err_code or (require_enabled and (a.ctrl_mode != 1 or a.arm_status != 0 or a.mode_feedback != 1)):
                raise ValueError('piper_controller_not_ready')
            if any(getattr(a, 'joint_{}_angle_limit'.format(i)) or
                   getattr(a, 'communication_status_joint_{}'.format(i)) for i in range(1,7)):
                raise ValueError('piper_joint_or_communication_fault')
            if require_enabled and not self.enabled:
                raise ValueError('enable_button_required')
            return self.joints.copy(), float(self.opening)

    def initialize_planner(self):
        if self.planner is None:
            self.planner = ExperimentPlanner(self.xml, self.config, self.system, self.root)
            self.display = Display(self.xml, self.planner.ik)
            self.planner.cancelled = lambda: self.cancel.is_set() or rospy.is_shutdown()

    def launch(self, key, path, args=()):
        child = self.children.get(key)
        if child is not None and child.poll() is None: return
        child_env = os.environ.copy()
        # The session node itself runs under /rekpiper/supervised. Nested
        # roslaunch must resolve camera and perception topics from the root.
        child_env.pop('ROS_NAMESPACE', None)
        self.children[key] = subprocess.Popen(['roslaunch', str(path)]+list(args),
                                              start_new_session=True, env=child_env)

    def command(self, req):
        try:
            args = json.loads(req.arguments_json or '{}')
            if req.command == 'stop':
                self.stop('operator_stop'); return SupervisedCommandResponse(True, '已请求停止，未松爪')
            if req.command == 'clear_trace':
                if self.display: self.display.clear()
                return SupervisedCommandResponse(True, '留影已清除')
            if not self.operation_lock.acquire(False): raise ValueError('operation_in_progress')
            try:
                if self.session.state in ('EXECUTING','PLANNING'):
                    raise ValueError('cancel_current_operation_before_changes')
                result = self.dispatch(req.command, args)
            finally:
                self.operation_lock.release()
            self.event(req.command, result=result)
            return SupervisedCommandResponse(True, str(result or '完成'))
        except Exception as exc:
            self.event('command_failed', command=req.command, error=str(exc))
            return SupervisedCommandResponse(False, str(exc))

    def dispatch(self, command, args):
        packages = rospkg.RosPack()
        launch_root = Path(packages.get_path('rekpiper_bringup'))/'launch'
        if command == 'connect':
            if self.connected: return '已连接'
            if time.monotonic()-self.heartbeat > .5: raise ValueError('operator_panel_heartbeat_missing')
            # No adoption of an existing control owner, even if it uses another namespace.
            state = rospy.get_master().getSystemState()[2]
            owners = [name for name, providers in state[2] if name.endswith(('/enable_srv','/stop_srv')) and providers]
            subscriptions = [name for name, subscribers in state[1]
                             if name.endswith(('/joint_ctrl_single','/joint_command','/pos_cmd')) and subscribers]
            if owners or subscriptions:
                raise ValueError('other_controller_present:'+','.join(owners+subscriptions))
            port = self.config['can_port']
            if not re.fullmatch(r'can[0-9]+', port): raise ValueError('invalid_can_interface')
            status = subprocess.check_output(['ip','-details','link','show','dev',port], text=True)
            flags = re.search(r'<([^>]+)>', status)
            if flags is None: raise ValueError('cannot_read_can_state')
            if 'UP' not in flags.group(1).split(','):
                subprocess.run(['pkexec', shutil.which('ip'), 'link','set',port,'up','type','can','bitrate','1000000'], check=True)
                status = subprocess.check_output(['ip','-details','link','show','dev',port], text=True)
            if 'can state ERROR-ACTIVE' not in status or not re.search(r'bitrate 1000000\b', status):
                raise ValueError('can_state_or_bitrate_invalid')
            self.launch('hardware', launch_root/'_supervised_hardware.launch',
                        ['session_id:='+self.control_id, 'can_port:='+port])
            rospy.wait_for_service(NS+'/driver/enable_srv', timeout=15.)
            self.connected = True
            return 'CAN已连接，尚未使能'
        if command == 'enable':
            self.enable_feedback()
            response = rospy.ServiceProxy(NS+'/driver/enable_srv', Enable)(True)
            if not response.enable_response: raise ValueError('six_motor_enable_not_confirmed')
            status = rospy.ServiceProxy(NS+'/driver/enable_status_srv', Trigger)()
            if not status.success: raise ValueError('six_motor_enable_status_not_ready: '+status.message)
            self.enabled = True
            self.session.invalidate('使能状态已改变，请重新预览')
            if self.joint_validation_error:
                self.warning = '已使能但禁止规划运动: '+self.joint_validation_error
                return '六电机使能已确认；'+self.warning
            self.warning = ''
            return '六电机使能已确认；没有执行轨迹'
        if command == 'disable':
            self.stop('operator_disable')
            response = rospy.ServiceProxy(NS+'/driver/enable_srv', Enable)(False)
            return '已失能' if response.enable_response else '失能反馈未确认'
        if command == 'cameras':
            self.camera_requested = True
            transforms, binding = load_experiment(self.config)
            self.transforms, self.calibration_binding = transforms, binding
            self.observations.transforms = transforms
            self.calibration = None
            for name in ('rs1','rs3'):
                serial = str(self.config['serials'][name])
                existing = rospy.get_param('/'+name+'/realsense2_camera/serial_no', None)
                if existing is not None and str(existing).lstrip('_') != serial:
                    raise ValueError('camera_serial_conflict:'+name)
            publishers=dict(rospy.get_master().getSystemState()[2][0])
            present=[bool(publishers.get('/'+name+'/color/image_raw')) for name in ('rs1','rs3')]
            if any(present) and not all(present):
                raise ValueError('partial_camera_stream_present')
            if all(present):
                for name in ('rs1','rs3'):
                    if not rospy.has_param('/'+name+'/realsense2_camera/serial_no'):
                        raise ValueError('existing_camera_identity_unverifiable:'+name)
            else:
                self.launch('cameras', launch_root/'_supervised_cameras.launch',
                            ['serial_rs1:='+str(self.config['serials']['rs1']),
                             'serial_rs3:='+str(self.config['serials']['rs3'])])
            self.initialize_planner()
            return ('已使用现有双相机' if all(present) else '相机启动中')+'；图像与点云可查看，按相机关键点执行前须复核坐标'
        if command == 'calibration_freeze':
            self.calibration_frames = self.observations.capture()
            for name,frame in self.calibration_frames.items():
                image=self.bridge.cv2_to_imgmsg(frame['rgb'],'rgb8')
                image.header.stamp=rospy.Time.from_sec(frame['rgb_stamp_s'])
                image.header.frame_id=frame['optical_frame']
                self.calibration_images[name].publish(image)
            return '已冻结同步图像用于已知点复核'
        if command == 'calibration_point':
            if self.calibration_frames is None: raise ValueError('freeze_calibration_images_first')
            base = np.asarray(args['base'],dtype=float)
            if base.shape != (3,) or not np.isfinite(base).all(): raise ValueError('invalid_reference_xyz')
            row = dict(base=base.tolist())
            for name in ('rs1','rs3'):
                row[name] = point_at(self.calibration_frames[name], args[name]).tolist()
            self.calibration_rows.append(row)
            return json.dumps(row, ensure_ascii=False)
        if command == 'calibration_clear':
            self.calibration_rows=[]; self.calibration=None
            self.session.invalidate('外参复核已清除'); return '已清除'
        if command == 'calibration_verify':
            errors = validate_known_points(self.calibration_rows)
            self.calibration = dict(binding=self.calibration_binding, rows=deepcopy(self.calibration_rows), errors=errors)
            (self.root/'calibration_check.json').write_text(json.dumps(self.calibration,indent=2))
            self.session.invalidate('实验外参现场复核通过')
            return json.dumps(errors)
        if command == 'keypoints':
            self.initialize_planner()
            self.observations.capture()
            if self.session.held: raise ValueError('finish_held_object_before_new_task')
            self.session.new_task(); self.selection=None; self.constraints=None; self.keys=[]; self.key_stamp=0.
            self.segmentation_snapshot=None; self.segmentation_hash=None
            self.rs3_candidate_image=None; self.rs3_image_stamp=0.
            self.tracking_stamp=0.; self.tracking_valid=0
            self.instruction=FIXED_TASK
            self.launch('perception', launch_root/'_supervised_perception.launch',
                        ['tracking_output:='+str(self.root/'tracking'),
                         'snapshot_output:='+str(self.root/'segmentation_snapshot.npz')])
            deadline=time.monotonic()+30
            while self.trigger_pub.get_num_connections()==0 and time.monotonic()<deadline:
                time.sleep(.1)
            if self.trigger_pub.get_num_connections()==0: raise ValueError('perception_start_timeout')
            self.trigger_pub.publish(String(self.instruction))
            return '已提交SAM/DINOv2关键点生成请求，等待感知结果'
        if command == 'task':
            with self.lock:
                if (not self.keys or self.candidate_image is None or self.mask is None
                        or max(self.key_stamp,self.image_stamp,self.mask_stamp)-min(self.key_stamp,self.image_stamp,self.mask_stamp) > .025):
                    raise ValueError('matching_keypoint_image_and_geometry_mask_required')
                keys=deepcopy(self.keys); image=self.candidate_image.copy(); stamp=self.key_stamp
                saved=self.segmentation_snapshot
                if saved is None or saved['rs3'] is None or self.rs3_candidate_image is None:
                    raise ValueError('matching_rs3_segmentation_and_candidate_image_required')
                if abs(self.rs3_image_stamp-saved['rs3']['stamp_ns']*1e-9) > .001:
                    raise ValueError('rs3_candidate_image_stamp_mismatch')
                rs3_image=self.rs3_candidate_image.copy()
                rs3_labels=saved['rs3']['mask'].copy()
            from rekpiper_planning.vlm_program_generator import create_backend
            image_path=self.root/('selection-'+uuid.uuid4().hex+'.png'); cv2.imwrite(str(image_path),image)
            prompt=('Fixed task: place the BLUE CUBE on the YELLOW DISK. Select one visible '
                    'numbered keypoint on the blue cube for grasp_id and one on the yellow disk '
                    'for target_id. Return only JSON {"grasp_id": integer, "target_id": integer}. '
                    'The keypoints must belong to DIFFERENT objects. Do not generate coordinates, '
                    'constraints, code or extra fields. Available IDs: '+str([k['id'] for k in keys]))
            backend=create_backend(self.system['vlm'])
            text=backend.generate_text(prompt,str(image_path))
            selection=select_keypoints(text,keys)
            rs3_path=self.root/('rs3-selection-'+uuid.uuid4().hex+'.png')
            cv2.imwrite(str(rs3_path), rs3_image)
            rs3_prompt=('This is the locked RS3 camera view of the same scene. '
                        'Identify the small cyan-blue cube among the SAM instance outlines labeled G1, G2, etc. '
                        'Return only JSON {"group_id": integer}. Never infer coordinates or select the yellow disk. '
                        'Available group IDs: '+str(sorted(int(v) for v in np.unique(rs3_labels) if v > 0)))
            rs3_text=backend.generate_text(rs3_prompt,str(rs3_path))
            rs3_group=select_instance_group(rs3_text,rs3_labels)
            with self.lock:
                if stamp != self.key_stamp: raise ValueError('keypoints_changed_during_vlm')
                self.selection=selection
                self.constraints=fixed_constraints(selection)
                self.session.invalidate('固定约束已生成，请预览预抓取')
            selection_file=self.root/'vlm_selection.json'
            selection_tmp=selection_file.with_suffix('.json.tmp')
            selection_tmp.write_text(json.dumps(dict(raw=text,selection=selection,
                rs3_raw=rs3_text,rs3_group_id=rs3_group,
                snapshot_sha256=self.segmentation_hash,
                constraints=self.constraints,stamp=stamp),ensure_ascii=False,indent=2))
            os.replace(str(selection_tmp),str(selection_file))
            return json.dumps(self.constraints,ensure_ascii=False)
        if command == 'task_changed':
            raise ValueError('fixed_blue_cube_yellow_disk_task')
        if command == 'speed':
            self.session.speed=Speed(float(args['velocity']),.10,.50,int(args['driver_percent']))
            self.session.invalidate('速度已改变，请重新预览'); return '速度已更新'
        if command == 'grasp':
            q,_=self.feedback(True)
            confirmed=args.get('confirmed') is True
            self.session.confirm_grasp(confirmed)
            if confirmed: self.planner.held(q)
            else: self.planner.held_local=None
            return self.session.reason
        raise ValueError('unknown_supervised_command')

    def binding(self, require_calibration=True):
        _, binding=load_experiment(self.config)
        if binding != self.calibration_binding:
            raise ValueError('experiment_extrinsics_changed_since_camera_start')
        if require_calibration and (self.calibration is None or self.calibration['binding'] != binding):
            raise ValueError('experiment_calibration_check_required')
        for name in ('rs1','rs3'):
            actual=str(rospy.get_param('/'+name+'/realsense2_camera/serial_no','')).lstrip('_')
            if actual != str(self.config['serials'][name]): raise ValueError('live_camera_identity_changed')
        if self.selection is None or self.constraints is None:
            raise ValueError('fixed_constraints_required')
        segmentation = None
        if self.config.get('extrinsics_status') == 'EXPERIMENTAL_PREVIEW_ONLY':
            if self.segmentation_snapshot is None or self.segmentation_hash != file_hash(
                    self.root/'segmentation_snapshot.npz'):
                raise ValueError('saved_segmentation_snapshot_required')
            segmentation = self.segmentation_hash
        return dict(extrinsics=digest(binding), calibration=digest(self.calibration),
                    constraints=digest(self.constraints), model=self.model_hash,
                    speed=vars(self.session.speed), key_stamp=self.key_stamp, segmentation=segmentation)

    def scene(self, joints, opening):
        held=(transform_points(self.planner.held_local,self.planner.ik.forward(joints))
              if self.planner.held_local is not None else None)
        return SegmentScene(self.observations.capture(),self.config,self.planner.sampler,joints,opening,held)

    def plan(self, goal):
        result=PlanSupervisedStageResult()
        if not self.operation_lock.acquire(False):
            result.status='operation_in_progress'; self.plan_action.set_aborted(result); return
        try:
            if (goal.session_id,goal.stage)!=(self.session.id,self.session.stage): raise ValueError('stale_stage_request')
            self.cancel.clear(); ticket=self.session.begin_plan()
            self.plan_action.publish_feedback(PlanSupervisedStageFeedback('采集并检查当前场景'))
            binding=self.binding(require_calibration=False); self.initialize_planner(); q,opening=self.feedback()
            self.planner.cancelled=lambda: self.cancel.is_set() or self.plan_action.is_preempt_requested() or rospy.is_shutdown()
            scene=self.scene(q,opening)
            if self.session.stage == 1:
                saved = getattr(self, 'segmentation_snapshot', None)
                mask = saved['mask'] if saved is not None else self.mask
                xyz = saved['xyz'] if saved is not None else None
                self.planner.support=scene.target_points(scene.frames['rs1'],mask,
                    self.selection[1]['rigid_group_id'],source_points=xyz)
                target=scene.target_points(scene.frames['rs1'],mask,
                    self.selection[0]['rigid_group_id'],source_points=xyz)
                for cloud,selected in ((target,self.selection[0]),(self.planner.support,self.selection[1])):
                    if cKDTree(cloud).query(selected['position'])[0] > .01:
                        raise ValueError('selected_object_moved_since_keypoint_snapshot')
                self.plan_action.publish_feedback(PlanSupervisedStageFeedback('检查抓取候选来源'))
                self.planner.target=target
            if self.session.stage != 1 and self.planner.candidate is None: raise ValueError('bound_grasp_candidate_missing')
            self.plan_action.publish_feedback(PlanSupervisedStageFeedback('ReKep规划、稠密IK与全路径碰撞验证'))
            job=dict(xml=self.xml,config=self.config,system=self.system,output=str(self.root),
                     target=self.planner.target,support=self.planner.support,held_local=self.planner.held_local,
                     grasp_matrix=self.planner.grasp_matrix,candidate=self.planner.candidate,
                     stage=self.session.stage,joints=q,opening=opening,scene=scene,speed=self.session.speed)
            smooth,detail,candidate=plan_in_process(job,self.planner.cancelled)
            self.planner.candidate=candidate
            if hasattr(candidate, 'target_points_base'):
                self.planner.target=candidate.target_points_base
            self.planner.check_cancel()
            if binding != self.binding(require_calibration=False): raise ValueError('inputs_changed_during_planning')
            preview=self.session.install(ticket,smooth.positions,smooth.times,binding,detail)
            self.preview_scene=scene
            self.display.ghosts(smooth.positions,detail['opening_m'],self.session.stage,detail)
            (self.root/(preview['id']+'.json')).write_text(json.dumps(preview,ensure_ascii=False,indent=2))
            for name,frame in scene.frames.items():
                np.savez_compressed(self.root/(preview['id']+'-'+name+'.npz'),**frame)
            self.event('preview_ready',preview_id=preview['id'],detail=detail)
            result.success=True;result.preview_id=preview['id'];result.status=('实验路径预览已生成；接触证据未证实，禁止执行'
                if not candidate.contact_membership_ok else '预览检查通过，等待点击执行本段')
            self.plan_action.set_succeeded(result)
        except Exception as exc:
            self.session.invalidate(str(exc));result.status=str(exc)
            self.event('planning_failed',error=str(exc))
            if isinstance(exc,InterruptedError) or self.plan_action.is_preempt_requested(): self.plan_action.set_preempted(result)
            else: self.plan_action.set_aborted(result)
        finally:
            self.operation_lock.release()

    def await_action(self, client, timeout):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if self.cancel.is_set() or self.execute_action.is_preempt_requested():
                client.cancel_goal();raise InterruptedError('operator_cancelled')
            self.feedback(True)
            if client.wait_for_result(rospy.Duration(.05)):
                result=client.get_result()
                if client.get_state()!=3 or result is None: raise ValueError('controller_action_failed:'+str(result))
                return result
        client.cancel_goal();raise ValueError('controller_action_timeout')

    def gripper_move(self, command, width):
        self.operation='gripper';self.lease_detail.update(gripper_command=int(command),gripper_width=float(width))
        if not self.gripper.wait_for_server(rospy.Duration(2.)): raise ValueError('gripper_action_unavailable')
        time.sleep(.1)
        self.gripper.send_goal(CommandGripperGoal(command=command,object_uuid='supervised-object',total_opening_m=width,effort=.5))
        result=self.await_action(self.gripper,8.)
        if not result.success: raise ValueError(result.status)
        return result

    def execute(self, goal):
        result=ExecuteSupervisedStageResult()
        if self.config.get('extrinsics_status') == 'EXPERIMENTAL_PREVIEW_ONLY':
            result.status='experimental_extrinsics_preview_only'
            self.execute_action.set_aborted(result)
            return
        if self.calibration is None:
            result.status='experiment_calibration_check_required'
            self.execute_action.set_aborted(result)
            return
        if not self.operation_lock.acquire(False):
            result.status='operation_in_progress';self.execute_action.set_aborted(result);return
        try:
            self.cancel.clear();self.planner.cancelled=lambda: self.cancel.is_set() or self.execute_action.is_preempt_requested() or rospy.is_shutdown()
            q,opening=self.feedback(True); binding=self.binding()
            p=self.session.preview
            if p is None: raise ValueError('preview_required')
            self.execute_action.publish_feedback(ExecuteSupervisedStageFeedback('复核当前场景与同一条预览轨迹'))
            fresh=self.scene(q,opening)
            self.preview_scene.unchanged(fresh)
            self.planner.audit(p['positions'],fresh,self.session.stage)
            if self.session.stage == 1:
                self.planner.sweep(q,fresh,1,opening,self.planner.candidate.suggested_preopen_width_m)
            if self.session.stage in (2,4):
                start=self.planner.candidate.suggested_preopen_width_m if self.session.stage==2 else opening
                end=max(0.,self.planner.candidate.predicted_width_m-.003) if self.session.stage==2 else self.planner.candidate.suggested_preopen_width_m
                self.planner.sweep(p['positions'][-1],fresh,self.session.stage,start,end)
            p=self.session.claim(goal.session_id,goal.stage,goal.preview_id,q,binding)
            self.preview_scene=fresh;self.monitor_time=time.monotonic();self.monitor_error=''
            self.lease_detail=dict(preview_id=p['id'],trajectory_sha256=p['trajectory_sha256'],
                                   velocity=self.session.speed.velocity,driver_percent=self.session.speed.driver_percent)
            self.event('execute_clicked',preview_id=p['id'])
            if self.session.stage==1:
                self.gripper_move(CommandGripperGoal.OPEN,self.planner.candidate.suggested_preopen_width_m)
            self.operation='trajectory'
            if not self.trajectory.wait_for_server(rospy.Duration(2.)): raise ValueError('trajectory_bridge_missing')
            time.sleep(.1)
            trajectory=FollowJointTrajectoryGoal();trajectory.trajectory.header.frame_id=p['id']
            trajectory.trajectory.joint_names=list(JOINT_NAMES)
            for values,stamp in zip(p['positions'],p['times']):
                trajectory.trajectory.points.append(JointTrajectoryPoint(positions=values,time_from_start=rospy.Duration(stamp)))
            self.trajectory.send_goal(trajectory)
            self.execute_action.publish_feedback(ExecuteSupervisedStageFeedback('执行当前已审核轨迹'))
            outcome=self.await_action(self.trajectory,p['times'][-1]+5.)
            if outcome.error_code: raise ValueError(outcome.error_string)
            actual,_=self.feedback(True)
            error=self.planner.ik._pose_errors(self.planner.ik.forward(actual),np.asarray(p['detail']['goal_matrix']))
            if error[0]>.003 or error[1]>.03: raise ValueError('tcp_arrival_not_confirmed')
            if self.session.stage==2:
                self.gripper_move(CommandGripperGoal.CLOSE,self.planner.candidate.predicted_width_m)
            elif self.session.stage==4:
                self.planner.supported(self.planner.ik.forward(actual))
                self.gripper_move(CommandGripperGoal.OPEN,self.planner.candidate.suggested_preopen_width_m)
            self.operation='';self.session.finish()
            self.event('stage_completed',preview_id=p['id'],actual_joints=actual)
            result.success=True;result.status=self.session.reason;self.execute_action.set_succeeded(result)
        except Exception as exc:
            self.stop(str(exc));result.status=str(exc)
            if isinstance(exc,InterruptedError):self.execute_action.set_preempted(result)
            else:self.execute_action.set_aborted(result)
        finally:
            self.operation='';self.operation_lock.release()

    def stop(self, reason):
        self.cancel.set();self.operation='';self.enabled=False
        self.armed_pub.publish(Bool(False))
        self.trajectory.cancel_all_goals();self.gripper.cancel_all_goals()
        # Invalidate synchronously; the driver independently expires its lease.
        self.session.invalidate(reason)
        self.event('stop',reason=reason)
        if self.connected:
            try:
                rospy.wait_for_service(NS+'/driver/stop_srv',timeout=.2)
                response=rospy.ServiceProxy(NS+'/driver/stop_srv',Trigger)()
                if not response.success:
                    self.warning='驱动停止未确认：'+response.message
            except (rospy.ROSException,rospy.ServiceException) as exc:
                self.warning='驱动停止服务失败：'+str(exc)

    def monitor(self, frames):
        q,opening=self.feedback(True)
        robot,radii,_=self.planner.sampler.contact_samples(q,opening)
        held=(transform_points(self.planner.held_local,self.planner.ik.forward(q))
              if self.planner.held_local is not None else None)
        points=[]
        for frame in frames.values():
            mask=mask_near_geometry(frame,robot,radii)
            if held is not None: mask|=mask_near_geometry(frame,held,np.full(len(held),.005))
            xyz,_,pixels=rgbd_points(frame)
            keep=~mask[pixels]
            keep &= np.all((xyz>=self.preview_scene.lower)&(xyz<=self.preview_scene.upper),axis=1)
            points.append(xyz[keep])
        points=np.vstack(points)
        if len(points)<300:raise ValueError('environment_monitor_insufficient_depth')
        distance=self.preview_scene.tree.query(points)[0]
        if np.count_nonzero(distance>.015)>max(20,.01*len(points)):
            raise ValueError('environment_changed_during_motion')
        self.monitor_time=time.monotonic();self.monitor_error=''

    def sensor_loop(self):
        while not rospy.is_shutdown():
            try:
                frames=self.observations.capture()
                self.observations.publish(frames)
                self.camera_error = ''
                if self.session.state=='EXECUTING': self.monitor(frames)
            except Exception as exc:
                if self.camera_requested:self.camera_error=str(exc)
                if self.session.state=='EXECUTING':self.monitor_error=str(exc)
            time.sleep(.10)

    def tick(self,_event):
        alive=time.monotonic()-self.heartbeat<=.5
        executing=self.session.state=='EXECUTING'
        if executing and (not alive or self.monitor_error or time.monotonic()-self.monitor_time>.5):
            reason=self.monitor_error or 'operator_or_scene_monitor_heartbeat_lost'
            self.stop(reason);executing=False
        self.sequence+=1
        lease=dict(self.lease_detail,session_id=self.control_id,sequence=self.sequence,
                   stamp=rospy.Time.now().to_sec(),alive=alive,executing=executing,operation=self.operation)
        self.lease_pub.publish(String(json.dumps(lease)))
        self.armed_pub.publish(Bool(executing and alive and bool(self.operation)))
        if self.sequence%4==0:
            status=self.session.status()
            # Trajectory arrays remain backend-owned; the panel receives summary only.
            if status['preview']:
                status['preview']={k:v for k,v in status['preview'].items() if k not in ('positions','times')}
            status.update(control_id=self.control_id,connected=self.connected,enabled=self.enabled,
                calibration_valid=self.calibration is not None,calibration_rows=self.calibration_rows,
                keypoints=self.keys,selection=self.selection,constraints=self.constraints,instruction=self.instruction,
                keypoint_stamp=self.key_stamp,tracking_stamp=self.tracking_stamp,
                tracking_valid=self.tracking_valid,warning=self.warning,camera_error=self.camera_error,
                session_directory=str(self.root),
                joints=None if self.joints is None else self.joints.tolist(),opening=self.opening,
                feedback_age_s=min(999.,time.monotonic()-self.feedback_time),
                raw_joints=None if self.raw_joints is None else self.raw_joints.tolist(),
                raw_feedback_age_s=min(999.,time.monotonic()-self.raw_feedback_time),
                joint_validation_error=self.joint_validation_error,
                arm_status=None if self.arm is None else int(self.arm.arm_status),
                teach_status=None if self.arm is None else int(self.arm.teach_status),
                ctrl_mode=None if self.arm is None else int(self.arm.ctrl_mode),
                camera_stamps={name:(frames[-1]['rgb_stamp_s'] if frames else 0.)
                               for name,frames in self.observations.frames.items()})
            self.status_pub.publish(String(json.dumps(status,ensure_ascii=False)))

    def shutdown(self):
        self.stop('workbench_shutdown')
        for process in self.children.values():
            if process.poll() is None:
                import signal
                os.killpg(process.pid,signal.SIGINT)


if __name__=='__main__':
    rospy.init_node('session')
    Workbench()
    rospy.spin()
