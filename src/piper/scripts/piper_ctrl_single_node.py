#!/usr/bin/env python3
# -*-coding:utf8-*-
# 本文件为控制单个机械臂节点，控制夹爪机械臂运动
from typing import (
    Optional,
)
import rospy
import rosnode
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import time
import threading
import argparse
import math
from piper_sdk import *
from piper_sdk import C_PiperInterface
from std_srvs.srv import Trigger, TriggerResponse
from piper_msgs.msg import PiperStatusMsg, PosCmd, PiperEulerPose
from rekpiper_acceptance import (
    AcceptanceError, assert_release_unchanged, validate_release_bundle)
from piper_msgs.srv import Enable, EnableResponse
from piper_msgs.srv import Gripper, GripperResponse
from piper_msgs.srv import GoZero, GoZeroResponse
from geometry_msgs.msg import Pose, PoseStamped
from tf.transformations import quaternion_from_euler  # 用于欧拉角到四元数的转换
import numpy as np

def check_ros_master():
    try:
        rosnode.rosnode_ping('rosout', max_count=1, verbose=False)
        rospy.loginfo("ROS Master is running.")
    except rosnode.ROSNodeIOException:
        rospy.logerr("ROS Master is not running.")
        raise RuntimeError("ROS Master is not running.")

class C_PiperRosNode():
    """机械臂ros节点
    """
    def __init__(self) -> None:
        check_ros_master()
        rospy.init_node('piper_ctrl_single_node', anonymous=True)
        # JointCtrl remains a high-rate position stream. MotionCtrl_2 is a
        # mode/speed selector and must not be reset for every trajectory point.
        self._motion_ctrl_refresh_s = float(rospy.get_param(
            "~motion_ctrl_refresh_s", 1.0))
        self._joint_command_default_speed = int(rospy.get_param(
            "~joint_command_default_speed_percent", 10))
        self._last_motion_ctrl = None
        self._last_motion_ctrl_time = 0.0
        # 外部param参数
        # can路由名称
        self.can_port = "can0"
        if rospy.has_param('~can_port'):
            self.can_port = rospy.get_param("~can_port")
            rospy.loginfo("%s is %s", rospy.resolve_name('~can_port'), self.can_port)
        else: 
            rospy.loginfo("未找到can_port参数")
            exit(0)
        # 是否自动使能，默认不自动使能
        self.auto_enable = False
        if rospy.has_param('~auto_enable'):
            if(rospy.get_param("~auto_enable")):
                self.auto_enable = True
        if self.auto_enable:
            raise rospy.ROSInitException(
                "auto_enable is forbidden; use /enable_srv and verify "
                "/enable_status_srv after signed preflight")
        rospy.loginfo("%s is %s", rospy.resolve_name('~auto_enable'), self.auto_enable)
        # 是否有夹爪，默认为有
        self.gripper_exist = True
        if rospy.has_param('~gripper_exist'):
            if(not rospy.get_param("~gripper_exist")):
                self.gripper_exist = False
        rospy.loginfo("%s is %s", rospy.resolve_name('~gripper_exist'), self.gripper_exist)
        self._hold_only = bool(rospy.get_param("~hold_only", False))
        self._hold_tolerance_rad = float(rospy.get_param(
            "~hold_tolerance_rad", 0.01))
        self._hold_reference = None
        self._command_armed = False
        self._armed_topic = str(rospy.get_param(
            "~armed_topic", "/rekpiper/execution/hardware_armed"))
        self._authorized_joint_caller = str(rospy.get_param(
            "~authorized_joint_command_caller",
            "/piper_trajectory_bridge"))
        self._authorized_gripper_caller = str(rospy.get_param(
            "~authorized_gripper_service_caller",
            "/piper_gripper_action"))
        self._allow_cartesian_commands = bool(rospy.get_param(
            "~allow_cartesian_commands", False))
        self._allow_go_zero_service = bool(rospy.get_param(
            "~allow_go_zero_service", False))
        # 是否是打开了rviz控制，默认为不是，如果打开了，gripper订阅的joint7关节消息会乘2倍-------已弃用
        self.rviz_ctrl_flag = False
        if rospy.has_param('~rviz_ctrl_flag'):
            if(rospy.get_param("~rviz_ctrl_flag")):
                self.rviz_ctrl_flag = True
        rospy.loginfo("%s is %s", rospy.resolve_name('~rviz_ctrl_flag'), self.rviz_ctrl_flag)
        # 夹爪的数值倍数，默认为1
        self.gripper_val_mutiple = 1  # 默认值
        if rospy.has_param('~gripper_val_mutiple'):
            gripper_val_mutiple = rospy.get_param("~gripper_val_mutiple")
            # 检查是否为数字（浮动数或整数）
            if isinstance(gripper_val_mutiple, (int, float)):
                # 确保值在合理范围内
                if gripper_val_mutiple <= 0:
                    rospy.logwarn("Invalid gripper_val_mutiple value: must be positive. Using default value of 1.")
                    self.gripper_val_mutiple = 1  # 设置为默认值
                else:
                    self.gripper_val_mutiple = gripper_val_mutiple
            else:
                rospy.logwarn("Invalid gripper_val_mutiple type. Expected int or float. Using default value of 1.")
                self.gripper_val_mutiple = 1  # 设置为默认值
        else:
            rospy.logwarn("No gripper_val_mutiple param. Using default value of 1.")
            self.gripper_val_mutiple = 1  # 设置为默认值
        rospy.loginfo("%s is %s", rospy.resolve_name('~gripper_val_mutiple'), self.gripper_val_mutiple)
        # publish
        self.joint_pub = rospy.Publisher('joint_states_single', JointState, queue_size=1)
        self.arm_status_pub = rospy.Publisher('arm_status', PiperStatusMsg, queue_size=1)
        # self.end_pose_euler_pub = rospy.Publisher('end_pose_euler', PosCmd, queue_size=1)
        self.end_pose_pub = rospy.Publisher('end_pose', PoseStamped, queue_size=1)
        self.end_pose_euler_pub = rospy.Publisher('end_pose_euler', PiperEulerPose, queue_size=1)
        # service
        self.enable_service = rospy.Service('enable_srv', Enable, self.handle_enable_service)  # 创建enable服务
        self.enable_status_service = rospy.Service(
            'enable_status_srv', Trigger, self.handle_enable_status_service)
        self.__enable_flag = False
        self._last_go_zero_failure = "not_started"
        self.gripper_service = rospy.Service('gripper_srv', Gripper, self.handle_gripper_service)  # 创建gripper服务
        self.stop_service = rospy.Service('stop_srv', Trigger, self.handle_stop_service)  # 创建stop服务
        self.reset_service = rospy.Service('reset_srv', Trigger, self.handle_reset_service)  # 创建reset服务
        self.go_zero_service = rospy.Service('go_zero_srv', GoZero, self.handle_go_zero_service)  # 创建reset服务
        # joint
        self.joint_states = JointState()
        self.joint_states.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper']
        self.joint_states.position = [0.0] * 7
        self.joint_states.velocity = [0.0] * 7
        self.joint_states.effort = [0.0] * 7
        
        # Validate the complete signed release before constructing the SDK or
        # opening CAN. roslaunch sibling ordering is not a safety boundary.
        try:
            self._release = validate_release_bundle(
                rospy.get_param("~release_bundle", ""),
                rospy.get_param("~acceptance_public_key", ""),
                rospy.get_param("~minimum_release_counter", ""),
                rospy.get_param("~robot_id", ""))
        except AcceptanceError as exc:
            raise rospy.ROSInitException(
                "signed autonomous release rejected before CAN: " + str(exc))
        # 创建piper类并打开can接口
        self.piper = C_PiperInterface(can_name=self.can_port)
        self.piper.ConnectPort()
        # Connecting the telemetry interface must not alter controller mode.
        # MotionCtrl is selected only after explicit, acknowledged enable.

        # 启动订阅线程
        sub_pos_th = threading.Thread(target=self.SubPosThread)
        sub_pos_th.daemon = True
        sub_pos_th.start()
        sub_joint_th = threading.Thread(target=self.SubJointThread)
        sub_enable_th = threading.Thread(target=self.SubEnableThread)
        
        sub_joint_th.daemon = True
        sub_enable_th.daemon = True
        
        sub_joint_th.start()
        sub_enable_th.start()

    def GetEnableFlag(self):
        return self.__enable_flag

    def _release_current(self):
        try:
            assert_release_unchanged(self._release)
            return True
        except AcceptanceError as exc:
            self.__enable_flag = False
            rospy.logerr_throttle(
                1.0, "signed release changed; hardware authorization revoked: %s",
                exc)
            return False

    def _ensure_joint_motion_mode(self, speed_percent, force=False):
        """Select CAN/MOVE-J only on change, first use or 1 s heartbeat."""
        if not self._release_current():
            return False
        speed = max(0, min(100, int(round(speed_percent))))
        mode = (0x01, 0x01, speed, 0x00)
        now = time.monotonic()
        changed = mode != self._last_motion_ctrl
        heartbeat = now - self._last_motion_ctrl_time >= self._motion_ctrl_refresh_s
        if force or changed or heartbeat:
            self.piper.MotionCtrl_2(*mode)
            self._last_motion_ctrl = mode
            self._last_motion_ctrl_time = now
            if changed or force:
                rospy.loginfo(
                    "Piper MOVE-J mode selected at %d%% (JointCtrl remains streamed)",
                    speed)
            else:
                rospy.logdebug("Piper MOVE-J 1 s heartbeat at %d%%", speed)
        return True

    def Pubilsh(self):
        """机械臂消息发布
        """
        rate = rospy.Rate(200)  # 200 Hz
        while not rospy.is_shutdown():
            # print(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
            # 发布消息
            self.PublishArmState()
            self.PublishArmEndPose()
            self.PublishArmJointAndGripper()
            rate.sleep()

    def PublishArmState(self):
        # 机械臂状态
        arm_status = PiperStatusMsg()
        arm_status.ctrl_mode = self.piper.GetArmStatus().arm_status.ctrl_mode
        arm_status.arm_status = self.piper.GetArmStatus().arm_status.arm_status
        arm_status.mode_feedback = self.piper.GetArmStatus().arm_status.mode_feed
        arm_status.teach_status = self.piper.GetArmStatus().arm_status.teach_status
        arm_status.motion_status = self.piper.GetArmStatus().arm_status.motion_status
        arm_status.trajectory_num = self.piper.GetArmStatus().arm_status.trajectory_num
        arm_status.err_code = self.piper.GetArmStatus().arm_status.err_code
        arm_status.joint_1_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_1_angle_limit
        arm_status.joint_2_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_2_angle_limit
        arm_status.joint_3_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_3_angle_limit
        arm_status.joint_4_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_4_angle_limit
        arm_status.joint_5_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_5_angle_limit
        arm_status.joint_6_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_6_angle_limit
        arm_status.communication_status_joint_1 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_1
        arm_status.communication_status_joint_2 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_2
        arm_status.communication_status_joint_3 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_3
        arm_status.communication_status_joint_4 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_4
        arm_status.communication_status_joint_5 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_5
        arm_status.communication_status_joint_6 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_6
        self.arm_status_pub.publish(arm_status)
        
    def PublishArmJointAndGripper(self):
        # 机械臂关节角和夹爪位置
        # Piper feedback is expressed in 0.001 degree units.
        millidegree_to_rad = math.pi / 180000.0
        joint_0:float = self.piper.GetArmJointMsgs().joint_state.joint_1 * millidegree_to_rad
        joint_1:float = self.piper.GetArmJointMsgs().joint_state.joint_2 * millidegree_to_rad
        joint_2:float = self.piper.GetArmJointMsgs().joint_state.joint_3 * millidegree_to_rad
        joint_3:float = self.piper.GetArmJointMsgs().joint_state.joint_4 * millidegree_to_rad
        joint_4:float = self.piper.GetArmJointMsgs().joint_state.joint_5 * millidegree_to_rad
        joint_5:float = self.piper.GetArmJointMsgs().joint_state.joint_6 * millidegree_to_rad
        joint_6:float = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle/1000000
        vel_0:float = self.piper.GetArmHighSpdInfoMsgs().motor_1.motor_speed/1000
        vel_1:float = self.piper.GetArmHighSpdInfoMsgs().motor_2.motor_speed/1000
        vel_2:float = self.piper.GetArmHighSpdInfoMsgs().motor_3.motor_speed/1000
        vel_3:float = self.piper.GetArmHighSpdInfoMsgs().motor_4.motor_speed/1000
        vel_4:float = self.piper.GetArmHighSpdInfoMsgs().motor_5.motor_speed/1000
        vel_5:float = self.piper.GetArmHighSpdInfoMsgs().motor_6.motor_speed/1000
        effort_0:float = self.piper.GetArmHighSpdInfoMsgs().motor_1.effort/1000
        effort_1:float = self.piper.GetArmHighSpdInfoMsgs().motor_2.effort/1000
        effort_2:float = self.piper.GetArmHighSpdInfoMsgs().motor_3.effort/1000
        effort_3:float = self.piper.GetArmHighSpdInfoMsgs().motor_4.effort/1000
        effort_4:float = self.piper.GetArmHighSpdInfoMsgs().motor_5.effort/1000
        effort_5:float = self.piper.GetArmHighSpdInfoMsgs().motor_6.effort/1000
        effort_6:float = self.piper.GetArmGripperMsgs().gripper_state.grippers_effort/1000
        self.joint_states.header.stamp = rospy.Time.now()
        self.joint_states.position = [joint_0,joint_1, joint_2, joint_3, joint_4, joint_5,joint_6]
        if self._hold_only and self._hold_reference is None:
            measured = np.asarray(self.joint_states.position[:6], dtype=float)
            if np.all(np.isfinite(measured)):
                self._hold_reference = measured.copy()
        self.joint_states.velocity = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, 0.0]
        self.joint_states.effort = [effort_0, effort_1, effort_2, effort_3, effort_4, effort_5, effort_6]
        # 发布所有消息
        self.joint_pub.publish(self.joint_states)
    
    def PublishArmEndPose(self):
        # 末端位姿
        endpos = PoseStamped()
        endpos.pose.position.x = self.piper.GetArmEndPoseMsgs().end_pose.X_axis/1000000
        endpos.pose.position.y = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis/1000000
        endpos.pose.position.z = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis/1000000
        roll = self.piper.GetArmEndPoseMsgs().end_pose.RX_axis/1000
        pitch = self.piper.GetArmEndPoseMsgs().end_pose.RY_axis/1000
        yaw = self.piper.GetArmEndPoseMsgs().end_pose.RZ_axis/1000
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        quaternion = quaternion_from_euler(roll, pitch, yaw)
        endpos.pose.orientation.x = quaternion[0]
        endpos.pose.orientation.y = quaternion[1]
        endpos.pose.orientation.z = quaternion[2]
        endpos.pose.orientation.w = quaternion[3]
        endpos.header.stamp = rospy.Time.now()
        self.end_pose_pub.publish(endpos)
        
        end_pose_euler = PiperEulerPose()
        end_pose_euler.header.stamp = rospy.Time.now()
        # end_pose_euler.header.seq = endpos.header.seq
        end_pose_euler.x = self.piper.GetArmEndPoseMsgs().end_pose.X_axis/1000000
        end_pose_euler.y = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis/1000000
        end_pose_euler.z = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis/1000000
        end_pose_euler.roll = roll
        end_pose_euler.pitch = pitch
        end_pose_euler.yaw = yaw
        self.end_pose_euler_pub.publish(end_pose_euler)
    
    def SubPosThread(self):
        """机械臂末端位姿订阅
        
        """
        rospy.Subscriber('pos_cmd', PosCmd, self.pos_callback, queue_size=1, tcp_nodelay=True)
        rospy.spin()
    
    def SubJointThread(self):
        """机械臂关节订阅
        
        """
        rospy.Subscriber('joint_ctrl_single', JointState, self.joint_callback, queue_size=1, tcp_nodelay=True)
        # rospy.Subscriber('/move_group/fake_controller_joint_states', JointState, self.joint_callback)
        rospy.spin()
    
    def SubEnableThread(self):
        """机械臂使能
        
        """
        rospy.Subscriber('enable_flag', Bool, self.enable_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self._armed_topic, Bool, self.armed_callback,
                         queue_size=1, tcp_nodelay=True)
        rospy.spin()

    @staticmethod
    def _caller_id(message):
        header = getattr(message, "_connection_header", None)
        return str(header.get("callerid", "")) if isinstance(header, dict) else ""

    def armed_callback(self, message):
        self._command_armed = bool(message.data)

    def pos_callback(self, pos_data):
        """机械臂末端位姿订阅回调函数

        Args:
            pos_data (): 
        """
        # rospy.loginfo("Received PosCmd:")
        # rospy.loginfo("x: %f", pos_data.x)
        # rospy.loginfo("y: %f", pos_data.y)
        # rospy.loginfo("z: %f", pos_data.z)
        # rospy.loginfo("roll: %f", pos_data.roll)
        # rospy.loginfo("pitch: %f", pos_data.pitch)
        # rospy.loginfo("yaw: %f", pos_data.yaw)
        # rospy.loginfo("gripper: %f", pos_data.gripper)
        # rospy.loginfo("mode1: %d", pos_data.mode1)
        # rospy.loginfo("mode2: %d", pos_data.mode2)
        if self._hold_only or not self._allow_cartesian_commands:
            rospy.logerr_throttle(
                1.0, "Cartesian commands are disabled in the ReKpiper driver")
            return
        factor = 180 / np.pi
        x = round(pos_data.x*1000) * 1000
        y = round(pos_data.y*1000) * 1000
        z = round(pos_data.z*1000) * 1000
        rx = round(pos_data.roll*1000*factor) 
        ry = round(pos_data.pitch*1000*factor)
        rz = round(pos_data.yaw*1000*factor)
        rospy.loginfo("Received PosCmd:")
        rospy.loginfo("x: %f", x)
        rospy.loginfo("y: %f", y)
        rospy.loginfo("z: %f", z)
        rospy.loginfo("roll: %f", rx)
        rospy.loginfo("pitch: %f", ry)
        rospy.loginfo("yaw: %f", rz)
        rospy.loginfo("gripper: %f", pos_data.gripper)
        rospy.loginfo("mode1: %d", pos_data.mode1)
        rospy.loginfo("mode2: %d", pos_data.mode2)
        if(self.GetEnableFlag() and self._release_current()):
            self.piper.MotionCtrl_1(0x00, 0x00, 0x00)
            self.piper.MotionCtrl_2(0x01, 0x00, 50)
            self.piper.EndPoseCtrl(x, y, z, 
                                    rx, ry, rz)
            gripper = round(pos_data.gripper*1000*1000)
            if(pos_data.gripper>80000): gripper = 80000
            if(pos_data.gripper<0): gripper = 0
            if(self.gripper_exist):
                self.piper.GripperCtrl(abs(gripper), 1000, 0x01, 0)
            self.piper.MotionCtrl_2(0x01, 0x00, 50)
    
    def joint_callback(self, joint_data):
        """机械臂关节角回调函数

        Args:
            joint_data (): 
        """
        if self._caller_id(joint_data) != self._authorized_joint_caller:
            rospy.logerr_throttle(
                1.0, "joint command rejected from unauthorized ROS caller")
            return
        if not self._command_armed:
            rospy.logwarn_throttle(
                1.0, "joint command rejected: execution authorization is false")
            return
        if (len(joint_data.position) < 6
                or not np.all(np.isfinite(
                    np.asarray(joint_data.position[:6], dtype=float)))):
            rospy.logerr_throttle(1.0, "invalid joint command rejected")
            return
        factor = 1000 * 180 / np.pi
        # rospy.loginfo("Received Joint States:")
        # rospy.loginfo("joint_0: %f", joint_data.position[0])
        # rospy.loginfo("joint_1: %f", joint_data.position[1])
        # rospy.loginfo("joint_2: %f", joint_data.position[2])
        # rospy.loginfo("joint_3: %f", joint_data.position[3])
        # rospy.loginfo("joint_4: %f", joint_data.position[4])
        # rospy.loginfo("joint_5: %f", joint_data.position[5])
        # rospy.loginfo("joint_6: %f", joint_data.position[6])
        # print(joint_data.position)
        joint_0 = round(joint_data.position[0]*factor)
        joint_1 = round(joint_data.position[1]*factor)
        joint_2 = round(joint_data.position[2]*factor)
        joint_3 = round(joint_data.position[3]*factor)
        joint_4 = round(joint_data.position[4]*factor)
        joint_5 = round(joint_data.position[5]*factor)
        if self._hold_only:
            requested = np.asarray(joint_data.position[:6], dtype=float)
            if (self._hold_reference is None
                    or requested.shape != (6,)
                    or not np.all(np.isfinite(requested))
                    or np.max(np.abs(requested - self._hold_reference))
                    > self._hold_tolerance_rad):
                rospy.logerr_throttle(
                    1.0, "non-hold joint command rejected in acceptance mode")
                return
        if(len(joint_data.position) >= 7):
            joint_6 = round(joint_data.position[6]*1000*1000)
            joint_6 = joint_6 * self.gripper_val_mutiple
            if(joint_6>80000): joint_6 = 80000
            if(joint_6<0): joint_6 = 0
        else: joint_6 = None
        if(self.GetEnableFlag() and self._release_current()):
            speed = self._joint_command_default_speed
            if len(joint_data.velocity) >= 7 and joint_data.velocity[6] > 0:
                speed = joint_data.velocity[6]
            if not self._ensure_joint_motion_mode(speed):
                return
            
            # 给定关节角位置
            self.piper.JointCtrl(joint_0, joint_1, joint_2, 
                                    joint_3, joint_4, joint_5)
            # 如果末端夹爪存在，则发送末端夹爪控制
            if(self.gripper_exist and joint_6 is not None):
                if abs(joint_6)<200:
                    joint_6=0
                if(len(joint_data.effort) >= 7):
                    gripper_effort = joint_data.effort[6]
                    gripper_effort = max(0.5, min(gripper_effort, 3))
                    # rospy.loginfo("gripper_effort: %f", gripper_effort)
                    gripper_effort = round(gripper_effort*1000)
                    self.piper.GripperCtrl(abs(joint_6), gripper_effort, 0x01, 0)
                # 默认1N
                else: self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        else:
            rospy.logwarn_throttle(
                2.0,
                "joint_ctrl_single ignored: Piper driver enable flag is false")
    
    def enable_callback(self, enable_flag:Bool):
        """机械臂使能回调函数

        Args:
            enable_flag (): 
        """
        rospy.loginfo("Received enable flag:")
        rospy.loginfo("enable_flag: %s", enable_flag.data)
        if(enable_flag.data):
            # The legacy Bool topic has no response channel and therefore
            # cannot carry the six-motor acknowledgement required before
            # motion.  Keep the topic for compatibility but require callers
            # to use /enable_srv for a confirmed enable transition.
            rospy.logerr(
                "enable_flag=true ignored; use /enable_srv and verify "
                "/enable_status_srv")
        else:
            self.__enable_flag = False
            self.piper.DisableArm(7)
            if(self.gripper_exist):
                self.piper.GripperCtrl(0,1000,0x00, 0)
    
    def handle_gripper_service(self,req):
        response = GripperResponse()
        response.code = 15999
        response.status = False
        if self._hold_only:
            response.code = 15904
            return response
        if (self._caller_id(req) != self._authorized_gripper_caller
                or not self._command_armed):
            response.code = 15905
            return response
        if(self.gripper_exist and self.GetEnableFlag()
                and self._release_current()):
            rospy.loginfo(f"-----------------------Gripper---------------------------")
            rospy.loginfo(f"Received request:")
            rospy.loginfo(f"PS: Piper should be enable.Please ensure piper is enable")
            rospy.loginfo(f"gripper_angle:{req.gripper_angle}, range is [0m, 0.07m]")
            rospy.loginfo(f"gripper_effort:{req.gripper_effort},range is [0.5N/m, 2N/m]")
            rospy.loginfo(f"gripper_code:{req.gripper_code}, range is [0, 1, 2, 3]\n \
                            0x00: Disable\n \
                            0x01: Enable\n \
                            0x03/0x02: Enable and clear error / Disable and clear error")
            rospy.loginfo(f"set_zero:{req.set_zero}, range is [0, 0xAE] \n \
                            0x00: Invalid value \n \
                            0xAE: Set zero point")
            rospy.loginfo(f"-----------------------Gripper---------------------------")
            gripper_angle = req.gripper_angle
            gripper_angle = round(max(0, min(req.gripper_angle, 0.07)) * 1e6)
            gripper_effort = req.gripper_effort
            gripper_effort = round(max(0.5, min(req.gripper_effort, 2)) * 1e3)
            if req.gripper_code not in [0x00, 0x01, 0x02, 0x03]:
                rospy.logwarn("gripper_code should be in [0, 1, 2, 3], default val is 1")
                gripper_code = 1
                response.code = 15901
            else: gripper_code = req.gripper_code
            if req.set_zero not in [0x00, 0xAE]:
                rospy.logwarn("set_zero should be in [0, 0xAE], default val is 0")
                set_zero = 0
                response.code = 15902
            else: set_zero = req.set_zero
            response.code = 15900
            self.piper.GripperCtrl(abs(gripper_angle), gripper_effort, gripper_code, set_zero)
            response.status = True
        else:
            rospy.logwarn(
                "gripper command rejected: gripper missing or Piper not "
                "confirmed enabled")
            response.code = 15903
            response.status = False
        rospy.loginfo(f"Returning GripperResponse: {response.code}, {response.status}")
        return response
    
    def _motor_enable_states(self):
        info = self.piper.GetArmLowSpdInfoMsgs()
        return [
            bool(info.motor_1.foc_status.driver_enable_status),
            bool(info.motor_2.foc_status.driver_enable_status),
            bool(info.motor_3.foc_status.driver_enable_status),
            bool(info.motor_4.foc_status.driver_enable_status),
            bool(info.motor_5.foc_status.driver_enable_status),
            bool(info.motor_6.foc_status.driver_enable_status),
        ]

    def handle_enable_status_service(self, _req):
        motors = self._motor_enable_states()
        # The hardware feedback is authoritative.  Never retain a software
        # enable flag after all six motor drivers report disabled.
        if self.GetEnableFlag() and not all(motors):
            self.__enable_flag = False
        status = self.piper.GetArmStatus().arm_status
        ctrl_mode = int(status.ctrl_mode)
        arm_status = int(status.arm_status)
        mode_feedback = int(status.mode_feed)
        err_code = int(status.err_code)
        ready = bool(
            self.GetEnableFlag() and all(motors)
            and ctrl_mode == 1 and arm_status == 0
            and mode_feedback == 1 and err_code == 0)
        detail = (
            "internal_enable={} motors={} ctrl_mode={} arm_status={} "
            "mode_feedback={} err_code={}").format(
                self.GetEnableFlag(),
                "".join("1" if value else "0" for value in motors),
                ctrl_mode, arm_status, mode_feedback, err_code)
        return TriggerResponse(success=ready, message=detail)

    def handle_enable_service(self,req):
        rospy.loginfo(f"Received request: {req.enable_request}")
        requested_enable = bool(req.enable_request)
        if requested_enable and not self._release_current():
            return EnableResponse(False)
        # Enabling selects 5% during acknowledgement. Force the next
        # trajectory command to select its requested speed immediately.
        self._last_motion_ctrl = None
        self._last_motion_ctrl_time = 0.0
        deadline = time.monotonic() + 5.0
        stable_enabled_samples = 0
        # Transmit first, then judge only feedback received after the request.
        # The previous order sampled stale enable bits before EnableArm(), which
        # could leave __enable_flag true while all real motors were disabled.
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if requested_enable:
                # Match the installed Piper SDK's official enable example.
                # EnablePiper() sends EnableArm(7) and returns the measured
                # six-motor acknowledgement; the explicit feedback check below
                # remains authoritative.
                self.piper.EnablePiper()
            else:
                self.piper.DisableArm(7)
            time.sleep(0.10)
            motors = self._motor_enable_states()
            enabled = all(motors)
            disabled = not any(motors)
            status = self.piper.GetArmStatus().arm_status
            rospy.loginfo(
                "Piper enable acknowledgement requested=%s motors=%s",
                requested_enable,
                "".join("1" if value else "0" for value in motors))
            if requested_enable and enabled and int(status.arm_status) == 0 and int(status.err_code) == 0:
                # A one-frame 111111 can occur while joint feedback is still
                # coming online.  Require a continuous acknowledgement before
                # authorising any motion.
                stable_enabled_samples += 1
                if stable_enabled_samples < 10:
                    time.sleep(0.10)
                    continue
                # Fresh startup does not need an emergency-stop recovery frame.
                # Select CAN/MOVE-J at 5%, then verify that enable remains
                # present after the mode transition before returning success.
                self.piper.MotionCtrl_2(0x01, 0x01, 5, 0x00)
                self._last_motion_ctrl = (0x01, 0x01, 5, 0x00)
                self._last_motion_ctrl_time = time.monotonic()
                time.sleep(0.50)
                post_motors = self._motor_enable_states()
                post_status = self.piper.GetArmStatus().arm_status
                post_ready = bool(
                    all(post_motors)
                    and int(post_status.ctrl_mode) == 1
                    and int(post_status.arm_status) == 0
                    and int(post_status.mode_feed) == 1
                    and int(post_status.err_code) == 0)
                if not post_ready:
                    self.__enable_flag = False
                    rospy.logerr(
                        "Piper enable did not persist after MOVE-J selection: "
                        "motors=%s ctrl_mode=%d arm_status=%d mode_feedback=%d "
                        "err_code=%d",
                        "".join("1" if value else "0" for value in post_motors),
                        int(post_status.ctrl_mode), int(post_status.arm_status),
                        int(post_status.mode_feed), int(post_status.err_code))
                    return EnableResponse(False)
                self.__enable_flag = True
                rospy.loginfo("Returning response: True")
                return EnableResponse(True)
            stable_enabled_samples = 0
            if not requested_enable and disabled:
                self.__enable_flag = False
                rospy.loginfo("Returning response: True")
                return EnableResponse(True)
            # Keep software authorization false until all six motors confirm.
            self.__enable_flag = False
            time.sleep(0.40)
        self.__enable_flag = False
        rospy.logerr("Piper enable request timed out without matching motor feedback")
        return EnableResponse(False)

    def handle_stop_service(self,req):
        response = TriggerResponse()
        response.success = False
        response.message = "stop piper failed"
        rospy.loginfo(f"-----------------------STOP---------------------------")
        rospy.loginfo(f"Stop piper.")
        rospy.loginfo(f"-----------------------STOP---------------------------")
        self.piper.MotionCtrl_1(0x01,0,0)
        self._last_motion_ctrl = None
        # A controller stop does not necessarily clear the six motor-enable
        # feedback bits.  Clear the driver's authorization explicitly so a
        # later trajectory cannot be accepted under a stale enable flag.
        self.__enable_flag = False
        response.success = True
        response.message = "stop piper success"
        rospy.loginfo(f"Returning StopResponse: {response.success}, {response.message}")
        return response

    def handle_reset_service(self,req):
        response = TriggerResponse()
        response.success = False
        response.message = "reset piper failed"
        rospy.loginfo(f"-----------------------RESET---------------------------")
        rospy.loginfo(f"reset piper.")
        rospy.loginfo(f"-----------------------RESET---------------------------")
        if not self._release_current():
            response.message = "signed release changed; reset authorization revoked"
            return response
        self.piper.MotionCtrl_1(0x02,0,0)#恢复
        self._last_motion_ctrl = None
        response.success = True
        response.message = "reset piper success"
        rospy.loginfo(f"Returning resetResponse: {response.success}, {response.message}")
        return response

    def handle_go_zero_service(self,req):
        response = GoZeroResponse()
        response.status = False
        response.code = 151000
        self._last_go_zero_failure = "not_started"
        if self._hold_only or not self._allow_go_zero_service:
            self._last_go_zero_failure = "go_zero_forbidden_in_hold_mode"
            response.code = 151007
            return response
        rospy.loginfo(f"-----------------------GOZERO---------------------------")
        if not self._release_current():
            self._last_go_zero_failure = "signed_release_changed"
            response.code = 151006
            return response
        rospy.loginfo(f"piper go zero .")
        rospy.loginfo(f"-----------------------GOZERO---------------------------")
        # Do not report success until measured feedback is actually at zero.
        # The controller receives the target repeatedly at 20 Hz because a
        # single CAN position frame is only an acknowledgement, not proof of
        # completed motion.
        if not self.GetEnableFlag() or not all(self._motor_enable_states()):
            rospy.logerr("GOZERO rejected: Piper motors are not confirmed enabled")
            self.__enable_flag = False
            self._last_go_zero_failure = "motor_enable_not_confirmed"
            response.code = 151002
            return response
        status = self.piper.GetArmStatus().arm_status
        if int(status.arm_status) != 0 or int(status.err_code) != 0:
            self._last_go_zero_failure = "arm_status={};err_code={}".format(
                int(status.arm_status), int(status.err_code))
            rospy.logerr("GOZERO rejected before motion: %s", self._last_go_zero_failure)
            self.__enable_flag = False
            response.code = 151005
            return response
        # Match the installed SDK's official go-zero sequence: after a verified
        # enable, select CAN/MOVE-J and send the zero joint target.  Do not send
        # another emergency-stop recovery frame here; that extra state change
        # previously caused enable to be lost on this arm.
        if(req.is_mit_mode):
            self.piper.MotionCtrl_2(0x01, 0x01, 5, 0xAD)
        else:
            self.piper.MotionCtrl_2(0x01, 0x01, 5, 0)
        deadline = time.monotonic() + 60.0
        stable_samples = 0
        tolerance_rad = 0.01
        while time.monotonic() < deadline and not rospy.is_shutdown():
            if not self.GetEnableFlag() or not all(self._motor_enable_states()):
                rospy.logerr("GOZERO aborted: Piper motor enable was lost")
                self.__enable_flag = False
                self._last_go_zero_failure = "motor_enable_lost"
                response.code = 151004
                return response
            status = self.piper.GetArmStatus().arm_status
            if int(status.arm_status) != 0 or int(status.err_code) != 0:
                rospy.logerr(
                    "GOZERO aborted: arm_status=%d err_code=%d",
                    int(status.arm_status), int(status.err_code))
                self._last_go_zero_failure = "arm_status={};err_code={}".format(
                    int(status.arm_status), int(status.err_code))
                self.__enable_flag = False
                response.code = 151005
                return response
            self.piper.MotionCtrl_2(
                0x01, 0x01, 5, 0xAD if req.is_mit_mode else 0)
            self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
            joint_state = self.piper.GetArmJointMsgs().joint_state
            raw = (
                joint_state.joint_1, joint_state.joint_2,
                joint_state.joint_3, joint_state.joint_4,
                joint_state.joint_5, joint_state.joint_6,
            )
            maximum_error = max(abs(value) for value in raw) * math.pi / 180000.0
            stable_samples = stable_samples + 1 if maximum_error <= tolerance_rad else 0
            if stable_samples >= 10:
                response.status = True
                response.code = 151001
                rospy.loginfo(
                    "GOZERO reached measured zero at 5%% speed; max error %.6f rad",
                    maximum_error)
                break
            time.sleep(0.05)
        if not response.status:
            rospy.logerr("GOZERO timeout: measured joints did not reach zero")
            self._last_go_zero_failure = "measured_zero_timeout"
            response.code = 151003
        rospy.loginfo(f"Returning GoZeroResponse: {response.status}, {response.code}")
        return response
    
if __name__ == '__main__':
    try:
        piper_signle = C_PiperRosNode()
        piper_signle.Pubilsh()
    except rospy.ROSInterruptException:
        pass
