#!/usr/bin/env python3
"""Publish dual-camera Piper masks with cuRobo RobotSegmenter."""

from pathlib import Path
import threading

from cv_bridge import CvBridge
import message_filters
import numpy as np
import rospy
import rospkg
from sensor_msgs.msg import CameraInfo, Image, JointState
import tf2_ros
import torch
import yaml

from rekpiper_camera.projection import transform_to_matrix


JOINT_NAMES = ["joint{}".format(index) for index in range(1, 9)]


class CuroboRobotSegmenterNode:
    def __init__(self):
        if not torch.cuda.is_available():
            raise rospy.ROSInitException("cuRobo robot segmentation requires CUDA")
        try:
            from curobo.types.base import TensorDeviceType
            from curobo.wrap.model.robot_segmenter import RobotSegmenter
        except (ImportError, OSError) as exc:
            raise rospy.ROSInitException(
                "cuRobo v0.7.6 is required for robot segmentation: {}".format(exc))

        self._bridge = CvBridge()
        self._lock = threading.RLock()
        self._inference = threading.Lock()
        self._joints = None
        self._joint_stamp = rospy.Time(0)
        self._maximum_joint_age = float(rospy.get_param(
            "~maximum_joint_state_age_s", 0.10))
        self._tf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf)
        self._base = str(rospy.get_param("~base_frame", "base_link"))
        tensor_args = TensorDeviceType.from_basic("cuda", 0)

        config_path = Path(rospy.get_param("~robot_config")).expanduser().resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        description = Path(rospkg.RosPack().get_path("piper_description")).resolve()
        kinematics = config["robot_cfg"]["kinematics"]
        kinematics["urdf_path"] = str(description / "urdf" / "piper_description.urdf")
        kinematics["asset_root_path"] = str(description)
        self._camera_names = tuple(rospy.get_param("~camera_names", ["rs1", "rs3"]))
        # RobotSegmenter owns mutable projection rays and CUDA graph buffers.
        # Each camera must retain its own intrinsics and inference state.
        self._segmenters = {camera: RobotSegmenter.from_robot_file(
            config,
            collision_sphere_buffer=float(rospy.get_param(
                "~collision_sphere_buffer_m", 0.005)),
            distance_threshold=float(rospy.get_param(
                "~distance_threshold_m", 0.015)),
            use_cuda_graph=bool(rospy.get_param("~use_cuda_graph", True)),
            tensor_args=tensor_args) for camera in self._camera_names}
        self._camera_models = {}
        self._tensor_args = tensor_args

        rospy.Subscriber(rospy.get_param(
            "~joint_state_topic", "/joint_states"),
            JointState, self._joint_callback, queue_size=10)
        self._publishers = {}
        self._syncs = []
        for camera in self._camera_names:
            prefix = "/{}".format(camera)
            depth = message_filters.Subscriber(
                prefix + "/aligned_depth_to_color/image_raw", Image,
                queue_size=1, buff_size=8 * 1024 * 1024)
            info = message_filters.Subscriber(
                prefix + "/color/camera_info", CameraInfo, queue_size=2)
            sync = message_filters.ApproximateTimeSynchronizer(
                [depth, info], queue_size=4, slop=0.02)
            sync.registerCallback(
                lambda depth_msg, info_msg, name=camera:
                self._camera_callback(name, depth_msg, info_msg))
            self._syncs.append((depth, info, sync))
            self._publishers[camera] = rospy.Publisher(
                "/rekpiper/mapping/{}/robot_mask".format(camera),
                Image, queue_size=2)

    def _joint_callback(self, message):
        mapping = dict(zip(message.name, message.position))
        if not all(name in mapping for name in JOINT_NAMES):
            return
        values = np.asarray([mapping[name] for name in JOINT_NAMES], dtype=np.float32)
        if not np.all(np.isfinite(values)):
            return
        with self._lock:
            self._joints = values
            self._joint_stamp = message.header.stamp or rospy.Time.now()

    def _camera_callback(self, camera, depth_message, info_message):
        if not self._inference.acquire(False):
            return
        try:
            with self._lock:
                joints = None if self._joints is None else self._joints.copy()
                joint_stamp = self._joint_stamp
            if joints is None:
                raise RuntimeError("Piper joint state is unavailable")
            stamp = depth_message.header.stamp
            if abs((stamp - joint_stamp).to_sec()) > self._maximum_joint_age:
                raise RuntimeError("Piper joint state is stale")
            depth = self._bridge.imgmsg_to_cv2(depth_message, "passthrough")
            if depth_message.encoding == "32FC1":
                depth = np.asarray(depth, dtype=np.float32) * 1000.0
            else:
                depth = np.asarray(depth, dtype=np.float32)
            transform = self._tf.lookup_transform(
                self._base, info_message.header.frame_id, stamp,
                rospy.Duration(0.08))
            camera_to_base = transform_to_matrix(transform.transform).astype(np.float32)

            from curobo.types.camera import CameraObservation
            from curobo.types.math import Pose
            from curobo.types.state import JointState as CuroboJointState
            observation = CameraObservation(
                depth_image=self._tensor_args.to_device(depth).unsqueeze(0),
                intrinsics=self._tensor_args.to_device(
                    np.asarray(info_message.K, dtype=np.float32).reshape(3, 3)),
                pose=Pose.from_matrix(self._tensor_args.to_device(camera_to_base)))
            state = CuroboJointState.from_numpy(
                JOINT_NAMES, joints, tensor_args=self._tensor_args)
            segmenter = self._segmenters[camera]
            model = (depth.shape, tuple(info_message.K))
            if self._camera_models.get(camera) != model:
                segmenter.update_camera_projection(observation)
                self._camera_models[camera] = model
            mask, _filtered = segmenter.get_robot_mask(observation, state)
            output = (mask[0].detach().to("cpu").numpy().astype(np.uint8) * 255)
            message = self._bridge.cv2_to_imgmsg(output, "mono8")
            message.header = depth_message.header
            self._publishers[camera].publish(message)
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0, "%s cuRobo robot mask rejected: %s", camera, exc)
        finally:
            self._inference.release()


if __name__ == "__main__":
    rospy.init_node("curobo_robot_segmenter")
    CuroboRobotSegmenterNode()
    rospy.spin()
