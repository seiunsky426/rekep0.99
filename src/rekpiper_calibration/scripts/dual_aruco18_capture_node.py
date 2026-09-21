#!/usr/bin/env python3
"""Read-only RGB-D capture for hybrid ArUco18 hand-eye calibration.

Each stopped pose records the four RGB corners and a compact set of metric
depth points sampled from the marker plane.  Full RGB/depth frames are never
written to the dataset.
"""

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import time
import traceback

import cv2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
import tf2_ros
import yaml

from rekpiper_camera.projection import transform_to_matrix
from rekpiper_calibration.robot_world_handeye import (
    average_transforms, check_pose_diversity, transform_deviation)


CAMERAS = ("rs1", "rs3")
SOFTWARE_REVISION = "dual-aruco18-v6-rgbd-joint-state"


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _camera_info(message):
    return {
        "width": int(message.width),
        "height": int(message.height),
        "distortion_model": str(message.distortion_model),
        "K": [float(item) for item in message.K],
        "D": [float(item) for item in message.D],
    }


def _edge_and_margin(corners, image_shape):
    points = np.asarray(corners, dtype=float).reshape(4, 2)
    lengths = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    height, width = image_shape[:2]
    margin = min(
        np.min(points[:, 0]), np.min(points[:, 1]),
        width - 1 - np.max(points[:, 0]), height - 1 - np.max(points[:, 1]))
    return float(np.min(lengths)), float(margin)


def _depth_metres(depth, encoding, scale):
    value = np.asarray(depth)
    if value.ndim != 2:
        raise ValueError("aligned depth must be single-channel")
    if encoding in ("16UC1", "mono16"):
        result = value.astype(np.float64) * float(scale)
    elif encoding == "32FC1":
        result = value.astype(np.float64)
    else:
        raise ValueError("unsupported aligned depth encoding " + str(encoding))
    result[~np.isfinite(result)] = np.nan
    result[result <= 0.0] = np.nan
    return result


def _marker_grid_pixels(corners, grid_size, inset):
    source = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=np.float32)
    destination = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    homography = cv2.getPerspectiveTransform(source, destination)
    coordinates = np.linspace(
        float(inset), 1.0 - float(inset), int(grid_size), dtype=np.float32)
    normalized = np.asarray(
        [(x, y) for y in coordinates for x in coordinates],
        dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(normalized, homography).reshape(-1, 2)


def _sample_marker_depth(corners, depth_m, camera_info, settings):
    """Return one deprojected point per fixed marker-grid location.

    Missing grid locations are represented by NaNs so the same physical
    locations can be robustly combined across the stopped-pose frame batch.
    """
    grid_size = int(settings["grid_size"])
    pixels = _marker_grid_pixels(
        corners, grid_size, float(settings["grid_inset_fraction"]))
    radius = int(settings["patch_radius_px"])
    minimum = float(settings["minimum_depth_m"])
    maximum = float(settings["maximum_depth_m"])
    height, width = depth_m.shape
    camera_matrix = np.asarray(
        camera_info["K"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(camera_info["D"], dtype=np.float64).reshape(-1)
    points = np.full((len(pixels), 3), np.nan, dtype=np.float64)
    for index, (u, v) in enumerate(pixels):
        column, row = int(round(float(u))), int(round(float(v)))
        x0, x1 = max(0, column - radius), min(width, column + radius + 1)
        y0, y1 = max(0, row - radius), min(height, row + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        patch = depth_m[y0:y1, x0:x1]
        valid = patch[
            np.isfinite(patch) & (patch >= minimum) & (patch <= maximum)]
        if valid.size == 0:
            continue
        z = float(np.median(valid))
        normalized = cv2.undistortPoints(
            np.asarray([[[u, v]]], dtype=np.float64),
            camera_matrix, distortion).reshape(2)
        points[index] = (normalized[0] * z, normalized[1] * z, z)
    return points


def _fit_plane_ransac(
        points, threshold_m, minimum_points, minimum_second_span_m):
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) < int(minimum_points):
        raise ValueError(
            "only {} valid marker depth points; require {}".format(
                len(values), minimum_points))
    rng = np.random.default_rng(180100)
    best = None
    iterations = max(96, len(values) * 8)
    for _ in range(iterations):
        selected = values[rng.choice(len(values), 3, replace=False)]
        normal = np.cross(selected[1] - selected[0], selected[2] - selected[0])
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        distance = np.abs((values - selected[0]) @ normal)
        inliers = distance <= float(threshold_m)
        score = (int(np.count_nonzero(inliers)),
                 -float(np.median(distance[inliers])))
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or np.count_nonzero(best[1]) < int(minimum_points):
        raise ValueError("marker depth plane RANSAC found too few inliers")
    inliers = best[1]
    selected = values[inliers]
    centroid = np.mean(selected, axis=0)
    _u, singular, vt = np.linalg.svd(selected - centroid)
    planar_spans = singular / math.sqrt(float(len(selected)))
    if len(planar_spans) < 2 or \
            planar_spans[1] < float(minimum_second_span_m):
        raise ValueError(
            "marker depth points lack two-dimensional plane span")
    normal = vt[-1]
    normal /= np.linalg.norm(normal)
    distance = np.abs((values - centroid) @ normal)
    inliers = distance <= float(threshold_m)
    if np.count_nonzero(inliers) < int(minimum_points):
        raise ValueError("refined marker depth plane found too few inliers")
    selected = values[inliers]
    centroid = np.mean(selected, axis=0)
    _u, _singular, vt = np.linalg.svd(selected - centroid)
    normal = vt[-1]
    normal /= np.linalg.norm(normal)
    if float(normal @ centroid) > 0.0:
        normal *= -1.0
    signed = (selected - centroid) @ normal
    return {
        "points_camera_m": selected,
        "centroid_camera_m": centroid,
        "normal_camera": normal,
        "rmse_m": float(np.sqrt(np.mean(np.square(signed)))),
        "mad_m": float(np.median(np.abs(signed - np.median(signed)))),
        "candidate_count": int(len(values)),
        "inlier_count": int(len(selected)),
        "principal_spans_m": planar_spans,
    }


class DualAruco18Capture:
    def __init__(self):
        rospy.set_param("~software_revision", SOFTWARE_REVISION)
        self._config = deepcopy(rospy.get_param("~config"))
        self._marker = self._config["marker"]
        self._robot = self._config["robot"]
        self._cameras = self._config["cameras"]
        self._capture_config = self._config.get("capture", {})
        self._require_joint_states = bool(rospy.get_param(
            "~require_joint_states_single", False))
        self._joint_state_topic = str(rospy.get_param(
            "~joint_state_topic", "/joint_states_single"))
        self._joint_state_timeout = float(rospy.get_param(
            "~joint_state_sync_tolerance_s", 0.10))
        self._joint_stationary_span = float(rospy.get_param(
            "~joint_stationary_span_rad_maximum", 0.002))
        self._joint_names = tuple(
            "joint{}".format(index) for index in range(1, 7))
        self._hybrid_config = self._config.get("hybrid_calibration", {})
        active_camera = str(rospy.get_param("~active_camera", "both")).strip()
        if active_camera == "both":
            self._active_cameras = CAMERAS
        elif active_camera in CAMERAS:
            self._active_cameras = (active_camera,)
        else:
            raise rospy.ROSInitException(
                "active_camera must be rs1, rs3, or both")
        required_cameras = rospy.get_param(
            "~required_cameras", list(CAMERAS))
        self._required_cameras = tuple(str(value) for value in required_cameras)
        if not self._required_cameras or \
                len(set(self._required_cameras)) != \
                len(self._required_cameras) or \
                not set(self._required_cameras).issubset(set(CAMERAS)):
            raise rospy.ROSInitException(
                "required_cameras must be a unique non-empty subset of rs1,rs3")

        self._frames_per_pose = int(
            self._capture_config.get("frames_per_pose", 50))
        self._capture_timeout = float(
            self._capture_config.get("capture_timeout_s", 15.0))
        self._tf_timeout = float(
            self._capture_config.get("tf_timeout_s", 0.10))
        self._minimum_edge = float(
            self._capture_config.get("minimum_marker_edge_px", 60.0))
        self._preferred_edge = float(
            self._capture_config.get("preferred_marker_edge_px", 80.0))
        self._minimum_margin = float(
            self._capture_config.get("image_margin_px", 60.0))
        self._preview_width = int(
            self._capture_config.get("preview_width_px", 640))
        self._preview_rate = float(
            self._capture_config.get("preview_rate_hz", 10.0))
        if self._preview_width <= 0 or self._preview_rate <= 0.0:
            raise rospy.ROSInitException(
                "preview_width_px and preview_rate_hz must be positive")
        self._preview_period = 1.0 / self._preview_rate
        self._last_preview_monotonic = None
        self._translation_limit = float(
            self._capture_config.get(
                "stationary_translation_span_m_maximum", 0.002))
        self._rotation_limit = float(
            self._capture_config.get(
                "stationary_rotation_span_deg_maximum", 0.20))
        self._minimum_optimization = int(self._capture_config.get(
            "minimum_optimization_poses_per_camera",
            self._capture_config.get("minimum_optimization_poses", 18)))
        self._minimum_validation = int(self._capture_config.get(
            "minimum_validation_poses_per_camera",
            self._capture_config.get("minimum_validation_poses", 0)))
        self._pose_diversity = self._config.get("pose_diversity")
        self._depth_settings = {
            "scale_to_m": float(
                self._hybrid_config.get("depth_scale_to_m", 0.001)),
            "minimum_depth_m": float(
                self._hybrid_config.get("minimum_depth_m", 0.15)),
            "maximum_depth_m": float(
                self._hybrid_config.get("maximum_depth_m", 2.0)),
            "grid_size": int(
                self._hybrid_config.get("depth_grid_size", 5)),
            "grid_inset_fraction": float(
                self._hybrid_config.get(
                    "depth_grid_inset_fraction", 0.14)),
            "patch_radius_px": int(
                self._hybrid_config.get("depth_patch_radius_px", 2)),
            "minimum_grid_points": int(
                self._hybrid_config.get(
                    "minimum_depth_grid_points", 12)),
            "minimum_frame_fraction": float(
                self._hybrid_config.get(
                    "minimum_depth_frame_fraction", 0.60)),
            "plane_outlier_threshold_m": float(
                self._hybrid_config.get(
                    "depth_plane_outlier_threshold_m", 0.008)),
            "plane_rmse_maximum_m": float(
                self._hybrid_config.get(
                    "depth_plane_rmse_maximum_m", 0.006)),
            "plane_minimum_second_span_m": float(
                self._hybrid_config.get(
                    "depth_plane_minimum_second_span_m", 0.012)),
        }
        self._validate_config()
        self._output_root = Path(
            self._capture_config.get(
                "output_root", "~/.ros/rekpiper/camera/dual_aruco18")
        ).expanduser()

        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, str(self._marker["dictionary"])))
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        self._bridge = CvBridge()
        self._condition = threading.Condition(threading.RLock())
        self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        self._capture_active = False
        self._capture_split = None
        self._capture_frames = []
        self._joint_history = deque(maxlen=2000)
        self._samples = []
        self._session_dir = None
        self._latest_valid = False
        self._latest_reason = "waiting_for_images"
        self._latest_quality = {}
        resume_dataset = str(
            rospy.get_param("~resume_dataset", "")).strip()
        if resume_dataset:
            self._restore_dataset(Path(resume_dataset).expanduser())

        self._status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self._annotated_pubs = {
            name: rospy.Publisher("~{}/annotated_image".format(name), Image, queue_size=1)
            for name in self._active_cameras
        }
        self._capture_pose_service = rospy.Service(
            "~capture_pose", Trigger, self._capture_optimization_pose)
        self._capture_validation_service = rospy.Service(
            "~capture_validation_pose", Trigger, self._capture_validation_pose)
        self._finish_service = rospy.Service(
            "~finish_session", Trigger, self._finish_session)
        if self._require_joint_states:
            self._joint_subscriber = rospy.Subscriber(
                self._joint_state_topic, JointState,
                self._joint_state_callback, queue_size=200,
                tcp_nodelay=True)

        subscribers = []
        for name in self._active_cameras:
            camera = self._cameras[name]
            subscribers.extend((
                message_filters.Subscriber(camera["image_topic"], Image, queue_size=2),
                message_filters.Subscriber(
                    camera["camera_info_topic"], CameraInfo, queue_size=2),
                message_filters.Subscriber(
                    camera["aligned_depth_topic"], Image, queue_size=2),
            ))
        self._sync = message_filters.ApproximateTimeSynchronizer(
            subscribers, queue_size=10,
            slop=float(self._capture_config.get("image_pair_slop_s", 0.08)),
            allow_headerless=False)
        self._sync.registerCallback(self._callback)
        self._publish_status("INITIALIZING")

    def _joint_state_callback(self, message):
        values = dict(zip(message.name, message.position))
        if any(name not in values for name in self._joint_names):
            return
        positions = np.asarray(
            [values[name] for name in self._joint_names], dtype=np.float64)
        if not np.all(np.isfinite(positions)):
            return
        stamp = message.header.stamp
        if stamp == rospy.Time():
            stamp = rospy.Time.now()
        with self._condition:
            self._joint_history.append((stamp.to_sec(), positions))

    def _joint_state_at(self, stamp):
        if not self._require_joint_states:
            return None
        target = stamp.to_sec()
        with self._condition:
            if not self._joint_history:
                raise ValueError("waiting for /joint_states_single")
            nearest_stamp, nearest = min(
                self._joint_history,
                key=lambda item: abs(item[0] - target))
        delta = abs(nearest_stamp - target)
        if delta > self._joint_state_timeout:
            raise ValueError(
                "/joint_states_single sync {:.3f}s > {:.3f}s".format(
                    delta, self._joint_state_timeout))
        return nearest.copy()

    def _restore_dataset(self, path):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise rospy.ROSInitException(
                "cannot resume dataset {}: {}".format(path, exc))
        if not isinstance(payload, dict) or payload.get("schema_version") != 2:
            raise rospy.ROSInitException(
                "hybrid RGB-D resume dataset must use schema_version 2")
        if payload.get("complete"):
            raise rospy.ROSInitException(
                "refusing to resume an already complete dataset")
        saved_marker = payload.get("marker", {})
        if str(saved_marker.get("dictionary")) != str(self._marker["dictionary"]) or \
                int(saved_marker.get("id", -1)) != int(self._marker["id"]) or \
                abs(float(saved_marker.get("marker_side_m", -1.0))
                    - float(self._marker["marker_side_m"])) > 1e-9:
            raise rospy.ROSInitException(
                "resume dataset marker does not match current configuration")
        if payload.get("robot") != self._robot:
            raise rospy.ROSInitException(
                "resume dataset robot frames do not match current configuration")
        if payload.get("hybrid_calibration") != self._hybrid_config:
            raise rospy.ROSInitException(
                "resume dataset hybrid calibration settings do not match")
        if payload.get("pose_diversity") != self._pose_diversity:
            raise rospy.ROSInitException(
                "resume dataset pose diversity settings do not match")
        if tuple(payload.get("required_cameras", CAMERAS)) != \
                self._required_cameras:
            raise rospy.ROSInitException(
                "resume dataset required_cameras do not match")
        saved_cameras = payload.get("cameras", {})
        for name in CAMERAS:
            saved = saved_cameras.get(name, {})
            current = self._cameras[name]
            for key in ("serial", "image_topic", "camera_info_topic",
                        "aligned_depth_topic",
                        "optical_frame", "link_frame"):
                if str(saved.get(key)) != str(current.get(key)):
                    raise rospy.ROSInitException(
                        "resume dataset {} {} does not match".format(name, key))
            for key in ("camera_info", "link_T_optical"):
                if key in saved:
                    current[key] = deepcopy(saved[key])
        samples = payload.get("samples", [])
        if not isinstance(samples, list) or any(
                item.get("split") not in ("optimization", "validation")
                for item in samples if isinstance(item, dict)):
            raise rospy.ROSInitException("resume dataset samples are invalid")
        if any(not isinstance(item, dict) for item in samples):
            raise rospy.ROSInitException("resume dataset samples are invalid")
        self._samples = deepcopy(samples)
        self._session_dir = path.parent
        rospy.set_param("~dataset_path", str(path))
        rospy.loginfo(
            "Resumed %d poses from %s", len(self._samples), path)

    def _validate_config(self):
        if set(self._cameras) != set(CAMERAS):
            raise rospy.ROSInitException("config requires exactly rs1 and rs3")
        if not isinstance(self._pose_diversity, dict):
            raise rospy.ROSInitException("pose_diversity settings are required")
        for key in (
                "optimization_translation_axis_span_m",
                "optimization_pair_rotation_deg",
                "validation_translation_axis_span_m",
                "validation_pair_rotation_deg"):
            if key not in self._pose_diversity:
                raise rospy.ROSInitException(
                    "pose_diversity/{} is required".format(key))
        if self._robot.get("base_frame") != "base_link" or \
                self._robot.get("effector_frame") != "link6":
            raise rospy.ROSInitException("robot frames must be base_link and link6")
        if str(self._marker.get("dictionary")) != "DICT_4X4_50" or \
                int(self._marker.get("id", -1)) != 18 or \
                abs(float(self._marker.get("marker_side_m", 0.0)) - 0.100) > 1e-9:
            raise rospy.ROSInitException(
                "this workflow is fixed to DICT_4X4_50 / ID 18 / 0.100 m")
        dictionary = str(self._marker.get("dictionary", ""))
        if not hasattr(cv2.aruco, dictionary):
            raise rospy.ROSInitException("unsupported ArUco dictionary: " + dictionary)
        for name in CAMERAS:
            for key in ("serial", "image_topic", "camera_info_topic",
                        "aligned_depth_topic",
                        "optical_frame", "link_frame"):
                if not self._cameras[name].get(key):
                    raise rospy.ROSInitException(
                        "{} {} is required".format(name, key))
        depth = self._depth_settings
        if depth["scale_to_m"] <= 0.0 or \
                not 0.0 < depth["minimum_depth_m"] < depth["maximum_depth_m"] or \
                depth["grid_size"] < 3 or \
                not 0.0 < depth["grid_inset_fraction"] < 0.5 or \
                depth["patch_radius_px"] < 0 or \
                depth["minimum_grid_points"] < 3 or \
                depth["minimum_grid_points"] > depth["grid_size"] ** 2 or \
                not 0.0 < depth["minimum_frame_fraction"] <= 1.0 or \
                depth["plane_outlier_threshold_m"] <= 0.0 or \
                depth["plane_rmse_maximum_m"] <= 0.0 or \
                depth["plane_minimum_second_span_m"] <= 0.0:
            raise rospy.ROSInitException(
                "invalid hybrid_calibration depth settings")

    def _driver_serial(self, name):
        for parameter in (
                "/{}/realsense2_camera/serial_no".format(name),
                "/{}/serial_no".format(name)):
            if rospy.has_param(parameter):
                return str(rospy.get_param(parameter)).lstrip("_")
        return None

    def _visual(self, name, image_message, info_message, depth_message):
        image = self._bridge.imgmsg_to_cv2(image_message, "bgr8")
        preview = image.copy()
        camera = self._cameras[name]
        if image_message.header.frame_id != camera["optical_frame"] or \
                info_message.header.frame_id != camera["optical_frame"] or \
                depth_message.header.frame_id != camera["optical_frame"]:
            return None, preview, "{} optical frame mismatch".format(name)
        if (image_message.width, image_message.height) != \
                (info_message.width, info_message.height):
            return None, preview, "{} image/CameraInfo size mismatch".format(name)
        if self._driver_serial(name) != str(camera["serial"]).lstrip("_"):
            return None, preview, "{} serial mismatch or missing".format(name)
        depth_raw = self._bridge.imgmsg_to_cv2(
            depth_message, desired_encoding="passthrough")
        if depth_raw.shape != image.shape[:2]:
            return None, preview, (
                "{} aligned depth shape {} != RGB shape {}"
            ).format(name, depth_raw.shape, image.shape[:2])
        depth_m = _depth_metres(
            depth_raw, depth_message.encoding,
            self._depth_settings["scale_to_m"])

        corners, ids, _rejected = self._detector.detectMarkers(image)
        detected = [] if ids is None else ids.reshape(-1).tolist()
        marker_id = int(self._marker["id"])
        if detected.count(marker_id) != 1:
            return None, preview, "{} needs one ID {}; detected {}".format(
                name, marker_id, detected)
        points = np.asarray(
            corners[detected.index(marker_id)], dtype=float).reshape(4, 2)
        edge, margin = _edge_and_margin(points, image.shape)
        accepted = edge >= self._minimum_edge and margin >= self._minimum_margin
        cv2.polylines(
            preview, [points.astype(np.int32)], True,
            (0, 190, 0) if accepted else (0, 0, 220), 3)
        if edge < self._minimum_edge:
            return None, preview, "{} marker {:.1f}px < {:.1f}px".format(
                name, edge, self._minimum_edge)
        if margin < self._minimum_margin:
            return None, preview, "{} margin {:.1f}px < {:.1f}px".format(
                name, margin, self._minimum_margin)
        info = _camera_info(info_message)
        depth_points = _sample_marker_depth(
            points, depth_m, info, self._depth_settings)
        valid_depth = int(np.count_nonzero(
            np.all(np.isfinite(depth_points), axis=1)))
        if valid_depth < self._depth_settings["minimum_grid_points"]:
            return None, preview, (
                "{} marker depth grid {}/{} < {}"
            ).format(
                name, valid_depth, len(depth_points),
                self._depth_settings["minimum_grid_points"])
        observation = {
            "stamp": image_message.header.stamp,
            "depth_stamp": depth_message.header.stamp,
            "corners_px": points,
            "edge_px": edge,
            "margin_px": margin,
            "camera_info": info,
            "depth_grid_points_camera_m": depth_points,
            "valid_depth_grid_points": valid_depth,
        }
        return observation, preview, None

    def _lookup(self, target, source, stamp):
        return transform_to_matrix(self._tf_buffer.lookup_transform(
            target, source, stamp, rospy.Duration(self._tf_timeout)).transform)

    def _publish_preview(self, name, preview, header, accepted, text):
        if preview.shape[1] != self._preview_width:
            scale = float(self._preview_width) / float(preview.shape[1])
            preview = cv2.resize(
                preview,
                (self._preview_width,
                 max(1, int(round(preview.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        color = (0, 190, 0) if accepted else (0, 0, 220)
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(
            preview, ("READY: " if accepted else "WAITING: ") + text[:115],
            (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)
        try:
            message = self._bridge.cv2_to_imgmsg(preview, "bgr8")
            message.header = header
            self._annotated_pubs[name].publish(message)
        except CvBridgeError:
            pass

    def _publish_status(self, state, reason=None, extra=None):
        payload = {
            "software_revision": SOFTWARE_REVISION,
            "active_cameras": list(self._active_cameras),
            "required_cameras": list(self._required_cameras),
            "state": state,
            "reason": reason,
            "capture_active": self._capture_active,
            "capture_split": self._capture_split,
            "captured_optimization": sum(
                item["split"] == "optimization" for item in self._samples),
            "captured_validation": sum(
                item["split"] == "validation" for item in self._samples),
            "captured_optimization_by_camera": {
                name: sum(
                    item["split"] == "optimization" and
                    name in item.get("cameras", {}) for item in self._samples)
                for name in CAMERAS
            },
            "captured_validation_by_camera": {
                name: sum(
                    item["split"] == "validation" and
                    name in item.get("cameras", {}) for item in self._samples)
                for name in CAMERAS
            },
            "required_frames_per_pose": self._frames_per_pose,
            "quality": self._latest_quality,
        }
        payload.update(extra or {})
        self._status_pub.publish(
            String(data=json.dumps(_plain(payload), sort_keys=True)))

    def _callback(self, *args):
        messages = {
            name: (args[index * 3], args[index * 3 + 1],
                   args[index * 3 + 2])
            for index, name in enumerate(self._active_cameras)
        }
        observations, previews, reasons = {}, {}, {}
        for name in self._active_cameras:
            try:
                observation, preview, reason = self._visual(name, *messages[name])
            except (CvBridgeError, ValueError, cv2.error) as exc:
                observation = None
                preview = np.zeros((180, 320, 3), dtype=np.uint8)
                reason = "{} image conversion/detection: {}".format(name, exc)
            observations[name], previews[name], reasons[name] = \
                observation, preview, reason

        pair_reason = next(
            (reasons[name] for name in self._active_cameras if reasons[name]), None)
        base_T_link6 = None
        joint_positions = None
        if pair_reason is None:
            try:
                stamps = [
                    observations[name]["stamp"] for name in self._active_cameras]
                midpoint = rospy.Time.from_sec(
                    sum(stamp.to_sec() for stamp in stamps) / len(stamps))
                base_T_link6 = self._lookup(
                    self._robot["base_frame"], self._robot["effector_frame"], midpoint)
                joint_positions = self._joint_state_at(midpoint)
                for name in self._active_cameras:
                    camera = self._cameras[name]
                    observations[name]["link_T_optical"] = self._lookup(
                        camera["link_frame"], camera["optical_frame"],
                        observations[name]["stamp"])
            except (tf2_ros.TransformException, ValueError) as exc:
                pair_reason = "TF/joint feedback: " + str(exc)

        pair_valid = pair_reason is None
        pair_delta_ms = None
        if len(self._active_cameras) == 2 and all(observations.values()):
            pair_delta_ms = abs(
                observations["rs1"]["stamp"].to_sec()
                - observations["rs3"]["stamp"].to_sec()) * 1000.0
        quality = {
            name: ({
                "edge_px": observations[name]["edge_px"],
                "margin_px": observations[name]["margin_px"],
                "valid_depth_grid_points":
                    observations[name]["valid_depth_grid_points"],
                "rgb_depth_delta_ms": abs(
                    observations[name]["stamp"].to_sec()
                    - observations[name]["depth_stamp"].to_sec()) * 1000.0,
            } if observations[name] else {"reason": reasons[name]})
            for name in self._active_cameras
        }
        if pair_delta_ms is not None:
            quality["pair_delta_ms"] = pair_delta_ms

        with self._condition:
            self._latest_valid = pair_valid
            self._latest_reason = pair_reason
            self._latest_quality = quality
            if self._capture_active and pair_valid and \
                    len(self._capture_frames) < self._frames_per_pose:
                self._capture_frames.append({
                    "base_T_link6": base_T_link6.copy(),
                    "joint_positions_rad": (
                        None if joint_positions is None
                        else joint_positions.copy()),
                    "observations": {
                        name: {
                            "corners_px": observations[name]["corners_px"].copy(),
                            "edge_px": observations[name]["edge_px"],
                            "margin_px": observations[name]["margin_px"],
                            "camera_info": observations[name]["camera_info"],
                            "depth_grid_points_camera_m":
                                observations[name][
                                    "depth_grid_points_camera_m"].copy(),
                            "link_T_optical":
                                observations[name]["link_T_optical"].copy(),
                        } for name in self._active_cameras
                    },
                })
                self._condition.notify_all()
            elif self._capture_active:
                self._condition.notify_all()
            count = len(self._capture_frames) if self._capture_active else 0
            split = self._capture_split

        now = time.monotonic()
        publish_preview = self._last_preview_monotonic is None or \
            now - self._last_preview_monotonic >= self._preview_period
        if publish_preview:
            self._last_preview_monotonic = now
            for name in self._active_cameras:
                if pair_valid:
                    item = observations[name]
                    text = "ID {} edge {:.0f}px margin {:.0f}px".format(
                        self._marker["id"], item["edge_px"], item["margin_px"])
                    text += " depth {}/{}".format(
                        item["valid_depth_grid_points"],
                        self._depth_settings["grid_size"] ** 2)
                    if pair_delta_ms is not None:
                        text += " dt {:.1f}ms".format(pair_delta_ms)
                    if item["edge_px"] < self._preferred_edge:
                        text += " | prefer >= {:.0f}px".format(
                            self._preferred_edge)
                    if self._capture_active:
                        text += " | {} {}/{}".format(
                            split, count, self._frames_per_pose)
                elif reasons[name]:
                    # Show each camera's own detection failure.  Previously the
                    # first pair failure (often rs1) was repeated in both views.
                    text = reasons[name]
                elif observations[name]:
                    item = observations[name]
                    text = (
                        "{} local OK edge {:.0f}px margin {:.0f}px; pair: {}"
                    ).format(
                        name, item["edge_px"], item["margin_px"],
                        pair_reason or "waiting")
                else:
                    text = pair_reason or "waiting for valid pair"
                self._publish_preview(
                    name, previews[name], messages[name][0].header,
                    pair_valid, text)
        self._publish_status(
            "CAPTURING" if self._capture_active else
            ("READY_TO_CAPTURE" if pair_valid else "WAITING_FOR_VALID_INPUT"),
            pair_reason, {"valid_frames": count})

    def _session(self):
        if self._session_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            self._session_dir = self._output_root / stamp
            self._session_dir.mkdir(parents=True, exist_ok=False)
        return self._session_dir

    def _write_dataset(self, complete=False):
        payload = {
            "schema_version": 2,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "complete": bool(complete),
            "marker": self._marker,
            "robot": self._robot,
            "cameras": self._cameras,
            "samples": self._samples,
            "read_only_capture": True,
            "observation_model":
                "rgb_corners_plus_aligned_depth_marker_plane",
            "hybrid_calibration": self._hybrid_config,
            "pose_diversity": self._pose_diversity,
            "required_cameras": list(self._required_cameras),
            "capture_mode": (
                "separate_single_camera" if any(
                    len(item.get("cameras", {})) == 1 for item in self._samples)
                else "simultaneous"),
        }
        path = self._session() / "dataset.yaml"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            yaml.safe_dump(_plain(payload), sort_keys=False), encoding="utf-8")
        temporary.replace(path)
        rospy.set_param("~dataset_path", str(path))
        return path

    def _stationary_average(self, frames):
        mean = average_transforms(
            [frame["base_T_link6"] for frame in frames])
        spans = [
            transform_deviation(mean, frame["base_T_link6"])
            for frame in frames
        ]
        translation = max(item["translation_m"] for item in spans)
        rotation = max(item["rotation_deg"] for item in spans)
        if translation > self._translation_limit:
            raise ValueError(
                "robot moved during capture: {:.6f}m > {:.6f}m".format(
                    translation, self._translation_limit))
        if rotation > self._rotation_limit:
            raise ValueError(
                "robot rotated during capture: {:.4f}deg > {:.4f}deg".format(
                    rotation, self._rotation_limit))
        return mean

    def _stationary_joint_state(self, frames):
        if not self._require_joint_states:
            return None
        values = np.asarray(
            [frame["joint_positions_rad"] for frame in frames],
            dtype=np.float64)
        if values.shape != (len(frames), 6) or \
                not np.all(np.isfinite(values)):
            raise ValueError(
                "invalid /joint_states_single samples during capture")
        span = float(np.max(np.ptp(values, axis=0)))
        if span > self._joint_stationary_span:
            raise ValueError(
                "/joint_states_single moved during capture: "
                "{:.6f}rad > {:.6f}rad".format(
                    span, self._joint_stationary_span))
        return {
            "topic": self._joint_state_topic,
            "names": list(self._joint_names),
            "positions_rad": np.median(values, axis=0),
            "sample_count": int(len(values)),
            "maximum_span_rad": span,
        }

    def _capture(self, split):
        with self._condition:
            if self._capture_active:
                return TriggerResponse(False, "capture already active")
            if not self._latest_valid:
                return TriggerResponse(
                    False, "not READY: " + str(self._latest_reason))
            self._capture_active = True
            self._capture_split = split
            self._capture_frames = []
            deadline = time.monotonic() + self._capture_timeout
            while len(self._capture_frames) < self._frames_per_pose and \
                    time.monotonic() < deadline and not rospy.is_shutdown():
                self._condition.wait(timeout=0.2)
            frames = list(self._capture_frames)
            self._capture_active = False
            self._capture_split = None
            self._capture_frames = []
        if len(frames) < self._frames_per_pose:
            reason = "capture timeout: {}/{} valid frames; last={}".format(
                len(frames), self._frames_per_pose, self._latest_reason)
            self._publish_status("CAPTURE_REJECTED", reason)
            return TriggerResponse(False, reason)
        try:
            base_T_link6 = self._stationary_average(frames)
            joint_state = self._stationary_joint_state(frames)
            camera_samples = {}
            for name in self._active_cameras:
                values = [frame["observations"][name] for frame in frames]
                median_corners = np.median(
                    [value["corners_px"] for value in values], axis=0)
                corner_delta = np.asarray(
                    [value["corners_px"] for value in values],
                    dtype=np.float64) - median_corners
                corner_sigma = 1.4826 * float(
                    np.median(np.abs(corner_delta)))
                depth_batches = np.asarray(
                    [value["depth_grid_points_camera_m"] for value in values],
                    dtype=np.float64)
                combined_depth = []
                required_frames = max(
                    1, int(np.ceil(
                        len(values) *
                        self._depth_settings["minimum_frame_fraction"])))
                for grid_index in range(depth_batches.shape[1]):
                    candidates = depth_batches[:, grid_index, :]
                    candidates = candidates[
                        np.all(np.isfinite(candidates), axis=1)]
                    if len(candidates) >= required_frames:
                        combined_depth.append(
                            np.median(candidates, axis=0))
                plane = _fit_plane_ransac(
                    combined_depth,
                    self._depth_settings["plane_outlier_threshold_m"],
                    self._depth_settings["minimum_grid_points"],
                    self._depth_settings[
                        "plane_minimum_second_span_m"])
                if plane["rmse_m"] > \
                        self._depth_settings["plane_rmse_maximum_m"]:
                    raise ValueError(
                        "{} marker depth plane RMSE {:.6f}m > {:.6f}m".format(
                            name, plane["rmse_m"],
                            self._depth_settings[
                                "plane_rmse_maximum_m"]))
                camera_samples[name] = {
                    "corners_px": median_corners,
                    "rgb_corner_sigma_px": corner_sigma,
                    "minimum_edge_px": float(np.median(
                        [value["edge_px"] for value in values])),
                    "minimum_margin_px": float(np.median(
                        [value["margin_px"] for value in values])),
                    "depth_plane": {
                        "points_camera_m": plane["points_camera_m"],
                        "centroid_camera_m": plane["centroid_camera_m"],
                        "normal_camera": plane["normal_camera"],
                        "fit_rmse_m": plane["rmse_m"],
                        "fit_mad_m": plane["mad_m"],
                        "candidate_count": plane["candidate_count"],
                        "inlier_count": plane["inlier_count"],
                        "principal_spans_m":
                            plane["principal_spans_m"],
                    },
                }
                if "camera_info" not in self._cameras[name]:
                    self._cameras[name]["camera_info"] = values[-1]["camera_info"]
                    self._cameras[name]["link_T_optical"] = \
                        values[-1]["link_T_optical"]
            sample = {
                "id": len(self._samples) + 1,
                "split": split,
                "base_T_link6": base_T_link6,
                "cameras": camera_samples,
            }
            if joint_state is not None:
                sample["joint_states_single"] = joint_state
            self._samples.append(sample)
            try:
                path = self._write_dataset(complete=False)
            except (OSError, ValueError, yaml.YAMLError):
                self._samples.pop()
                raise
        except (OSError, ValueError, yaml.YAMLError) as exc:
            reason = str(exc)
            self._publish_status("CAPTURE_REJECTED", reason)
            return TriggerResponse(False, reason)
        message = "accepted {} pose {}; dataset={}".format(
            split, sample["id"], path)
        self._publish_status(
            "POSE_ACCEPTED", extra={"sample_id": sample["id"], "dataset": str(path)})
        return TriggerResponse(True, message)

    def _capture_optimization_pose(self, _request):
        return self._safe_trigger(
            "capture optimization pose",
            lambda: self._capture("optimization"))

    def _capture_validation_pose(self, _request):
        return self._safe_trigger(
            "capture validation pose",
            lambda: self._capture("validation"))

    def _finish_session(self, _request):
        return self._safe_trigger("finish session", self._finish_session_impl)

    def _finish_session_impl(self):
        optimization = {
            name: sum(item["split"] == "optimization" and
                      name in item.get("cameras", {}) for item in self._samples)
            for name in CAMERAS
        }
        validation = {
            name: sum(item["split"] == "validation" and
                      name in item.get("cameras", {}) for item in self._samples)
            for name in CAMERAS
        }
        if any(optimization[name] < self._minimum_optimization or
               validation[name] < self._minimum_validation
               for name in self._required_cameras):
            return TriggerResponse(
                False,
                "requires {}/{} optimization/validation poses for {}; "
                "got rs1={}/{}, rs3={}/{}".format(
                    self._minimum_optimization, self._minimum_validation,
                    ",".join(self._required_cameras),
                    optimization["rs1"], validation["rs1"],
                    optimization["rs3"], validation["rs3"]))
        for name in self._required_cameras:
            for split, minimum in (
                    ("optimization", self._minimum_optimization),
                    ("validation", self._minimum_validation)):
                poses = [item["base_T_link6"] for item in self._samples
                         if item["split"] == split
                         and name in item.get("cameras", {})]
                prefix = split + "_"
                try:
                    check_pose_diversity(
                        poses,
                        self._pose_diversity[
                            prefix + "translation_axis_span_m"],
                        self._pose_diversity[prefix + "pair_rotation_deg"],
                        minimum)
                except ValueError as exc:
                    return TriggerResponse(
                        False, "{} {}: {}".format(name, split, exc))
        path = self._write_dataset(complete=True)
        self._publish_status(
            "SESSION_COMPLETE", extra={"dataset": str(path)})
        return TriggerResponse(True, str(path))

    def _safe_trigger(self, label, callback):
        try:
            return callback()
        except Exception as exc:  # keep ROS service failures visible to operators
            with self._condition:
                self._capture_active = False
                self._capture_split = None
                self._capture_frames = []
            detail = "{}: {}: {}".format(label, type(exc).__name__, exc)
            rospy.logerr("%s\n%s", detail, traceback.format_exc())
            self._publish_status("INTERNAL_ERROR", detail)
            return TriggerResponse(False, detail)


if __name__ == "__main__":
    rospy.init_node("dual_aruco18_capture")
    DualAruco18Capture()
    rospy.spin()
