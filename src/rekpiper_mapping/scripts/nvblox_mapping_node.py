#!/usr/bin/env python3
"""Fuse safety-masked D435 depth into nvblox and serve observed SDF queries."""

import json
import hashlib
from functools import partial
from pathlib import Path
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET

from cv_bridge import CvBridge, CvBridgeError
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import message_filters
import numpy as np
import rospy
import rospkg
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String
import tf2_ros
import torch
import yaml

from rekpiper_acceptance import (
    AcceptanceError, assert_release_unchanged, validate_release_bundle)
from rekpiper_camera.projection import transform_to_matrix
from rekpiper_mapping.frame_preparation import (
    normalize_camera_sources, normalize_sdf_query, prepare_static_depth)
from rekpiper_mapping.gripper_obstacle_range import project_tsdf_surface_range
from rekpiper_mapping.visualization import (
    colorize_esdf_slice, colorize_local_esdf_contours, colorize_static_depth)
from rekpiper_mapping.sdf_conventions import inclusive_grid_axes, sanitize_nvblox_sdf
from rekpiper_mapping.safe_map_state import SafeMapState, SafeMapSupervisor
from rekpiper_msgs.msg import SDFGrid, SafeMappingStatus, TrackedObjectArray
from rekpiper_msgs.srv import (CaptureSDFSnapshot, CaptureSDFSnapshotResponse,
                            QuerySDF, QuerySDFResponse, RebuildMap,
                            RebuildMapResponse, RebuildMapRequest)

CAPTURE_TIMEOUT_S = 10.0
CAPTURE_FRAMES_PER_CAMERA = 5


class NvbloxMappingNode:
    def __init__(self):
        self._bridge = CvBridge()
        self._lock = threading.RLock()
        self._input_lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._pending_frames = {}
        self._capture_lock = threading.Lock()
        self._capture_updated = threading.Event()
        self._capture_active = False
        self._snapshot_ready = False
        self._capture_after_stamp = None
        self._mask_lock = threading.Lock()
        self._visualization_state_lock = threading.Lock()
        self._target_frame = rospy.get_param("~target_frame", "base_link")
        self._camera_frame = rospy.get_param("~camera_frame", "camera_color_optical_frame")
        default_camera_source = {
            "name": "camera",
            "camera_frame": self._camera_frame,
            "depth_topic": rospy.get_param(
                "~depth_topic", "/camera/aligned_depth_to_color/image_raw"),
            "camera_info_topic": rospy.get_param(
                "~camera_info_topic", "/camera/color/camera_info"),
            "robot_mask_topic": rospy.get_param(
                "~robot_mask_topic", "/rekpiper/mapping/robot_mask"),
            "dynamic_mask_topic": rospy.get_param(
                "~dynamic_mask_topic", "/rekpiper/mapping/dynamic_object_mask"),
        }
        try:
            self._camera_sources = normalize_camera_sources(
                rospy.get_param("~cameras", None), default_camera_source)
        except ValueError as exc:
            raise rospy.ROSInitException(str(exc))
        self._camera_source_names = [source["name"] for source in self._camera_sources]
        self._render_camera_name = str(rospy.get_param(
            "~render_camera_name", self._camera_source_names[0]))
        if self._render_camera_name not in self._camera_source_names:
            raise rospy.ROSInitException("render_camera_name is not a configured camera")
        self._bounds_min = self._vector("~workspace_bounds_min")
        self._bounds_max = self._vector("~workspace_bounds_max")
        if np.any(self._bounds_min >= self._bounds_max):
            raise rospy.ROSInitException("workspace bounds must be ordered")
        self._voxel_size_m = float(rospy.get_param("~voxel_size_m", 0.015))
        self._depth_scale = float(rospy.get_param("~depth_scale", 0.001))
        self._min_depth_m = float(rospy.get_param("~min_depth_m", 0.10))
        self._max_depth_m = float(rospy.get_param("~max_depth_m", 2.00))
        self._depth_period_s = 1.0 / self._positive_param("~depth_rate_hz", 10.0)
        self._esdf_period_s = 1.0 / self._positive_param("~esdf_rate_hz", 2.0)
        self._max_map_age_s = self._positive_param("~max_map_age_s", 1.0)
        self._tf_timeout_s = self._positive_param("~tf_timeout_s", 0.10)
        self._mask_max_age_s = self._positive_param("~mask_max_age_s", 0.15)
        self._mask_sync_slop_s = self._positive_param("~mask_sync_slop_s", 0.01)
        self._mask_stamp_tolerance_s = self._positive_param(
            "~mask_stamp_tolerance_s", 0.002)
        self._min_retained_fraction = float(rospy.get_param("~min_retained_fraction", 0.05))
        self._mask_dilation_px = int(rospy.get_param("~mask_dilation_px", 3))
        self._max_queries = int(rospy.get_param("~max_queries", 4096))
        self._unknown_occupied_distance_m = self._positive_param(
            "~unknown_occupied_distance_m", 1.0)
        self._esdf_slice_pixels = int(rospy.get_param("~esdf_slice_pixels", 192))
        self._esdf_slice_z_m = float(rospy.get_param("~esdf_slice_z_m", 0.10))
        self._follow_esdf_slice_anchor = bool(rospy.get_param(
            "~follow_esdf_slice_anchor", False))
        self._esdf_slice_anchor_label = str(rospy.get_param(
            "~esdf_slice_anchor_label", "gripper_base"))
        self._esdf_slice_anchor_max_age_s = self._positive_param(
            "~esdf_slice_anchor_max_age_s", 0.25)
        self._esdf_slice_truncation_m = self._positive_param(
            "~esdf_slice_truncation_m", 0.15)
        self._tsdf_render_width = int(rospy.get_param("~tsdf_render_width", 320))
        self._tsdf_render_height = int(rospy.get_param("~tsdf_render_height", 240))
        self._tsdf_visualization_period_s = 1.0 / self._positive_param(
            "~tsdf_visualization_rate_hz", 1.0)
        self._local_map_pixels = int(rospy.get_param("~local_map_pixels", 256))
        self._local_map_half_extent_m = self._positive_param(
            "~local_map_half_extent_m", 0.30)
        self._obstacle_range_period_s = 1.0 / self._positive_param(
            "~obstacle_range_rate_hz", 1.0)
        self._require_robot_mask = bool(rospy.get_param("~require_robot_mask", True))
        self._require_dynamic_mask = bool(rospy.get_param("~require_dynamic_mask", True))
        self._publish_raw_tsdf_diagnostic = bool(rospy.get_param(
            "~publish_raw_tsdf_diagnostic", False))
        self._planning_safe_inputs = self._require_robot_mask and self._require_dynamic_mask
        self._safe_supervision_enabled = bool(rospy.get_param(
            "~safe_supervision_enabled", False))
        self._observation_only = bool(rospy.get_param("~observation_only", False))
        self._stepwise_capture = bool(rospy.get_param("~stepwise_capture", False))
        self._planning_safe_inputs &= self._safe_supervision_enabled and not self._observation_only
        self._instance_uuid = str(uuid.uuid4())
        self._safe_supervisor = SafeMapSupervisor(
            camera_names=tuple(self._camera_source_names),
            warmup_frames_per_camera=int(rospy.get_param("~warmup_frames_per_camera", 30)),
            build_frames_per_camera=int(rospy.get_param("~build_frames_per_camera", 150)),
            minimum_esdf_updates=int(rospy.get_param("~minimum_esdf_updates", 10)))
        self._global_object_tracking_safe = False if self._safe_supervision_enabled else True
        self._exclusion_signature = ""
        self._camera_info_snapshots = {}
        self._attached_geometry_path = str(rospy.get_param(
            "~attached_collision_geometry", ""))
        self._precision_extrinsics_files = list(rospy.get_param(
            "~precision_extrinsics_files", []))
        self._acceptance_manifest_path = str(rospy.get_param(
            "~safe_acceptance_manifest", ""))
        self._mode = str(rospy.get_param("~mode", "shadow")).strip().lower()
        if self._observation_only and self._mode != "shadow":
            raise rospy.ROSInitException("observation_only requires shadow mode")
        if self._stepwise_capture and not (
                self._observation_only and self._mode == "shadow"
                and not self._safe_supervision_enabled):
            raise rospy.ROSInitException("stepwise_capture requires observation-only shadow mode")
        self._release_bundle_path = str(rospy.get_param("~release_bundle", ""))
        self._release = None
        self._signed_safe_map = None
        if self._mode == "autonomous":
            try:
                self._release = validate_release_bundle(
                    self._release_bundle_path,
                    str(rospy.get_param("~acceptance_public_key", "")),
                    str(rospy.get_param("~minimum_release_counter", "")),
                    expected_robot_id=str(rospy.get_param(
                        "~robot_id", "piper-rekpiper")))
                self._signed_safe_map = self._release[
                    "verified_artifacts"]["safe_map"]["payload"]
                workspace_artifact = self._release[
                    "verified_artifacts"]["workspace"]
                workspace = workspace_artifact["payload"]
                if self._signed_safe_map.get("workspace_payload_sha256") != \
                        workspace_artifact["payload_sha256"]:
                    raise ValueError("safe-map and workspace acceptance differ")
                self._bounds_min = np.asarray(
                    workspace["workspace_bounds_min"], dtype=float)
                self._bounds_max = np.asarray(
                    workspace["workspace_bounds_max"], dtype=float)
                if (self._bounds_min.shape != (3,)
                        or self._bounds_max.shape != (3,)
                        or np.any(self._bounds_min >= self._bounds_max)):
                    raise ValueError("signed workspace bounds are invalid")
            except (AcceptanceError, KeyError, TypeError, ValueError) as exc:
                raise rospy.ROSInitException(
                    "signed safe-map acceptance rejected: {}".format(exc))
        # Optional startup rebuilding still observes every live mask,
        # calibration, freshness and acceptance gate.
        self._auto_rebuild_on_startup = bool(rospy.get_param(
            "~auto_rebuild_on_startup", False))
        self._startup_rebuild_inflight = False
        self._startup_rebuild_completed = False
        self._built_configuration_files = None
        self._built_robot_description_sha = ""
        self._built_collision_mesh_stats = None
        if self._voxel_size_m <= 0.0 or not 0.0 <= self._min_retained_fraction <= 1.0:
            raise rospy.ROSInitException("invalid voxel size or retained-depth fraction")
        if self._max_queries <= 0 or self._mask_dilation_px < 0:
            raise rospy.ROSInitException("query limit must be positive and mask dilation nonnegative")
        if (not 32 <= self._esdf_slice_pixels <= 512
                or not np.isfinite(self._esdf_slice_z_m)
                or not self._bounds_min[2] <= self._esdf_slice_z_m <= self._bounds_max[2]):
            raise rospy.ROSInitException("ESDF slice pixels/z are outside configured limits")
        if (not 64 <= self._tsdf_render_width <= 1280
                or not 48 <= self._tsdf_render_height <= 720):
            raise rospy.ROSInitException("TSDF render dimensions are outside configured limits")
        if not 64 <= self._local_map_pixels <= 512:
            raise rospy.ROSInitException("local map pixels are outside configured limits")
        if not torch.cuda.is_available():
            raise rospy.ROSInitException("CUDA is required by the pinned nvblox_torch runtime")

        nvblox_root = Path(rospy.get_param("~nvblox_torch_python_root")).expanduser().resolve()
        if not nvblox_root.is_dir():
            raise rospy.ROSInitException("nvblox_torch Python root does not exist: {}".format(nvblox_root))
        sys.path.insert(0, str(nvblox_root))
        try:
            from nvblox_torch.mapper import Mapper
        except (ImportError, OSError) as exc:
            raise rospy.ROSInitException("cannot load nvblox_torch: {}".format(exc))

        self._device = torch.device("cuda:0")
        self._mapper = Mapper(
            voxel_sizes=[self._voxel_size_m], integrator_types=["tsdf"],
            free_on_destruction=True,
        )
        # This second map is diagnostic-only and is never reachable from the
        # SDF query service or the planning-safe status.
        self._raw_diagnostic_mapper = (
            Mapper(
                voxel_sizes=[self._voxel_size_m],
                integrator_types=["tsdf"],
                free_on_destruction=True,
            )
            if self._publish_raw_tsdf_diagnostic else None)
        slice_x = np.linspace(
            self._bounds_min[0], self._bounds_max[0], self._esdf_slice_pixels,
            dtype=np.float32)
        slice_y = np.linspace(
            self._bounds_max[1], self._bounds_min[1], self._esdf_slice_pixels,
            dtype=np.float32)
        grid_x, grid_y = np.meshgrid(slice_x, slice_y)
        slice_queries = np.column_stack((
            grid_x.reshape(-1), grid_y.reshape(-1),
            np.full(grid_x.size, self._esdf_slice_z_m, dtype=np.float32),
            np.zeros(grid_x.size, dtype=np.float32),
        ))
        self._esdf_slice_queries = torch.from_numpy(slice_queries).to(self._device)
        self._esdf_slice_closest = torch.zeros(
            (grid_x.size, 4), dtype=torch.float32, device=self._device)
        local_axis = np.linspace(-1.0, 1.0, self._local_map_pixels, dtype=np.float32)
        local_x, local_y = np.meshgrid(local_axis, local_axis[::-1])
        self._local_query_offsets = np.column_stack((
            local_x.reshape(-1), local_y.reshape(-1))).astype(np.float32)
        self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        self._last_depth_monotonic = {
            name: -np.inf for name in self._camera_source_names}
        self._last_integrated_monotonic = {
            name: None for name in self._camera_source_names}
        self._last_integrated_stamp_by_camera = {
            name: None for name in self._camera_source_names}
        self._last_esdf_monotonic = -np.inf
        self._last_tsdf_visualization_monotonic = -np.inf
        self._last_obstacle_range_monotonic = -np.inf
        self._obstacle_projection_running = False
        self._obstacle_range_minimum_m = None
        self._map_generation = 0
        self._last_esdf_monotonic_completed = None
        self._last_esdf_stamp = None
        self._integrated_frames = 0
        self._integrated_frames_by_camera = {
            name: 0 for name in self._camera_source_names}
        self._esdf_updates = 0
        self._esdf_rate_hz = 0.0
        self._rejected_frames = 0
        self._last_status_state = None
        self._render_states = {name: None for name in self._camera_source_names}
        self._tsdf_rendered_pixels_by_camera = {
            name: 0 for name in self._camera_source_names}
        self._filter_counts_by_camera = {name: {} for name in self._camera_source_names}
        self._esdf_slice_anchor_base = None
        self._esdf_slice_anchor_received_monotonic = -np.inf
        self._current_esdf_slice_z_m = self._esdf_slice_z_m
        self._current_esdf_slice_anchor_base = None

        self._status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self._safe_status_pub = rospy.Publisher(
            "~safe_status", SafeMappingStatus, queue_size=1, latch=True)
        self._static_depth_pub = rospy.Publisher("~static_depth", Image, queue_size=1)
        self._camera_publishers = {}
        for name in self._camera_source_names:
            self._camera_publishers[name] = {
                "static_depth": rospy.Publisher(
                    "~{}/static_depth".format(name), Image, queue_size=1),
                "filter_reasons": rospy.Publisher(
                    "~{}/filter_reasons".format(name), Image, queue_size=1),
                "tsdf_render_depth": rospy.Publisher(
                    "~{}/tsdf_render_depth".format(name), Image, queue_size=1, latch=True),
                "tsdf_render_color": rospy.Publisher(
                    "~{}/tsdf_render_color".format(name), Image, queue_size=1, latch=True),
                "raw_tsdf_render_depth": rospy.Publisher(
                    "~{}/raw_tsdf_render_depth".format(name),
                    Image, queue_size=1, latch=True),
                "raw_tsdf_render_color": rospy.Publisher(
                    "~{}/raw_tsdf_render_color".format(name),
                    Image, queue_size=1, latch=True),
            }
        self._esdf_slice_pub = rospy.Publisher("~esdf_slice", Image, queue_size=1, latch=True)
        self._esdf_slice_observed_pub = rospy.Publisher(
            "~esdf_slice_observed", Image, queue_size=1, latch=True)
        self._esdf_slice_color_pub = rospy.Publisher(
            "~esdf_slice_color", Image, queue_size=1, latch=True)
        self._local_esdf_color_pub = rospy.Publisher(
            "~local_esdf_color", Image, queue_size=1, latch=True)
        self._obstacle_range_color_pub = rospy.Publisher(
            "~obstacle_range_color", Image, queue_size=1, latch=True)
        self._tsdf_render_depth_pub = rospy.Publisher(
            "~tsdf_render_depth", Image, queue_size=1, latch=True)
        self._tsdf_render_color_pub = rospy.Publisher(
            "~tsdf_render_color", Image, queue_size=1, latch=True)
        self._diagnostics_pub = rospy.Publisher("/diagnostics", DiagnosticArray, queue_size=2)
        if self._follow_esdf_slice_anchor:
            rospy.Subscriber(
                rospy.get_param(
                    "~esdf_slice_anchor_topic",
                    "/rekpiper/mapping/gripper_center_probe"),
                PointStamped, self._esdf_slice_anchor_callback, queue_size=1)
        self._synchronizers = []
        self._camera_info_subscribers = []
        for source in self._camera_sources:
            # Cache immutable calibration independently of the depth/mask
            # synchronizer.  A map generation can then be requested as soon as
            # both camera drivers are alive; integration still waits for the
            # complete depth + robot-mask + dynamic-mask tuple below.
            self._camera_info_subscribers.append(rospy.Subscriber(
                source["camera_info_topic"], CameraInfo,
                partial(self._camera_info_callback, source["name"]),
                queue_size=1))
            synchronized_inputs = [
                message_filters.Subscriber(
                    source["depth_topic"], Image, queue_size=1),
                message_filters.Subscriber(
                    source["camera_info_topic"], CameraInfo, queue_size=1),
            ]
            synchronized_mask_labels = []
            if self._require_robot_mask:
                synchronized_inputs.append(message_filters.Subscriber(
                    source.get("robot_mask_topic", default_camera_source["robot_mask_topic"]),
                    Image, queue_size=1))
                synchronized_mask_labels.append("robot")
            if self._require_dynamic_mask:
                synchronized_inputs.append(message_filters.Subscriber(
                    source.get("dynamic_mask_topic", default_camera_source["dynamic_mask_topic"]),
                    Image, queue_size=1))
                synchronized_mask_labels.append("dynamic")
            synchronization_slop = (
                self._mask_sync_slop_s if synchronized_mask_labels
                else float(rospy.get_param("~sync_slop_s", 0.05)))
            synchronizer = message_filters.ApproximateTimeSynchronizer(
                synchronized_inputs, queue_size=10,
                slop=synchronization_slop, allow_headerless=False)
            synchronizer.registerCallback(partial(
                self._queue_depth_frame, source, tuple(synchronized_mask_labels)))
            self._synchronizers.append(synchronizer)
        self._query_service = rospy.Service("~query_sdf", QuerySDF, self._query_sdf)
        self._rebuild_service = rospy.Service(
            "~rebuild_map", RebuildMap, self._rebuild_map)
        if self._stepwise_capture:
            axes = inclusive_grid_axes(
                self._bounds_min, self._bounds_max, self._voxel_size_m)
            self._snapshot_shape = tuple(len(axis) for axis in axes)
            xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            queries = np.column_stack((xyz, np.zeros(len(xyz), dtype=np.float32)))
            self._snapshot_queries = torch.from_numpy(queries).to(self._device)
            self._snapshot_closest = torch.zeros_like(self._snapshot_queries)
            self._snapshot_pub = rospy.Publisher(
                "~diagnostic_sdf_grid", SDFGrid, queue_size=1, latch=True)
            self._capture_service = rospy.Service(
                "~capture_sdf_snapshot", CaptureSDFSnapshot, self._capture_sdf_snapshot)
        if self._safe_supervision_enabled:
            rospy.Subscriber("/rekpiper/objects/exclusion_signature", String,
                             self._exclusion_callback, queue_size=1)
            rospy.Subscriber("/rekpiper/objects/registry", TrackedObjectArray,
                             self._registry_callback, queue_size=1)
        self._publish_status("waiting_for_safe_inputs", DiagnosticStatus.WARN)
        self._publish_safe_status()
        self._status_timer = rospy.Timer(
            rospy.Duration(min(0.5, max(0.1, self._max_map_age_s / 2.0))),
            self._status_watchdog,
        )
        self._startup_rebuild_timer = rospy.Timer(
            rospy.Duration(0.5), self._startup_rebuild_timer_callback,
        )
        self._input_timer = rospy.Timer(
            rospy.Duration(self._depth_period_s / len(self._camera_sources)),
            self._process_pending_frame)
        rospy.loginfo(
            "nvblox mapping ready: voxel=%.3f m, depth=%.1f Hz/camera, ESDF=%.1f Hz, frame=%s, cameras=%s",
            self._voxel_size_m, 1.0 / self._depth_period_s,
            1.0 / self._esdf_period_s, self._target_frame,
            ",".join(self._camera_source_names),
        )

    @staticmethod
    def _positive_param(name, default):
        value = float(rospy.get_param(name, default))
        if not np.isfinite(value) or value <= 0.0:
            raise rospy.ROSInitException("{} must be finite and positive".format(name))
        return value

    @staticmethod
    def _vector(name):
        value = np.asarray(rospy.get_param(name, None), dtype=np.float32)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise rospy.ROSInitException("{} must contain three finite values".format(name))
        return value

    def _mask_for_stamp(self, message, stamp, required, label):
        if message is None:
            if required:
                raise ValueError("{}_mask_missing".format(label))
            return None
        skew = abs((message.header.stamp - stamp).to_sec())
        allowed_skew = self._mask_stamp_tolerance_s if required else self._mask_max_age_s
        if not np.isfinite(skew) or skew > allowed_skew:
            if required:
                raise ValueError("{}_mask_stale".format(label))
            return None
        mask = self._bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        if mask.ndim != 2:
            raise ValueError("{}_mask_not_mono".format(label))
        return mask != 0

    def _esdf_slice_anchor_callback(self, message):
        if message.header.frame_id != self._target_frame:
            rospy.logwarn_throttle(2.0, "ESDF slice anchor frame mismatch: %s", message.header.frame_id)
            return
        point = np.asarray([message.point.x, message.point.y, message.point.z], dtype=np.float32)
        if (not np.all(np.isfinite(point))
                or np.any(point < self._bounds_min)
                or np.any(point > self._bounds_max)):
            rospy.logwarn_throttle(2.0, "ESDF slice anchor lies outside configured workspace")
            return
        with self._lock:
            self._esdf_slice_anchor_base = point
            self._esdf_slice_anchor_received_monotonic = time.monotonic()

    def _active_esdf_slice_anchor(self):
        if (not self._follow_esdf_slice_anchor
                or self._esdf_slice_anchor_base is None
                or time.monotonic() - self._esdf_slice_anchor_received_monotonic
                > self._esdf_slice_anchor_max_age_s):
            return None
        return self._esdf_slice_anchor_base.copy()

    def _reject(self, reason):
        self._rejected_frames += 1
        self._publish_status(reason, DiagnosticStatus.WARN)
        rospy.logwarn_throttle(2.0, "nvblox frame rejected: %s", reason)

    def _publish_status(self, state, level=DiagnosticStatus.OK, extra=None):
        self._last_status_state = state
        now = time.monotonic()
        values = {
            "state": state,
            "target_frame": self._target_frame,
            "integrated_frames": self._integrated_frames,
            "integrated_frames_by_camera": dict(self._integrated_frames_by_camera),
            "filter_counts_by_camera": {
                name: dict(counts) for name, counts in self._filter_counts_by_camera.items()},
            "camera_age_s": {
                name: (None if stamp is None else max(0.0, now - stamp))
                for name, stamp in self._last_integrated_monotonic.items()},
            "last_integrated_stamp_by_camera": dict(
                self._last_integrated_stamp_by_camera),
            "esdf_updates": self._esdf_updates,
            "esdf_rate_hz": self._esdf_rate_hz,
            "rejected_frames": self._rejected_frames,
            "map_query_allowed": bool(state == "esdf_ready" and self._planning_safe_inputs),
            "planning_safe_inputs": self._planning_safe_inputs,
        }
        if extra:
            values.update(extra)
        self._status_pub.publish(String(data=json.dumps(values, sort_keys=True)))
        diagnostic = DiagnosticStatus(
            level=level, name="rekpiper_mapping/nvblox", message=state,
            hardware_id="cuda_nvblox_torch",
            values=[KeyValue(str(key), str(value)) for key, value in sorted(values.items())],
        )
        self._diagnostics_pub.publish(DiagnosticArray(
            header=Header(stamp=rospy.Time.now()), status=[diagnostic]))

    def _publish_safe_status(self, reason=None):
        supervisor = self._safe_supervisor
        if (self._safe_supervision_enabled and self._built_configuration_files is not None
                and supervisor.state not in (SafeMapState.DIRTY, SafeMapState.CLEARING)):
            robot_sha = hashlib.sha256(str(rospy.get_param(
                "/robot_description", "")).encode("utf-8")).hexdigest()
            try:
                files_changed = self._configuration_file_hashes() != self._built_configuration_files
                mesh_changed = (self._collision_mesh_stats(str(rospy.get_param(
                    "/robot_description", ""))) != self._built_collision_mesh_stats)
            except (OSError, ValueError, ET.ParseError, rospkg.ResourceNotFound):
                files_changed = True
                mesh_changed = True
            if (files_changed or mesh_changed
                    or robot_sha != self._built_robot_description_sha):
                supervisor.mark_dirty("configuration_source_changed")
        message = SafeMappingStatus()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self._target_frame
        effective_state = supervisor.state
        effective_reason = supervisor.reason if reason is None else str(reason)
        if (self._safe_supervision_enabled and not self._global_object_tracking_safe
                and effective_state not in (SafeMapState.DIRTY, SafeMapState.CLEARING)):
            effective_state = SafeMapState.PAUSED_UNSAFE
            effective_reason = "object_tracking_or_identity_unsafe"
        message.state = int(effective_state)
        message.state_name = SafeMapState(effective_state).name
        message.map_generation_uuid = supervisor.generation_uuid
        message.configuration_hash = supervisor.configuration_hash
        acceptance_complete = self._acceptance_complete()
        if effective_state == SafeMapState.READY and not acceptance_complete:
            effective_state = SafeMapState.VALIDATING
            effective_reason = "engineering_acceptance_manifest_incomplete"
            message.state = int(effective_state)
            message.state_name = effective_state.name
        now = time.monotonic()
        cameras_current = self._all_cameras_current(now)
        ready = (effective_state == SafeMapState.READY and acceptance_complete
                 and self._planning_safe_inputs and cameras_current)
        if ready:
            effective_reason = "SAFE_TSDF_READY"
        elif effective_state == SafeMapState.READY and not cameras_current:
            stale = [
                name for name, stamp in self._last_integrated_monotonic.items()
                if stamp is None or now - stamp > self._max_map_age_s]
            effective_reason = "camera_source_stale:" + ",".join(stale)
        elif effective_state == SafeMapState.READY and not self._planning_safe_inputs:
            effective_reason = "planning_safe_inputs_missing"
        message.map_query_allowed = ready
        message.planning_safe = ready
        message.rs1_valid_frames = int(supervisor.valid_frames.get("rs1", 0))
        message.rs3_valid_frames = int(supervisor.valid_frames.get("rs3", 0))
        message.esdf_updates = int(supervisor.esdf_updates)
        message.reason = effective_reason
        if self._observation_only:
            message.state = int(SafeMapState.VALIDATING if self._esdf_updates
                                else SafeMapState.WAITING_FOR_INPUTS)
            message.state_name = SafeMapState(message.state).name
            message.map_generation_uuid = "mapper-{}-{}".format(
                self._instance_uuid, self._map_generation)
            message.rs1_valid_frames = self._integrated_frames_by_camera.get("rs1", 0)
            message.rs3_valid_frames = self._integrated_frames_by_camera.get("rs3", 0)
            message.esdf_updates = self._esdf_updates
            message.reason = ("observation_only_not_for_planning" if cameras_current
                              else "observation_inputs_unavailable_or_stale")
            if self._stepwise_capture:
                message.reason = ("stepwise_snapshot_ready_for_preview" if self._snapshot_ready
                                  else "stepwise_collecting" if self._capture_active
                                  else "stepwise_waiting_for_capture")
            message.map_query_allowed = message.planning_safe = False
        self._safe_status_pub.publish(message)

    def _exclusion_callback(self, message):
        with self._lock:
            supplied = str(message.data)
            if supplied == self._exclusion_signature:
                return
            if self._exclusion_signature and supplied != self._exclusion_signature:
                self._safe_supervisor.mark_dirty("dynamic_exclusion_set_changed")
                self._global_object_tracking_safe = False
            self._exclusion_signature = supplied
            self._publish_safe_status()

    def _registry_callback(self, message):
        with self._lock:
            tracking_safe = bool(message.all_excluded_objects_safe)
            if tracking_safe == self._global_object_tracking_safe:
                return
            self._global_object_tracking_safe = tracking_safe
            self._publish_safe_status()

    @staticmethod
    def _hash_file(path):
        value = Path(path).expanduser().resolve()
        if not value.is_file():
            raise ValueError("configuration file missing: {}".format(value))
        return hashlib.sha256(value.read_bytes()).hexdigest()

    def _acceptance_complete(self):
        if self._mode == "autonomous":
            try:
                assert_release_unchanged(self._release)
            except AcceptanceError:
                return False
            values = self._signed_safe_map
            required = (
                "e3_robot_mask_accepted",
                "e4_multicamera_tracking_accepted",
                "grasp_release_slip_accepted",
                "tsdf_clear_and_residuals_accepted",
                "dual_camera_independent_3d_accepted",
            )
            return (isinstance(values, dict)
                    and bool(values.get("accepted", False))
                    and all(bool(values.get(key, False)) for key in required))
        if not self._acceptance_manifest_path:
            return False
        path = Path(self._acceptance_manifest_path).expanduser()
        if not path.is_file():
            return False
        try:
            values = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError):
            return False
        required = (
            "e3_robot_mask_accepted",
            "e4_multicamera_tracking_accepted",
            "grasp_release_slip_accepted",
            "tsdf_clear_and_residuals_accepted",
            "dual_camera_independent_3d_accepted",
        )
        if not isinstance(values, dict) or not all(bool(values.get(key, False))
                                                   for key in required):
            return False
        reports = values.get("reports", {})
        if not isinstance(reports, dict):
            return False
        base = path.parent
        def report_path(key):
            value = Path(str(reports.get(key, ""))).expanduser()
            return value if value.is_absolute() else base / value
        return all(report_path(key).is_file() for key in required)

    def _acceptance_report_paths(self):
        if not self._acceptance_manifest_path:
            return []
        values = yaml.safe_load(Path(self._acceptance_manifest_path).expanduser().read_text())
        reports = values.get("reports", {}) if isinstance(values, dict) else {}
        if not isinstance(reports, dict):
            return []
        base = Path(self._acceptance_manifest_path).expanduser().parent
        resolved = []
        for supplied in reports.values():
            if not str(supplied).strip():
                continue
            path = Path(str(supplied)).expanduser()
            resolved.append(str(path if path.is_absolute() else base / path))
        return resolved

    def _configuration_file_hashes(self):
        acceptance_path = (self._release_bundle_path
                           if self._mode == "autonomous"
                           else self._acceptance_manifest_path)
        paths = ([self._attached_geometry_path, acceptance_path]
                 + list(self._precision_extrinsics_files)
                 + ([] if self._mode == "autonomous"
                    else self._acceptance_report_paths()))
        return tuple(self._hash_file(path) for path in paths if path)

    @staticmethod
    def _collision_mesh_paths(robot_description):
        root = ET.fromstring(robot_description)
        package_index = rospkg.RosPack()
        paths = []
        for mesh in root.findall(".//collision/geometry/mesh"):
            uri = mesh.attrib.get("filename", "")
            if uri.startswith("package://"):
                package_and_path = uri[len("package://"):].split("/", 1)
                if len(package_and_path) != 2:
                    raise ValueError("invalid package mesh URI: " + uri)
                path = Path(package_index.get_path(package_and_path[0])) / package_and_path[1]
            elif uri.startswith("file://"):
                path = Path(uri[len("file://"):])
            else:
                raise ValueError("unsupported collision mesh URI: " + uri)
            if not path.is_file():
                raise ValueError("collision mesh missing: {}".format(path))
            paths.append((uri, path.resolve()))
        return tuple(paths)

    @classmethod
    def _collision_mesh_stats(cls, robot_description):
        return tuple((uri, str(path), path.stat().st_size, path.stat().st_mtime_ns)
                     for uri, path in cls._collision_mesh_paths(robot_description))

    @classmethod
    def _robot_geometry_hash(cls, robot_description):
        digest = hashlib.sha256(robot_description.encode("utf-8"))
        for uri, path in cls._collision_mesh_paths(robot_description):
            digest.update(uri.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _configuration_hash(self):
        if not self._exclusion_signature:
            raise ValueError("object exclusion signature unavailable")
        if set(self._camera_info_snapshots) != set(self._camera_source_names):
            missing = sorted(
                set(self._camera_source_names) - set(self._camera_info_snapshots))
            raise ValueError(
                "CameraInfo snapshots unavailable for: {}".format(
                    ",".join(missing)))
        robot_description = rospy.get_param("/robot_description", "")
        if not robot_description:
            raise ValueError("robot_description unavailable")
        if not self._attached_geometry_path:
            raise ValueError("attached collision geometry manifest is not configured")
        with open(Path(self._attached_geometry_path).expanduser(), "r") as stream:
            attachment_manifest = stream.read()
        attachment_config = yaml.safe_load(attachment_manifest)
        if not isinstance(attachment_config, dict) or not bool(
                attachment_config.get("operator_verified_complete", False)):
            raise ValueError("attached collision geometry inventory not operator verified")
        payload = {
            "cameras": self._camera_sources,
            "camera_info": self._camera_info_snapshots,
            "robot_geometry_sha256": self._robot_geometry_hash(robot_description),
            "attachment_sha256": hashlib.sha256(
                attachment_manifest.encode("utf-8")).hexdigest(),
            "extrinsics_sha256": (
                [self._release["verified_artifacts"][
                    "camera_extrinsics_" + name]["payload_sha256"]
                 for name in self._camera_source_names]
                if self._mode == "autonomous" else
                [self._hash_file(path)
                 for path in self._precision_extrinsics_files]),
            "safe_acceptance_sha256": (
                self._release["verified_artifacts"]["safe_map"][
                    "payload_sha256"]
                if self._mode == "autonomous" else
                self._hash_file(self._acceptance_manifest_path)),
            "exclusion_signature": self._exclusion_signature,
            "mask": {
                "dilation_px": self._mask_dilation_px,
                "stamp_tolerance_s": self._mask_stamp_tolerance_s,
                "minimum_retained_fraction": self._min_retained_fraction,
            },
            "voxel_size_m": self._voxel_size_m,
            "workspace_min": self._bounds_min.tolist(),
            "workspace_max": self._bounds_max.tolist(),
        }
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def _publish_cleared_outputs(self):
        stamp = rospy.Time.now()
        slice_shape = (self._esdf_slice_pixels, self._esdf_slice_pixels)
        local_shape = (self._local_map_pixels, self._local_map_pixels)
        render_shape = (self._tsdf_render_height, self._tsdf_render_width)
        for publisher, array, encoding in (
                (self._esdf_slice_pub, np.zeros(slice_shape, np.float32), "32FC1"),
                (self._esdf_slice_observed_pub, np.zeros(slice_shape, np.uint8), "mono8"),
                (self._esdf_slice_color_pub, np.zeros(slice_shape + (3,), np.uint8), "bgr8"),
                (self._local_esdf_color_pub, np.zeros(local_shape + (3,), np.uint8), "bgr8"),
                (self._obstacle_range_color_pub,
                 np.zeros(local_shape + (3,), np.uint8), "bgr8"),
                (self._tsdf_render_depth_pub, np.zeros(render_shape, np.float32), "32FC1"),
                (self._tsdf_render_color_pub, np.zeros(render_shape + (3,), np.uint8), "bgr8")):
            message = self._bridge.cv2_to_imgmsg(array, encoding=encoding)
            message.header = Header(stamp=stamp, frame_id=self._target_frame)
            publisher.publish(message)
        for publishers in self._camera_publishers.values():
            for key, array, encoding in (
                    ("tsdf_render_depth", np.zeros(render_shape, np.float32), "32FC1"),
                    ("tsdf_render_color", np.zeros(render_shape + (3,), np.uint8), "bgr8"),
                    ("raw_tsdf_render_depth",
                     np.zeros(render_shape, np.float32), "32FC1"),
                    ("raw_tsdf_render_color",
                     np.zeros(render_shape + (3,), np.uint8), "bgr8")):
                message = self._bridge.cv2_to_imgmsg(array, encoding=encoding)
                message.header = Header(stamp=stamp, frame_id=self._target_frame)
                publishers[key].publish(message)

    def _clear_map_data(self):
        """Caller owns the mapper lock; invalidate all data from the old map."""
        self._mapper.clear(0)
        if self._raw_diagnostic_mapper is not None:
            self._raw_diagnostic_mapper.clear(0)
        with self._visualization_state_lock:
            self._map_generation += 1
            self._obstacle_range_minimum_m = None
        self._integrated_frames = 0
        self._integrated_frames_by_camera = {
            name: 0 for name in self._camera_source_names}
        self._last_depth_monotonic = {name: -np.inf for name in self._camera_source_names}
        self._last_integrated_monotonic = {name: None for name in self._camera_source_names}
        self._last_integrated_stamp_by_camera = {
            name: None for name in self._camera_source_names}
        self._last_esdf_monotonic = -np.inf
        self._last_esdf_monotonic_completed = None
        self._last_esdf_stamp = None
        self._esdf_updates = 0
        self._esdf_rate_hz = 0.0
        self._render_states = {name: None for name in self._camera_source_names}
        self._tsdf_rendered_pixels_by_camera = {name: 0 for name in self._camera_source_names}
        self._filter_counts_by_camera = {name: {} for name in self._camera_source_names}
        with self._input_lock:
            self._pending_frames.clear()
        self._publish_cleared_outputs()

    def _capture_sdf_snapshot(self, _request):
        """Capture a fresh map for one planning batch, without motion authority."""
        if not (self._stepwise_capture and self._observation_only
                and self._mode == "shadow" and not self._safe_supervision_enabled):
            return CaptureSDFSnapshotResponse(False, "stepwise_capture_disabled", SDFGrid())
        if not self._capture_lock.acquire(False):
            return CaptureSDFSnapshotResponse(False, "capture_in_progress", SDFGrid())
        started = time.monotonic()
        deadline = started + CAPTURE_TIMEOUT_S
        requested_stamp = rospy.Time.now()
        try:
            self._snapshot_ready = False
            self._snapshot_pub.publish(SDFGrid(status="capture_started", valid=False))
            if not self._lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
                raise TimeoutError("capture_timeout_waiting_for_mapper")
            try:
                self._clear_map_data()
                self._capture_after_stamp = requested_stamp
                self._capture_updated.clear()
                self._capture_active = True
            finally:
                self._lock.release()
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                self._capture_updated.wait(min(0.1, max(0.0, deadline - time.monotonic())))
                self._capture_updated.clear()
                if not self._lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
                    break
                try:
                    enough = all(count >= CAPTURE_FRAMES_PER_CAMERA for count in
                                 self._integrated_frames_by_camera.values())
                    now = rospy.Time.now().to_sec()
                    fresh = all(stamp is not None and 0.0 <= now - stamp <= self._max_map_age_s
                                for stamp in self._last_integrated_stamp_by_camera.values())
                    if not (enough and fresh):
                        continue
                    self._capture_active = False
                    self._mapper.update_esdf(0)
                    self._mapper.update_hashmaps()
                    raw = self._mapper.query_sdf(
                        self._snapshot_queries, self._snapshot_closest, True, mapper_id=0)
                    # The CPU copy completes the query before integration can resume.
                    safe = sanitize_nvblox_sdf(raw.detach().cpu().numpy(),
                        unknown_occupied_distance_m=self._unknown_occupied_distance_m)
                    if not np.any(safe.observed):
                        raise ValueError("snapshot_has_no_observed_cells")
                    self._last_esdf_monotonic_completed = time.monotonic()
                    self._last_esdf_stamp = rospy.Time.from_sec(min(
                        self._last_integrated_stamp_by_camera.values()))
                    self._esdf_updates += 1
                    generation = "mapper-{}-{}".format(self._instance_uuid, self._map_generation)
                    grid = SDFGrid(
                        header=Header(stamp=self._last_esdf_stamp, frame_id=self._target_frame),
                        map_generation_uuid=generation,
                        bounds_min=Point(*self._bounds_min.tolist()),
                        bounds_max=Point(*self._bounds_max.tolist()),
                        resolution_m=self._voxel_size_m,
                        size_x=self._snapshot_shape[0], size_y=self._snapshot_shape[1],
                        size_z=self._snapshot_shape[2],
                        distances_m=safe.distances_m.reshape(-1).tolist(),
                        observed=safe.observed.astype(np.uint8).reshape(-1).tolist(),
                        valid=False, status="stepwise_snapshot_for_planning_preview")
                    details = dict(
                        state="snapshot_ready", request_stamp_s=requested_stamp.to_sec(),
                        frames_by_camera=dict(self._integrated_frames_by_camera),
                        source_stamps_s=dict(self._last_integrated_stamp_by_camera),
                        elapsed_s=time.monotonic() - started, generation_uuid=generation)
                    if time.monotonic() >= deadline:
                        raise TimeoutError("capture_timeout_exporting_grid")
                    self._snapshot_pub.publish(grid)
                    self._snapshot_ready = True
                    self._publish_status("stepwise_snapshot_ready", DiagnosticStatus.OK)
                    return CaptureSDFSnapshotResponse(True, json.dumps(details), grid)
                finally:
                    self._lock.release()
            raise TimeoutError("capture_timeout_waiting_for_fresh_cameras:{}".format(
                self._integrated_frames_by_camera))
        except Exception as exc:
            self._snapshot_ready = False
            message = "{}:{}".format(type(exc).__name__, exc)
            self._snapshot_pub.publish(SDFGrid(valid=False, status=message))
            return CaptureSDFSnapshotResponse(False, message, SDFGrid(status=message))
        finally:
            self._capture_active = False
            self._capture_lock.release()

    def _rebuild_map(self, request):
        if not self._safe_supervision_enabled:
            return RebuildMapResponse(False, "", "safe_supervision_disabled")
        with self._lock:
            try:
                configuration_hash = self._configuration_hash()
                generation = self._safe_supervisor.begin_rebuild(configuration_hash)
                self._publish_safe_status(request.reason or "clearing")
                self._clear_map_data()
                self._built_configuration_files = self._configuration_file_hashes()
                self._built_robot_description_sha = hashlib.sha256(str(
                    rospy.get_param("/robot_description", "")).encode("utf-8")).hexdigest()
                self._built_collision_mesh_stats = self._collision_mesh_stats(str(
                    rospy.get_param("/robot_description", "")))
                self._safe_supervisor.cleared()
                self._publish_safe_status()
                return RebuildMapResponse(True, generation,
                                          "cleared_warmup_started")
            except Exception as exc:
                self._safe_supervisor.mark_dirty("clear_failed:" + str(exc))
                self._publish_safe_status()
                return RebuildMapResponse(False, "", str(exc))

    def _all_cameras_current(self, now=None):
        now = time.monotonic() if now is None else now
        return all(
            stamp is not None and now - stamp <= self._max_map_age_s
            for stamp in self._last_integrated_monotonic.values())

    def _status_watchdog(self, _event):
        """Withdraw the latched motion-ready state as soon as the ESDF is stale."""
        completed = self._last_esdf_monotonic_completed
        if completed is not None and self._last_status_state == "esdf_ready":
            now = time.monotonic()
            if not self._all_cameras_current(now):
                self._publish_status("camera_source_stale", DiagnosticStatus.WARN)
            elif now - completed > self._max_map_age_s:
                self._publish_status("map_stale", DiagnosticStatus.WARN)
        if getattr(self, "_safe_supervision_enabled", False) or self._observation_only:
            self._publish_safe_status()

    def _startup_rebuild_timer_callback(self, _event):
        """Build one fresh map once its immutable startup inputs are known.

        Retry quietly while camera calibration or the registry signature is
        unavailable.  A successful attempt is never repeated by this timer;
        later task/session rebuilds retain their existing ownership.
        """
        if (not self._auto_rebuild_on_startup
                or not self._safe_supervision_enabled):
            return
        with self._lock:
            if (self._startup_rebuild_completed
                    or self._startup_rebuild_inflight
                    or self._safe_supervisor.state != SafeMapState.DIRTY
                    or self._safe_supervisor.generation_uuid
                    or not self._global_object_tracking_safe):
                return
            try:
                self._configuration_hash()
            except (OSError, ValueError, yaml.YAMLError):
                return
            self._startup_rebuild_inflight = True
        response = None
        try:
            response = self._rebuild_map(RebuildMapRequest(
                reason="startup_auto_safe_map"))
            if response.success:
                rospy.loginfo(
                    "startup safe-map rebuild started: %s",
                    response.map_generation_uuid)
        finally:
            with self._lock:
                self._startup_rebuild_inflight = False
                self._startup_rebuild_completed = bool(
                    response is not None and response.success)

    def _publish_esdf_slice(self, stamp):
        anchor = self._active_esdf_slice_anchor()
        slice_z_m = float(anchor[2]) if anchor is not None else self._esdf_slice_z_m
        queries = self._esdf_slice_queries
        if anchor is not None:
            # Keep the x/y grid but move the queried plane to the live anchor z.
            queries = self._esdf_slice_queries.clone()
            queries[:, 2].fill_(slice_z_m)
        with torch.inference_mode():
            raw = self._mapper.query_sdf(
                queries, self._esdf_slice_closest,
                True, mapper_id=0)
            raw_numpy = raw.detach().cpu().numpy()
        safe = sanitize_nvblox_sdf(
            raw_numpy, unknown_occupied_distance_m=self._unknown_occupied_distance_m)
        shape = (self._esdf_slice_pixels, self._esdf_slice_pixels)
        distances = safe.distances_m.reshape(shape)
        observed = safe.observed.reshape(shape)
        color = colorize_esdf_slice(
            distances, observed, self._esdf_slice_truncation_m)
        if anchor is not None:
            x_fraction = ((float(anchor[0]) - float(self._bounds_min[0])) /
                          float(self._bounds_max[0] - self._bounds_min[0]))
            y_fraction = ((float(self._bounds_max[1]) - float(anchor[1])) /
                          float(self._bounds_max[1] - self._bounds_min[1]))
            center_x = int(round(x_fraction * (self._esdf_slice_pixels - 1)))
            center_y = int(round(y_fraction * (self._esdf_slice_pixels - 1)))
            for offset in range(-4, 5):
                if 0 <= center_x + offset < self._esdf_slice_pixels:
                    color[center_y, center_x + offset] = (255, 255, 255)
                if 0 <= center_y + offset < self._esdf_slice_pixels:
                    color[center_y + offset, center_x] = (255, 255, 255)
        self._current_esdf_slice_z_m = slice_z_m
        self._current_esdf_slice_anchor_base = anchor
        for publisher, array, encoding in (
                (self._esdf_slice_pub, distances, "32FC1"),
                (self._esdf_slice_observed_pub, observed.astype(np.uint8) * 255, "mono8"),
                (self._esdf_slice_color_pub, color, "bgr8")):
            message = self._bridge.cv2_to_imgmsg(array, encoding=encoding)
            message.header = Header(stamp=stamp, frame_id=self._target_frame)
            publisher.publish(message)
        return float(np.mean(observed))

    def _render_tsdf_depth(self, mapper, pose_tensor, intrinsics_tensor):
        with torch.no_grad():
            rendered = mapper.render_depth_image(
                0, pose_tensor, intrinsics_tensor,
                self._tsdf_render_height, self._tsdf_render_width,
                max_ray_length=self._max_depth_m, max_steps=100)
            depth = rendered.detach().cpu().numpy().astype(
                np.float32, copy=False)
        valid = (np.isfinite(depth) & (depth >= self._min_depth_m)
                 & (depth <= self._max_depth_m))
        output = np.zeros(depth.shape, dtype=np.float32)
        output[valid] = depth[valid]
        return output

    def _publish_tsdf_render(self, camera_name, stamp, pose_tensor,
                             camera_matrix, source_shape, camera_frame):
        source_height, source_width = source_shape
        intrinsics = np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3).copy()
        intrinsics[0, :] *= float(self._tsdf_render_width) / float(source_width)
        intrinsics[1, :] *= float(self._tsdf_render_height) / float(source_height)
        render_intrinsics = torch.from_numpy(intrinsics)
        safe_depth = self._render_tsdf_depth(
            self._mapper, pose_tensor, render_intrinsics)
        color = colorize_static_depth(
            safe_depth, safe_depth.shape, self._min_depth_m, self._max_depth_m)
        publishers = self._camera_publishers[camera_name]
        outputs = [
            (publishers["tsdf_render_depth"], safe_depth, "32FC1"),
            (publishers["tsdf_render_color"], color, "bgr8"),
        ]
        if camera_name == self._render_camera_name:
            outputs.extend([
                (self._tsdf_render_depth_pub, safe_depth, "32FC1"),
                (self._tsdf_render_color_pub, color, "bgr8"),
            ])
        for publisher, array, encoding in outputs:
            message = self._bridge.cv2_to_imgmsg(array, encoding=encoding)
            message.header = Header(stamp=stamp, frame_id=camera_frame)
            publisher.publish(message)
        if self._raw_diagnostic_mapper is not None:
            raw_depth = self._render_tsdf_depth(
                self._raw_diagnostic_mapper, pose_tensor, render_intrinsics)
            raw_color = colorize_static_depth(
                raw_depth, raw_depth.shape,
                self._min_depth_m, self._max_depth_m)
            for publisher, array, encoding in (
                    (publishers["raw_tsdf_render_depth"],
                     raw_depth, "32FC1"),
                    (publishers["raw_tsdf_render_color"],
                     raw_color, "bgr8")):
                message = self._bridge.cv2_to_imgmsg(array, encoding=encoding)
                message.header = Header(
                    stamp=stamp, frame_id=camera_frame)
                publisher.publish(message)
        return int(np.count_nonzero(safe_depth))

    def _project_obstacle_range(self, vertices, triangles, anchor, observed,
                                stamp, generation):
        """Rasterize the CPU mesh without holding the mapper integration lock."""
        try:
            obstacle_color, _, minimum = project_tsdf_surface_range(
                vertices, triangles, anchor, observed,
                half_extent_m=self._local_map_half_extent_m)
            with self._visualization_state_lock:
                current_generation = self._map_generation
            if generation != current_generation or rospy.is_shutdown():
                return
            obstacle_message = self._bridge.cv2_to_imgmsg(
                obstacle_color, encoding="bgr8")
            obstacle_message.header = Header(stamp=stamp, frame_id=self._target_frame)
            self._obstacle_range_color_pub.publish(obstacle_message)
            with self._visualization_state_lock:
                self._obstacle_range_minimum_m = minimum
        except Exception as exc:
            rospy.logwarn_throttle(
                5.0, "gripper obstacle range projection failed: {}".format(exc))
        finally:
            with self._visualization_state_lock:
                self._obstacle_projection_running = False

    def _publish_local_maps(self, stamp, now):
        anchor = self._active_esdf_slice_anchor()
        if anchor is None:
            return 0.0, None
        xy = anchor[None, :2] + self._local_query_offsets * self._local_map_half_extent_m
        queries_numpy = np.column_stack((
            xy,
            np.full(len(xy), float(anchor[2]), dtype=np.float32),
            np.zeros(len(xy), dtype=np.float32),
        )).astype(np.float32)
        queries = torch.from_numpy(queries_numpy).to(self._device)
        closest = torch.zeros((len(xy), 4), dtype=torch.float32, device=self._device)
        with torch.inference_mode():
            raw = self._mapper.query_sdf(queries, closest, True, mapper_id=0)
            raw_numpy = raw.detach().cpu().numpy()
        safe = sanitize_nvblox_sdf(
            raw_numpy, unknown_occupied_distance_m=self._unknown_occupied_distance_m)
        shape = (self._local_map_pixels, self._local_map_pixels)
        distances = safe.distances_m.reshape(shape)
        observed = safe.observed.reshape(shape)
        local_color = colorize_local_esdf_contours(distances, observed)
        local_message = self._bridge.cv2_to_imgmsg(local_color, encoding="bgr8")
        local_message.header = Header(stamp=stamp, frame_id=self._target_frame)
        self._local_esdf_color_pub.publish(local_message)

        with self._visualization_state_lock:
            projection_due = (
                not self._obstacle_projection_running
                and now - self._last_obstacle_range_monotonic
                >= self._obstacle_range_period_s)
            mesh_minimum = self._obstacle_range_minimum_m
            if projection_due:
                self._obstacle_projection_running = True
                self._last_obstacle_range_monotonic = now
                generation = self._map_generation
        if projection_due:
            try:
                # GPU mesh extraction stays serialized with integration. The
                # more expensive CPU triangle rasterization runs independently.
                self._mapper.update_mesh(0)
                mesh = self._mapper.get_mesh(0)
                vertices = mesh["vertices"].detach().cpu().numpy().astype(
                    np.float32, copy=True)
                triangles = mesh["triangles"].detach().cpu().numpy().astype(
                    np.int64, copy=True)
                worker = threading.Thread(
                    target=self._project_obstacle_range,
                    args=(vertices, triangles, anchor.copy(), observed.copy(),
                          stamp, generation),
                    name="gripper_obstacle_range", daemon=True)
                worker.start()
            except Exception as exc:
                with self._visualization_state_lock:
                    self._obstacle_projection_running = False
                rospy.logwarn_throttle(
                    5.0, "nvblox mesh extraction failed: {}".format(exc))
        return float(np.mean(observed)), mesh_minimum

    @staticmethod
    def _camera_info_snapshot(message):
        return {
            "frame_id": message.header.frame_id,
            "width": int(message.width), "height": int(message.height),
            "K": [float(value) for value in message.K],
            "D": [float(value) for value in message.D],
            "distortion_model": message.distortion_model,
        }

    def _camera_info_callback(self, camera_name, message):
        camera_snapshot = self._camera_info_snapshot(message)
        if not self._lock.acquire(False):
            return
        try:
            previous_snapshot = self._camera_info_snapshots.get(camera_name)
            self._camera_info_snapshots[camera_name] = camera_snapshot
            if (self._safe_supervision_enabled and previous_snapshot is not None
                    and previous_snapshot != camera_snapshot
                    and self._safe_supervisor.generation_uuid):
                self._safe_supervisor.mark_dirty(
                    camera_name + ":CameraInfo_changed")
                self._publish_safe_status()
        finally:
            self._lock.release()

    def _queue_depth_frame(self, source, labels, depth, info, *masks):
        # Never run GPU work while message_filters holds its input queue lock.
        with self._input_lock:
            self._pending_frames[source["name"]] = (source, labels, depth, info, *masks)

    def _process_pending_frame(self, _event):
        if self._stepwise_capture and not self._capture_active:
            return
        if not self._worker_lock.acquire(False):
            return
        try:
            with self._input_lock:
                if not self._pending_frames:
                    return
                camera = min(self._pending_frames,
                             key=lambda name: self._last_depth_monotonic[name])
                frame = self._pending_frames.pop(camera)
            self._depth_callback(*frame)
        except Exception as exc:
            self._reject("mapping_worker:" + str(exc))
        finally:
            self._worker_lock.release()

    def _depth_callback(self, source, synchronized_mask_labels,
                        depth_msg, info_msg, *mask_messages):
        now = time.monotonic()
        camera_name = source["name"]
        camera_frame = source["camera_frame"]
        if now - self._last_depth_monotonic[camera_name] < self._depth_period_s:
            return
        if not self._lock.acquire(False):
            return
        try:
            if self._stepwise_capture and (
                    not self._capture_active
                    or depth_msg.header.stamp <= self._capture_after_stamp):
                return
            self._last_depth_monotonic[camera_name] = now
            age = (rospy.Time.now() - depth_msg.header.stamp).to_sec()
            if not 0.0 <= age <= self._max_map_age_s:
                return self._reject(camera_name + ":depth_capture_stale")
            self._camera_info_callback(camera_name, info_msg)
            if self._safe_supervision_enabled:
                if not self._global_object_tracking_safe:
                    self._publish_safe_status("object_tracking_or_identity_unsafe")
                    return
                if self._safe_supervisor.state in (
                        SafeMapState.DIRTY, SafeMapState.WAITING_FOR_INPUTS,
                        SafeMapState.CLEARING, SafeMapState.PAUSED_UNSAFE):
                    self._publish_safe_status()
                    return
            if depth_msg.header.frame_id and depth_msg.header.frame_id != camera_frame:
                return self._reject(camera_name + ":unexpected_depth_frame")
            if abs((depth_msg.header.stamp - info_msg.header.stamp).to_sec()) > 0.05:
                return self._reject(camera_name + ":depth_camera_info_timestamp_skew")
            synchronized_masks = dict(zip(synchronized_mask_labels, mask_messages))
            robot_message = synchronized_masks.get("robot")
            dynamic_message = synchronized_masks.get("dynamic")
            try:
                robot_mask = self._mask_for_stamp(
                    robot_message, depth_msg.header.stamp, self._require_robot_mask, "robot")
                dynamic_mask = self._mask_for_stamp(
                    dynamic_message, depth_msg.header.stamp, self._require_dynamic_mask, "dynamic")
                depth_raw = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                transform = self._tf_buffer.lookup_transform(
                    self._target_frame, camera_frame, depth_msg.header.stamp,
                    rospy.Duration(self._tf_timeout_s),
                )
                labelled_masks = [
                    (label, mask) for label, mask in (
                        ("robot", robot_mask), ("dynamic", dynamic_mask))
                    if mask is not None]
                prepared = prepare_static_depth(
                    depth_raw, self._depth_scale, info_msg.K,
                    transform_to_matrix(transform.transform), self._bounds_min, self._bounds_max,
                    [mask for _, mask in labelled_masks], self._mask_dilation_px,
                    self._min_depth_m, self._max_depth_m,
                    exclusion_labels=[label for label, _ in labelled_masks],
                )
                raw_prepared = None
                if self._raw_diagnostic_mapper is not None:
                    raw_prepared = prepare_static_depth(
                        depth_raw, self._depth_scale, info_msg.K,
                        transform_to_matrix(transform.transform),
                        self._bounds_min, self._bounds_max,
                        exclusion_masks=(), mask_dilation_px=0,
                        min_depth_m=self._min_depth_m,
                        max_depth_m=self._max_depth_m,
                        exclusion_labels=(),
                    )
            except (CvBridgeError, ValueError, tf2_ros.TransformException) as exc:
                return self._reject(camera_name + ":" + str(exc))
            if prepared.retained_fraction < self._min_retained_fraction:
                return self._reject(camera_name + ":insufficient_safe_depth")

            if (self._safe_supervision_enabled
                    and self._safe_supervisor.state == SafeMapState.WARMUP):
                self._safe_supervisor.record_valid_frame(camera_name)
                self._publish_safe_status()
                return

            depth_tensor = torch.from_numpy(prepared.depth_m).to(self._device)
            pose_tensor = torch.from_numpy(
                transform_to_matrix(transform.transform).astype(np.float32)).to(self._device)
            intrinsics_tensor = torch.tensor(info_msg.K, dtype=torch.float32).reshape(3, 3)
            self._mapper.add_depth_frame(depth_tensor, pose_tensor, intrinsics_tensor, mapper_id=0)
            if self._raw_diagnostic_mapper is not None:
                raw_depth_tensor = torch.from_numpy(
                    raw_prepared.depth_m).to(self._device)
                self._raw_diagnostic_mapper.add_depth_frame(
                    raw_depth_tensor, pose_tensor, intrinsics_tensor,
                    mapper_id=0)
            self._integrated_frames += 1
            self._integrated_frames_by_camera[camera_name] += 1
            self._last_integrated_monotonic[camera_name] = time.monotonic()
            self._last_integrated_stamp_by_camera[camera_name] = (
                depth_msg.header.stamp.to_sec())
            if self._stepwise_capture:
                self._capture_updated.set()
            self._filter_counts_by_camera[camera_name] = dict(prepared.reason_counts)
            if self._safe_supervision_enabled:
                self._safe_supervisor.record_valid_frame(camera_name)
            static_depth_message = self._bridge.cv2_to_imgmsg(
                prepared.depth_m, encoding="32FC1")
            static_depth_message.header = depth_msg.header
            self._camera_publishers[camera_name]["static_depth"].publish(static_depth_message)
            reason_message = self._bridge.cv2_to_imgmsg(
                prepared.filter_reasons, encoding="mono8")
            reason_message.header = depth_msg.header
            self._camera_publishers[camera_name]["filter_reasons"].publish(reason_message)
            if camera_name == self._render_camera_name:
                self._static_depth_pub.publish(static_depth_message)
            self._render_states[camera_name] = (
                depth_msg.header.stamp, pose_tensor, list(info_msg.K),
                prepared.depth_m.shape, camera_frame)
            if now - self._last_esdf_monotonic >= self._esdf_period_s:
                previous_esdf_completed = self._last_esdf_monotonic_completed
                self._mapper.update_esdf(0)
                self._mapper.update_hashmaps()
                if self._raw_diagnostic_mapper is not None:
                    self._raw_diagnostic_mapper.update_hashmaps()
                torch.cuda.synchronize(self._device)
                self._last_esdf_monotonic = now
                self._last_esdf_monotonic_completed = time.monotonic()
                self._last_esdf_stamp = depth_msg.header.stamp
                self._esdf_updates += 1
                if previous_esdf_completed is not None:
                    interval = self._last_esdf_monotonic_completed - previous_esdf_completed
                    if interval > 0.0:
                        instantaneous = 1.0 / interval
                        self._esdf_rate_hz = (instantaneous if self._esdf_rate_hz <= 0.0
                                              else 0.8 * self._esdf_rate_hz
                                              + 0.2 * instantaneous)
                if self._safe_supervision_enabled:
                    self._safe_supervisor.record_esdf_update()
                slice_observed_fraction = self._publish_esdf_slice(depth_msg.header.stamp)
                local_observed_fraction, obstacle_range_minimum_m = self._publish_local_maps(
                    depth_msg.header.stamp, now)
                if (now - self._last_tsdf_visualization_monotonic
                        >= self._tsdf_visualization_period_s):
                    for render_camera, render_state in self._render_states.items():
                        if render_state is None:
                            continue
                        self._tsdf_rendered_pixels_by_camera[render_camera] = (
                            self._publish_tsdf_render(render_camera, *render_state))
                    self._last_tsdf_visualization_monotonic = now
                all_cameras_current = self._all_cameras_current()
                state = "esdf_ready" if all_cameras_current else "waiting_for_all_cameras"
                self._publish_status(
                    state,
                    DiagnosticStatus.OK if all_cameras_current else DiagnosticStatus.WARN,
                    extra={
                    "retained_pixels": prepared.retained_pixels,
                    "excluded_pixels": prepared.explicitly_excluded_pixels,
                    "last_integrated_camera": camera_name,
                    "esdf_slice_z_m": self._current_esdf_slice_z_m,
                    "esdf_slice_following_anchor": bool(
                        self._current_esdf_slice_anchor_base is not None),
                    "esdf_slice_anchor_label": self._esdf_slice_anchor_label,
                    "esdf_slice_anchor_base_m": (
                        None if self._current_esdf_slice_anchor_base is None
                        else [float(value) for value
                              in self._current_esdf_slice_anchor_base]),
                    # Compatibility for existing status consumers. New code
                    # must use esdf_slice_following_anchor and anchor_label.
                    "esdf_slice_following_gripper_base": bool(
                        self._current_esdf_slice_anchor_base is not None),
                    "esdf_slice_observed_fraction": slice_observed_fraction,
                    "local_esdf_observed_fraction": local_observed_fraction,
                    "obstacle_range_minimum_m": obstacle_range_minimum_m,
                    "tsdf_rendered_pixels_by_camera": dict(
                        self._tsdf_rendered_pixels_by_camera),
                })
                if self._safe_supervision_enabled:
                    self._publish_safe_status()
        finally:
            self._lock.release()

    def _safe_query_response(self, count, status):
        stamp = self._last_esdf_stamp if self._last_esdf_stamp is not None else rospy.Time(0)
        return QuerySDFResponse(
            header=Header(stamp=stamp, frame_id=self._target_frame),
            distances_m=[self._unknown_occupied_distance_m] * count,
            observed=[False] * count, map_valid=False,
            map_generation_uuid="", status=status,
        )

    def _query_sdf(self, request):
        count = len(request.points)
        if request.header.frame_id != self._target_frame:
            return self._safe_query_response(count, "query_frame_mismatch")
        try:
            xyz, radii = normalize_sdf_query(
                [[point.x, point.y, point.z] for point in request.points],
                request.radii_m, self._max_queries,
            )
        except ValueError as exc:
            return self._safe_query_response(count, "invalid_query: {}".format(exc))
        if self._safe_supervision_enabled:
            if (self._safe_supervisor.state != SafeMapState.READY
                    or not self._global_object_tracking_safe
                    or not self._acceptance_complete()):
                return self._safe_query_response(len(xyz), "safe_map_not_ready")
        if self._last_esdf_monotonic_completed is None:
            return self._safe_query_response(len(xyz), "map_unavailable")
        if not self._all_cameras_current():
            return self._safe_query_response(len(xyz), "camera_source_stale")
        if time.monotonic() - self._last_esdf_monotonic_completed > self._max_map_age_s:
            return self._safe_query_response(len(xyz), "map_stale")
        with self._lock:
            queries = torch.from_numpy(np.column_stack((xyz, radii)).astype(np.float32)).to(self._device)
            closest = torch.zeros((len(xyz), 4), dtype=torch.float32, device=self._device)
            raw = self._mapper.query_sdf(queries, closest, True, mapper_id=0)
            raw_numpy = raw.detach().cpu().numpy()
        safe = sanitize_nvblox_sdf(
            raw_numpy, unknown_occupied_distance_m=self._unknown_occupied_distance_m)
        all_observed = bool(np.all(safe.observed))
        planning_valid = (self._planning_safe_inputs
                          and (not self._safe_supervision_enabled
                               or self._safe_supervisor.state == SafeMapState.READY))
        if not self._planning_safe_inputs:
            status = "acceptance_map_not_safe_for_motion"
        else:
            status = "ok" if all_observed else "ok_unknown_is_occupied"
        generation = (self._safe_supervisor.generation_uuid
                      if self._safe_supervision_enabled else
                      "mapper-{}-{}".format(self._instance_uuid, self._map_generation))
        return QuerySDFResponse(
            header=Header(stamp=self._last_esdf_stamp, frame_id=self._target_frame),
            distances_m=safe.distances_m.reshape(-1).tolist(),
            observed=safe.observed.reshape(-1).tolist(), map_valid=planning_valid,
            map_generation_uuid=generation, status=status,
        )


if __name__ == "__main__":
    rospy.init_node("nvblox_mapping")
    try:
        NvbloxMappingNode()
        rospy.spin()
    except rospy.ROSInitException as exc:
        rospy.logfatal(str(exc))
        raise
