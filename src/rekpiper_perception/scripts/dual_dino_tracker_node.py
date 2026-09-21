#!/usr/bin/env python3
"""Fixed-reference DINOv2 keypoint tracking for the configured observation camera."""

from collections import deque
from copy import deepcopy
import threading
import time

from cv_bridge import CvBridge
import cv2
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
import torch
import torch.nn.functional as functional
import tf2_ros

from rekpiper_camera.projection import transform_to_matrix
from rekpiper_msgs.msg import (
    Keypoint3DArray, SceneSnapshot, TrackedObject, TrackedObjectArray)
from rekpiper_perception.official_keypoint_adapter import load_local_dinov2
from rekpiper_perception.tracking_geometry import organized_xyz
from rekpiper_perception.snapshot_contract import groups_in_seed_bounds
from rekpiper_perception.keypoint_tracking import (
    LOST, OBSERVED, PROPAGATED, STALE_OCCLUDED,
    capture_stamp_is_fresh,
    nearest_reference_frames, tensor_feature_observation, tensor_reference_descriptor,
    anchor_points_to_gripper, attached_points_in_base,
)


class DenseDino:
    patch_size = 14

    def __init__(self, repo, weights, device):
        self.model = load_local_dinov2(repo, weights, device)
        self.device = next(self.model.parameters()).device

    @torch.inference_mode()
    def __call__(self, bgr):
        height, width = bgr.shape[:2]
        ph, pw = height // self.patch_size, width // self.patch_size
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (pw * 14, ph * 14)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            tokens = self.model.forward_features(tensor)["x_norm_patchtokens"]
        grid = tokens.reshape(1, ph, pw, -1).permute(0, 3, 1, 2)
        return functional.interpolate(
            grid, size=(height, width), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1).squeeze(0)


class DualDinoTracker:
    def __init__(self):
        if not torch.cuda.is_available():
            raise rospy.ROSInitException("DINO realtime tracker requires CUDA")
        self._lock = threading.RLock()
        self._inference = threading.Lock()
        self._bridge = CvBridge()
        self._camera_names = tuple(rospy.get_param("~camera_names", ["rs1"]))
        if not self._camera_names or len(set(self._camera_names)) != len(self._camera_names):
            raise rospy.ROSInitException("camera_names must be unique and non-empty")
        self._latest = {name: None for name in self._camera_names}
        self._frame_buffers = {name: deque(maxlen=160) for name in self._latest}
        self._group_maps = {name: deque(maxlen=12) for name in self._latest}
        self._groups = set(rospy.get_param("~rigid_group_ids", []))
        self._freeze_snapshot = bool(rospy.get_param("~freeze_snapshot", False))
        self._minimum_stamp = rospy.get_param("~minimum_snapshot_stamp_s", 0.0)
        self._use_object_masks = bool(rospy.get_param("~use_object_masks", False))
        self._seed_lower = rospy.get_param("~seed_bounds_min", None)
        self._seed_upper = rospy.get_param("~seed_bounds_max", None)
        self._last_processed_stamps = None
        self._snapshot = None
        self._tracks = {}
        self._attached_group = 0
        self._attached_local_points = {}
        self._tf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf)
        self._base_frame = rospy.get_param("~base_frame", "base_link")
        self._gripper_frame = rospy.get_param(
            "~gripper_frame", "gripper_base")
        self._maximum_attached_visual_error = float(rospy.get_param(
            "~maximum_attached_visual_error_m", 0.015))
        self._rate = float(rospy.get_param("~rate_hz", 20.0))
        self._maximum_frame_age = float(rospy.get_param(
            "~maximum_frame_age_s", 0.15))
        self._maximum_reference_offset = float(rospy.get_param(
            "~maximum_reference_offset_s", 0.15))
        self._radius = float(rospy.get_param("~reference_radius_m", 0.020))
        self._similarity = float(rospy.get_param("~similarity_threshold", 0.60))
        self._top_k = int(rospy.get_param("~top_k", 100))
        self._mad = float(rospy.get_param("~mad_multiplier", 2.0))
        self._window = int(rospy.get_param("~smoothing_window", 10))
        self._lost_after = int(rospy.get_param("~lost_after_frames", 10))
        self._dino = DenseDino(
            rospy.get_param("~dinov2_repo"),
            rospy.get_param("~dinov2_weights"), "cuda")
        self._publisher = rospy.Publisher(
            "/rekpiper/tracking/keypoints", Keypoint3DArray, queue_size=1)
        rospy.Subscriber("/rekpiper/perception/scene_snapshot", SceneSnapshot,
                         self._snapshot_cb, queue_size=1)
        rospy.Subscriber("/rekpiper/objects/registry", TrackedObjectArray,
                         self._registry_cb, queue_size=1)
        for camera in self._camera_names:
            rospy.Subscriber("/rekpiper/objects/{}/rigid_group_map".format(camera),
                             Image, self._group_map_cb, callback_args=camera, queue_size=1)
            rgb = message_filters.Subscriber(rospy.get_param(
                "~{}_rgb_topic".format(camera),
                "/{}/color/image_raw".format(camera)), Image, queue_size=1)
            cloud = message_filters.Subscriber(rospy.get_param(
                "~{}_points_topic".format(camera),
                "/rekpiper/camera/{}/points_recognition".format(camera)),
                PointCloud2, queue_size=1)
            sync = message_filters.ApproximateTimeSynchronizer([rgb, cloud], 4, 0.05)
            sync.registerCallback(lambda a, b, name=camera: self._frame_cb(name, a, b))
            setattr(self, "_sync_" + camera, sync)
        rospy.Timer(rospy.Duration(1.0 / self._rate), self._tick)

    def _frame_cb(self, camera, rgb, cloud):
        try:
            bgr = self._bridge.imgmsg_to_cv2(rgb, "bgr8")
            xyz = organized_xyz(cloud).copy()
            if bgr.shape[:2] != xyz.shape[:2]:
                return
            with self._lock:
                frame = (rgb.header.stamp, bgr.copy(), xyz)
                self._latest[camera] = frame
                history = self._frame_buffers[camera]
                if not history or (frame[0] - history[-1][0]).to_sec() >= 0.09:
                    history.append(frame)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "%s tracker frame rejected: %s", camera, exc)

    def _snapshot_cb(self, snapshot):
        if not snapshot.valid or not snapshot.immutable_layout:
            return
        with self._lock:
            if snapshot.header.stamp.to_sec() < self._minimum_stamp:
                return
            if self._snapshot is not None and (
                    self._freeze_snapshot or self._snapshot.snapshot_id == snapshot.snapshot_id):
                return
            self._snapshot = deepcopy(snapshot)
            if self._seed_lower is not None and self._seed_upper is not None:
                self._groups = groups_in_seed_bounds(
                    snapshot.instances.instances, self._seed_lower, self._seed_upper)
                if not self._groups:
                    self._snapshot = None
                    return
            self._tracks = {}
            self._attached_local_points = {}
            self._last_processed_stamps = None

    def _group_map_cb(self, message, camera):
        labels = self._bridge.imgmsg_to_cv2(message, "passthrough").copy()
        with self._lock:
            self._group_maps[camera].append((message.header.stamp, labels))

    def _registry_cb(self, registry):
        attached = [item for item in registry.objects
                    if item.state == TrackedObject.ATTACHED]
        group = int(attached[0].rigid_group_id) if len(attached) == 1 else 0
        with self._lock:
            if group != self._attached_group:
                previous_group = self._attached_group
                for track in self._tracks.values():
                    if int(track["template"].rigid_group_id) == previous_group:
                        track["history"].clear()
                self._attached_group = group
                self._attached_local_points = {}

    def _gripper_transform(self):
        transform = self._tf.lookup_transform(
            self._base_frame, self._gripper_frame, rospy.Time(0),
            rospy.Duration(0.05))
        return transform_to_matrix(transform.transform)

    def _initialize(self, snapshot, frames, feature_maps):
        tracks = {}
        for message in snapshot.keypoints.keypoints:
            if self._groups and message.rigid_group_id not in self._groups:
                continue
            point = np.array([message.position.x, message.position.y,
                              message.position.z], dtype=float)
            reference = tensor_reference_descriptor(
                [feature_maps[camera] for camera in self._camera_names],
                [frames[camera][2] for camera in self._camera_names],
                point, self._radius)
            if reference is None:
                raise RuntimeError("K{} is invisible in configured reference views".format(message.id))
            tracks[int(message.id)] = {
                "template": deepcopy(message), "reference": point,
                "feature": reference, "history": deque([point], maxlen=self._window),
                "point": point, "covariance": np.eye(3) * 1e-6,
                "state": OBSERVED, "stale": 0,
            }
        with self._lock:
            if self._snapshot.snapshot_id == snapshot.snapshot_id:
                self._tracks = tracks
                rospy.loginfo("Initialized %d fixed-reference keypoints for %s",
                              len(tracks), snapshot.snapshot_id)

    def _tick(self, _event):
        if not self._inference.acquire(False):
            return
        try:
            with self._lock:
                snapshot = deepcopy(self._snapshot)
                frames = dict(self._latest)
                tracks = self._tracks
                buffers = {name: list(values) for name, values in self._frame_buffers.items()}
                group_maps = {name: list(values) for name, values in self._group_maps.items()}
            if snapshot is None or any(frames[name] is None for name in self._camera_names):
                return
            if not tracks:
                reference_frames = nearest_reference_frames(
                    buffers, snapshot.snapshot_stamp_ns, self._maximum_reference_offset)
                if reference_frames is None:
                    raise RuntimeError("matching snapshot reference frames are not buffered")
                features = {name: self._dino(reference_frames[name][1]) for name in frames}
                self._initialize(snapshot, reference_frames, features)
                return
            if self._use_object_masks:
                # Cutie and DINO must inspect the same captured scene. Select the
                # buffered RGB-D frame closest to each incoming group mask.
                for name in frames:
                    if not group_maps[name]:
                        return
                    stamp, _ = group_maps[name][-1]
                    frames[name] = min(buffers[name] + [frames[name]],
                        key=lambda frame: abs((frame[0] - stamp).to_sec()))
            stamps = tuple(frames[name][0].to_nsec() for name in self._camera_names)
            delta = (max(stamps) - min(stamps)) / 1e9
            if delta > 0.10:
                raise RuntimeError("camera frames are not time-consistent")
            if stamps == self._last_processed_stamps:
                return
            newest = max(frames[name][0] for name in self._camera_names)
            if not capture_stamp_is_fresh(
                    rospy.Time.now().to_nsec(), newest.to_nsec(),
                    self._maximum_frame_age):
                raise RuntimeError("camera frames are stale")
            self._last_processed_stamps = stamps
            feature_maps = {name: self._dino(frames[name][1])
                            for name in self._camera_names}
            attached_group = int(self._attached_group)
            base_from_gripper = None
            if attached_group > 0:
                base_from_gripper = self._gripper_transform()
                if not self._attached_local_points:
                    identifiers = [identifier for identifier, track in tracks.items()
                                   if int(track["template"].rigid_group_id)
                                   == attached_group]
                    if not identifiers:
                        raise RuntimeError(
                            "attached rigid group has no ReKep keypoints")
                    points = np.asarray([tracks[value]["point"]
                                         for value in identifiers])
                    local = anchor_points_to_gripper(
                        points, base_from_gripper)
                    self._attached_local_points = dict(zip(identifiers, local))
                attached_predictions = dict(zip(
                    self._attached_local_points,
                    attached_points_in_base(
                        np.asarray(list(self._attached_local_points.values())),
                        base_from_gripper)))
            else:
                attached_predictions = {}
            observations = {}
            for identifier, track in tracks.items():
                masks = None
                if self._use_object_masks:
                    masks = []
                    for name in self._camera_names:
                        stamp, labels = min(group_maps[name], key=lambda item:
                            abs((item[0] - frames[name][0]).to_sec()))
                        mask = labels == track["template"].rigid_group_id
                        if abs((stamp - frames[name][0]).to_sec()) > 0.05:
                            mask = np.zeros(labels.shape, dtype=bool)
                        masks.append(mask)
                observed = tensor_feature_observation(
                    track["feature"],
                    [feature_maps[name] for name in self._camera_names],
                    [frames[name][2] for name in self._camera_names],
                    self._similarity, self._top_k, self._mad, masks=masks)
                observations[identifier] = (None if observed is None else
                    (observed.point, observed.covariance, "masked_" + "+".join(self._camera_names)))
            messages = []
            for identifier, track in sorted(tracks.items()):
                observed = observations[identifier]
                attached_prediction = identifier in attached_predictions
                if attached_prediction:
                    point = attached_predictions[identifier]
                    if (observed is not None
                            and np.linalg.norm(observed[0] - point)
                            > self._maximum_attached_visual_error):
                        covariance = track["covariance"] + np.eye(3) * 1e-4
                        evidence = "attached_visual_rigid_mismatch"
                        state = LOST
                    else:
                        covariance = np.eye(3) * 4e-6
                        evidence = "attached_gripper_rigid"
                        state = PROPAGATED
                    track["stale"] = 0
                elif observed is not None:
                    point, covariance, evidence = observed
                    state = OBSERVED
                    track["stale"] = 0
                else:
                    point, covariance = track["point"], track["covariance"] + np.eye(3) * 1e-5
                    track["stale"] += 1
                    state = LOST if track["stale"] >= self._lost_after else STALE_OCCLUDED
                    evidence = "no_visible_object_observation"
                if state in (OBSERVED, PROPAGATED) and not attached_prediction:
                    track["history"].append(point)
                    point = np.mean(np.stack(track["history"]), axis=0)
                elif state == PROPAGATED and attached_prediction:
                    track["history"].clear()
                    track["history"].append(point)
                track.update(point=point, covariance=covariance, state=state)
                message = deepcopy(track["template"])
                message.header.stamp = newest
                message.position.x, message.position.y, message.position.z = point
                message.covariance = covariance.reshape(-1).tolist()
                message.confidence = 1.0 if state == OBSERVED else 0.5 if state == PROPAGATED else 0.0
                message.source = "dinov2_{}_{}".format(evidence, state)
                message.valid = state in (OBSERVED, PROPAGATED)
                message.in_workspace = message.valid and not message.outside_workspace
                messages.append(message)
            header = deepcopy(messages[0].header)
            header.frame_id = self._base_frame
            valid = all(message.valid for message in messages)
            self._publisher.publish(Keypoint3DArray(
                header=header, keypoints=messages, all_valid=valid,
                motion_allowed=False,
                status="dinov2_tracking" if valid else "keypoint_tracking_lost"))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "dual DINO tracker tick rejected: %s", exc)
        finally:
            self._inference.release()


if __name__ == "__main__":
    rospy.init_node("dual_dino_tracker")
    DualDinoTracker()
    rospy.spin()
