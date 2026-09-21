#!/usr/bin/env python3
"""SAM initialization and Cutie propagation for configured tracking cameras.

The node fails closed per camera: when an excluded object might be visible but
cannot be tracked, no dynamic mask for that RGB-D frame is published.
"""

from dataclasses import dataclass, field
import math
import threading

import cv2
from cv_bridge import CvBridge
import message_filters
import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import tf2_ros
import torch

from rekpiper_camera.projection import transform_to_matrix
from rekpiper_perception.cutie_backend import CutieBackend
from rekpiper_perception.tracking_geometry import mask_patch_weights
from rekpiper_msgs.msg import (ObjectSeed, TrackedObject, TrackedObjectArray,
                            TrackedObjectCloud, TrackedObjectCloudArray)


@dataclass
class CameraFrame:
    bgr: np.ndarray
    depth_m: np.ndarray
    info: CameraInfo
    stamp: rospy.Time


@dataclass
class Track:
    object_uuid: str
    rigid_group_id: int
    label: int
    display_name: str
    excluded: bool
    masks: dict
    centroid_base: object = None
    descriptor: object = None
    cloud_base: object = None
    lost_frames: dict = field(default_factory=dict)


def depth_to_meters(image, encoding):
    values = np.asarray(image)
    if encoding == "16UC1" or values.dtype == np.uint16:
        return values.astype(np.float32) * 0.001
    if encoding == "32FC1" or values.dtype == np.float32:
        return values.astype(np.float32)
    raise ValueError("unsupported aligned depth encoding {}".format(encoding))


def masked_points_camera(mask, depth_m, k, stride=2):
    rows, cols = np.nonzero(mask)
    if stride > 1:
        rows, cols = rows[::stride], cols[::stride]
    z = depth_m[rows, cols]
    valid = np.isfinite(z) & (z > 0.10) & (z < 2.0)
    rows, cols, z = rows[valid], cols[valid], z[valid]
    if z.size < 20:
        raise ValueError("mask contains fewer than 20 valid depth samples")
    x = (cols.astype(np.float32) - k[2]) * z / k[0]
    y = (rows.astype(np.float32) - k[5]) * z / k[4]
    return np.column_stack((x, y, z)).astype(np.float32)


def transform_points(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


class MultiCameraTracker:
    def __init__(self):
        self._bridge = CvBridge()
        self._lock = threading.RLock()
        self._tf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf)
        self._base_frame = rospy.get_param("~base_frame", "base_link")
        self._camera_names = tuple(rospy.get_param(
            "~camera_names", ["rs1", "rs3"]))
        if (not self._camera_names
                or any(name not in {"rs1", "rs3"}
                       for name in self._camera_names)
                or len(set(self._camera_names)) != len(self._camera_names)):
            raise rospy.ROSInitException(
                "camera_names must be a unique non-empty subset of [rs1, rs3]")
        self._camera_frames = {name: None for name in self._camera_names}
        self._processors = {}
        self._tracks = {}
        self._next_label = 1
        self._object_states = {}
        self._object_exclusions = {}
        self._attached_uuid = ""
        self._attached_points_gripper = None
        self._minimum_pixels = int(rospy.get_param("~minimum_mask_pixels", 200))
        self._minimum_confidence = float(rospy.get_param("~minimum_track_confidence", 0.80))
        self._centroid_limit = float(rospy.get_param("~maximum_cross_camera_centroid_m", 0.020))
        self._occlusion_margin = float(rospy.get_param("~occlusion_depth_margin_m", 0.030))
        self._reacquire_every_frames = int(rospy.get_param(
            "~reacquire_every_frames", 15))
        self._maximum_reacquire_displacement = float(rospy.get_param(
            "~maximum_reacquire_displacement_m", 0.050))
        self._maximum_frame_age = float(rospy.get_param("~maximum_frame_age_s", 0.30))

        backend = CutieBackend(
            rospy.get_param("~cutie_source_root"),
            rospy.get_param("~cutie_weights_path"),
            int(rospy.get_param("~max_internal_size", 480)),
            bool(rospy.get_param("~require_cuda", True)))
        self._backend = backend
        self._processors = {name: backend.new_processor() for name in self._camera_names}

        self._dino = None
        self._minimum_dino = float(rospy.get_param("~minimum_dino_cosine_similarity", 0.65))
        if bool(rospy.get_param("~require_dinov2", True)):
            from rekpiper_perception.official_keypoint_adapter import load_local_dinov2
            self._dino = load_local_dinov2(
                rospy.get_param("~dinov2_repo_path"),
                rospy.get_param("~dinov2_weights_path"), str(backend.device))

        self._label_pubs = {}
        self._group_pubs = {}
        self._mask_pubs = {}
        self._syncs = []
        for name in self._camera_names:
            prefix = rospy.get_param("~{}_prefix".format(name), "/{}".format(name))
            rgb_topic = rospy.get_param("~{}_rgb_topic".format(name),
                                        prefix + "/color/image_raw")
            depth_topic = rospy.get_param("~{}_depth_topic".format(name),
                                          prefix + "/aligned_depth_to_color/image_raw")
            info_topic = rospy.get_param("~{}_camera_info_topic".format(name),
                                         prefix + "/color/camera_info")
            subscribers = [message_filters.Subscriber(rgb_topic, Image, queue_size=1,
                                                       buff_size=8 * 1024 * 1024),
                           message_filters.Subscriber(depth_topic, Image, queue_size=1,
                                                       buff_size=8 * 1024 * 1024),
                           message_filters.Subscriber(info_topic, CameraInfo, queue_size=1)]
            sync = message_filters.ApproximateTimeSynchronizer(
                subscribers, queue_size=8, slop=0.020, allow_headerless=False)
            sync.registerCallback(lambda rgb, depth, info, n=name:
                                  self._frame_callback(n, rgb, depth, info))
            self._syncs.append((subscribers, sync))
            self._label_pubs[name] = rospy.Publisher(
                "/rekpiper/objects/{}/label_map".format(name), Image, queue_size=2)
            self._mask_pubs[name] = rospy.Publisher(
                "/rekpiper/objects/{}/dynamic_mask".format(name), Image, queue_size=2)
            self._group_pubs[name] = rospy.Publisher(
                "/rekpiper/objects/{}/rigid_group_map".format(name), Image, queue_size=2)
        self._updates_pub = rospy.Publisher(
            "/rekpiper/objects/tracker_updates", TrackedObjectArray, queue_size=2)
        self._cloud_pub = rospy.Publisher(
            "/rekpiper/objects/tracked_clouds", TrackedObjectCloudArray, queue_size=2)
        rospy.Subscriber("/rekpiper/objects/seeds", ObjectSeed, self._seed_callback,
                         queue_size=8)
        rospy.Subscriber("/rekpiper/objects/registry", TrackedObjectArray,
                         self._registry_callback, queue_size=2)
        rospy.Subscriber("/rekpiper/objects/attached_object_state", TrackedObject,
                         self._attached_state_callback, queue_size=1)
        rospy.Subscriber("/rekpiper/objects/attached_collision_cloud", PointCloud2,
                         self._attached_cloud_callback, queue_size=1)

    def _registry_callback(self, message):
        with self._lock:
            self._object_states = {item.object_uuid: item.state for item in message.objects}
            self._object_exclusions = {
                item.object_uuid: bool(item.excluded_from_static_map)
                for item in message.objects}
            for identity, track in self._tracks.items():
                if identity in self._object_exclusions:
                    track.excluded = self._object_exclusions[identity]

    def _attached_state_callback(self, message):
        with self._lock:
            if message.state == TrackedObject.ATTACHED:
                self._attached_uuid = message.object_uuid
            elif message.object_uuid == self._attached_uuid:
                self._attached_uuid = ""

    def _attached_cloud_callback(self, message):
        values = list(point_cloud2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True))
        with self._lock:
            self._attached_points_gripper = (np.asarray(values, np.float32)
                                             if values else None)

    def _lookup(self, target, source, stamp):
        msg = self._tf.lookup_transform(target, source, stamp, rospy.Duration(0.08))
        return transform_to_matrix(msg.transform).astype(np.float32)

    def _cloud_base(self, camera, frame, mask):
        points = masked_points_camera(mask, frame.depth_m, frame.info.K)
        base_from_camera = self._lookup(self._base_frame, frame.info.header.frame_id,
                                        frame.stamp)
        return transform_points(base_from_camera, points)

    @torch.inference_mode()
    def _descriptor(self, bgr, mask):
        if self._dino is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self._backend.device)
        features = self._dino.forward_features(tensor)["x_norm_patchtokens"][0]
        side = int(round(math.sqrt(features.shape[0])))
        weights = torch.from_numpy(mask_patch_weights(mask, side)).to(features.device)
        value = (features * weights[:, None]).sum(dim=0)
        value = value / value.norm().clamp_min(1e-8)
        return value.detach().cpu().numpy()

    def _mask_from_seed(self, seed, frame):
        if seed.seed_mask.data:
            mask = self._bridge.imgmsg_to_cv2(seed.seed_mask, "mono8") > 0
            if mask.shape != frame.depth_m.shape:
                raise ValueError("seed mask dimensions do not match source camera")
            return mask
        raise ValueError("Cutie registration requires an original SAM mask")

    def _project_seed_mask(self, points_base, target_frame):
        camera_from_base = self._lookup(target_frame.info.header.frame_id,
                                        self._base_frame, target_frame.stamp)
        points = transform_points(camera_from_base, points_base)
        valid = points[:, 2] > 0.05
        points = points[valid]
        if len(points) < 20:
            raise ValueError("object seed is outside target camera")
        u = target_frame.info.K[0] * points[:, 0] / points[:, 2] + target_frame.info.K[2]
        v = target_frame.info.K[4] * points[:, 1] / points[:, 2] + target_frame.info.K[5]
        height, width = target_frame.depth_m.shape
        pixels = np.column_stack((u, v))
        pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
        hull = cv2.convexHull(np.rint(pixels).astype(np.int32))
        if len(hull) < 3:
            raise ValueError("target camera projected SAM seed is degenerate")
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        return mask.astype(bool)

    def _seed_callback(self, seed):
        with self._lock:
            if seed.action == ObjectSeed.REMOVE:
                self._tracks.pop(seed.object_uuid, None)
                self._reinitialize_processors()
                return
            if seed.action != ObjectSeed.REGISTER:
                return
            try:
                if seed.source_camera not in self._camera_names:
                    raise ValueError("unknown seed camera")
                if self._next_label > np.iinfo(np.uint16).max:
                    raise ValueError("session label namespace exhausted; restart session")
                if any(frame is None for frame in self._camera_frames.values()):
                    raise ValueError("configured RGB-D camera frame is required")
                source = self._camera_frames[seed.source_camera]
                source_mask = self._mask_from_seed(seed, source)
                if int(source_mask.sum()) < self._minimum_pixels:
                    raise ValueError("source SAM mask is too small")
                source_cloud = self._cloud_base(seed.source_camera, source, source_mask)
                source_center = np.median(source_cloud, axis=0)
                source_desc = self._descriptor(source.bgr, source_mask)
                masks = {seed.source_camera: source_mask}
                cloud_base = source_cloud
                centroid_base = source_center
                if len(self._camera_names) == 2:
                    target_name = (
                        "rs3" if seed.source_camera == "rs1" else "rs1")
                    target = self._camera_frames[target_name]
                    target_mask = self._project_seed_mask(source_cloud, target)
                    target_cloud = self._cloud_base(target_name, target, target_mask)
                    target_center = np.median(target_cloud, axis=0)
                    separation = float(np.linalg.norm(source_center - target_center))
                    if separation > self._centroid_limit:
                        raise ValueError(
                            "cross-camera centroid mismatch {:.1f}mm".format(
                                separation * 1000.0))
                    target_desc = self._descriptor(target.bgr, target_mask)
                    if source_desc is not None:
                        similarity = float(np.dot(source_desc, target_desc))
                        if similarity < self._minimum_dino:
                            raise ValueError(
                                "cross-camera DINO similarity {:.3f}".format(
                                    similarity))
                    masks[target_name] = target_mask
                    cloud_base = np.vstack((source_cloud, target_cloud))
                    centroid_base = (source_center + target_center) * 0.5
                self._tracks[seed.object_uuid] = Track(
                    object_uuid=seed.object_uuid,
                    rigid_group_id=int(seed.rigid_group_id),
                    label=self._next_label,
                    display_name=seed.display_name,
                    excluded=seed.excluded_from_static_map, masks=masks,
                    centroid_base=centroid_base,
                    descriptor=source_desc,
                    cloud_base=cloud_base,
                    lost_frames={name: 0 for name in self._camera_names})
                self._next_label += 1
                self._reinitialize_processors()
                rospy.loginfo("Object %s initialized in cameras %s",
                              seed.object_uuid, ",".join(self._camera_names))
            except Exception as exc:
                rospy.logerr("Object %s initialization rejected: %s",
                             seed.object_uuid, exc)
                self._publish_initialization_failure(seed, str(exc))

    def _reinitialize_processors(self):
        self._processors = {name: self._backend.new_processor()
                            for name in self._camera_names}
        for name in self._camera_names:
            frame = self._camera_frames[name]
            if frame is None or not self._tracks:
                continue
            labels = np.zeros(frame.depth_m.shape, dtype=np.uint16)
            object_labels = []
            for track in self._tracks.values():
                mask = track.masks.get(name)
                if mask is None or mask.shape != labels.shape:
                    continue
                labels[mask] = track.label
                object_labels.append(track.label)
            if object_labels:
                self._backend.step(self._processors[name], frame.bgr, labels,
                                   object_labels)

    def _trusted_not_visible(self, track, frame):
        if track.centroid_base is None:
            return False
        camera_from_base = self._lookup(frame.info.header.frame_id, self._base_frame,
                                        frame.stamp)
        point = transform_points(camera_from_base,
                                 np.asarray(track.centroid_base).reshape(1, 3))[0]
        if point[2] <= 0.05:
            return True
        u = frame.info.K[0] * point[0] / point[2] + frame.info.K[2]
        v = frame.info.K[4] * point[1] / point[2] + frame.info.K[5]
        h, w = frame.depth_m.shape
        if u < 0 or v < 0 or u >= w or v >= h:
            return True
        observed = frame.depth_m[int(round(v)), int(round(u))]
        return bool(np.isfinite(observed) and observed > 0
                    and observed < point[2] - self._occlusion_margin)

    def _attached_prediction(self, track, frame):
        if (self._object_states.get(track.object_uuid) != TrackedObject.ATTACHED
                or self._attached_uuid != track.object_uuid
                or self._attached_points_gripper is None):
            return None
        camera_from_gripper = self._lookup(
            frame.info.header.frame_id, "gripper_base", frame.stamp)
        points = transform_points(camera_from_gripper, self._attached_points_gripper)
        points = points[points[:, 2] > 0.05]
        if len(points) < 8:
            return np.zeros(frame.depth_m.shape, dtype=bool)
        u = frame.info.K[0] * points[:, 0] / points[:, 2] + frame.info.K[2]
        v = frame.info.K[4] * points[:, 1] / points[:, 2] + frame.info.K[5]
        h, w = frame.depth_m.shape
        inside = (u >= 0) & (v >= 0) & (u < w) & (v < h)
        if int(inside.sum()) < 8:
            return np.zeros(frame.depth_m.shape, dtype=bool)
        pixels = np.column_stack((u[inside], v[inside])).astype(np.int32)
        hull = cv2.convexHull(pixels)
        projected = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(projected, hull, 1)
        predicted_depth = float(np.median(points[inside, 2]))
        observed = frame.depth_m
        protected_foreground = (np.isfinite(observed) & (observed > 0)
                                & (observed < predicted_depth - 0.020))
        return projected.astype(bool) & ~protected_foreground

    def _try_reacquire(self, track, name, frame):
        track.lost_frames[name] = track.lost_frames.get(name, 0) + 1
        if (self._reacquire_every_frames <= 0
                or track.lost_frames[name] % self._reacquire_every_frames != 0
                or track.cloud_base is None):
            return None
        candidate = self._project_seed_mask(track.cloud_base, frame)
        if int(candidate.sum()) < self._minimum_pixels:
            return None
        candidate_cloud = self._cloud_base(name, frame, candidate)
        candidate_center = np.median(candidate_cloud, axis=0)
        displacement = float(np.linalg.norm(candidate_center - track.centroid_base))
        if displacement > self._maximum_reacquire_displacement:
            return None
        descriptor = self._descriptor(frame.bgr, candidate)
        if (track.descriptor is not None
                and float(np.dot(track.descriptor, descriptor)) < self._minimum_dino):
            return None
        # The same immutable UUID/label is explicitly re-seeded only after all
        # checks pass; no nearest-instance or label reassignment is performed.
        track.masks[name] = candidate
        track.centroid_base = candidate_center
        track.cloud_base = candidate_cloud
        track.lost_frames[name] = 0
        self._reinitialize_processors()
        return candidate

    def _frame_callback(self, name, rgb_msg, depth_msg, info):
        # Drop work under GPU contention instead of processing a TCP backlog.
        if not self._lock.acquire(False):
            return
        try:
            age = (rospy.Time.now() - depth_msg.header.stamp).to_sec()
            if not 0.0 <= age <= self._maximum_frame_age:
                return
            bgr = self._bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            raw_depth = self._bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            if rgb_msg.header.stamp != depth_msg.header.stamp:
                # The resulting mask must carry the depth stamp; intra-camera slop is
                # allowed only while receiving, never represented as exact equality.
                if abs((rgb_msg.header.stamp - depth_msg.header.stamp).to_sec()) > 0.020:
                    return
            frame = CameraFrame(bgr, depth_to_meters(raw_depth, depth_msg.encoding),
                                info, depth_msg.header.stamp)
            if frame.bgr.shape[:2] != frame.depth_m.shape:
                return
            if int(info.width) != frame.depth_m.shape[1] or int(info.height) != frame.depth_m.shape[0]:
                return
            with self._lock:
                self._camera_frames[name] = frame
                if not self._tracks:
                    self._publish_empty(name, depth_msg.header)
                    return
                labels, confidence = self._backend.step(self._processors[name], frame.bgr)
                self._publish_tracking(name, frame, depth_msg.header, labels, confidence)
        except Exception as exc:
            rospy.logerr_throttle(2.0, "%s tracking frame rejected: %s", name, exc)
        finally:
            self._lock.release()

    def _publish_empty(self, name, header):
        frame = self._camera_frames[name]
        labels = np.zeros(frame.depth_m.shape, dtype=np.uint16)
        mask = np.zeros(frame.depth_m.shape, dtype=np.uint8)
        self._publish_image(self._label_pubs[name], labels, "mono16", header)
        self._publish_image(self._group_pubs[name], labels, "mono16", header)
        self._publish_image(self._mask_pubs[name], mask, "mono8", header)

    def _publish_image(self, publisher, image, encoding, header):
        message = self._bridge.cv2_to_imgmsg(np.ascontiguousarray(image), encoding)
        message.header = header
        publisher.publish(message)

    def _publish_tracking(self, name, frame, header, labels, probabilities):
        updates = TrackedObjectArray()
        updates.header = header
        updates.header.frame_id = self._base_frame
        clouds = TrackedObjectCloudArray()
        clouds.header = updates.header
        dynamic = np.zeros(labels.shape, dtype=np.uint8)
        group_labels = np.zeros(labels.shape, dtype=np.uint16)
        all_safe = True
        for track in self._tracks.values():
            object_mask = labels == track.label
            pixels = int(object_mask.sum())
            score = float(np.median(probabilities[object_mask])) if pixels else 0.0
            item = TrackedObject()
            item.header = updates.header
            item.object_uuid = track.object_uuid
            item.display_name = track.display_name
            item.rigid_group_id = track.rigid_group_id
            # Lifecycle state belongs to grasp_state_monitor; zero means no update.
            item.state = 0
            item.excluded_from_static_map = track.excluded
            item.parent_frame = self._base_frame
            item.camera_names = [name]
            item.camera_confidence = [score]
            if pixels >= self._minimum_pixels and score >= self._minimum_confidence:
                try:
                    points = self._cloud_base(name, frame, object_mask)
                    center = np.median(points, axis=0)
                    track.centroid_base = center
                    track.cloud_base = points
                    track.masks[name] = object_mask
                    track.lost_frames[name] = 0
                    item.camera_status = [TrackedObject.CAMERA_TRACKED]
                    item.pose_base.pose.position.x = float(center[0])
                    item.pose_base.pose.position.y = float(center[1])
                    item.pose_base.pose.position.z = float(center[2])
                    item.pose_base.pose.orientation.w = 1.0
                    item.status = "tracked"
                    group_labels[object_mask] = track.rigid_group_id
                    cloud = TrackedObjectCloud()
                    cloud.header = updates.header
                    cloud.object_uuid = track.object_uuid
                    cloud.cloud_base = point_cloud2.create_cloud_xyz32(
                        updates.header, points[::max(1, len(points) // 4000)])
                    cloud.confidence = score
                    clouds.objects.append(cloud)
                    if track.excluded:
                        dynamic[object_mask] = 255
                except Exception as exc:
                    item.camera_status = [TrackedObject.CAMERA_LOST]
                    item.status = "depth_or_tf_invalid:" + str(exc)
            elif self._trusted_not_visible(track, frame):
                item.camera_status = [TrackedObject.CAMERA_TRUSTED_NOT_VISIBLE]
                item.status = "trusted_not_visible"
            else:
                reacquired = self._try_reacquire(track, name, frame)
                predicted = (None if reacquired is not None
                             else self._attached_prediction(track, frame))
                if reacquired is not None:
                    labels[reacquired] = track.label
                    if track.excluded:
                        dynamic[reacquired] = 255
                    item.camera_status = [TrackedObject.CAMERA_TRACKED]
                    item.camera_confidence = [1.0]
                    item.status = "projected_seed_reacquired_uuid_verified"
                elif predicted is not None:
                    if np.any(predicted):
                        labels[predicted] = track.label
                        if track.excluded:
                            dynamic[predicted] = 255
                    item.camera_status = [TrackedObject.CAMERA_TRACKED]
                    item.camera_confidence = [1.0]
                    item.status = "attached_predicted_from_tcp_cloud"
                else:
                    item.camera_status = [TrackedObject.CAMERA_LOST]
                    item.status = "possibly_visible_tracking_lost"
            if (item.camera_status[0] == TrackedObject.CAMERA_TRACKED
                    and track.centroid_base is not None
                    and np.all(np.isfinite(track.centroid_base))):
                center = np.asarray(track.centroid_base, dtype=float)
                item.pose_base.pose.position.x = float(center[0])
                item.pose_base.pose.position.y = float(center[1])
                item.pose_base.pose.position.z = float(center[2])
                item.pose_base.pose.orientation.w = 1.0
            if track.excluded and item.camera_status[0] not in (
                    TrackedObject.CAMERA_TRACKED,
                    TrackedObject.CAMERA_TRUSTED_NOT_VISIBLE):
                all_safe = False
            updates.objects.append(item)
        updates.all_excluded_objects_safe = all_safe
        updates.status = "{}:{}".format(name, "safe" if all_safe else "unsafe")
        label_msg = self._bridge.cv2_to_imgmsg(labels, "mono16")
        label_msg.header = header
        self._label_pubs[name].publish(label_msg)
        self._publish_image(self._group_pubs[name], group_labels, "mono16", header)
        if all_safe:
            self._publish_image(self._mask_pubs[name], dynamic, "mono8", header)
        self._updates_pub.publish(updates)
        self._cloud_pub.publish(clouds)

    def _publish_initialization_failure(self, seed, reason):
        update = TrackedObjectArray()
        update.header.stamp = rospy.Time.now()
        update.header.frame_id = self._base_frame
        item = TrackedObject()
        item.header = update.header
        item.object_uuid = seed.object_uuid
        item.display_name = seed.display_name
        item.rigid_group_id = seed.rigid_group_id
        item.state = TrackedObject.LOST
        item.excluded_from_static_map = seed.excluded_from_static_map
        item.parent_frame = self._base_frame
        item.camera_names = list(self._camera_names)
        item.camera_status = [TrackedObject.CAMERA_LOST] * len(self._camera_names)
        item.camera_confidence = [0.0] * len(self._camera_names)
        item.status = "initialization_failed:" + reason
        update.objects = [item]
        update.all_excluded_objects_safe = False
        update.status = item.status
        self._updates_pub.publish(update)


if __name__ == "__main__":
    rospy.init_node("multicamera_object_tracker", anonymous=False)
    MultiCameraTracker()
    rospy.spin()
