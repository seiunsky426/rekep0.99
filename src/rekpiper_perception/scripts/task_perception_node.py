#!/usr/bin/env python3
"""Original SAM masks followed by paper-v2 DINOv2 ViT-S/14 reg4 candidates."""

import json
import threading
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, Point32
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
import torch

from rekpiper_msgs.msg import (
    InstanceGeometry, InstanceGeometryArray, Keypoint3D, Keypoint3DArray,
)
from rekpiper_perception.official_keypoint_adapter import OfficialKeypointProposerAdapter
from rekpiper_perception.sam_automatic import (
    SAMAutomaticSegmenter, arbitrate_quality_ordered_masks,
    masks_to_label_map)
from rekpiper_perception.scene_change import SceneChangeMonitor
from rekpiper_perception.task_geometry import (
    capture_follows_task_trigger, clean_organized_mask, estimate_instance_geometry, organized_workspace_mask,
    prepare_sam_visual_mask, workspace_mask_area_ratio,
)
from rekpiper_perception.visualization import annotate_vlm_candidates, draw_state_banner


class TaskPerceptionNode:
    DETECTING = "DETECTING"
    LOCKED = "LOCKED"
    WAITING_FOR_TASK = "WAITING_FOR_TASK"
    WAITING_FOR_STABLE_SCENE = "WAITING_FOR_STABLE_SCENE"

    def __init__(self):
        self._bridge = CvBridge()
        self._callback_lock = threading.Lock()
        self._base_frame = rospy.get_param("~base_frame", "base_link")
        self._bounds_min = self._required_vector("~workspace_bounds_min", 3)
        self._bounds_max = self._required_vector("~workspace_bounds_max", 3)
        if np.any(self._bounds_min >= self._bounds_max):
            raise rospy.ROSInitException("workspace bounds are not ordered")
        requested_device = rospy.get_param("~device", "auto")
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise rospy.ROSInitException("CUDA requested but PyTorch CUDA is unavailable")
        self._min_period_s = float(rospy.get_param("~min_inference_period_s", 0.5))
        self._last_inference_monotonic = -np.inf
        monitor_rate_hz = float(rospy.get_param("~scene_monitor/rate_hz", 2.0))
        if monitor_rate_hz <= 0.0:
            raise rospy.ROSInitException("scene_monitor/rate_hz must be positive")
        self._monitor_period_s = 1.0 / monitor_rate_hz
        self._last_monitor_monotonic = -np.inf
        self._scene_monitor = SceneChangeMonitor(
            size=(int(rospy.get_param("~scene_monitor/width", 160)),
                  int(rospy.get_param("~scene_monitor/height", 120))),
            depth_change_m=rospy.get_param("~scene_monitor/depth_change_m", 0.04),
            depth_change_ratio=rospy.get_param("~scene_monitor/depth_change_ratio", 0.05),
            rgb_change_level=rospy.get_param("~scene_monitor/rgb_change_level", 35.0),
            rgb_change_ratio=rospy.get_param("~scene_monitor/rgb_change_ratio", 0.15),
            change_frames=rospy.get_param("~scene_monitor/change_frames", 3),
            stable_depth_change_m=rospy.get_param("~scene_monitor/stable_depth_change_m", 0.02),
            stable_depth_change_ratio=rospy.get_param("~scene_monitor/stable_depth_change_ratio", 0.01),
            stable_rgb_mean_difference=rospy.get_param(
                "~scene_monitor/stable_rgb_mean_difference", 8.0),
            stable_frames=rospy.get_param("~scene_monitor/stable_frames", 5),
        )
        self._wait_for_task_trigger = bool(
            rospy.get_param("~wait_for_task_trigger", False))
        self._state = (
            self.WAITING_FOR_TASK if self._wait_for_task_trigger else self.DETECTING)
        self._task_request_serial = 0
        self._task_trigger_stamp_ns = 0
        self._processed_task_serial = 0
        self._task_instruction = ""
        self._last_wait_preview_monotonic = -np.inf
        self._successful_locks = 0
        self._locked_snapshot = None
        self._initial_stability_started = False
        self._initial_scene_stable = False
        self._min_valid_pixels = int(rospy.get_param("~scene/min_valid_pixels", 150))
        self._minimum_confidence = float(rospy.get_param("~scene/minimum_confidence", 0.70))
        self._visual_closing_px = int(
            rospy.get_param("~scene/visual_closing_px", 5))
        self._geometry_erosion_px = int(
            rospy.get_param("~scene/geometry_erosion_px", 3))
        if (self._visual_closing_px < 0 or self._visual_closing_px > 15
                or (self._visual_closing_px not in (0, 1)
                    and self._visual_closing_px % 2 == 0)):
            raise rospy.ROSInitException(
                "scene/visual_closing_px must be 0, 1, or an odd value in [3, 15]")
        if not 0 <= self._geometry_erosion_px <= 15:
            raise rospy.ROSInitException(
                "scene/geometry_erosion_px must lie in [0, 15]")
        self._max_workspace_mask_ratio = float(rospy.get_param(
            "~scene/max_workspace_mask_ratio", 0.30))
        if not 0.0 < self._max_workspace_mask_ratio < 1.0:
            raise rospy.ROSInitException(
                "scene/max_workspace_mask_ratio must lie strictly between 0 and 1")
        self._sam = SAMAutomaticSegmenter(
            rospy.get_param("~scene/sam_model_path"), requested_device,
            model_type=rospy.get_param("~scene/sam_model_type", "vit_h"),
            use_float16=rospy.get_param("~scene/sam_use_float16", True),
            min_area_px=rospy.get_param("~scene/min_area_px", 150),
            max_area_ratio=rospy.get_param("~scene/max_area_ratio", 0.50),
            max_masks=rospy.get_param("~scene/max_masks", 64),
            pred_iou_thresh=rospy.get_param("~scene/pred_iou_thresh", 0.88),
            stability_score_thresh=rospy.get_param(
                "~scene/stability_score_thresh", 0.95),
            points_per_side=rospy.get_param("~scene/points_per_side", 32),
        )
        config = {
            "num_candidates_per_mask": int(rospy.get_param("~keypoint_proposer/num_candidates_per_mask", 3)),
            "min_dist_bt_keypoints": float(rospy.get_param("~keypoint_proposer/min_dist_bt_keypoints", 0.05)),
            "max_mask_ratio": float(rospy.get_param("~keypoint_proposer/max_mask_ratio", 0.50)),
            "mean_shift_jobs": int(rospy.get_param("~keypoint_proposer/mean_shift_jobs", 1)),
            "device": requested_device, "bounds_min": self._bounds_min.tolist(),
            "bounds_max": self._bounds_max.tolist(), "seed": int(rospy.get_param("~keypoint_proposer/seed", 0)),
        }
        self._proposer = OfficialKeypointProposerAdapter(
            rospy.get_param("~keypoint_proposer/official_root"),
            rospy.get_param("~keypoint_proposer/dinov2_repo"),
            rospy.get_param("~keypoint_proposer/dinov2_weights"), config)
        self._keypoints_pub = rospy.Publisher("~keypoints", Keypoint3DArray, queue_size=1, latch=True)
        self._instances_pub = rospy.Publisher(
            "~instances", InstanceGeometryArray, queue_size=1, latch=True)
        # scene_mask remains a compatibility alias for the VLM-facing mask.
        self._scene_mask_pub = rospy.Publisher("~scene_mask", Image, queue_size=1, latch=True)
        self._sam_visual_mask_pub = rospy.Publisher(
            "~sam_visual_mask", Image, queue_size=1, latch=True)
        self._geometry_valid_mask_pub = rospy.Publisher(
            "~geometry_valid_mask", Image, queue_size=1, latch=True)
        self._candidate_pub = rospy.Publisher("~candidate_image", Image, queue_size=1, latch=True)
        self._source_image_pub = rospy.Publisher("~source_image", Image, queue_size=1, latch=True)
        self._task_status_pub = rospy.Publisher(
            "~task_status", String, queue_size=1, latch=True)
        self._task_request_sub = rospy.Subscriber(
            "~task_request", String, self._task_request_callback, queue_size=1)
        rgb = message_filters.Subscriber(rospy.get_param("~rgb_topic", "/camera/color/image_raw"), Image, queue_size=1)
        xyz = message_filters.Subscriber(rospy.get_param("~points_topic", "/rgbd_projection/points_base"), PointCloud2, queue_size=1)
        self._sync_slop_s = float(rospy.get_param("~sync_slop_s", 0.05))
        self._sync_queue_size = int(rospy.get_param("~sync_queue_size", 5))
        if self._sync_slop_s <= 0.0 or self._sync_queue_size < 2:
            raise rospy.ROSInitException("sync_slop_s must be positive and sync_queue_size at least 2")
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb, xyz], self._sync_queue_size, self._sync_slop_s)
        self._sync.registerCallback(self._callback)
        self._publish_task_status(self._state)
        rospy.loginfo(
            "SAM+DINOv2 scene perception ready on %s (state=%s task_gate=%s)",
            requested_device, self._state, self._wait_for_task_trigger)

    @staticmethod
    def _required_vector(name, length):
        value = np.asarray(rospy.get_param(name, None), dtype=float)
        if value.shape != (length,) or not np.all(np.isfinite(value)):
            raise rospy.ROSInitException("{} must contain {} finite values".format(name, length))
        return value

    def _publish_task_status(self, state, detail=""):
        snapshot_stamp_ns = 0
        if state == self.LOCKED and self._locked_snapshot is not None:
            snapshot_stamp_ns = (
                self._locked_snapshot["keypoints"].header.stamp.to_nsec())
        self._task_status_pub.publish(String(data=json.dumps({
            "state": str(state),
            "task_serial": int(self._task_request_serial),
            "instruction": self._task_instruction,
            "detail": str(detail),
            "snapshot_stamp_ns": int(snapshot_stamp_ns),
        }, ensure_ascii=False, sort_keys=True)))

    def _task_request_callback(self, message):
        instruction = str(message.data).strip()
        if not instruction:
            rospy.logwarn("Ignored empty task request")
            return
        # Serialize the trigger with RGB-D processing so a completed inference
        # can always be associated with exactly one task serial.
        with self._callback_lock:
            previous_state = self._state
            self._task_trigger_stamp_ns = rospy.Time.now().to_nsec()
            self._task_request_serial += 1
            self._task_instruction = instruction
            if self._locked_snapshot is not None:
                snapshot = self._locked_snapshot
                header = snapshot["keypoints"].header
                self._keypoints_pub.publish(Keypoint3DArray(
                    header=header, keypoints=[], all_valid=False,
                    motion_allowed=False,
                    status="task_requested_waiting_for_candidates"))
                self._instances_pub.publish(InstanceGeometryArray(
                    header=header, instances=[], all_valid=False,
                    status="task_requested_waiting_for_candidates"))
                waiting_image = draw_state_banner(
                    snapshot["source_bgr"], "NEW TASK - REFRESHING", (0, 180, 255))
                self._publish_images(
                    header, snapshot["source_bgr"],
                    np.zeros(snapshot["source_bgr"].shape[:2], dtype=np.uint16),
                    np.zeros(snapshot["source_bgr"].shape[:2], dtype=np.uint16),
                    waiting_image)
                self._locked_snapshot = None
            self._state = (
                self.WAITING_FOR_STABLE_SCENE
                if previous_state == self.WAITING_FOR_STABLE_SCENE
                else self.DETECTING)
            self._last_inference_monotonic = -np.inf
            detail = (
                "task_received_waiting_for_stability"
                if self._state == self.WAITING_FOR_STABLE_SCENE
                else "task_received")
            self._publish_task_status(self._state, detail)
        rospy.loginfo(
            "Received task request serial=%d; next stable RGB-D snapshot "
            "will run SAM+DINOv2", self._task_request_serial)

    @staticmethod
    def _organized_xyz(message):
        if [(f.name, f.offset) for f in message.fields] != [("x", 0), ("y", 4), ("z", 8)] or message.point_step != 12:
            raise ValueError("PointCloud2 is not the ReKep contiguous XYZ layout")
        if message.height <= 1 or message.row_step != message.width * 12 or len(message.data) != message.height * message.row_step:
            raise ValueError("PointCloud2 must be organized without row padding")
        return np.frombuffer(message.data, np.float32).reshape(message.height, message.width, 3)

    def _failure(self, header, reason):
        self._keypoints_pub.publish(Keypoint3DArray(header=header, keypoints=[], all_valid=False,
                                                    motion_allowed=False, status=reason))
        self._instances_pub.publish(InstanceGeometryArray(
            header=header, instances=[], all_valid=False, status=reason))
        if self._wait_for_task_trigger and self._task_request_serial:
            self._publish_task_status(self.DETECTING, reason + "_retrying")
        rospy.logwarn_throttle(2.0, "Perception blocked motion: %s", reason)

    def _reject_input(self, header, detail):
        # A single malformed/synchronization frame must not destroy a valid lock.
        if self._state == self.LOCKED:
            rospy.logwarn_throttle(2.0, "Ignored RGB-D frame while candidates remain locked: %s", detail)
            return
        if self._state == self.WAITING_FOR_TASK:
            rospy.logwarn_throttle(
                2.0, "Ignored malformed RGB-D frame while waiting for a task: %s", detail)
            return
        if self._state == self.WAITING_FOR_STABLE_SCENE:
            self._failure(header, "scene_changed_waiting_for_stability")
            return
        rospy.logwarn_throttle(2.0, "Candidate refresh input rejected: %s", detail)
        self._failure(header, "candidate_refresh_failed")

    def _publish_images(self, header, bgr, visual_label_map,
                        geometry_label_map, candidate_bgr):
        visual_msg = self._bridge.cv2_to_imgmsg(visual_label_map, "16UC1")
        visual_msg.header = header
        self._scene_mask_pub.publish(visual_msg)
        self._sam_visual_mask_pub.publish(visual_msg)
        geometry_msg = self._bridge.cv2_to_imgmsg(geometry_label_map, "16UC1")
        geometry_msg.header = header
        self._geometry_valid_mask_pub.publish(geometry_msg)
        source_msg = self._bridge.cv2_to_imgmsg(bgr, "bgr8")
        source_msg.header = header
        self._source_image_pub.publish(source_msg)
        candidate_msg = self._bridge.cv2_to_imgmsg(candidate_bgr, "bgr8")
        candidate_msg.header = header
        self._candidate_pub.publish(candidate_msg)

    def _invalidate_locked_snapshot(self, header, bgr):
        self._locked_snapshot = None
        self._keypoints_pub.publish(Keypoint3DArray(
            header=header, keypoints=[], all_valid=False, motion_allowed=False,
            status="scene_changed_waiting_for_stability"))
        self._instances_pub.publish(InstanceGeometryArray(
            header=header, instances=[], all_valid=False,
            status="scene_changed_waiting_for_stability"))
        invalid_image = draw_state_banner(bgr, "SCENE CHANGED", (0, 0, 255))
        empty = np.zeros(bgr.shape[:2], dtype=np.uint16)
        self._publish_images(header, bgr, empty, empty, invalid_image)

    def _monitor_is_due(self, now):
        if now - self._last_monitor_monotonic < self._monitor_period_s:
            return False
        self._last_monitor_monotonic = now
        return True

    def _callback(self, rgb_msg, points_msg):
        # message_filters callbacks can arrive from ROS subscriber threads while
        # GPU inference is still running. Drop frames so at most one model
        # pass can execute at a time.
        if not self._callback_lock.acquire(False):
            return
        try:
            self._process_callback(rgb_msg, points_msg)
        finally:
            self._callback_lock.release()

    def _process_callback(self, rgb_msg, points_msg):
        if not capture_follows_task_trigger(
                rgb_msg.header.stamp.to_nsec(), points_msg.header.stamp.to_nsec(),
                self._task_trigger_stamp_ns):
            return
        if points_msg.header.frame_id != self._base_frame:
            return self._reject_input(
                points_msg.header, "pointcloud_not_in_{}".format(self._base_frame))
        if abs((rgb_msg.header.stamp - points_msg.header.stamp).to_sec()) > self._sync_slop_s:
            return self._reject_input(points_msg.header, "rgb_pointcloud_timestamp_skew")
        try:
            bgr = self._bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            xyz = self._organized_xyz(points_msg)
            if bgr.shape[:2] != xyz.shape[:2]:
                raise ValueError("RGB and point cloud dimensions differ")
        except (CvBridgeError, ValueError) as exc:
            return self._reject_input(points_msg.header, "perception_error: {}".format(exc))

        rospy.loginfo_throttle(
            5.0,
            "Received synchronized RGB-D: frame=%s shape=%dx%d",
            self._base_frame, bgr.shape[1], bgr.shape[0],
        )

        now = time.monotonic()
        if self._wait_for_task_trigger and self._task_request_serial == 0:
            if now - self._last_wait_preview_monotonic >= 0.2:
                self._last_wait_preview_monotonic = now
                waiting_image = draw_state_banner(
                    bgr, "WAITING FOR TASK", (0, 180, 255))
                self._publish_images(
                    points_msg.header, bgr,
                    np.zeros(bgr.shape[:2], dtype=np.uint16),
                    np.zeros(bgr.shape[:2], dtype=np.uint16), waiting_image)
            rospy.loginfo_throttle(
                5.0, "Waiting for /rekpiper/perception/task_request; "
                "SAM and DINOv2 have not run")
            return

        if self._state == self.LOCKED:
            if not self._monitor_is_due(now):
                return
            try:
                scene_changed = self._scene_monitor.update_locked(bgr, xyz)
            except ValueError as exc:
                return self._reject_input(points_msg.header, "scene_monitor_error: {}".format(exc))
            if not scene_changed:
                return
            self._invalidate_locked_snapshot(points_msg.header, bgr)
            self._scene_monitor.begin_waiting_for_stability(bgr, xyz)
            self._state = self.WAITING_FOR_STABLE_SCENE
            self._publish_task_status(
                self.WAITING_FOR_STABLE_SCENE, "scene_change_confirmed")
            rospy.logwarn("Scene change confirmed; old candidates cleared (state=%s)", self._state)
            return

        if self._state == self.WAITING_FOR_STABLE_SCENE:
            if not self._monitor_is_due(now):
                return
            try:
                scene_stable = self._scene_monitor.update_waiting(bgr, xyz)
            except ValueError as exc:
                return self._reject_input(points_msg.header, "scene_monitor_error: {}".format(exc))
            if not scene_stable:
                return
            self._state = self.DETECTING
            self._last_inference_monotonic = -np.inf
            self._publish_task_status(self.DETECTING, "scene_stable_refreshing")
            rospy.loginfo("Scene stable; refreshing candidates once (state=%s)", self._state)

        # Do not spend the one initial model pass on D435 startup exposure/depth
        # transients.  This reuses the configured adjacent-frame stability test
        # and does not add a fourth state or run either neural network.
        if self._successful_locks == 0 and not self._initial_scene_stable:
            if not self._monitor_is_due(now):
                return
            if not self._initial_stability_started:
                self._scene_monitor.begin_waiting_for_stability(bgr, xyz)
                self._initial_stability_started = True
                rospy.loginfo("Received first RGB-D frame; waiting for stable scene")
                return
            try:
                self._initial_scene_stable = self._scene_monitor.update_waiting(bgr, xyz)
            except ValueError as exc:
                return self._reject_input(points_msg.header, "scene_monitor_error: {}".format(exc))
            if not self._initial_scene_stable:
                return
            rospy.loginfo("Initial RGB-D scene is stable; running first SAM+DINOv2 pass")

        if now - self._last_inference_monotonic < self._min_period_s:
            return
        self._last_inference_monotonic = now
        try:
            scene_masks = arbitrate_quality_ordered_masks(
                self._sam.segment(bgr))
            workspace_pixels = organized_workspace_mask(
                xyz, self._bounds_min, self._bounds_max)
            visual_masks, geometry_masks = [], []
            stats, instance_estimates = [], []
            for mask in scene_masks:
                raw_mask = np.asarray(mask, dtype=bool)
                visual_mask = prepare_sam_visual_mask(
                    raw_mask, closing_px=self._visual_closing_px)
                # The 3-D branch starts independently from the original SAM
                # mask. Largest-component selection and closing belong only to
                # the VLM-facing visual branch.
                geometry_seed = raw_mask & workspace_pixels
                area_ratio = workspace_mask_area_ratio(
                    geometry_seed, workspace_pixels)
                if area_ratio > self._max_workspace_mask_ratio:
                    rospy.loginfo(
                        "Rejected oversized scene mask: workspace_ratio=%.3f limit=%.3f",
                        area_ratio, self._max_workspace_mask_ratio)
                    continue
                clean = clean_organized_mask(
                    xyz, geometry_seed,
                    erosion_px=self._geometry_erosion_px)
                count = int(clean.sum())
                if count < self._min_valid_pixels:
                    continue
                confidence = min(1.0, count / float(max(self._min_valid_pixels, 1)))
                confidence *= count / float(max(int(geometry_seed.sum()), 1))
                if confidence < self._minimum_confidence:
                    continue
                points = xyz[clean]
                covariance = np.cov(points.T) / count if count > 1 else np.full((3, 3), np.inf)
                visual_masks.append(visual_mask)
                geometry_masks.append(clean)
                stats.append((count, float(confidence), covariance))
                instance_estimates.append(
                    estimate_instance_geometry(
                        xyz, clean, visual_mask=visual_mask))
            if not geometry_masks:
                raise RuntimeError("no_scene_masks_with_valid_depth")
            visual_masks = np.stack(visual_masks)
            geometry_masks = np.stack(geometry_masks)
            result = self._proposer.propose(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), xyz, geometry_masks)
            if not len(result.points):
                raise RuntimeError("dinov2_no_candidates")
        except (CvBridgeError, RuntimeError, ValueError, FileNotFoundError) as exc:
            rospy.logwarn("SAM+DINOv2 candidate refresh failed: %s", exc)
            return self._failure(points_msg.header, "candidate_refresh_failed")
        candidate_points = [np.asarray(point, dtype=float) for point in result.points]
        candidate_pixels = [tuple(int(value) for value in pixel)
                            for pixel in result.pixels_rc]
        candidate_groups = [int(group_id) + 1 for group_id in result.rigid_group_ids]
        candidate_sources = ["paper_v2_dinov2_vits14_reg4"] * len(candidate_points)

        keypoints = []
        for candidate_id, (point_xyz, pixel_rc, group_id, source) in enumerate(zip(
                candidate_points, candidate_pixels, candidate_groups, candidate_sources)):
            count, confidence, covariance = stats[group_id - 1]
            valid = bool(np.all(np.isfinite(point_xyz)))
            if not valid:
                return self._failure(points_msg.header, "candidate_refresh_failed")
            keypoints.append(Keypoint3D(id=candidate_id, header=points_msg.header,
                name="K{}".format(candidate_id), position=Point(*point_xyz.tolist()),
                confidence=confidence, covariance=np.asarray(covariance).reshape(-1).tolist(),
                valid_depth_pixels=count, rigid_group_id=group_id,
                pixel_x=int(pixel_rc[1]), pixel_y=int(pixel_rc[0]), source=source,
                outside_workspace=False,
                in_workspace=True, valid=valid))
        keypoint_array = Keypoint3DArray(
            header=points_msg.header, keypoints=keypoints, all_valid=True,
            motion_allowed=False, status="locked_static_snapshot")

        instance_messages = []
        for group_offset, estimate in enumerate(instance_estimates):
            x, y, width, height = estimate.bbox_xywh
            instance_messages.append(InstanceGeometry(
                rigid_group_id=group_offset + 1,
                mask_area_pixels=estimate.mask_area_pixels,
                visual_mask_area_pixels=estimate.visual_mask_area_pixels,
                geometry_valid_pixels=estimate.geometry_valid_pixels,
                bbox_x=x, bbox_y=y, bbox_width=width, bbox_height=height,
                surface_medoid=Point(*estimate.surface_medoid.tolist()),
                medoid_pixel_x=estimate.medoid_pixel_rc[1],
                medoid_pixel_y=estimate.medoid_pixel_rc[0],
                bounds_min=Point(*estimate.bounds_min.tolist()),
                bounds_max=Point(*estimate.bounds_max.tolist()),
                contour_pixels=[Point32(x=float(xy[0]), y=float(xy[1]), z=0.0)
                                for xy in estimate.contour_xy],
                valid=True,
            ))
        instance_array = InstanceGeometryArray(
            header=points_msg.header, instances=instance_messages, all_valid=True,
            status="locked_static_snapshot")
        visual_label_map = masks_to_label_map(visual_masks)
        geometry_label_map = masks_to_label_map(geometry_masks)
        banner = "STATIC LOCK" if self._successful_locks == 0 else "REFRESHED"
        candidate_bgr = draw_state_banner(
            annotate_vlm_candidates(
                bgr, visual_masks, candidate_pixels,
                candidate_groups, candidate_sources),
            banner, (0, 220, 0))
        self._keypoints_pub.publish(keypoint_array)
        self._instances_pub.publish(instance_array)
        self._publish_images(
            points_msg.header, bgr, visual_label_map,
            geometry_label_map, candidate_bgr)
        self._locked_snapshot = {
            "keypoints": keypoint_array,
            "instances": instance_array,
            "source_bgr": bgr.copy(),
            "visual_label_map": visual_label_map.copy(),
            "geometry_label_map": geometry_label_map.copy(),
            "candidate_bgr": candidate_bgr.copy(),
        }
        self._scene_monitor.lock_baseline(bgr, xyz)
        self._state = self.LOCKED
        self._processed_task_serial = self._task_request_serial
        self._successful_locks += 1
        self._publish_task_status(
            self.LOCKED,
            "locked_static_snapshot_with_{}_candidates".format(len(keypoints)))
        rospy.loginfo("Locked %d static candidates after one SAM+DINOv2 pass (state=%s)",
                      len(keypoints), self._state)


if __name__ == "__main__":
    rospy.init_node("task_perception")
    TaskPerceptionNode()
    rospy.spin()
