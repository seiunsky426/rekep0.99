#!/usr/bin/env python3
"""Authoritative session UUID registry for dynamic objects.

This node has no robot command publishers or service clients.  Numeric tracker
labels are deliberately private and never escape as object identity.
"""

import copy
import json
import math
import threading
import time
import uuid

import rospy
from std_msgs.msg import String

from rekpiper_perception.object_registry import ObjectRegistry, visually_tracked
from rekpiper_msgs.msg import ObjectSeed, TrackedObject, TrackedObjectArray
from rekpiper_msgs.srv import ObjectCommand, ObjectCommandResponse
from rekpiper_msgs.srv import RegisterObject, RegisterObjectResponse


class RegistryNode:
    def __init__(self):
        cameras = tuple(rospy.get_param("~camera_names", ["rs1", "rs3"]))
        maximum = int(rospy.get_param("~maximum_active_objects", 4))
        self._registry = ObjectRegistry(cameras, maximum_active=maximum)
        self._session_uuid = str(uuid.uuid4())
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._messages = {}
        self._initialization_results = {}
        self._pending_initializations = set()
        self._initialization_timeout_s = float(rospy.get_param(
            "~initialization_timeout_s", 20.0))
        self._maximum_track_age_s = float(rospy.get_param(
            "~maximum_track_age_s", 0.30))
        self._camera_update_wall = {}
        self._visual_lifecycle_only = bool(rospy.get_param("~visual_lifecycle_only", False))

        self._registry_pub = rospy.Publisher(
            "/rekpiper/objects/registry", TrackedObjectArray, queue_size=1, latch=True)
        self._seed_pub = rospy.Publisher(
            "/rekpiper/objects/seeds", ObjectSeed, queue_size=8, latch=False)
        self._signature_pub = rospy.Publisher(
            "/rekpiper/objects/exclusion_signature", String, queue_size=1, latch=True)
        rospy.Subscriber("/rekpiper/objects/tracker_updates", TrackedObjectArray,
                         self._merge_updates, queue_size=4)
        rospy.Subscriber("/rekpiper/objects/grasp_updates", TrackedObjectArray,
                         self._merge_updates, queue_size=4)
        rospy.Service("/rekpiper/objects/register", RegisterObject, self._register)
        rospy.Service("/rekpiper/objects/unregister", ObjectCommand, self._unregister)
        rospy.Timer(rospy.Duration(0.10), self._tracking_watchdog)
        self._publish("empty_registry")

    @staticmethod
    def _has_prompt(request):
        return (bool(request.seed_mask.data)
                or (request.roi.width > 0 and request.roi.height > 0)
                or bool(request.positive_pixels))

    def _register(self, request):
        with self._condition:
            if self._seed_pub.get_num_connections() != 1:
                return RegisterObjectResponse(
                    success=False, object_uuid="",
                    message="exactly_one_multicamera_tracker_must_be_connected")
            if not self._has_prompt(request):
                return RegisterObjectResponse(
                    success=False, object_uuid="",
                    message="registration_requires_seed_mask_roi_or_positive_click")
            try:
                record = self._registry.register(
                    request.source_camera, request.display_name,
                    request.rigid_group_id,
                    request.excluded_from_static_map)
            except (ValueError, KeyError) as exc:
                return RegisterObjectResponse(False, "", str(exc))
            message = TrackedObject()
            message.header.stamp = rospy.Time.now()
            message.header.frame_id = "base_link"
            message.object_uuid = record.object_uuid
            message.display_name = record.display_name
            message.rigid_group_id = record.rigid_group_id
            message.state = TrackedObject.FREE_TRACKED
            message.excluded_from_static_map = record.excluded_from_static_map
            message.parent_frame = "base_link"
            message.camera_names = list(self._registry.camera_names)
            message.camera_status = [TrackedObject.CAMERA_UNKNOWN] * len(message.camera_names)
            message.camera_confidence = [0.0] * len(message.camera_names)
            message.status = "initialization_pending"
            self._messages[record.object_uuid] = message
            self._pending_initializations.add(record.object_uuid)

            seed = ObjectSeed()
            seed.header.stamp = rospy.Time.now()
            seed.action = ObjectSeed.REGISTER
            seed.object_uuid = record.object_uuid
            seed.display_name = record.display_name
            seed.rigid_group_id = record.rigid_group_id
            seed.source_camera = request.source_camera
            seed.excluded_from_static_map = record.excluded_from_static_map
            seed.seed_mask = request.seed_mask
            seed.roi = request.roi
            seed.positive_pixels = request.positive_pixels
            seed.negative_pixels = request.negative_pixels
            self._seed_pub.publish(seed)
            self._publish("registration_pending")
            deadline = time.monotonic() + self._initialization_timeout_s
            while record.object_uuid not in self._initialization_results:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0 or rospy.is_shutdown():
                    break
                self._condition.wait(timeout=remaining)
            result = self._initialization_results.pop(record.object_uuid, None)
            self._pending_initializations.discard(record.object_uuid)
            if result and result[0]:
                return RegisterObjectResponse(True, record.object_uuid, result[1])

            # A provisional UUID is still never reused, but failed initialization
            # must not leave an unsafe active exclusion record behind.
            try:
                self._registry.retire(record.object_uuid)
            except KeyError:
                pass
            self._messages.pop(record.object_uuid, None)
            remove = ObjectSeed()
            remove.header.stamp = rospy.Time.now()
            remove.action = ObjectSeed.REMOVE
            remove.object_uuid = record.object_uuid
            remove.rigid_group_id = record.rigid_group_id
            self._seed_pub.publish(remove)
            failure = result[1] if result else "tracker_initialization_timeout"
            self._publish("registration_rejected_map_dirty")
            return RegisterObjectResponse(False, "", failure)

    def _unregister(self, request):
        with self._lock:
            try:
                record = self._registry.retire(request.object_uuid)
            except KeyError as exc:
                return ObjectCommandResponse(False, 0, str(exc))
            self._messages.pop(record.object_uuid, None)
            seed = ObjectSeed()
            seed.header.stamp = rospy.Time.now()
            seed.action = ObjectSeed.REMOVE
            seed.object_uuid = record.object_uuid
            seed.rigid_group_id = record.rigid_group_id
            self._seed_pub.publish(seed)
            self._publish("object_unregistered_map_dirty")
            return ObjectCommandResponse(True, 0,
                                         "unregistered_full_tsdf_rebuild_required")

    def _merge_updates(self, update):
        with self._condition:
            for incoming in update.objects:
                if incoming.object_uuid not in self._messages:
                    continue
                current = self._messages[incoming.object_uuid]
                if (int(incoming.rigid_group_id) != 0
                        and int(incoming.rigid_group_id)
                        != int(current.rigid_group_id)):
                    rospy.logerr_throttle(
                        1.0, "rejected rigid-group identity mutation for %s: %d != %d",
                        incoming.object_uuid, incoming.rigid_group_id,
                        current.rigid_group_id)
                    current.state = TrackedObject.FAULT
                    current.status = "rigid_group_identity_mismatch"
                    continue
                # Tracker owns camera/pose fields; grasp monitor owns lifecycle state.
                if incoming.camera_names:
                    current.header = incoming.header
                    position = incoming.pose_base.pose.position
                    pose_is_finite = all(math.isfinite(value) for value in (
                        position.x, position.y, position.z))
                    has_tracked_pose = any(
                        status == TrackedObject.CAMERA_TRACKED
                        for status in incoming.camera_status)
                    if has_tracked_pose and pose_is_finite:
                        current.pose_base = incoming.pose_base
                    current.parent_frame = incoming.parent_frame or "base_link"
                    status_by_camera = dict(zip(current.camera_names,
                                                current.camera_status))
                    confidence_by_camera = dict(zip(current.camera_names,
                                                    current.camera_confidence))
                    for name, value, confidence in zip(
                            incoming.camera_names, incoming.camera_status,
                            incoming.camera_confidence):
                        if name in self._registry.camera_names:
                            status_by_camera[name] = value
                            confidence_by_camera[name] = confidence
                            self._camera_update_wall[(incoming.object_uuid, name)] = time.monotonic()
                    current.camera_names = list(self._registry.camera_names)
                    current.camera_status = [status_by_camera.get(
                        name, TrackedObject.CAMERA_UNKNOWN)
                        for name in current.camera_names]
                    current.camera_confidence = [confidence_by_camera.get(name, 0.0)
                                                 for name in current.camera_names]
                if incoming.state:
                    current.state = incoming.state
                    # An ungrasped target remains part of the scene. It becomes
                    # a dynamic exclusion only after attachment is physically
                    # verified, and returns to the map only after release
                    # verification reaches FREE_TRACKED.
                    desired_exclusion = None
                    if incoming.state == TrackedObject.ATTACHED:
                        desired_exclusion = True
                    elif incoming.state == TrackedObject.FREE_TRACKED:
                        desired_exclusion = False
                    if desired_exclusion is not None:
                        self._registry.set_excluded_from_static_map(
                            incoming.object_uuid, desired_exclusion)
                        current.excluded_from_static_map = desired_exclusion
                if incoming.status:
                    current.status = incoming.status
                if (incoming.object_uuid in self._pending_initializations
                        and (incoming.status.startswith("initialization_failed:")
                             or incoming.state == TrackedObject.LOST)):
                    self._initialization_results[incoming.object_uuid] = (
                        False, incoming.status or "tracker_initialization_failed")
                elif (incoming.object_uuid in self._pending_initializations
                      and len(current.camera_status) == len(self._registry.camera_names)
                      and all(value == TrackedObject.CAMERA_TRACKED
                              for value in current.camera_status)):
                    self._initialization_results[incoming.object_uuid] = (
                        True, "registered_and_verified_in_configured_cameras")
            self._publish(update.status or "updated")
            self._condition.notify_all()

    def _publish(self, status):
        now = rospy.Time.now()
        array = TrackedObjectArray()
        array.header.stamp = now
        array.header.frame_id = "base_link"
        array.session_uuid = self._session_uuid
        array.objects = [copy.deepcopy(value) for value in self._messages.values()]
        if self._visual_lifecycle_only:
            for item in array.objects:
                if item.object_uuid in self._pending_initializations:
                    continue
                if item.state in (TrackedObject.FREE_TRACKED, TrackedObject.LOST):
                    ages = [time.monotonic() - self._camera_update_wall.get(
                        (item.object_uuid, name), -float("inf")) for name in item.camera_names]
                    visible = visually_tracked(item.camera_status, ages, self._maximum_track_age_s)
                    item.state = TrackedObject.FREE_TRACKED if visible else TrackedObject.LOST
                    item.status = "visual_free_tracked" if visible else "visual_lost"
        all_safe = True
        for item in array.objects:
            if not item.excluded_from_static_map:
                continue
            if item.state in (TrackedObject.LOST, TrackedObject.AMBIGUOUS,
                              TrackedObject.FAULT):
                all_safe = False
                break
            if len(item.camera_status) != len(self._registry.camera_names):
                all_safe = False
                break
            if any(value not in (TrackedObject.CAMERA_TRACKED,
                                 TrackedObject.CAMERA_TRUSTED_NOT_VISIBLE)
                   for value in item.camera_status):
                all_safe = False
                break
            now = time.monotonic()
            if any(now - self._camera_update_wall.get(
                    (item.object_uuid, name), -float("inf"))
                   > self._maximum_track_age_s for name in item.camera_names):
                all_safe = False
                break
        array.all_excluded_objects_safe = all_safe
        array.status = str(status)
        self._registry_pub.publish(array)
        signature = {
            "session_uuid": self._session_uuid,
            # Tracking a non-excluded object must not invalidate a valid map.
            "revision": self._registry.exclusion_revision,
            "excluded_object_uuids": list(self._registry.exclusion_signature()),
        }
        self._signature_pub.publish(String(data=json.dumps(signature, sort_keys=True)))

    def _tracking_watchdog(self, _event):
        with self._lock:
            self._publish("tracking_age_watchdog")


if __name__ == "__main__":
    rospy.init_node("object_registry", anonymous=False, disable_signals=False)
    RegistryNode()
    rospy.spin()
