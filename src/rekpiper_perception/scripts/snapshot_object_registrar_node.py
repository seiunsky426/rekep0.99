#!/usr/bin/env python3
"""Register each immutable SAM instance with the configured Cutie tracker."""

from cv_bridge import CvBridge
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image

from rekpiper_msgs.msg import SceneSnapshot
from rekpiper_perception.snapshot_contract import groups_in_seed_bounds
from rekpiper_msgs.srv import (
    ObjectCommand, RebuildMap, RegisterObject, RegisterObjectRequest)


class SnapshotObjectRegistrar:
    def __init__(self):
        self._bridge = CvBridge()
        self._active = []
        self._processed_snapshot_id = ""
        self._groups = set(rospy.get_param("~rigid_group_ids", []))
        self._freeze_snapshot = bool(rospy.get_param("~freeze_snapshot", False))
        self._rebuild_map = bool(rospy.get_param("~rebuild_map", True))
        self._minimum_stamp = rospy.get_param("~minimum_snapshot_stamp_s", 0.0)
        self._seed_lower = rospy.get_param("~seed_bounds_min", None)
        self._seed_upper = rospy.get_param("~seed_bounds_max", None)
        self._register = rospy.ServiceProxy(
            "/rekpiper/objects/register", RegisterObject)
        self._unregister = rospy.ServiceProxy(
            "/rekpiper/objects/unregister", ObjectCommand)
        self._rebuild = rospy.ServiceProxy(
            "/rekpiper/mapping/rebuild_safe_tsdf", RebuildMap)
        snapshot = message_filters.Subscriber(
            "/rekpiper/perception/scene_snapshot", SceneSnapshot, queue_size=2)
        labels = message_filters.Subscriber(rospy.get_param(
            "~label_map_topic", "/rekpiper/perception/scene_mask"),
            Image, queue_size=2)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [snapshot, labels], 4, 0.10)
        self._sync.registerCallback(self._callback)

    def _callback(self, snapshot, labels_message):
        if not snapshot.valid or not snapshot.immutable_layout:
            return
        if snapshot.snapshot_id == self._processed_snapshot_id:
            return
        if snapshot.header.stamp.to_sec() < self._minimum_stamp:
            return
        if self._freeze_snapshot and self._processed_snapshot_id:
            return
        labels = self._bridge.imgmsg_to_cv2(labels_message, "passthrough")
        if labels.ndim != 2:
            return
        if self._seed_lower is not None and self._seed_upper is not None:
            self._groups = groups_in_seed_bounds(
                snapshot.instances.instances, self._seed_lower, self._seed_upper)
            if not self._groups:
                rospy.logerr("No instances inside the configured initial object region")
                return
        for object_uuid in self._active:
            try:
                self._unregister(object_uuid)
            except rospy.ServiceException as exc:
                rospy.logwarn("failed to unregister stale object %s: %s",
                              object_uuid, exc)
        self._active = []
        expected_groups = {int(item.rigid_group_id)
                           for item in snapshot.instances.instances
                           if not self._groups or item.rigid_group_id in self._groups}
        registered_groups = set()
        for index, instance in enumerate(snapshot.instances.instances):
            group = int(instance.rigid_group_id)
            if group not in expected_groups:
                continue
            mask = np.asarray(labels == group, dtype=np.uint8) * 255
            if not np.any(mask):
                continue
            mask_message = self._bridge.cv2_to_imgmsg(mask, "mono8")
            mask_message.header = labels_message.header
            request = RegisterObjectRequest(
                source_camera="rs1", display_name="rigid_group_{}".format(group),
                rigid_group_id=group,
                excluded_from_static_map=False, seed_mask=mask_message)
            try:
                response = self._register(request)
            except rospy.ServiceException as exc:
                rospy.logerr("rigid group %d registration failed: %s", group, exc)
                continue
            if response.success:
                self._active.append(response.object_uuid)
                registered_groups.add(group)
            else:
                rospy.logerr("rigid group %d registration rejected: %s",
                             group, response.message)
        if registered_groups != expected_groups:
            rospy.logerr(
                "initial safe map withheld: registered groups %s expected %s",
                sorted(registered_groups), sorted(expected_groups))
            return
        if not self._rebuild_map:
            self._processed_snapshot_id = snapshot.snapshot_id
            rospy.loginfo("Observation objects registered for snapshot %s: groups %s",
                          snapshot.snapshot_id, sorted(registered_groups))
            return
        try:
            rospy.wait_for_service(
                "/rekpiper/mapping/rebuild_safe_tsdf", timeout=20.0)
            response = self._rebuild("initial_scene_objects_registered")
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logerr("initial safe map rebuild unavailable: %s", exc)
            return
        if not response.success or not response.map_generation_uuid:
            rospy.logerr("initial safe map rebuild rejected: %s", response.message)
            return
        self._processed_snapshot_id = snapshot.snapshot_id
        rospy.loginfo("initial safe map generation %s is building",
                      response.map_generation_uuid)


if __name__ == "__main__":
    rospy.init_node("snapshot_object_registrar")
    SnapshotObjectRegistrar()
    rospy.spin()
