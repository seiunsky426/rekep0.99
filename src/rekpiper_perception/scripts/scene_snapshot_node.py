#!/usr/bin/env python3
"""Publish immutable task snapshots from one synchronized perception result."""

from copy import deepcopy
import hashlib
import json

import message_filters
import rospy

from rekpiper_msgs.msg import InstanceGeometryArray, Keypoint3DArray, SceneSnapshot
from rekpiper_perception.snapshot_contract import validate_snapshot_layout


def layout_digest(keypoints, instances):
    payload = {
        "keypoints": [(int(k.id), str(k.name), int(k.rigid_group_id))
                      for k in keypoints.keypoints],
        "groups": [int(item.rigid_group_id) for item in instances.instances],
    }
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"),
                                     sort_keys=True).encode("utf-8")).hexdigest()


class SceneSnapshotNode:
    def __init__(self):
        self._published_stamp = 0
        # Initial proposals are currently produced from rs1 only.  The online
        # tracker validates its own independent rs1/rs3 reference frames.
        self._camera_names = list(rospy.get_param("~camera_names", ["rs1"]))
        if self._camera_names != ["rs1"]:
            raise rospy.ROSInitException(
                "scene snapshot may only claim the rs1 proposal camera")
        self._annotated_topic = rospy.get_param(
            "~annotated_image_topic", "/rekpiper/perception/candidate_image")
        self._mask_topics = list(rospy.get_param(
            "~instance_mask_topics", ["/rekpiper/perception/scene_mask"]))
        self._sync_slop_s = float(rospy.get_param("~sync_slop_s", 0.02))
        self._publisher = rospy.Publisher(
            "/rekpiper/perception/scene_snapshot", SceneSnapshot,
            queue_size=1, latch=True)
        keypoints = message_filters.Subscriber(
            rospy.get_param("~keypoints_topic", "/rekpiper/perception/keypoints"),
            Keypoint3DArray, queue_size=2)
        instances = message_filters.Subscriber(
            rospy.get_param("~instances_topic", "/rekpiper/perception/instances"),
            InstanceGeometryArray, queue_size=2)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [keypoints, instances], 4, self._sync_slop_s)
        self._sync.registerCallback(self._callback)

    def _callback(self, keypoints, instances):
        stamp = keypoints.header.stamp.to_nsec()
        if stamp <= self._published_stamp:
            return
        instance_stamp = instances.header.stamp.to_nsec()
        time_consistent = bool(
            stamp > 0 and instance_stamp > 0
            and abs(stamp - instance_stamp) <= int(self._sync_slop_s * 1e9))
        valid = bool(keypoints.all_valid and instances.all_valid and keypoints.keypoints)
        layout_valid, layout_status = validate_snapshot_layout(
            keypoints.keypoints, instances.instances)
        valid = valid and layout_valid and time_consistent
        status = (layout_status if not layout_valid else
                  "perception_messages_are_not_time_consistent"
                  if not time_consistent else
                  "immutable_snapshot_locked" if valid else
                  "invalid_perception_input")
        digest = layout_digest(keypoints, instances)
        snapshot_id = "snap-{}-{}".format(stamp, digest[:12])
        message = SceneSnapshot(
            header=deepcopy(keypoints.header), snapshot_id=snapshot_id,
            snapshot_stamp_ns=stamp, keypoints=deepcopy(keypoints),
            instances=deepcopy(instances), camera_names=self._camera_names,
            camera_stamp_ns=[stamp],
            annotated_image_topic=self._annotated_topic,
            instance_mask_topics=self._mask_topics,
            time_consistent=time_consistent, immutable_layout=True,
            valid=valid, status=status)
        self._publisher.publish(message)
        self._published_stamp = stamp


if __name__ == "__main__":
    rospy.init_node("scene_snapshot")
    SceneSnapshotNode()
    rospy.spin()
