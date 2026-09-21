#!/usr/bin/env python3
"""Select the closest fully-audited native AnyGrasp pose to GPT-4o's K index."""

from copy import deepcopy
import threading
import rospy

from rekpiper_grasp.selection import (
    GraspBinding, horizon_requests_grasp, select_keypoint_candidate)
from rekpiper_perception.keypoint_tracking import capture_stamp_is_fresh

from rekpiper_msgs.msg import (
    ClosedLoopStatus, GraspCandidate, GraspCandidateArray,
    Keypoint3DArray, ReKepHorizon, ReKepProgram, SafeMappingStatus,
    SceneSnapshot, TrackedObject,
    TrackedObjectArray,
)


class KeypointAnyGraspSelector:
    def __init__(self):
        self._lock = threading.RLock()
        self._snapshot = None
        self._program = None
        self._registry = None
        self._map_status = None
        self._tracked = None
        self._horizon = None
        self._candidates = None
        self._stage = int(rospy.get_param("~stage", 1))
        self._grasp_attempt = 0
        self._maximum_distance = float(rospy.get_param(
            "~maximum_keypoint_distance_m", 0.10))
        self._publisher = rospy.Publisher(
            "/rekpiper/grasp/selected", GraspCandidate, queue_size=1, latch=True)
        rospy.Subscriber("/rekpiper/perception/scene_snapshot", SceneSnapshot,
                         self._snapshot_cb, queue_size=1)
        rospy.Subscriber("/rekpiper/program/current", ReKepProgram,
                         self._program_cb, queue_size=1)
        rospy.Subscriber("/rekpiper/tracking/keypoints", Keypoint3DArray,
                         self._tracked_cb, queue_size=1)
        rospy.Subscriber("/rekpiper/execution/status", ClosedLoopStatus,
                         self._status_cb, queue_size=1)
        rospy.Subscriber("/rekpiper/planning/horizon", ReKepHorizon,
                         self._horizon_cb, queue_size=1)
        rospy.Subscriber("/rekpiper/objects/registry", TrackedObjectArray,
                         self._registry_cb, queue_size=1)
        rospy.Subscriber("/rekpiper/mapping/safe_status", SafeMappingStatus,
                         self._map_status_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param(
            "~candidates_topic", "/rekpiper/grasp/candidates"),
            GraspCandidateArray, self._candidates_cb, queue_size=1)

    def _snapshot_cb(self, value):
        self._set_and_evaluate("snapshot", value)

    def _program_cb(self, value):
        self._set_and_evaluate("program", value)

    def _tracked_cb(self, value):
        self._set_and_evaluate("tracked", value)

    def _status_cb(self, value):
        with self._lock:
            self._stage = max(1, int(value.stage_index))
            self._grasp_attempt = int(value.grasp_attempt)
        self._evaluate()

    def _horizon_cb(self, value):
        self._set_and_evaluate("horizon", value)

    def _registry_cb(self, value):
        self._set_and_evaluate("registry", value)

    def _map_status_cb(self, value):
        self._set_and_evaluate("map_status", value)

    def _candidates_cb(self, values):
        self._set_and_evaluate("candidates", values)

    def _set_and_evaluate(self, name, value):
        with self._lock:
            setattr(self, "_" + name, deepcopy(value))
        self._evaluate()

    def _evaluate(self):
        with self._lock:
            snapshot, program, tracked = (
                deepcopy(self._snapshot), deepcopy(self._program),
                deepcopy(self._tracked))
            registry, map_status = (
                deepcopy(self._registry), deepcopy(self._map_status))
            horizon, values = (
                deepcopy(self._horizon), deepcopy(self._candidates))
            stage_value, attempt = self._stage, self._grasp_attempt
        if (snapshot is None or program is None or tracked is None or registry is None
                or map_status is None or horizon is None or values is None
                or not program.approved
                or not tracked.all_valid
                or not capture_stamp_is_fresh(
                    rospy.Time.now().to_nsec(),
                    tracked.header.stamp.to_nsec(), 0.15)
                or map_status.state != SafeMappingStatus.READY
                or not map_status.planning_safe
                or not map_status.map_query_allowed):
            return
        stage = max(1, min(stage_value, int(program.num_stages)))
        keypoint_index = int(program.grasp_keypoints[stage - 1])
        if (keypoint_index < 0
                or keypoint_index >= len(snapshot.keypoints.keypoints)
                or keypoint_index >= len(tracked.keypoints)):
            return
        keypoint = tracked.keypoints[keypoint_index].position
        group = int(tracked.keypoints[keypoint_index].rigid_group_id)
        if group != int(snapshot.keypoints.keypoints[keypoint_index].rigid_group_id):
            rospy.logerr_throttle(1.0, "tracked keypoint rigid group changed")
            return
        objects = [item for item in registry.objects
                   if int(item.rigid_group_id) == group]
        if (group <= 0 or len(objects) != 1
                or objects[0].state != TrackedObject.FREE_TRACKED):
            rospy.logerr_throttle(1.0, "grasp group has no unique FREE_TRACKED UUID")
            return
        target = objects[0]
        binding = GraspBinding(
            group, target.object_uuid, program.session_id,
            program.program_sha256, snapshot.snapshot_id,
            map_status.map_generation_uuid, stage, attempt)
        if not horizon_requests_grasp(horizon, binding):
            return
        selected = select_keypoint_candidate(
            values.candidates, [keypoint.x, keypoint.y, keypoint.z], binding,
            self._maximum_distance)
        if selected is None:
            rospy.logwarn_throttle(2.0, "no audited native AnyGrasp pose near K%d", keypoint_index)
            return
        selected = deepcopy(selected)
        selected.planning_authorized = False
        self._publisher.publish(selected)


if __name__ == "__main__":
    rospy.init_node("keypoint_anygrasp_selector")
    KeypointAnyGraspSelector()
    rospy.spin()
