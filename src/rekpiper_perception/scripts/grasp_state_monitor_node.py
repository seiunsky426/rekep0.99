#!/usr/bin/env python3
"""Evidence-only grasp/release monitor.

Safety invariant: this file contains no arm enable/disable, motion service,
JointState command publisher, or CAN API.  Faults only withdraw planning safety.
"""

from collections import defaultdict, deque
import copy
import threading

import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Header
from piper_msgs.msg import PiperStatusMsg
import tf2_ros
import yaml

from rekpiper_acceptance import (
    AcceptanceError, assert_release_unchanged, validate_release_bundle)
from rekpiper_camera.projection import transform_to_matrix
from rekpiper_perception.grasp_state import (
    GraspEvidence, GraspState, GraspStateMachine, GripperBaseline)
from rekpiper_msgs.msg import (TrackedObject, TrackedObjectArray,
                            TrackedObjectCloudArray)
from rekpiper_msgs.srv import (ObjectCommand, ObjectCommandResponse,
                            ResolveObjectState, ResolveObjectStateResponse)


def quaternion_angle_deg(a, b):
    dot = abs(float(np.dot(a, b)))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))


def voxel_and_inflate(points, voxel_m=0.005, inflation_m=0.010):
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if not len(xyz):
        return xyz
    keys = np.floor(xyz / float(voxel_m)).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    base = xyz[np.sort(indices)]
    offsets = np.asarray([[0, 0, 0], [inflation_m, 0, 0], [-inflation_m, 0, 0],
                          [0, inflation_m, 0], [0, -inflation_m, 0],
                          [0, 0, inflation_m], [0, 0, -inflation_m]], np.float32)
    return (base[:, None, :] + offsets[None, :, :]).reshape(-1, 3)


class GraspMonitor:
    def __init__(self):
        configured_cameras = rospy.get_param("~camera_names", ["rs1", "rs3"])
        self._camera_names = tuple(str(name).strip() for name in configured_cameras)
        if (not self._camera_names
                or len(set(self._camera_names)) != len(self._camera_names)):
            raise rospy.ROSInitException("camera_names must be non-empty and unique")
        mode = str(rospy.get_param("~mode", "shadow")).strip().lower()
        self._release = None
        if mode == "autonomous":
            try:
                self._release = validate_release_bundle(
                    str(rospy.get_param("~release_bundle", "")),
                    str(rospy.get_param("~acceptance_public_key", "")),
                    str(rospy.get_param("~minimum_release_counter", "")),
                    expected_robot_id=str(rospy.get_param(
                        "~robot_id", "piper-rekpiper")))
                self._baseline_config = self._release[
                    "verified_artifacts"]["gripper_baseline"]["payload"]
            except (AcceptanceError, KeyError, TypeError, ValueError) as exc:
                raise rospy.ROSInitException(
                    "signed gripper baseline rejected: {}".format(exc))
        else:
            baseline_path = rospy.get_param("~gripper_baseline")
            with open(baseline_path, "r") as stream:
                self._baseline_config = yaml.safe_load(stream) or {}
        self._baseline_ready = bool(self._baseline_config.get("calibrated", False))
        self._evidence_mode = str(self._baseline_config.get(
            "contact_evidence_mode", "encoder_effort_vision"))
        baseline = GripperBaseline(
            float(self._baseline_config["empty_closed_opening_p99_m"]),
            float(self._baseline_config["empty_effort_p95"]),
            float(self._baseline_config.get("minimum_opening_margin_m", 0.002)),
            float(self._baseline_config.get("minimum_effort_margin", 0.2)))
        minimum_release_stable_updates = int(rospy.get_param(
            "~minimum_release_stable_updates", 10))
        self._machines = defaultdict(lambda: GraspStateMachine(
            baseline,
            minimum_release_stable_updates=minimum_release_stable_updates,
            contact_evidence_mode=self._evidence_mode))
        self._region_radius = float(self._baseline_config.get(
            "gripper_region_radius_m", 0.18))
        self._lock = threading.RLock()
        self._registry = {}
        self._clouds = {}
        self._history = defaultdict(lambda: deque(maxlen=100))
        self._gripper_history = deque()
        self._grasp_start_opening = None
        self._gripper_opening = 0.0
        self._gripper_effort = 0.0
        self._joint_stamp = rospy.Time(0)
        self._arm_stamp = rospy.Time(0)
        self._arm_error = "arm_status_missing"
        self._tf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf)
        self._base = rospy.get_param("~base_frame", "base_link")
        self._gripper = rospy.get_param("~gripper_frame", "gripper_base")

        self._updates = rospy.Publisher(
            "/rekpiper/objects/grasp_updates", TrackedObjectArray, queue_size=2)
        self._attached_cloud = rospy.Publisher(
            "/rekpiper/objects/attached_collision_cloud", PointCloud2,
            queue_size=1, latch=True)
        self._attached_state = rospy.Publisher(
            "/rekpiper/objects/attached_object_state", TrackedObject,
            queue_size=1, latch=True)
        rospy.Subscriber("/rekpiper/objects/registry", TrackedObjectArray,
                         self._registry_callback, queue_size=2)
        rospy.Subscriber("/rekpiper/objects/tracked_clouds", TrackedObjectCloudArray,
                         self._cloud_callback, queue_size=2)
        rospy.Subscriber("/joint_states_single", JointState,
                         self._joint_callback, queue_size=10)
        rospy.Subscriber("/arm_status", PiperStatusMsg,
                         self._arm_callback, queue_size=10)
        rospy.Service("/rekpiper/objects/begin_grasp", ObjectCommand,
                      self._begin_grasp)
        rospy.Service("/rekpiper/objects/begin_release", ObjectCommand,
                      self._begin_release)
        rospy.Service("/rekpiper/objects/cancel_grasp", ObjectCommand,
                      self._cancel_grasp)
        rospy.Service("/rekpiper/objects/confirm_attachment", ObjectCommand,
                      self._confirm_attachment)
        rospy.Service("/rekpiper/objects/confirm_release", ObjectCommand,
                      self._confirm_release)
        rospy.Service("/rekpiper/objects/resolve_ambiguous", ResolveObjectState,
                      self._resolve)
        rospy.Timer(rospy.Duration(0.05), self._tick)

    def _registry_callback(self, message):
        with self._lock:
            self._registry = {item.object_uuid: copy.deepcopy(item)
                              for item in message.objects}
            for identity in self._registry:
                self._machines[identity]

    def _cloud_callback(self, message):
        with self._lock:
            for item in message.objects:
                values = list(point_cloud2.read_points(
                    item.cloud_base, field_names=("x", "y", "z"), skip_nans=True))
                if values:
                    stamp = (item.header.stamp if item.header.stamp != rospy.Time(0)
                             else item.cloud_base.header.stamp)
                    self._clouds[item.object_uuid] = (
                        stamp, np.asarray(values, np.float32))

    def _joint_callback(self, message):
        try:
            index = message.name.index("gripper")
        except ValueError:
            return
        with self._lock:
            # The driver reports joint7 / one-finger travel.  Evidence and
            # contact thresholds use the physical gap between both fingers.
            # The Piper driver publishes grippers_angle / 1e6 as the total
            # jaw opening in metres.  It is not a single-finger displacement.
            self._gripper_opening = float(message.position[index])
            self._gripper_effort = abs(float(message.effort[index])) if len(message.effort) > index else 0.0
            self._joint_stamp = message.header.stamp or rospy.Time.now()

    def _arm_callback(self, message):
        with self._lock:
            self._arm_stamp = rospy.Time.now()
            if int(message.err_code) != 0:
                self._arm_error = "arm_err_code_{}".format(message.err_code)
            elif any((message.joint_1_angle_limit, message.joint_2_angle_limit,
                      message.joint_3_angle_limit, message.joint_4_angle_limit,
                      message.joint_5_angle_limit, message.joint_6_angle_limit)):
                self._arm_error = "joint_limit"
            elif any((message.communication_status_joint_1,
                      message.communication_status_joint_2,
                      message.communication_status_joint_3,
                      message.communication_status_joint_4,
                      message.communication_status_joint_5,
                      message.communication_status_joint_6)):
                # PiperStatusMsg uses True for a reported communication fault;
                # the normal hardware sample observed on this system is False.
                self._arm_error = "joint_communication_fault"
            else:
                self._arm_error = ""

    def _require_object(self, identity):
        if identity not in self._registry:
            raise ValueError("unknown active object_uuid")
        return self._machines[identity]

    def _begin_grasp(self, request):
        with self._lock:
            try:
                if not self._baseline_ready:
                    raise ValueError("calibrated gripper baseline is required")
                machine = self._require_object(request.object_uuid)
                state = machine.request_grasp()
                self._history[request.object_uuid].clear()
                self._gripper_history.clear()
                self._grasp_start_opening = self._gripper_opening
                return ObjectCommandResponse(True, int(state), machine.reason)
            except ValueError as exc:
                return ObjectCommandResponse(False, 0, str(exc))

    def _begin_release(self, request):
        with self._lock:
            try:
                machine = self._require_object(request.object_uuid)
                state = machine.request_release()
                self._history[request.object_uuid].clear()
                return ObjectCommandResponse(True, int(state), machine.reason)
            except ValueError as exc:
                return ObjectCommandResponse(False, 0, str(exc))

    def _cancel_grasp(self, request):
        with self._lock:
            try:
                machine = self._require_object(request.object_uuid)
                item = self._registry[request.object_uuid]
                status_by_camera = dict(zip(item.camera_names,
                                            item.camera_status))
                vision_free = all(status_by_camera.get(name) in (
                    TrackedObject.CAMERA_TRACKED,
                    TrackedObject.CAMERA_TRUSTED_NOT_VISIBLE)
                    for name in self._camera_names)
                gripper_open = self._gripper_opening > (
                    self._machines[request.object_uuid].baseline.
                    empty_closed_opening_p99_m + 0.010)
                state = machine.cancel_grasp(gripper_open, vision_free)
                self._history[request.object_uuid].clear()
                self._gripper_history.clear()
                self._grasp_start_opening = None
                return ObjectCommandResponse(
                    state == GraspState.FREE_TRACKED, int(state), machine.reason)
            except (ValueError, KeyError) as exc:
                return ObjectCommandResponse(False, 0, str(exc))

    def _confirm_attachment(self, request):
        with self._lock:
            try:
                machine = self._require_object(request.object_uuid)
                state = machine.confirm_attachment()
                self._freeze_attached_cloud(
                    request.object_uuid, rospy.Time.now())
                return ObjectCommandResponse(True, int(state), machine.reason)
            except Exception as exc:
                machine = self._machines.get(request.object_uuid)
                if machine is not None:
                    machine.fault("attachment_confirmation_failed:" + str(exc))
                return ObjectCommandResponse(False, 0, str(exc))

    def _confirm_release(self, request):
        with self._lock:
            try:
                machine = self._require_object(request.object_uuid)
                state = machine.confirm_release()
                self._clear_attached_cloud(rospy.Time.now())
                return ObjectCommandResponse(True, int(state), machine.reason)
            except (ValueError, KeyError) as exc:
                return ObjectCommandResponse(False, 0, str(exc))

    def _resolve(self, request):
        with self._lock:
            try:
                machine = self._require_object(request.object_uuid)
                machine.resolve(request.resolved_state, request.operator_note)
                return ResolveObjectStateResponse(True, machine.reason)
            except (ValueError, KeyError) as exc:
                return ResolveObjectStateResponse(False, str(exc))

    def _gripper_transform(self, stamp):
        message = self._tf.lookup_transform(
            self._base, self._gripper, stamp, rospy.Duration(0.05))
        base_from_gripper = transform_to_matrix(message.transform).astype(np.float32)
        return base_from_gripper, np.linalg.inv(base_from_gripper).astype(np.float32)

    def _evidence(self, identity, item, now):
        base_from_gripper, gripper_from_base = self._gripper_transform(now)
        center = np.asarray([item.pose_base.pose.position.x,
                             item.pose_base.pose.position.y,
                             item.pose_base.pose.position.z], np.float32)
        center_gripper = center @ gripper_from_base[:3, :3].T + gripper_from_base[:3, 3]
        tcp_position = base_from_gripper[:3, 3]
        rotation = base_from_gripper[:3, :3]
        # Matrix-to-quaternion is only used for an angular span; trace formula suffices.
        angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1, 1))))
        self._history[identity].append((now.to_sec(), center, center_gripper,
                                        tcp_position, angle))
        while self._history[identity] and now.to_sec() - self._history[identity][0][0] > 2.0:
            self._history[identity].popleft()
        history = list(self._history[identity])
        def span(index):
            values = np.asarray([entry[index] for entry in history])
            return float(np.linalg.norm(values.max(axis=0) - values.min(axis=0))) if len(values) else 0.0
        tcp_rotation = (max(entry[4] for entry in history) - min(entry[4] for entry in history)) if history else 0.0
        status_by_camera = dict(zip(item.camera_names, item.camera_status))
        vision_safe = (all(
            status_by_camera.get(name) in (
                TrackedObject.CAMERA_TRACKED,
                TrackedObject.CAMERA_TRUSTED_NOT_VISIBLE)
            for name in self._camera_names)
            and any(status_by_camera.get(name) == TrackedObject.CAMERA_TRACKED
                    for name in self._camera_names))
        gripper_history = list(self._gripper_history)
        if gripper_history:
            gripper_stable_duration = max(
                0.0, now.to_sec() - gripper_history[0][0])
            opening_values = [entry[1] for entry in gripper_history]
            gripper_opening_span = max(opening_values) - min(opening_values)
        else:
            gripper_stable_duration = 0.0
            gripper_opening_span = float("inf")
        closure_from_start = (0.0 if self._grasp_start_opening is None else
                              max(0.0, self._grasp_start_opening
                                  - self._gripper_opening))
        return GraspEvidence(
            self._gripper_opening, self._gripper_effort, vision_safe,
            float(np.linalg.norm(center_gripper)) <= self._region_radius,
            span(3), tcp_rotation, span(2), span(1),
            gripper_stable_duration, gripper_opening_span,
            closure_from_start)

    def _freeze_attached_cloud(self, identity, stamp):
        cached = self._clouds.get(identity)
        if cached is None:
            raise ValueError("attached object has no valid visual point cloud")
        cloud_stamp, cloud = cached
        if ((stamp - cloud_stamp).to_sec() > 0.30 or not len(cloud)):
            raise ValueError("attached object visual point cloud is stale")
        _, gripper_from_base = self._gripper_transform(stamp)
        local = cloud @ gripper_from_base[:3, :3].T + gripper_from_base[:3, 3]
        inflated = voxel_and_inflate(local)
        header = Header(stamp=stamp, frame_id=self._gripper)
        self._attached_cloud.publish(point_cloud2.create_cloud_xyz32(header, inflated))

    def _clear_attached_cloud(self, stamp):
        header = Header(stamp=stamp, frame_id=self._gripper)
        self._attached_cloud.publish(
            point_cloud2.create_cloud_xyz32(header, []))

    def _tick(self, _event):
        if self._release is not None:
            try:
                assert_release_unchanged(self._release)
            except AcceptanceError as exc:
                self._baseline_ready = False
                rospy.logfatal(
                    "gripper baseline release changed after validation: %s",
                    exc)
                rospy.signal_shutdown("signed gripper release changed")
                return
        now = rospy.Time.now()
        with self._lock:
            output = TrackedObjectArray()
            output.header.stamp = now
            output.header.frame_id = self._base
            output.status = "grasp_monitor"
            stale = ((now - self._joint_stamp).to_sec() > 0.3
                     or (now - self._arm_stamp).to_sec() > 0.3)
            if not stale:
                self._gripper_history.append((now.to_sec(),
                                              self._gripper_opening))
                history_window = 2.5
                while (self._gripper_history and now.to_sec()
                       - self._gripper_history[0][0] > history_window):
                    self._gripper_history.popleft()
            for identity, item in self._registry.items():
                # Registration publishes an object with UNKNOWN camera status
                # before the tracker receives and validates its seed.  That is
                # not a tracking loss, so the grasp monitor must not turn it
                # into LOST and race the registration service.
                if item.status == "initialization_pending":
                    continue
                machine = self._machines[identity]
                previous = machine.state
                try:
                    if stale:
                        machine.fault("robot_feedback_stale")
                    elif self._arm_error:
                        machine.fault(self._arm_error)
                    else:
                        machine.update(self._evidence(identity, item, now))
                    if previous != GraspState.ATTACHED and machine.state == GraspState.ATTACHED:
                        self._freeze_attached_cloud(identity, now)
                    if (previous != GraspState.FREE_TRACKED
                            and machine.state == GraspState.FREE_TRACKED):
                        self._clear_attached_cloud(now)
                except Exception as exc:
                    machine.fault("monitor_error:" + str(exc))
                update = TrackedObject()
                update.header = output.header
                update.object_uuid = identity
                update.rigid_group_id = item.rigid_group_id
                update.state = int(machine.state)
                update.status = machine.reason
                output.objects.append(update)
                if machine.state == GraspState.ATTACHED:
                    attached = copy.deepcopy(item)
                    attached.state = int(machine.state)
                    attached.status = machine.reason
                    self._attached_state.publish(attached)
                elif (previous != GraspState.FREE_TRACKED
                      and machine.state == GraspState.FREE_TRACKED):
                    released = copy.deepcopy(item)
                    released.state = int(machine.state)
                    released.status = machine.reason
                    self._attached_state.publish(released)
            self._updates.publish(output)


if __name__ == "__main__":
    rospy.init_node("grasp_state_monitor", anonymous=False)
    GraspMonitor()
    rospy.spin()
