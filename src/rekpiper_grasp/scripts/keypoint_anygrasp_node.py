#!/usr/bin/env python3
"""Asynchronous native AnyGrasp proposals around GPT-4o's selected K index."""

from copy import deepcopy
import threading

import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Header
from tf.transformations import quaternion_from_matrix
import tf2_ros

from rekpiper_grasp.anygrasp_adapter import AnyGraspAdapter
from rekpiper_grasp.target_geometry import target_region_mask
from rekpiper_grasp.horizontal_grasp import HorizontalGraspPolicy
from rekpiper_grasp.piper_gripper import PiperGripperGeometry
from rekpiper_planning.continuous_ik import interpolate_pose
from rekpiper_planning.contact_policy import TargetContactPolicy
from rekpiper_planning.trajectory_audit import audit_joint_path
from rekpiper_planning.motion_backend import MotionBackend
from rekpiper_grasp.grasp_geometry import anygrasp_to_piper_pose
from rekpiper_grasp.selection import (
    GraspBinding, horizon_requests_grasp, inference_request_key)
from rekpiper_perception.keypoint_tracking import capture_stamp_is_fresh
from rekpiper_msgs.msg import (
    ClosedLoopStatus, GraspCandidate, GraspCandidateArray,
    Keypoint3DArray, ReKepHorizon, ReKepProgram, SafeMappingStatus, SceneSnapshot,
    TrackedObject, TrackedObjectArray, TrackedObjectCloudArray, SDFGrid,
)
from rekpiper_msgs.srv import QuerySDF, QuerySDFRequest
from rekpiper_planning.piper_collision_sampling import PiperCollisionSampler
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver


JOINT_NAMES = ["joint{}".format(index) for index in range(1, 7)]


def _transform_matrix(value):
    q = value.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    result = np.eye(4)
    result[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    result[:3, 3] = [value.translation.x, value.translation.y, value.translation.z]
    return result


def _pose_message(matrix):
    from geometry_msgs.msg import Pose
    message = Pose()
    message.position.x, message.position.y, message.position.z = matrix[:3, 3]
    q = quaternion_from_matrix(matrix)
    message.orientation.x, message.orientation.y = q[:2]
    message.orientation.z, message.orientation.w = q[2:]
    return message


class KeypointAnyGraspNode:
    def __init__(self):
        self._lock = threading.RLock()
        self._worker = None
        self._snapshot = self._program = self._tracked = self._cloud = None
        self._horizon = None
        self._registry = self._map_status = None
        self._joints = None
        self._object_clouds = None
        self._joint_stamp = rospy.Time(0)
        self._sdf = None
        self._stage = 1
        self._grasp_attempt = 0
        self._last_request = None
        self._base_frame = "base_link"
        self._camera_frame = rospy.get_param(
            "~inference_frame", "rs1_color_optical_frame")
        self._points_topic = rospy.get_param(
            "~points_topic", "/rekpiper/camera/rs1/points_recognition")
        self._source_camera = rospy.get_param("~source_camera", "rs1")
        self._source_cameras = list(rospy.get_param("~source_cameras", [self._source_camera]))
        self._restrict_horizontal = bool(rospy.get_param("~restrict_horizontal", False))
        self._horizontal_angle_deg = float(rospy.get_param(
            "~maximum_horizontal_angle_deg", 10.0))
        if not 0 < self._horizontal_angle_deg <= 30:
            raise ValueError('invalid_horizontal_grasp_angle')
        self._maximum_candidates = int(rospy.get_param("~maximum_candidates", 40))
        self._minimum_clearance = float(rospy.get_param("~minimum_clearance_m", 0.01))
        if not rospy.has_param("/robot_description"):
            raise rospy.ROSInitException("Piper robot_description is unavailable")
        robot_xml = str(rospy.get_param("/robot_description"))
        self._ik = PiperURDFIKSolver.from_urdf_xml(
            robot_xml, self._base_frame, "rekep_tcp", JOINT_NAMES,
            position_tolerance=0.002, orientation_tolerance=0.03)
        self._gripper_geometry = PiperGripperGeometry(robot_xml, self._ik)
        self._adapter = AnyGraspAdapter(
            rospy.get_param("~anygrasp_sdk_root"),
            rospy.get_param("~anygrasp_checkpoint"),
            rospy.get_param("~anygrasp_license_dir"),
            max_gripper_width_m=self._gripper_geometry.maximum_opening_m,
            gripper_height_m=self._gripper_geometry.finger_height_m)
        self._backend = MotionBackend(robot_xml)
        self._collision_sampler = PiperCollisionSampler(
            robot_xml, self._ik,
            voxel_size_m=float(rospy.get_param("~collision_voxel_size_m", 0.025)),
            maximum_points_per_link=int(rospy.get_param(
                "~maximum_collision_points_per_link", 160)))
        self._query = rospy.ServiceProxy(
            "/rekpiper/mapping/query_sdf", QuerySDF, persistent=False)
        self._tf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self._listener = tf2_ros.TransformListener(self._tf)
        self._publisher = rospy.Publisher(
            "/rekpiper/grasp/candidates", GraspCandidateArray,
            queue_size=1, latch=True)
        rospy.Subscriber("/rekpiper/perception/scene_snapshot", SceneSnapshot,
                         self._set, callback_args="snapshot", queue_size=1)
        rospy.Subscriber("/rekpiper/program/current", ReKepProgram,
                         self._set, callback_args="program", queue_size=1)
        rospy.Subscriber("/rekpiper/tracking/keypoints", Keypoint3DArray,
                         self._set, callback_args="tracked", queue_size=1)
        rospy.Subscriber(self._points_topic, PointCloud2,
                         self._cloud_cb, queue_size=1)
        rospy.Subscriber("/joint_states_single", JointState,
                         self._joint_cb, queue_size=10)
        rospy.Subscriber("/rekpiper/execution/status", ClosedLoopStatus,
                         self._status_cb, queue_size=1)
        rospy.Subscriber("/rekpiper/planning/horizon", ReKepHorizon,
                         self._set, callback_args="horizon", queue_size=1)
        rospy.Subscriber("/rekpiper/objects/registry", TrackedObjectArray,
                         self._set, callback_args="registry", queue_size=1)
        rospy.Subscriber("/rekpiper/objects/tracked_clouds", TrackedObjectCloudArray,
                         self._set, callback_args="object_clouds", queue_size=1)
        rospy.Subscriber('/rekpiper/mapping/sdf_grid',SDFGrid,
                         self._set,callback_args='sdf',queue_size=1)
        rospy.Subscriber("/rekpiper/mapping/safe_status", SafeMappingStatus,
                         self._set, callback_args="map_status", queue_size=1)

    def _set(self, value, name):
        with self._lock:
            setattr(self, "_" + name, deepcopy(value))
        self._maybe_start()

    def _cloud_cb(self, value):
        points = np.asarray(list(point_cloud2.read_points(
            value, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float32)
        if points.ndim == 2 and points.shape[1] == 3:
            with self._lock:
                self._cloud = (deepcopy(value.header), points)
            self._maybe_start()

    def _joint_cb(self, value):
        mapping = dict(zip(value.name, value.position))
        if all(name in mapping for name in JOINT_NAMES):
            with self._lock:
                self._joints = np.asarray([mapping[name] for name in JOINT_NAMES])
                self._joint_stamp = value.header.stamp

    def _status_cb(self, value):
        with self._lock:
            self._stage = max(1, int(value.stage_index))
            self._grasp_attempt = int(value.grasp_attempt)
        self._maybe_start()

    def _maybe_start(self):
        with self._lock:
            if (self._worker is not None and self._worker.is_alive()) or any(
                    value is None for value in (
                        self._snapshot, self._program, self._tracked,
                        self._cloud, self._joints, self._registry,
                        self._map_status, self._horizon, self._object_clouds, self._sdf)):
                return
            if not self._program.approved or self._stage > self._program.num_stages:
                return
            if (not self._snapshot.valid
                    or not self._sdf.valid
                    or self._sdf.map_generation_uuid != self._map_status.map_generation_uuid
                    or self._snapshot.snapshot_id != self._program.snapshot_id
                    or self._map_status.state != SafeMappingStatus.READY
                    or not self._map_status.planning_safe
                    or not self._map_status.map_query_allowed
                    or not self._map_status.map_generation_uuid):
                return
            now = rospy.Time.now()
            if not all(capture_stamp_is_fresh(
                    now.to_nsec(), stamp.to_nsec(), 0.15) for stamp in (
                        self._tracked.header.stamp, self._cloud[0].stamp,
                        self._map_status.header.stamp, self._joint_stamp,
                        self._object_clouds.header.stamp)):
                return
            keypoint = int(self._program.grasp_keypoints[self._stage - 1])
            if keypoint < 0:
                return
            if keypoint >= len(self._tracked.keypoints):
                return
            group = int(self._tracked.keypoints[keypoint].rigid_group_id)
            objects = [item for item in self._registry.objects
                       if int(item.rigid_group_id) == group]
            if (group <= 0 or len(objects) != 1
                    or objects[0].state != TrackedObject.FREE_TRACKED):
                rospy.logerr_throttle(
                    1.0, "grasp K%d does not resolve to one FREE_TRACKED UUID",
                    keypoint)
                return
            target = objects[0]
            status_by_camera = dict(zip(target.camera_names,
                                        target.camera_status))
            if (not all(value in (
                    TrackedObject.CAMERA_TRACKED,
                    TrackedObject.CAMERA_TRUSTED_NOT_VISIBLE)
                    for value in status_by_camera.values())
                    or not any(value == TrackedObject.CAMERA_TRACKED
                               for value in status_by_camera.values())):
                return
            binding = GraspBinding(
                group, target.object_uuid, self._program.session_id,
                self._program.program_sha256, self._snapshot.snapshot_id,
                self._map_status.map_generation_uuid, self._stage,
                self._grasp_attempt)
            if not horizon_requests_grasp(self._horizon, binding):
                return
            request_key = inference_request_key(binding)
            if request_key == self._last_request:
                return
            self._last_request = request_key
            inputs = (deepcopy(self._snapshot), deepcopy(self._program),
                      deepcopy(self._tracked), (deepcopy(self._cloud[0]),
                      self._cloud[1].copy()), self._joints.copy(), keypoint,
                      target.object_uuid, self._program.session_id,
                      self._map_status.map_generation_uuid, self._stage,
                      self._grasp_attempt, deepcopy(self._object_clouds))
            self._worker = threading.Thread(
                target=self._infer, args=inputs, daemon=True)
            self._worker.start()

    def _query_sdf(self, points, radii):
        from geometry_msgs.msg import Point
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        radii = np.asarray(radii, dtype=float).reshape(-1)
        response = self._query(QuerySDFRequest(
            header=Header(frame_id="base_link"),
            points=[Point(*value.tolist()) for value in points],
            radii_m=radii.tolist()))
        if (not response.map_valid
                or response.map_generation_uuid != self._map_status.map_generation_uuid
                or len(response.distances_m) != len(points)):
            return False, 0.0
        observed = np.asarray(response.observed, dtype=bool)
        clearance = -np.asarray(response.distances_m) - radii
        return (bool(np.all(observed)
                     and np.min(clearance) >= self._minimum_clearance),
                float(np.min(clearance)))

    def _sdf_audit(self, candidate):
        approach = np.linspace(candidate.pregrasp_pose[:3, 3],
                               candidate.grasp_pose[:3, 3], 8)
        # AnyGrasp was invoked with collision_detection=True over the RS1
        # scene.  The final samples are intentional target contact, so the ESDF
        # separately audits only the pre-contact portion of the approach.
        samples = approach[:-2]
        return self._query_sdf(samples, np.full(len(samples), 0.008))

    def _candidate_message(self, candidate, header, rigid_group_id,
                           object_uuid, session_id, program, snapshot,
                           map_generation_uuid, stage_index, grasp_attempt):
        from geometry_msgs.msg import Point
        return GraspCandidate(
            header=header, candidate_id=candidate.candidate_id,
            candidate_origin=candidate.candidate_origin,
            interaction_region_id=candidate.interaction_region_id,
            part_name=candidate.part_name,
            rigid_group_id=int(rigid_group_id),
            object_uuid=object_uuid,
            session_id=session_id,
            program_sha256=program.program_sha256,
            snapshot_id=snapshot.snapshot_id,
            map_generation_uuid=map_generation_uuid,
            stage_index=int(stage_index),
            grasp_attempt=int(grasp_attempt),
            grasp_pose=_pose_message(candidate.grasp_pose),
            pregrasp_pose=_pose_message(candidate.pregrasp_pose),
            tcp_position=Point(*candidate.tcp_position.tolist()),
            predicted_width_m=candidate.predicted_width_m,
            suggested_preopen_width_m=candidate.suggested_preopen_width_m,
            insertion_depth_m=candidate.insertion_depth_m,
            network_score=candidate.network_score,
            joint_motion_cost=getattr(candidate,'joint_motion_cost',float('inf')),
            source_cameras=candidate.source_cameras,
            cross_view_center_error_m=0.0,
            cross_view_orientation_error_rad=0.0,
            cross_view_width_error_m=0.0,
            part_membership_ok=candidate.part_membership_ok,
            contact_membership_ok=candidate.contact_membership_ok,
            contact_points=[Point(*value) for value in candidate.contact_points_base],
            width_ok=candidate.width_ok,
            finger_collision_free=candidate.finger_collision_free,
            approach_clear=candidate.approach_clear,
            ik_ok=candidate.ik_ok,
            robot_collision_free=candidate.robot_collision_free,
            esdf_clear=candidate.esdf_clear,
            cross_view_consistent=False,
            perception_valid=candidate.perception_valid,
            planning_safe=candidate.planning_safe,
            planning_authorized=False,
            rejection_reasons=sorted(set(candidate.rejection_reasons)))

    def _infer(self, snapshot, program, tracked, cloud, joints, keypoint_index,
               object_uuid, session_id, map_generation_uuid, stage_index,
               grasp_attempt, object_clouds):
        try:
            header, points_base = cloud
            # These clouds originate from current tracked instance masks.
            owned, others = [], []
            for item in object_clouds.objects:
                if (item.cloud_base.header.frame_id != self._base_frame
                        or abs((item.cloud_base.header.stamp-header.stamp).to_sec()) > .15):
                    raise RuntimeError('object_cloud_frame_or_stamp_invalid')
                points = np.asarray(list(point_cloud2.read_points(
                    item.cloud_base, field_names=('x','y','z'), skip_nans=True)))
                (owned if item.object_uuid == object_uuid else others).append(points)
            if not owned:
                raise RuntimeError('target_mask_cloud_missing')
            object_points = np.vstack(owned)
            mask = target_region_mask(points_base, object_points,
                                      np.vstack(others) if others else None)
            group = int(tracked.keypoints[keypoint_index].rigid_group_id)
            transform = self._tf.lookup_transform(
                self._base_frame, self._camera_frame, rospy.Time(0), rospy.Duration(0.5))
            base_from_camera = _transform_matrix(transform.transform)
            camera_from_base = np.linalg.inv(base_from_camera)
            points_camera = (points_base @ camera_from_base[:3, :3].T
                             + camera_from_base[:3, 3]).astype(np.float32)
            geometry_policy = HorizontalGraspPolicy(
                points_base, mask, base_from_camera[:3, 3], self._gripper_geometry,
                maximum_horizontal_angle_deg=self._horizontal_angle_deg,
                restrict_horizontal=self._restrict_horizontal)
            raw = []
            for direction_camera in geometry_policy.directions_in_camera(base_from_camera):
                raw.extend(self._adapter.infer(
                    points_camera, geometry_policy.region_mask, self._source_camera,
                    approach_steering=direction_camera,
                    approach_thresh_rad=(np.deg2rad(self._horizontal_angle_deg)
                                         if self._restrict_horizontal else np.pi),
                    max_candidates=self._maximum_candidates, dense_grasp=True))
            with self._lock:
                grid = deepcopy(self._sdf)
            if (not grid.valid or grid.map_generation_uuid != map_generation_uuid
                    or not capture_stamp_is_fresh(rospy.Time.now().to_nsec(),grid.header.stamp.to_nsec(),.15)):
                raise RuntimeError('grasp_audit_requires_fresh_safe_grid')
            keypoint = tracked.keypoints[keypoint_index].position
            anchor = np.array([keypoint.x, keypoint.y, keypoint.z])
            audited = []
            for index, item in enumerate(raw):
                candidate = anygrasp_to_piper_pose(
                    item, base_from_camera,
                    candidate_id="K{}-{:03d}".format(keypoint_index, index),
                    interaction_region_id="rigid_group_{}".format(group),
                    part_name="GPT4o_selected_K{}".format(keypoint_index),
                    physical_opening_m=self._gripper_geometry.maximum_opening_m)
                candidate.source_cameras = list(self._source_cameras)
                geometry_audit = geometry_policy.audit(candidate)
                if not geometry_audit['valid']:
                    candidate.rejection_reasons.extend(
                        'grasp_geometry:'+reason for reason in geometry_audit['rejection_reasons'])
                    audited.append(candidate)
                    continue
                half = 0.5 * candidate.predicted_width_m
                contacts = [candidate.tcp_position - candidate.closing_axis_base * half,
                            candidate.tcp_position + candidate.closing_axis_base * half]
                candidate.contact_points_base = [value.tolist() for value in contacts]
                nearest = [float(np.min(np.linalg.norm(object_points-value, axis=1)))
                           for value in contacts]
                candidate.contact_membership_ok = max(nearest) <= 0.020
                candidate.part_membership_ok = np.linalg.norm(candidate.tcp_position-anchor) <= 0.10
                try:
                    pre = self._ik.solve(candidate.pregrasp_pose,
                                         initial_joint_pos=joints, max_iterations=200)
                    final = self._ik.solve(candidate.grasp_pose,
                                           initial_joint_pos=pre.cspace_position, max_iterations=200)
                    candidate.ik_ok = bool(pre.success and final.success)
                    if candidate.ik_ok:
                        poses = []
                        for a,b in ((self._ik.forward(joints), candidate.pregrasp_pose),
                                    (candidate.pregrasp_pose,candidate.grasp_pose)):
                            count = max(2, int(np.ceil(np.linalg.norm(b[:3,3]-a[:3,3])/.005))+1)
                            poses.extend(interpolate_pose(a,b,t) for t in np.linspace(0,1,count))
                        continuous = self._ik.validate_pose_sequence(poses,joints)
                        candidate.ik_ok = continuous['valid']
                        candidate.joint_motion_cost = (sum(s['maximum_joint_step_rad']
                            for s in continuous['steps']) if continuous['valid'] else float('inf'))
                    if candidate.ik_ok:
                        samples = []
                        radii = []
                        endpoint_links = tuple(
                            name for name in self._collision_sampler.sampled_links
                            if name not in ("link6", "gripper_base"))
                        for solution, links in (
                                (pre.cspace_position, None),
                                (final.cspace_position, endpoint_links)):
                            sampled, sampled_radii = self._collision_sampler.samples(
                                solution[:6], links=links)
                            samples.append(sampled)
                            radii.append(sampled_radii)
                        candidate.robot_collision_free, _ = self._query_sdf(
                            np.vstack(samples), np.concatenate(radii))
                    else:
                        candidate.robot_collision_free = False
                except Exception:
                    candidate.ik_ok = candidate.robot_collision_free = False
                candidate.esdf_clear, clearance = self._sdf_audit(candidate)
                candidate.approach_clear = candidate.esdf_clear
                # The licensed SDK returns candidates only after its RS1
                # point-cloud collision_detection pass.
                candidate.finger_collision_free = True
                if candidate.ik_ok:
                    try:
                        from geometry_msgs.msg import Point
                        from types import SimpleNamespace
                        policy = TargetContactPolicy(self._ik,SimpleNamespace(
                            contact_points=[Point(*v) for v in contacts]),object_points,points_base,
                            [v[0] for v in self._collision_sampler._finger_samples],
                            candidate.suggested_preopen_width_m,candidate.grasp_pose,
                            other_object_points=np.vstack(others) if others else None)
                        qs = np.array([s['joint_positions'] for s in continuous['steps']])
                        if not all(self._backend.self_clear(q,policy.opening_m) for q in qs):
                            raise RuntimeError('grasp_path_self_collision')
                        audit_joint_path(qs,self._collision_sampler,grid,contact_policy=policy)
                        candidate.robot_collision_free = candidate.esdf_clear = True
                        candidate.approach_clear = candidate.finger_collision_free = True
                    except Exception as exc:
                        candidate.robot_collision_free = candidate.esdf_clear = False
                        candidate.approach_clear = candidate.finger_collision_free = False
                        candidate.rejection_reasons.append('full_grasp_path:'+str(exc))
                candidate.perception_valid = bool(candidate.part_membership_ok
                                                  and candidate.contact_membership_ok)
                candidate.planning_safe = all((
                    candidate.width_ok, candidate.finger_collision_free,
                    candidate.approach_clear, candidate.ik_ok,
                    candidate.robot_collision_free, candidate.esdf_clear,
                    candidate.perception_valid))
                checks = {
                    "contact_membership": candidate.contact_membership_ok,
                    "keypoint_membership": candidate.part_membership_ok,
                    "width": candidate.width_ok, "ik": candidate.ik_ok,
                    "robot_collision": candidate.robot_collision_free,
                    "esdf": candidate.esdf_clear,
                }
                candidate.rejection_reasons.extend(
                    name for name, passed in checks.items() if not passed)
                audited.append(candidate)
            audited.sort(key=lambda item: (
                not item.planning_safe,
                getattr(item,'joint_motion_cost',float('inf')),
                -item.network_score, item.candidate_id))
            with self._lock:
                if (self._map_status.map_generation_uuid != map_generation_uuid
                        or self._program.program_sha256 != program.program_sha256
                        or self._snapshot.snapshot_id != snapshot.snapshot_id
                        or self._stage != stage_index or self._grasp_attempt != grasp_attempt
                        or not capture_stamp_is_fresh(rospy.Time.now().to_nsec(),
                                                      self._tracked.header.stamp.to_nsec(), .15)
                        or np.max(np.abs(self._joints-joints)) > .01):
                    raise RuntimeError('anygrasp_result_stale_or_binding_changed')
                latest = self._tracked.keypoints[keypoint_index].position
                if np.linalg.norm(np.array([latest.x,latest.y,latest.z])-anchor) > .005:
                    raise RuntimeError('anygrasp_target_moved_during_inference')
                # Keep capture provenance in header; execution rebuilds contact
                # checks from fresh mask-derived clouds before any movement.
            messages = [self._candidate_message(
                item, header, group, object_uuid, session_id, program,
                snapshot, map_generation_uuid, stage_index, grasp_attempt)
                for item in audited]
            safe_available = any(item.planning_safe for item in audited)
            self._publisher.publish(GraspCandidateArray(
                header=header, instruction="stage {} grasp K{}".format(
                    stage_index, keypoint_index),
                session_id=session_id, program_sha256=program.program_sha256,
                snapshot_id=snapshot.snapshot_id,
                map_generation_uuid=map_generation_uuid,
                stage_index=stage_index, grasp_attempt=grasp_attempt,
                candidates=messages,
                status=("safe_anygrasp_candidates_available" if safe_available
                        else "no_safe_anygrasp_candidates"),
                planning_authorized=False))
        except Exception as exc:
            rospy.logerr("AnyGrasp generation failed closed: %s", exc)
            self._publisher.publish(GraspCandidateArray(
                header=cloud[0], session_id=session_id,
                program_sha256=program.program_sha256,
                snapshot_id=snapshot.snapshot_id,
                map_generation_uuid=map_generation_uuid,
                stage_index=stage_index, grasp_attempt=grasp_attempt,
                status="{}:{}".format(type(exc).__name__, exc),
                planning_authorized=False))


if __name__ == "__main__":
    rospy.init_node("keypoint_anygrasp")
    KeypointAnyGraspNode()
    rospy.spin()
