#!/usr/bin/env python3
"""Program-approved, short-horizon ReKep closed-loop coordinator."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import actionlib
from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from geometry_msgs.msg import Pose, PoseArray
import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Bool, Header
from std_srvs.srv import Trigger, TriggerResponse
import tf2_ros
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rekpiper_msgs.msg import (
    ClosedLoopStatus, CommandGripperAction, CommandGripperGoal,
    ExecuteReKepAction, ExecuteReKepFeedback,
    ExecuteReKepResult, GraspCandidate, GraspCandidateArray, Keypoint3DArray,
    ReKepHorizon,
    ReKepProgram, SafeMappingStatus, SDFGrid, SceneSnapshot, TrackedObject,
    TrackedObjectArray, TrackedObjectCloudArray,
)
from rekpiper_msgs.srv import ObjectCommand, RebuildMap
from rekpiper_acceptance import (
    AcceptanceError, assert_release_unchanged, validate_program_approval,
    validate_release_bundle)
from rekpiper_planning import (
    PersistentReKepPlanner, PlanningGeneration, RealtimePlanningRequest,
    SolverAcceptanceError, TrajectoryAuditError, audit_joint_path,
    load_acceptance, load_workspace_table_height,
)
from rekpiper_planning.solver_acceptance import (
    paper_real_source_sha256, sha256_json, validate_acceptance)
from rekpiper_planning.upstream import EXPECTED_OFFICIAL_COMMIT
from rekpiper_planning.official_program import load_official_stage_constraints
from rekpiper_planning.realtime_planner import grasp_target_constraints
from rekpiper_planning.official_program import (
    LOCAL_SAFETY_CONTRACT_VERSION, OFFICIAL_PROMPT_SHA256,
    compute_program_sha256)
from rekpiper_planning.piper_collision_sampling import PiperCollisionSampler
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver
from rekpiper_planning.motion_backend import MotionBackend
from rekpiper_mapping.sdf_conventions import inclusive_grid_axes
from rekpiper_execution.planning_worker import (
    PlanningWorker, validate_returned_context, solve_with_history, restore_history)
from rekpiper_execution.contact_policy import (
    TargetContactPolicy, validate_preopen, validate_support_geometry)
from rekpiper_execution.trajectory import (
    TrajectoryValidationError, normalize_feedback_to_joint_limits,
    short_horizon_joint_prefix, smooth_pchip_trajectory, validate_trajectory)
from rekpiper_execution.grasp_approach import validate_path_constraints
from rekpiper_execution.coordinator import ReKepCoordinatorCore, pose_reached
from rekpiper_execution.lifecycle_contract import (
    grasp_batch_matches_binding,
    validate_grasp_candidate_binding)
from rekpiper_execution.rate_monitor import EventRate
from rekpiper_execution.runtime_acceptance import (
    RuntimeAcceptanceError, load_runtime_acceptance, validate_evidence)


JOINT_NAMES = ["joint{}".format(index) for index in range(1, 7)]


def _pose_message(values):
    message = Pose()
    message.position.x, message.position.y, message.position.z = values[:3]
    message.orientation.x, message.orientation.y = values[3], values[4]
    message.orientation.z, message.orientation.w = values[5], values[6]
    return message


def _grasp_matrix(candidate):
    pose = candidate.grasp_pose
    q = np.asarray([pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w], dtype=float)
    if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-9:
        raise ValueError("AnyGrasp quaternion is invalid")
    x, y, z, w = q / np.linalg.norm(q)
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return matrix


def _pose_matrix(pose):
    wrapper = type("PoseWrapper", (), {"grasp_pose": pose})()
    return _grasp_matrix(wrapper)


def _ee_matrix(values):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = values[:3]
    pose.orientation.x, pose.orientation.y = values[3], values[4]
    pose.orientation.z, pose.orientation.w = values[5], values[6]
    return _pose_matrix(pose)


class ClosedLoopNode:
    def __init__(self):
        self._lock = threading.RLock()
        self._planning_lock = threading.Lock()
        self._mode = str(rospy.get_param("~mode", "shadow"))
        if self._mode not in ("shadow", "autonomous"):
            raise rospy.ROSInitException("mode must be shadow or autonomous")
        self._allow_commands = bool(rospy.get_param("~allow_hardware_commands", False))
        self._acceptance_bundle_type = os.environ.get(
            "REKPIPER_ACCEPTANCE_BUNDLE_TYPE", "release_bundle")
        if self._acceptance_bundle_type == "hardware_acceptance_bundle":
            # Bootstrap acceptance only authorizes the dedicated hold bridge.
            self._allow_commands = False
        if self._mode != "autonomous" and self._allow_commands:
            raise rospy.ROSInitException("hardware commands are legal only in autonomous mode")
        self._rate_hz = float(rospy.get_param("~planning_rate_hz", 10.0))
        self._tracking_max_age = float(rospy.get_param("~tracking_max_age_s", 0.15))
        self._map_max_age = float(rospy.get_param("~map_max_age_s", 0.15))
        self._deadline = float(rospy.get_param("~planning_deadline_s", 0.20))
        self._maximum_deadline_misses = int(rospy.get_param("~maximum_deadline_misses", 3))
        self._tolerance = float(rospy.get_param(
            "~backtracking_constraint_tolerance", 0.10))
        self._goal_position_tolerance = float(rospy.get_param(
            "~goal_position_tolerance_m", 0.01))
        self._goal_rotation_tolerance = float(rospy.get_param(
            "~goal_rotation_tolerance_rad", 0.10))
        self._success_ticks_required = int(rospy.get_param("~success_ticks_required", 3))
        self._joint_velocity = float(rospy.get_param("~maximum_velocity_rad_s", 0.30))
        self._joint_feedback_boundary_tolerance = float(rospy.get_param(
            "/rekpiper/execution/joint_feedback_boundary_tolerance_rad",
            0.01))
        self._base_frame = str(rospy.get_param(
            "~system/base_frame", "base_link"))
        self._tip_frame = str(rospy.get_param(
            "~system/tip_frame", "rekep_tcp"))
        reset_joints = np.asarray(rospy.get_param(
            "~system/reset_joint_positions",
            [0.0, 1.57, -1.40, 0.0, 0.0, 0.0, 0.0]), dtype=float)
        if reset_joints.shape != (7,) or not np.all(np.isfinite(reset_joints)):
            raise rospy.ROSInitException(
                "system/reset_joint_positions must contain seven finite values")

        description_file = str(rospy.get_param("~robot_description_file", ""))
        if rospy.has_param("/robot_description"):
            robot_xml = str(rospy.get_param("/robot_description"))
        elif description_file:
            robot_xml = Path(description_file).read_text(encoding="utf-8")
        else:
            raise rospy.ROSInitException("Piper robot_description is unavailable")
        self._ik = PiperURDFIKSolver.from_urdf_xml(
            robot_xml, self._base_frame, self._tip_frame, JOINT_NAMES)
        self._robot_xml = robot_xml
        self._motion_backend = None
        self._planning_worker = PlanningWorker(float(rospy.get_param('~cold_plan_timeout_s',60.)))
        rospy.on_shutdown(self._planning_worker.cancel)
        self._joint_stamp = rospy.Time(0)
        self._gripper_opening_m = None
        self._object_clouds = None
        self._scene_cloud = None
        self._contact_policy = None
        self._preopened_grasp_key = None
        self._approach_opening_m = None
        self._grasp_target_key = None
        self._grasp_target_anchor = None
        self._release_probe = False
        self._grasp_may_hold = False
        self._collision_sampler = PiperCollisionSampler(
            robot_xml, self._ik,
            voxel_size_m=float(rospy.get_param("~collision_voxel_size_m", 0.025)),
            maximum_points_per_link=int(rospy.get_param(
                "~maximum_collision_points_per_link", 160)))
        self._paper_real_arm_links = tuple(
            name for name in self._collision_sampler.sampled_links
            if name not in ("base_link", "link6", "gripper_base"))
        if not self._paper_real_arm_links:
            raise rospy.ROSInitException(
                "full-arm table collision samples are unavailable")
        subgoal_config = rospy.get_param("~subgoal_solver")
        path_config = rospy.get_param("~path_solver")
        self._solver_profile = str(rospy.get_param(
            "~solver_profile", "official_exact"))
        self._paper_real_acceptance_path = str(rospy.get_param(
            "~paper_real_acceptance", ""))
        self._paper_real_expected_inputs = None
        self._runtime_acceptance_path = str(rospy.get_param(
            "~runtime_performance_acceptance", ""))
        self._runtime_software_acceptance_path = str(rospy.get_param(
            "~runtime_performance_software_acceptance",
            self._runtime_acceptance_path))
        self._runtime_hardware_acceptance_path = str(rospy.get_param(
            "~runtime_performance_hardware_acceptance", ""))
        self._release_bundle_path = str(rospy.get_param("~release_bundle", ""))
        self._acceptance_public_key = str(rospy.get_param(
            "~acceptance_public_key", ""))
        self._minimum_release_counter = str(rospy.get_param(
            "~minimum_release_counter", ""))
        self._robot_id = str(rospy.get_param("~robot_id", ""))
        self._release = None
        self._program_approval_guard = None
        paper_weights = None
        table_height = None
        if self._solver_profile == "paper_real":
            workspace_path = str(rospy.get_param("~workspace_config", ""))
            try:
                if self._mode == "autonomous":
                    self._release = validate_release_bundle(
                        self._release_bundle_path, self._acceptance_public_key,
                        self._minimum_release_counter, self._robot_id)
                    artifacts = self._release["verified_artifacts"]
                    workspace_artifact = artifacts["workspace"]
                    workspace_values = workspace_artifact["payload"]
                    if (workspace_values.get("status") != "ACCEPTED"
                            or not bool(workspace_values.get(
                                "precision_operation_allowed", False))):
                        raise SolverAcceptanceError(
                            "signed workspace is not accepted")
                    table_height = float(workspace_values["table_height_m"])
                    workspace_sha = workspace_artifact["payload_sha256"]
                    paper_weights_value = validate_acceptance(
                        artifacts["paper_real_solver"]["payload"],
                        require_approved=True)
                else:
                    table_height, workspace_sha = load_workspace_table_height(
                        workspace_path)
                    _candidate, paper_weights_value = load_acceptance(
                        self._paper_real_acceptance_path,
                        require_approved=False)
            except (AcceptanceError, KeyError, TypeError, ValueError,
                    SolverAcceptanceError) as exc:
                raise rospy.ROSInitException(str(exc))
            paper_weights = {
                "subgoal_consistency": paper_weights_value.subgoal_consistency,
                "path_consistency": paper_weights_value.path_consistency,
                "table_clearance": paper_weights_value.table_clearance,
            }
            self._paper_real_expected_inputs = {
                "source_tree_sha256": paper_real_source_sha256(),
                "workspace_sha256": workspace_sha,
                "robot_description_sha256": hashlib.sha256(
                    robot_xml.encode("utf-8")).hexdigest(),
                "solver_config_sha256": sha256_json({
                    "subgoal_solver": subgoal_config,
                    "path_solver": path_config,
                    "official_commit": EXPECTED_OFFICIAL_COMMIT,
                }),
            }
        self._planner = PersistentReKepPlanner(
            subgoal_config, path_config,
            self._ik, reset_joints,
            continuity_guard_enabled=bool(rospy.get_param(
                "~continuity_guard_enabled", True)),
            maximum_position_step_m=0.005,
            maximum_rotation_step_rad=np.deg2rad(1.0),
            solver_profile=self._solver_profile,
            paper_real_weights=paper_weights,
            table_height_m=table_height,
            robot_collision_points_fn=self._paper_real_robot_points)

        self._tf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf)
        self._program = self._snapshot = self._tracked = self._sdf = None
        self._registry = self._map_status = None
        self._joints = self._grasp = self._grasp_batch = None
        self._raw_joints = None
        self._joint_feedback_error = "joint_feedback_missing"
        self._attached_collision_points = np.empty((0, 3), dtype=float)
        self._state_sequence = 0
        self._armed = self._paused = self._active = False
        self._stage = 1
        self._held_keypoint = -1
        self._held_group_id = 0
        self._attached_grasp = None
        self._grasp_failures = {}
        self._grasp_attempt = 0
        self._coordinator = None
        self._success_ticks = self._deadline_misses = self._backtracks = 0
        self._tick_count = 0
        self._reason = "initialized_disarmed"
        self._task_finished = None
        self._last_plan_time = 0.0
        self._last_plan_latency = 0.0
        self._last_tracking_stamp = rospy.Time(0)
        self._last_map_stamp = rospy.Time(0)
        self._tracking_rate = EventRate()
        self._mapping_rate = EventRate()
        self._planning_rate = EventRate()

        self._horizon_pub = rospy.Publisher(
            "/rekpiper/planning/horizon", ReKepHorizon, queue_size=1)
        self._status_pub = rospy.Publisher(
            "/rekpiper/execution/status", ClosedLoopStatus, queue_size=1, latch=True)
        self._armed_pub = rospy.Publisher(
            "/rekpiper/execution/hardware_armed", Bool, queue_size=1, latch=True)
        rospy.Subscriber("/rekpiper/program/current", ReKepProgram,
                         self._program_cb, queue_size=1)
        rospy.Subscriber("/rekpiper/perception/scene_snapshot", SceneSnapshot,
                         self._snapshot_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param(
            "~tracked_keypoints_topic", "/rekpiper/tracking/keypoints"),
            Keypoint3DArray, self._tracked_cb, queue_size=1)
        rospy.Subscriber("/rekpiper/mapping/sdf_grid", SDFGrid,
                         self._sdf_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param("~joint_state_topic", "/joint_states_single"),
                         JointState, self._joint_cb, queue_size=10)
        rospy.Subscriber("/rekpiper/grasp/selected", GraspCandidate,
                         self._grasp_cb, queue_size=1)
        rospy.Subscriber("/rekpiper/grasp/candidates", GraspCandidateArray,
                         self._grasp_batch_cb, queue_size=1)
        rospy.Subscriber("/rekpiper/objects/attached_collision_cloud", PointCloud2,
                         self._attached_cloud_cb, queue_size=1)
        rospy.Subscriber("/rekpiper/objects/registry", TrackedObjectArray,
                         self._registry_cb, queue_size=1)
        rospy.Subscriber('/rekpiper/objects/tracked_clouds',TrackedObjectCloudArray,
                         self._object_clouds_cb,queue_size=1)
        rospy.Subscriber('/rekpiper/camera/fused/points_base',PointCloud2,
                         self._scene_cloud_cb,queue_size=1)
        rospy.Subscriber("/rekpiper/mapping/safe_status", SafeMappingStatus,
                         self._map_status_cb, queue_size=1)
        self._begin_grasp = rospy.ServiceProxy(
            "/rekpiper/objects/begin_grasp", ObjectCommand)
        self._cancel_grasp = rospy.ServiceProxy(
            "/rekpiper/objects/cancel_grasp", ObjectCommand)
        self._begin_release = rospy.ServiceProxy(
            "/rekpiper/objects/begin_release", ObjectCommand)
        self._confirm_attachment = rospy.ServiceProxy(
            "/rekpiper/objects/confirm_attachment", ObjectCommand)
        self._confirm_release = rospy.ServiceProxy(
            "/rekpiper/objects/confirm_release", ObjectCommand)
        self._rebuild_map = rospy.ServiceProxy(
            "/rekpiper/mapping/rebuild_safe_tsdf", RebuildMap)
        self._trajectory_client = actionlib.SimpleActionClient(
            rospy.get_param("~trajectory_action",
                            "/manipulator_controller/follow_joint_trajectory"),
            FollowJointTrajectoryAction)
        self._gripper_client = actionlib.SimpleActionClient(
            "/rekpiper/execution/command_gripper", CommandGripperAction)
        self._server = actionlib.SimpleActionServer(
            "/rekpiper/execution/execute_rekep", ExecuteReKepAction,
            execute_cb=self._execute, auto_start=False)
        self._server.start()
        rospy.Service("/rekpiper/execution/arm", Trigger, self._arm)
        rospy.Service("/rekpiper/execution/pause", Trigger, self._pause)
        rospy.Service("/rekpiper/execution/resume", Trigger, self._resume)
        rospy.Service("/rekpiper/execution/abort", Trigger, self._abort_service)
        rospy.Timer(rospy.Duration(1.0 / self._rate_hz), self._tick)
        self._publish_status()

    def _paper_real_robot_points(self, joints):
        """Return moving arm samples; EE/attachment samples are passed separately."""
        return self._collision_sampler.samples(
            joints, links=self._paper_real_arm_links)[0]

    def _program_cb(self, value):
        with self._lock:
            if self._program and self._program.program_sha256 != value.program_sha256:
                self._safe_hold("program_generation_changed")
                self._planner.invalidate()
                self._grasp = self._grasp_batch = None
            self._program = deepcopy(value)

    def _snapshot_cb(self, value):
        with self._lock:
            if self._snapshot and self._snapshot.snapshot_id != value.snapshot_id and self._active:
                self._safe_hold("task_snapshot_changed")
                self._planner.invalidate()
                self._grasp = self._grasp_batch = None
            self._snapshot = deepcopy(value)

    def _tracked_cb(self, value):
        with self._lock:
            self._tracked = deepcopy(value)
            self._last_tracking_stamp = value.header.stamp
            self._tracking_rate.mark(time.monotonic())

    def _sdf_cb(self, value):
        with self._lock:
            self._sdf = deepcopy(value)
            self._last_map_stamp = value.header.stamp
            self._mapping_rate.mark(time.monotonic())

    def _joint_cb(self, value):
        mapping = dict(zip(value.name, value.position))
        if any(name not in mapping for name in JOINT_NAMES):
            return
        raw = np.asarray([mapping[name] for name in JOINT_NAMES], dtype=float)
        try:
            normalized = normalize_feedback_to_joint_limits(
                raw, self._joint_feedback_boundary_tolerance)
            error = ""
        except TrajectoryValidationError as exc:
            normalized = None
            error = str(exc)
        with self._lock:
            self._raw_joints = raw
            self._joints = normalized
            self._joint_feedback_error = error
            self._state_sequence += 1
            self._joint_stamp = value.header.stamp
            width=mapping.get('gripper')
            self._gripper_opening_m = (float(np.clip(width,0.,.070))
                if width is not None and np.isfinite(width) and -.003 <= width <= .073 else None)
            if error and self._active:
                self._safe_hold("joint_feedback_invalid")
        if error:
            rospy.logerr_throttle(1.0, "Piper feedback rejected: %s", error)

    def _grasp_cb(self, value):
        with self._lock:
            fields = ('session_id', 'program_sha256', 'snapshot_id',
                      'map_generation_uuid', 'stage_index', 'grasp_attempt')
            if self._grasp is not None and all(
                    getattr(self._grasp, name) == getattr(value, name) for name in fields):
                return  # Freeze the first selected target for this attempt.
            self._grasp = deepcopy(value)

    def _grasp_batch_cb(self, value):
        with self._lock:
            self._grasp_batch = deepcopy(value)

    def _attached_cloud_cb(self, value):
        if value.header.frame_id != "gripper_base":
            rospy.logwarn_throttle(
                2.0, "attached collision cloud must be in gripper_base")
            return
        points = np.asarray(list(point_cloud2.read_points(
            value, field_names=("x", "y", "z"), skip_nans=True)), dtype=float)
        if points.size == 0:
            points = np.empty((0, 3), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            return
        with self._lock:
            self._attached_collision_points = points

    def _registry_cb(self, value):
        with self._lock:
            self._registry = deepcopy(value)

    def _map_status_cb(self, value):
        with self._lock:
            if (self._map_status is not None
                    and self._map_status.map_generation_uuid
                    != value.map_generation_uuid):
                self._planner.invalidate()
                self._grasp = self._grasp_batch = None
            self._map_status = deepcopy(value)

    def _preflight(self):
        if self._program is None or not self._program.approved:
            return False, "approved_program_missing"
        if (compute_program_sha256(self._program.program_directory)
                != self._program.program_sha256):
            return False, "approved_program_hash_changed"
        if self._mode == "autonomous":
            try:
                if self._release is not None:
                    assert_release_unchanged(self._release)
                self._program_approval_guard = validate_program_approval(
                    str(Path(self._program.program_directory)
                        / "program_approval.yaml"),
                    self._acceptance_public_key, self._program.session_id,
                    self._program.snapshot_id, self._program.program_sha256,
                    OFFICIAL_PROMPT_SHA256,
                    LOCAL_SAFETY_CONTRACT_VERSION)
            except (AcceptanceError, OSError, ValueError) as exc:
                return False, "signed_program_approval_rejected:" + str(exc)
            if self._solver_profile != "paper_real":
                return False, "paper_real_solver_required"
            try:
                release = validate_release_bundle(
                    self._release_bundle_path, self._acceptance_public_key,
                    self._minimum_release_counter, self._robot_id)
                artifacts = release["verified_artifacts"]
                validate_acceptance(
                    artifacts["paper_real_solver"]["payload"],
                    expected_inputs=self._paper_real_expected_inputs,
                    require_approved=True)
                validate_evidence(
                    artifacts["runtime_performance_software"]["payload"],
                    expected_profile="software")
                if self._acceptance_bundle_type == "release_bundle":
                    validate_evidence(
                        artifacts["runtime_performance_hardware"]["payload"],
                        expected_profile="hardware_hold")
            except (AcceptanceError, RuntimeAcceptanceError,
                    SolverAcceptanceError, KeyError, TypeError, ValueError) as exc:
                return False, "signed_release_rejected:" + str(exc)
            import torch
            if not torch.cuda.is_available():
                return False, "cuda_unavailable_realtime_forbidden"
            if not self._allow_commands:
                return False, "hardware_commands_disabled"
        return True, "ok"

    def _arm(self, _request):
        with self._lock:
            ok, reason = self._preflight()
            self._armed = bool(ok)
            self._paused = False
            self._reason = "armed" if ok else reason
            self._armed_pub.publish(Bool(data=self._armed and self._allow_commands))
            self._publish_status()
            return TriggerResponse(ok, self._reason)

    def _pause(self, _request):
        with self._lock:
            self._paused = True
            self._safe_hold("operator_pause")
        return TriggerResponse(True, "paused")

    def _resume(self, _request):
        with self._lock:
            ok, reason = self._preflight()
            if not self._armed or not ok:
                return TriggerResponse(False, reason)
            self._paused = False
            self._reason = "resumed"
            self._armed_pub.publish(Bool(data=self._allow_commands))
        return TriggerResponse(True, "resumed")

    def _abort_service(self, _request):
        with self._lock:
            self._active = False
            self._armed = False
            self._task_finished = (False, "operator_abort")
            self._safe_hold("operator_abort")
        return TriggerResponse(True, "aborted_and_disarmed")

    def _safe_hold(self, reason):
        self._paused = True
        if hasattr(self, '_planning_worker'):
            self._planning_worker.cancel()
        self._trajectory_client.cancel_all_goals()
        if hasattr(self,'_gripper_client'):
            self._gripper_client.cancel_all_goals()
        self._reason = str(reason)
        self._armed_pub.publish(Bool(data=False))

    def _backend(self):
        if self._motion_backend is None:
            self._motion_backend = MotionBackend(self._robot_xml)
        return self._motion_backend

    def _object_clouds_cb(self, value):
        with self._lock:
            self._object_clouds = deepcopy(value)

    def _scene_cloud_cb(self, value):
        with self._lock:
            self._scene_cloud = deepcopy(value)

    def _owned_cloud(self, object_uuid):
        with self._lock:
            clouds = deepcopy(self._object_clouds)
        if clouds is None:
            raise RuntimeError('mask_derived_object_cloud_missing')
        points = []
        for item in clouds.objects:
            if item.object_uuid != object_uuid:
                continue
            age = (rospy.Time.now()-item.cloud_base.header.stamp).to_sec()
            if not 0 <= age <= .15 or item.cloud_base.header.frame_id != self._base_frame:
                raise RuntimeError('object_cloud_stale_or_wrong_frame')
            points.extend(point_cloud2.read_points(item.cloud_base,
                          field_names=('x','y','z'),skip_nans=True))
        if len(points) < 120:
            raise RuntimeError('mask_derived_object_cloud_incomplete')
        return np.asarray(points)

    def _make_contact_policy(self, candidate, opening):
        target = self._owned_cloud(candidate.object_uuid)
        with self._lock:
            cloud = deepcopy(self._scene_cloud)
        if (cloud is None or cloud.header.frame_id != self._base_frame
                or not 0 <= (rospy.Time.now()-cloud.header.stamp).to_sec() <= .15):
            raise RuntimeError('contact_scene_cloud_stale')
        scene = np.asarray(list(point_cloud2.read_points(cloud,
                               field_names=('x','y','z'),skip_nans=True)))
        others=self._other_owned_points(candidate.object_uuid)
        return TargetContactPolicy(self._ik,candidate,target,scene,
            [item[0] for item in self._collision_sampler._finger_samples],
            opening,_pose_matrix(candidate.grasp_pose),other_object_points=others)

    def _other_owned_points(self, excluded_uuid):
        with self._lock:
            identities={c.object_uuid for c in self._object_clouds.objects
                        if c.object_uuid != excluded_uuid}
        chunks=[self._owned_cloud(identity) for identity in identities]
        return np.vstack(chunks) if chunks else np.empty((0,3))

    def _check_supported_release(self, attached):
        # Complete object surface / metrology are site acceptance inputs, not
        # inferred from the VLM's K10 label or a partially visible top surface.
        config = (self._release or {}).get('verified_artifacts',{}).get(
            'workspace',{}).get('payload',{}).get('placement_geometry',{})
        if (not config.get('accepted',False)
                or config.get('program_sha256') != self._program.program_sha256
                or config.get('object_uuid') != attached.object_uuid):
            raise RuntimeError('placement_geometry_acceptance_missing')
        anchor = int(config.get('anchor_keypoint',-1))
        if not 0 <= anchor < len(self._tracked.keypoints):
            raise RuntimeError('placement_anchor_missing')
        group = self._tracked.keypoints[anchor].rigid_group_id
        support = self._unique_object_for_group(self._registry,group,TrackedObject.FREE_TRACKED)
        return validate_support_geometry(self._owned_cloud(attached.object_uuid),
            self._owned_cloud(support.object_uuid),float(config['uncertainty_m']))

    def _fresh_motion_inputs(self):
        with self._lock:
            now = rospy.Time.now()
            for name, stamp, limit in (
                    ('joints',self._joint_stamp,.15),
                    ('tracking',self._last_tracking_stamp,self._tracking_max_age),
                    ('map',self._last_map_stamp,self._map_max_age)):
                age = (now-stamp).to_sec()
                if not 0 <= age <= limit:
                    raise RuntimeError('motion_input_stale:'+name)
            if (self._joints is None or self._sdf is None or not self._sdf.valid
                    or self._gripper_opening_m is None
                    or self._map_status is None or not self._map_status.planning_safe
                    or not self._map_status.map_query_allowed
                    or self._map_status.state != SafeMappingStatus.READY
                    or self._map_status.map_generation_uuid != self._sdf.map_generation_uuid
                    or self._tracked is None or not self._tracked.all_valid
                    or self._registry is None
                    or any(o.state in (TrackedObject.LOST,TrackedObject.AMBIGUOUS,TrackedObject.FAULT)
                           for o in self._registry.objects)):
                raise RuntimeError('motion_input_not_accepted')
            return self._joints.copy(), deepcopy(self._sdf)

    def _audit_executable(self, positions, sdf, keypoints, contact_policy=None):
        if contact_policy is not None:
            target=self._owned_cloud(contact_policy.object_uuid)
            with self._lock:
                cloud=deepcopy(self._scene_cloud)
            if (cloud is None or cloud.header.frame_id != self._base_frame
                    or not 0 <= (rospy.Time.now()-cloud.header.stamp).to_sec() <= .15):
                raise RuntimeError('contact_scene_stale_before_audit')
            scene=np.asarray(list(point_cloud2.read_points(cloud,
                                   field_names=('x','y','z'),skip_nans=True)))
            contact_policy.refresh_scene(target,scene,self._other_owned_points(contact_policy.object_uuid))
        backend = self._backend()
        if not all(backend.self_clear(q,contact_policy.opening_m if contact_policy else self._gripper_opening_m)
                   for q in positions):
            raise RuntimeError('trajectory_self_collision')
        audit_joint_path(positions,self._collision_sampler,sdf,
                         minimum_clearance_m=.01,
                         attached_points_local=(contact_policy.probe_points_local
                             if contact_policy is not None and contact_policy.probe_frame is not None
                             else self._attached_collision_points
                             if self._held_keypoint >= 0 and not self._release_probe else None),
                         attached_frame=('rekep_tcp' if contact_policy is not None
                             and contact_policy.probe_frame is not None else 'gripper_base'),
                         contact_policy=contact_policy)
        poses = []
        current = self._ik.forward(self._joints)
        for q in positions:
            matrix = self._ik.forward(q)
            position, quaternion = self._planner.modules.transform_utils.mat2pose(matrix)
            poses.append(np.r_[position,quaternion])
        _subgoals, constraints = load_official_stage_constraints(
            self._program.program_directory,self._stage,len(keypoints),self._grasp_cost)
        movable = np.zeros(len(keypoints),dtype=bool)
        if self._held_group_id > 0 and not self._release_probe:
            movable = np.array([k.rigid_group_id == self._held_group_id
                                for k in self._tracked.keypoints])
        tolerance = float(self._planner.path_config.get('constraint_tolerance',.0001))
        for pose,q in zip(poses,positions):
            updated = np.asarray(keypoints).copy()
            transform = self._ik.forward(q) @ np.linalg.inv(current)
            updated[movable] = updated[movable] @ transform[:3,:3].T+transform[:3,3]
            validate_path_constraints([pose],updated,constraints,tolerance)

    def _execute(self, goal):
        with self._lock:
            if (not self._armed or self._program is None
                    or goal.session_id != self._program.session_id
                    or goal.program_sha256 != self._program.program_sha256):
                self._server.set_aborted(ExecuteReKepResult(
                    success=False, status="not_armed_or_program_mismatch"))
                return
            self._active = True
            self._paused = False
            self._stage = 1
            self._held_keypoint = -1
            self._held_group_id = 0
            self._attached_grasp = None
            self._grasp_failures = {}
            self._grasp_attempt = 0
            self._grasp = self._grasp_batch = None
            self._coordinator = ReKepCoordinatorCore(
                int(self._program.num_stages))
            self._task_finished = None
            self._success_ticks = self._deadline_misses = self._backtracks = 0
            self._reason = "closed_loop_started"
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self._server.is_preempt_requested():
                with self._lock:
                    self._active = False
                    self._safe_hold("action_preempted")
                self._server.set_preempted(ExecuteReKepResult(
                    success=False, status="preempted"))
                return
            with self._lock:
                finished = self._task_finished
                feedback = ExecuteReKepFeedback(
                    stage_index=self._stage, tick=self._tick_count,
                    phase="paused" if self._paused else "closed_loop",
                    planning_rate_hz=self._rate_hz,
                    planning_latency_s=self._last_plan_latency,
                    reason=self._reason)
                self._server.publish_feedback(feedback)
            if finished is not None:
                success, status = finished
                result = ExecuteReKepResult(
                    success=success, status=status,
                    audit_directory=self._program.program_directory,
                    completed_stages=max(0, self._stage - (0 if success else 1)),
                    backtrack_count=self._backtracks)
                (self._server.set_succeeded if success else self._server.set_aborted)(
                    result, status)
                return
            rate.sleep()

    def _current_ee(self):
        transform = self._tf.lookup_transform(
            self._base_frame, self._tip_frame, rospy.Time(0), rospy.Duration(0.05))
        value = transform.transform
        result = np.asarray([value.translation.x, value.translation.y,
                             value.translation.z, value.rotation.x,
                             value.rotation.y, value.rotation.z,
                             value.rotation.w], dtype=float)
        norm = float(np.linalg.norm(result[3:]))
        if not np.all(np.isfinite(result)) or norm < 1e-9:
            raise RuntimeError("current end-effector transform is invalid")
        result[3:] /= norm
        return result

    def _grasp_cost(self, index):
        if self._held_group_id <= 0 or self._snapshot is None:
            return 1.0
        keypoints = self._snapshot.keypoints.keypoints
        index = int(index)
        if index < 0 or index >= len(keypoints):
            return 1.0
        return (0.0 if int(keypoints[index].rigid_group_id)
                == self._held_group_id else 1.0)

    @staticmethod
    def _unique_object_for_group(registry, rigid_group_id, required_state=None):
        matches = [item for item in registry.objects
                   if int(item.rigid_group_id) == int(rigid_group_id)]
        if len(matches) != 1:
            raise RuntimeError(
                "rigid_group_{}_must_bind_one_object_uuid".format(
                    rigid_group_id))
        if required_state is not None and matches[0].state != required_state:
            raise RuntimeError(
                "object_{}_state_{}_expected_{}".format(
                    matches[0].object_uuid, matches[0].state, required_state))
        return matches[0]

    def _synchronize_attachment(self, registry, program, snapshot):
        attached = [item for item in registry.objects
                    if item.state == TrackedObject.ATTACHED]
        unsafe = [item for item in registry.objects
                  if item.state in (TrackedObject.AMBIGUOUS,
                                    TrackedObject.LOST,
                                    TrackedObject.FAULT)]
        if unsafe:
            raise RuntimeError("object_registry_contains_unsafe_lifecycle_state")
        if len(attached) > 1:
            raise RuntimeError("multiple_attached_objects_are_not_supported")
        if not attached:
            self._held_keypoint = -1
            self._held_group_id = 0
            return None
        item = attached[0]
        group = int(item.rigid_group_id)
        candidates = [int(index) for index in program.grasp_keypoints
                      if int(index) >= 0
                      and int(index) < len(snapshot.keypoints.keypoints)
                      and int(snapshot.keypoints.keypoints[
                          int(index)].rigid_group_id) == group]
        if not candidates:
            raise RuntimeError("attached_object_has_no_program_grasp_keypoint")
        self._held_keypoint = candidates[-1]
        self._held_group_id = group
        return item

    def _validate_grasp_binding(self, candidate, registry, map_status,
                                program, snapshot, stage, keypoint_index):
        group = int(snapshot.keypoints.keypoints[
            keypoint_index].rigid_group_id)
        target = self._unique_object_for_group(
            registry, group, TrackedObject.FREE_TRACKED)
        validate_grasp_candidate_binding(
            candidate, group, target.object_uuid, program.session_id,
            program.program_sha256, snapshot.snapshot_id,
            map_status.map_generation_uuid, stage, self._grasp_attempt)
        return target

    def _prepare_grasp_target(self, candidate, registry, map_status,
                              program, snapshot, keypoints, grasp_index):
        self._validate_grasp_binding(candidate, registry, map_status,
                                     program, snapshot, self._stage, grasp_index)
        matrix = _grasp_matrix(candidate)
        tcp = np.array([candidate.tcp_position.x, candidate.tcp_position.y,
                        candidate.tcp_position.z])
        if (candidate.header.frame_id != self._base_frame
                or not np.all(np.isfinite(matrix))
                or not np.all(np.isfinite(tcp))
                or not np.allclose(tcp, matrix[:3, 3], atol=1e-6, rtol=0.)
                or np.linalg.norm(tcp - keypoints[grasp_index]) > .10):
            raise RuntimeError('invalid_bound_grasp_tcp')
        xyz, quat = self._planner.modules.transform_utils.mat2pose(matrix)
        pose = np.r_[xyz, quat]
        key = (candidate.candidate_id, candidate.session_id, candidate.program_sha256,
               candidate.snapshot_id, candidate.object_uuid, candidate.map_generation_uuid,
               self._stage, self._grasp_attempt, tuple(pose))
        if key != self._grasp_target_key:
            self._grasp_target_key = key
            self._grasp_target_anchor = keypoints[grasp_index].copy()
            self._preopened_grasp_key = None
            self._approach_opening_m = None
            self._success_ticks = 0
            self._planner.invalidate()
        elif np.linalg.norm(keypoints[grasp_index] - self._grasp_target_anchor) > .005:
            raise RuntimeError('grasp_keypoint_moved_replan_required')
        opening = float(candidate.suggested_preopen_width_m)
        if not 0 < opening <= .070:
            raise RuntimeError('preopen_width_out_of_range')
        self._contact_policy = self._make_contact_policy(candidate, opening)
        return pose

    def _request_grasp_target(self, program, snapshot, sdf, sequence):
        self._planning_worker.cancel()
        self._contact_policy = None
        self._approach_opening_m = None
        self._success_ticks = 0
        self._horizon_pub.publish(ReKepHorizon(
            header=sdf.header, session_id=program.session_id,
            program_sha256=program.program_sha256, snapshot_id=snapshot.snapshot_id,
            map_generation_uuid=sdf.map_generation_uuid,
            state_sequence=sequence, stage_index=self._stage,
            valid=True, authorized=False, status='grasp_target_pending'))
        self._reason = 'waiting_for_bound_anygrasp_target_before_planning'

    def _ensure_grasp_open(self, candidate):
        """Open once before dispatch, then require live aperture feedback."""
        if self._grasp_target_key is None:
            raise RuntimeError('grasp_target_missing_before_preopen')
        opening = float(candidate.suggested_preopen_width_m)
        if not 0 < opening <= .070:
            raise RuntimeError('preopen_width_out_of_range')
        if self._preopened_grasp_key != self._grasp_target_key:
            preopen = self._send_gripper_goal(CommandGripperGoal.OPEN,
                candidate.object_uuid, opening, 1.0)
            validate_preopen(preopen, opening)
            self._preopened_grasp_key = self._grasp_target_key
        self._approach_opening_m = opening
        self._check_approach_opening()
        self._contact_policy = self._make_contact_policy(candidate, self._gripper_opening_m)

    def _check_approach_opening(self):
        if self._approach_opening_m is not None and (
                self._gripper_opening_m is None
                or not np.isfinite(self._gripper_opening_m)
                or abs(self._gripper_opening_m - self._approach_opening_m) > .003):
            raise RuntimeError('gripper_not_open_during_approach')

    def _require_grasp_arrival(self, candidate, target_pose):
        """Recheck measured TCP immediately before the close event."""
        if (self._grasp_target_key is None
                or self._preopened_grasp_key != self._grasp_target_key
                or self._approach_opening_m is None):
            raise RuntimeError('grasp_preopen_not_confirmed')
        matrix = _grasp_matrix(candidate)
        xyz, quat = self._planner.modules.transform_utils.mat2pose(matrix)
        candidate_pose = np.r_[xyz, quat]
        if not pose_reached(candidate_pose, target_pose, 1e-6, 1e-6):
            raise RuntimeError('grasp_target_changed_before_close')
        joints, _sdf = self._fresh_motion_inputs()
        xyz, quat = self._planner.modules.transform_utils.mat2pose(self._ik.forward(joints))
        if not pose_reached(np.r_[xyz, quat], candidate_pose,
                            min(.003, self._goal_position_tolerance),
                            min(.03, self._goal_rotation_tolerance)):
            raise RuntimeError('grasp_tcp_not_reached_before_close')
        self._check_approach_opening()

    def _wait_for_object_state(self, object_uuid, expected, timeout_s):
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline and not rospy.is_shutdown():
            with self._lock:
                registry = deepcopy(self._registry)
            if registry is not None:
                matches = [item for item in registry.objects
                           if item.object_uuid == object_uuid]
                if len(matches) == 1:
                    if matches[0].state == expected:
                        return matches[0]
                    if matches[0].state in (
                            TrackedObject.AMBIGUOUS, TrackedObject.LOST,
                            TrackedObject.FAULT):
                        raise RuntimeError(
                            "object_lifecycle_became_unsafe:{}".format(
                                matches[0].status))
            rospy.sleep(0.02)
        raise RuntimeError("object_state_transition_timeout")

    def _wait_for_lifecycle_evidence(self, object_uuid, state, reason,
                                     timeout_s=2.0):
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline and not rospy.is_shutdown():
            with self._lock:
                registry = deepcopy(self._registry)
            if registry is not None:
                matches = [item for item in registry.objects
                           if item.object_uuid == object_uuid]
                if (len(matches) == 1 and matches[0].state == state
                        and matches[0].status == reason):
                    return matches[0]
                if (len(matches) == 1 and matches[0].state in (
                        TrackedObject.AMBIGUOUS, TrackedObject.LOST,
                        TrackedObject.FAULT)):
                    raise RuntimeError("lifecycle_evidence_became_unsafe:" +
                                       matches[0].status)
            rospy.sleep(0.02)
        raise RuntimeError("lifecycle_evidence_timeout:" + str(reason))

    def _rebuild_and_wait(self, reason):
        # Registry and exclusion-signature topics are independent ROS
        # transports.  Wait until the mapper has consumed the lifecycle change
        # and withdrawn the old generation before asking it to rebuild.
        with self._lock:
            previous = ("" if self._map_status is None else
                        self._map_status.map_generation_uuid)
        invalidation_deadline = time.monotonic() + 2.0
        while time.monotonic() < invalidation_deadline and not rospy.is_shutdown():
            with self._lock:
                status = deepcopy(self._map_status)
            if (status is not None
                    and (status.state != SafeMappingStatus.READY
                         or not status.planning_safe
                         or status.map_generation_uuid != previous)):
                break
            rospy.sleep(0.02)
        else:
            raise RuntimeError("old_map_generation_not_invalidated")
        response = self._rebuild_map(str(reason))
        if not response.success or not response.map_generation_uuid:
            raise RuntimeError("safe_map_rebuild_rejected:" + response.message)
        expected = response.map_generation_uuid
        deadline = time.monotonic() + float(rospy.get_param(
            "~map_rebuild_timeout_s", 20.0))
        while time.monotonic() < deadline and not rospy.is_shutdown():
            with self._lock:
                status = deepcopy(self._map_status)
                sdf = deepcopy(self._sdf)
            if (status is not None
                    and status.map_generation_uuid == expected
                    and status.state == SafeMappingStatus.READY
                    and status.map_query_allowed and status.planning_safe
                    and sdf is not None and sdf.valid
                    and sdf.map_generation_uuid == expected):
                return expected
            rospy.sleep(0.05)
        raise RuntimeError("safe_map_rebuild_ready_timeout")

    def _send_gripper_goal(self, command, object_uuid, opening_m, effort=1.0):
        self._assert_motion_authorized()
        joints,_sdf = self._fresh_motion_inputs()
        # Opening/closing also changes robot geometry while the arm is still.
        for width in np.linspace(self._gripper_opening_m,float(opening_m),36):
            if not self._backend().self_clear(joints,width):
                raise RuntimeError('gripper_aperture_sweep_self_collision')
        goal = CommandGripperGoal(
            command=command, object_uuid=object_uuid,
            total_opening_m=float(opening_m), effort=float(effort))
        self._gripper_client.send_goal(goal)
        if not self._gripper_client.wait_for_result(rospy.Duration(8.0)):
            self._gripper_client.cancel_goal()
            raise RuntimeError("gripper_action_timeout")
        result = self._gripper_client.get_result()
        if result is None or not result.success:
            raise RuntimeError("gripper_action_failed:{}".format(
                "no_result" if result is None else result.status))
        return result

    def _assert_motion_authorized(self):
        if (self._mode != 'autonomous' or not self._allow_commands
                or not self._armed or self._paused or not self._active
                or self._program is None or not self._program.approved
                or self._snapshot is None or not self._snapshot.valid
                or self._program.snapshot_id != self._snapshot.snapshot_id):
            raise RuntimeError('motion_authority_revoked')
        assert_release_unchanged(self._release)
        assert_release_unchanged(self._program_approval_guard)
        if compute_program_sha256(self._program.program_directory) != self._program.program_sha256:
            raise RuntimeError('approved_program_changed_before_dispatch')

    def _send_local_trajectory(self, smooth, sdf, keypoints):
        """Only audited 100 ms prefixes, including grasp and verification moves."""
        started = time.monotonic()
        goal = np.asarray(smooth.positions[-1])
        cursor = 0.
        identity = (self._program.program_sha256,self._snapshot.snapshot_id,
                    tuple((k.id,k.rigid_group_id) for k in self._tracked.keypoints))
        reference = np.asarray(keypoints).copy()
        moving_group = (self._contact_policy.rigid_group_id
            if self._contact_policy is not None and self._contact_policy.probe_frame is not None
            else self._held_group_id)
        movable = (np.array([k.rigid_group_id == moving_group for k in self._tracked.keypoints])
                   if moving_group > 0 and not self._release_probe else np.zeros(len(reference),bool))
        while time.monotonic()-started < smooth.duration_s+8.:
            self._assert_motion_authorized()
            self._check_approach_opening()
            if identity != (self._program.program_sha256,self._snapshot.snapshot_id,
                            tuple((k.id,k.rigid_group_id) for k in self._tracked.keypoints)):
                raise RuntimeError('motion_identity_changed')
            joints,latest = self._fresh_motion_inputs()
            if latest.map_generation_uuid != sdf.map_generation_uuid:
                raise RuntimeError('local_motion_map_generation_changed')
            # Reaching is measured, not inferred from a successful send.
            position_error,rotation_error = self._ik._pose_errors(
                self._ik.forward(joints),self._ik.forward(goal))
            if (np.max(np.abs(joints-goal)) <= .003
                    and position_error <= .003 and rotation_error <= .03):
                return
            expected = np.array([np.interp(cursor,smooth.times,smooth.positions[:,j])
                                 for j in range(6)])
            if np.max(np.abs(expected-joints)) > .01:
                raise RuntimeError('local_motion_tracking_deviation')
            stop = min(cursor+.10,smooth.duration_s)
            if stop <= cursor:
                raise RuntimeError('trajectory_finished_but_feedback_not_at_goal')
            times = np.linspace(cursor,stop,6)
            positions = np.column_stack([np.interp(times,smooth.times,smooth.positions[:,j])
                                         for j in range(6)])
            # Preserve the time-parameterized path instead of restarting a
            # zero-speed polynomial at every prefix (which would barely move).
            validate_trajectory(JOINT_NAMES,positions,times-cursor,joints,
                start_tolerance_rad=.01,maximum_velocity_rad_s=self._joint_velocity,
                maximum_acceleration_rad_s2=.10,maximum_jerk_rad_s3=.50)
            prefix = JointTrajectory(joint_names=JOINT_NAMES)
            prefix.header.frame_id = latest.map_generation_uuid
            for values,stamp in zip(positions,times-cursor):
                p=JointTrajectoryPoint(); p.positions=values.tolist()
                p.time_from_start=rospy.Duration(float(stamp)); prefix.points.append(p)
            current_points = np.array([[k.position.x,k.position.y,k.position.z]
                                       for k in self._tracked.keypoints])
            if np.any(np.linalg.norm(current_points[~movable]-reference[~movable],axis=1)>.005):
                raise RuntimeError('unheld_object_moved_replan_required')
            self._audit_executable([p.positions for p in prefix.points],latest,
                                   current_points,self._contact_policy)
            self._assert_motion_authorized()
            self._trajectory_client.send_goal(FollowJointTrajectoryGoal(trajectory=prefix))
            self._reason='executing_audited_100ms_prefix'
            self._publish_status()
            if not self._trajectory_client.wait_for_result(rospy.Duration(1.)):
                self._trajectory_client.cancel_goal()
                raise RuntimeError('local_prefix_timeout')
            outcome = self._trajectory_client.get_result()
            if outcome is None or outcome.error_code != outcome.SUCCESSFUL:
                raise RuntimeError('local_prefix_failed')
            cursor = stop
        raise RuntimeError('local_trajectory_timeout')

    def _verification_probe(self, candidate, sdf, joints):
        joints,sdf = self._fresh_motion_inputs()
        grasp_pose = _pose_matrix(candidate.grasp_pose)
        # Piper +Z is the approach axis, including coincident grasp/pregrasp.
        direction = -grasp_pose[:3, 2]
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm < 1e-6:
            raise RuntimeError("AnyGrasp approach direction is invalid")
        direction /= norm
        current_pose = self._ik.forward(joints)
        direction = (np.array([0.,0.,1.]) if self._release_probe else
                     current_pose[:3,:3] @ grasp_pose[:3,:3].T @ direction)
        poses = []
        for offset in np.linspace(0.0, 0.020, 6):
            pose = current_pose.copy()
            pose[:3, 3] += direction * offset
            poses.append(pose)
        validation = self._ik.validate_pose_sequence(poses, joints)
        if not validation["valid"]:
            raise RuntimeError("attachment_probe_ik_failed")
        joint_path = np.asarray([
            step["joint_positions"] for step in validation["steps"]],
            dtype=float)
        keypoints = np.array([[k.position.x,k.position.y,k.position.z]
                              for k in self._tracked.keypoints])
        self._audit_executable(joint_path,sdf,keypoints,self._contact_policy)
        actual_start = self._ik.forward(joint_path[0])[:3, 3]
        actual_end = self._ik.forward(joint_path[-1])[:3, 3]
        actual_delta = actual_end - actual_start
        if (float(np.linalg.norm(actual_delta)) < 0.019
                or float(np.dot(actual_delta, direction)) < 0.018):
            raise RuntimeError("attachment_probe_actual_motion_below_20mm")
        smooth = smooth_pchip_trajectory(
            JOINT_NAMES, joint_path, joints,
            maximum_velocity_rad_s=min(self._joint_velocity, 0.05),
            maximum_acceleration_rad_s2=0.10,
            maximum_jerk_rad_s3=0.50)
        self._audit_executable(smooth.positions,sdf,keypoints,self._contact_policy)
        self._send_local_trajectory(smooth,sdf,keypoints)

    def _execute_grasp_approach(self, candidate, sdf, joints, keypoints):
        """Execute the real-environment grasp action after ReKep's subgoal."""
        joints,sdf = self._fresh_motion_inputs()
        current_matrix = self._ik.forward(joints)
        pregrasp = _pose_matrix(candidate.pregrasp_pose)
        grasp = _pose_matrix(candidate.grasp_pose)
        if np.linalg.norm(current_matrix[:3, 3] - pregrasp[:3, 3]) > 0.10:
            raise RuntimeError("AnyGrasp pregrasp is not local to ReKep subgoal")
        vectors = []
        for start, end in ((current_matrix, pregrasp), (pregrasp, grasp)):
            start_pose = self._planner.modules.transform_utils.mat2pose(start)
            end_pose = self._planner.modules.transform_utils.mat2pose(end)
            start_vector = np.r_[start_pose[0], start_pose[1]]
            end_vector = np.r_[end_pose[0], end_pose[1]]
            count = self._planner.modules.utils.get_linear_interpolation_steps(
                start_vector, end_vector, 0.005, np.deg2rad(1.0))
            segment = self._planner.modules.utils.linear_interpolate_poses(
                start_vector, end_vector, count)
            vectors.extend(segment if not vectors else segment[1:])
        vectors = np.asarray(vectors, dtype=float)
        _subgoals, path_constraints = load_official_stage_constraints(
            self._program.program_directory, self._stage, len(keypoints),
            self._grasp_cost)
        validate_path_constraints(
            vectors, keypoints, path_constraints, self._tolerance)
        matrices = self._planner.modules.transform_utils.convert_pose_quat2mat(
            vectors)
        validation = self._ik.validate_pose_sequence(matrices, joints)
        if not validation["valid"]:
            raise RuntimeError("AnyGrasp approach IK failed")
        joint_path = np.asarray([
            step["joint_positions"] for step in validation["steps"]], dtype=float)
        self._audit_executable(joint_path,sdf,keypoints,self._contact_policy)
        smooth = smooth_pchip_trajectory(
            JOINT_NAMES, joint_path, joints,
            maximum_velocity_rad_s=min(self._joint_velocity, 0.05),
            maximum_acceleration_rad_s2=0.10,
            maximum_jerk_rad_s3=0.50)
        self._audit_executable(smooth.positions,sdf,keypoints,self._contact_policy)
        self._send_local_trajectory(smooth,sdf,keypoints)

    def _current_constraints(self, stage, ee, keypoints):
        subgoals, paths = load_official_stage_constraints(
            self._program.program_directory, stage, len(keypoints), self._grasp_cost)
        return ([float(fn(ee[:3], keypoints)) for fn in subgoals],
                [float(fn(ee[:3], keypoints)) for fn in paths])

    def _backtrack_if_needed(self, ee, keypoints):
        _subgoals, current_paths = self._current_constraints(self._stage, ee, keypoints)
        if all(np.isfinite(value) and value <= self._tolerance for value in current_paths):
            return False
        if self._stage <= 1:
            return False
        for candidate in range(self._stage - 1, 0, -1):
            _goals, paths = self._current_constraints(candidate, ee, keypoints)
            if all(np.isfinite(value) and value <= self._tolerance for value in paths):
                self._stage = candidate
                if self._coordinator is not None:
                    self._coordinator.enter_stage(candidate)
                    self._grasp_attempt = self._coordinator.grasp_attempt
                self._success_ticks = 0
                self._grasp = self._grasp_batch = None
                self._backtracks += 1
                self._reason = "constraint_backtrack_stage_{}".format(candidate)
                return True
        # The official coordinator falls back to stage one when no earlier
        # path constraint set remains satisfied; it does not invent a paused
        # terminal state.
        self._stage = 1
        if self._coordinator is not None:
            self._coordinator.enter_stage(1)
            self._grasp_attempt = self._coordinator.grasp_attempt
        self._success_ticks = 0
        self._grasp = self._grasp_batch = None
        self._backtracks += 1
        self._reason = "constraint_backtrack_stage_1"
        return True

    def _prefix(self, joint_path, current, map_generation_uuid):
        positions, times = short_horizon_joint_prefix(
            joint_path, current, self._joint_velocity, 0.10)
        smooth = smooth_pchip_trajectory(JOINT_NAMES,positions,current,
            maximum_velocity_rad_s=self._joint_velocity,
            maximum_acceleration_rad_s2=.10,maximum_jerk_rad_s3=.50)
        times = np.linspace(0.,min(.10,smooth.duration_s),6)
        positions = np.column_stack([np.interp(times,smooth.times,smooth.positions[:,j])
                                     for j in range(6)])
        validate_trajectory(JOINT_NAMES,positions,times,current,
            maximum_velocity_rad_s=self._joint_velocity,
            maximum_acceleration_rad_s2=.10,maximum_jerk_rad_s3=.50)
        trajectory = JointTrajectory(joint_names=JOINT_NAMES)
        # This restricted bridge uses header.frame_id as the immutable map
        # generation authorization token; no TF lookup is performed on it.
        trajectory.header.frame_id = str(map_generation_uuid)
        for values, stamp in zip(positions, times):
            point = JointTrajectoryPoint()
            point.positions = values.tolist()
            point.time_from_start = rospy.Duration(float(stamp))
            trajectory.points.append(point)
        return trajectory

    def _cancel_failed_grasp(self, object_uuid, stage, reason):
        if self._grasp_may_hold:
            self._safe_hold('grasp_state_uncertain_no_automatic_open:'+reason)
            raise RuntimeError('grasp_state_uncertain_requires_verified_recovery')
        # Cancellation is legal only after the gripper is physically reopened;
        # the monitor independently verifies opening and dual-camera freedom.
        self._send_gripper_goal(
            CommandGripperGoal.OPEN, object_uuid, 0.070, 1.0)
        response = self._cancel_grasp(object_uuid)
        if not response.success:
            raise RuntimeError("cancel_grasp_rejected:" + response.message)
        failures = self._grasp_failures.get(stage, 0) + 1
        self._grasp_failures[stage] = failures
        if self._coordinator is None:
            raise RuntimeError("coordinator_is_unavailable")
        self._grasp_attempt = self._coordinator.record_grasp_failure()
        self._success_ticks = 0
        self._grasp = self._grasp_batch = None
        self._reason = "grasp_retry_{}/3:{}".format(failures, reason)
        if failures >= 3:
            raise RuntimeError("grasp_failed_three_times_paused")
        return False

    def _record_grasp_inference_failure(self, stage, reason):
        """Advance attempt without touching hardware after an empty inference."""
        failures = self._grasp_failures.get(stage, 0) + 1
        self._grasp_failures[stage] = failures
        if self._coordinator is None:
            raise RuntimeError("coordinator_is_unavailable")
        self._grasp_attempt = self._coordinator.record_grasp_failure()
        self._success_ticks = 0
        self._grasp = self._grasp_batch = None
        self._reason = "anygrasp_retry_{}/3:{}".format(failures, reason)
        if failures >= 3:
            raise RuntimeError("anygrasp_failed_three_times_paused")

    def _stage_event(self, program, snapshot, registry, map_status, sdf,
                     joints, keypoints, selected_grasp,
                     semantic_subgoal_pose):
        stage = int(self._stage)
        grasp_index = int(program.grasp_keypoints[stage - 1])
        release_index = int(program.release_keypoints[stage - 1])
        if grasp_index < 0 and release_index < 0:
            return True
        if self._mode != "autonomous" or not self._allow_commands:
            raise RuntimeError("shadow_stage_event_requires_physical_lifecycle")

        if grasp_index >= 0:
            target = self._validate_grasp_binding(
                selected_grasp, registry, map_status, program, snapshot,
                stage, grasp_index)
            grasp_matrix = _grasp_matrix(selected_grasp)
            semantic = np.asarray(semantic_subgoal_pose, dtype=float)
            tcp = np.asarray([
                selected_grasp.tcp_position.x,
                selected_grasp.tcp_position.y,
                selected_grasp.tcp_position.z,
            ], dtype=float)
            if (semantic.shape != (7,) or not np.all(np.isfinite(semantic))
                    or np.linalg.norm(
                        grasp_matrix[:3, 3] - semantic[:3]) > 0.10):
                raise RuntimeError(
                    "AnyGrasp candidate is not local to the ReKep subgoal")
            if (not np.all(np.isfinite(tcp))
                    or np.linalg.norm(tcp - keypoints[grasp_index]) > 0.10):
                raise RuntimeError(
                    "AnyGrasp candidate is no longer local to grasp K")
            self._require_grasp_arrival(selected_grasp, semantic_subgoal_pose)
            response = self._begin_grasp(target.object_uuid)
            if not response.success:
                raise RuntimeError("begin_grasp_rejected:" + response.message)
            try:
                self._require_grasp_arrival(selected_grasp, semantic_subgoal_pose)
                self._approach_opening_m = None
                self._grasp_may_hold = True
                outcome = self._send_gripper_goal(
                    CommandGripperGoal.CLOSE, target.object_uuid,
                    selected_grasp.predicted_width_m, 1.0)
                if not outcome.stable_contact:
                    raise RuntimeError("contact_evidence_not_stable")
                joints,sdf = self._fresh_motion_inputs()
                self._contact_policy = self._make_contact_policy(selected_grasp,outcome.final_opening_m)
                self._contact_policy.begin_attachment_probe(
                    self._ik.forward(joints),self._owned_cloud(target.object_uuid))
                # This is the real-world implementation of ReKep's grasp
                # action/is_grasping environment interface, not a new stage.
                self._verification_probe(selected_grasp, sdf, joints)
                self._wait_for_lifecycle_evidence(
                    target.object_uuid, TrackedObject.ATTACH_VERIFYING,
                    "rigid_tcp_attachment_evidence_ready")
                confirmation = self._confirm_attachment(target.object_uuid)
                if not confirmation.success:
                    raise RuntimeError(
                        "attachment_confirmation_rejected:" +
                        confirmation.message)
                attached = self._wait_for_object_state(
                    target.object_uuid, TrackedObject.ATTACHED, 4.0)
                if int(attached.rigid_group_id) != int(target.rigid_group_id):
                    raise RuntimeError("attached_object_identity_changed")
            except Exception as exc:
                return self._cancel_failed_grasp(
                    target.object_uuid, stage,
                    "{}:{}".format(type(exc).__name__, exc))
            self._attached_grasp = deepcopy(selected_grasp)
            self._held_keypoint = grasp_index
            self._held_group_id = int(target.rigid_group_id)
            self._rebuild_and_wait("attached_object_dynamic_exclusion")
            self._grasp_failures.pop(stage, None)
            return True

        attached = self._synchronize_attachment(registry, program, snapshot)
        if attached is None:
            raise RuntimeError("release_requires_one_attached_object")
        release_group = int(snapshot.keypoints.keypoints[
            release_index].rigid_group_id)
        if release_group != int(attached.rigid_group_id):
            raise RuntimeError("release_keypoint_does_not_match_attached_object")
        if self._attached_grasp is None:
            raise RuntimeError("release_withdrawal_direction_is_unavailable")
        self._fresh_motion_inputs()
        self._check_supported_release(attached)
        # Rebase the original finger contact pair to the current placed pose.
        release_candidate = deepcopy(self._attached_grasp)
        current_matrix = self._ik.forward(self._joints)
        original_matrix = _pose_matrix(release_candidate.grasp_pose)
        transform = current_matrix @ np.linalg.inv(original_matrix)
        for point in release_candidate.contact_points:
            xyz = transform[:3,:3] @ np.array([point.x,point.y,point.z])+transform[:3,3]
            point.x,point.y,point.z = xyz
        xyz,quat = self._planner.modules.transform_utils.mat2pose(current_matrix)
        release_candidate.grasp_pose = _pose_message(np.r_[xyz,quat])
        response = self._begin_release(attached.object_uuid)
        if not response.success:
            raise RuntimeError("begin_release_rejected:" + response.message)
        opened = self._send_gripper_goal(
            CommandGripperGoal.OPEN, attached.object_uuid, 0.070, 1.0)
        validate_preopen(opened,.070)
        self._contact_policy = self._make_contact_policy(release_candidate,.070)
        self._release_probe = True
        # A short audited withdrawal supplies the relative-motion evidence that
        # distinguishes a released object from one still following the gripper.
        with self._lock:
            release_joints = (self._joints.copy() if self._joints is not None
                              else joints.copy())
            release_sdf = deepcopy(self._sdf)
        self._verification_probe(self._attached_grasp, release_sdf,
                                 release_joints)
        self._wait_for_lifecycle_evidence(
            attached.object_uuid, TrackedObject.RELEASE_VERIFYING,
            "release_evidence_ready")
        confirmation = self._confirm_release(attached.object_uuid)
        if not confirmation.success:
            raise RuntimeError("release_confirmation_rejected:" +
                               confirmation.message)
        released = self._wait_for_object_state(
            attached.object_uuid, TrackedObject.FREE_TRACKED, 4.0)
        if released.excluded_from_static_map:
            raise RuntimeError("released_object_remains_dynamic_exclusion")
        self._held_keypoint = -1
        self._held_group_id = 0
        self._attached_grasp = None
        self._grasp_may_hold = False
        self._release_probe = False
        self._contact_policy = None
        self._rebuild_and_wait("released_object_returned_to_static_map")
        return True

    def _tick(self, _event):
        if not self._planning_lock.acquire(False):
            return
        try:
            with self._lock:
                if not self._active or self._paused or not self._armed:
                    self._publish_status()
                    return
                if self._mode == "autonomous":
                    assert_release_unchanged(self._release)
                    assert_release_unchanged(self._program_approval_guard)
                program, snapshot = deepcopy(self._program), deepcopy(self._snapshot)
                tracked, sdf = deepcopy(self._tracked), deepcopy(self._sdf)
                registry = deepcopy(self._registry)
                map_status = deepcopy(self._map_status)
                joints = None if self._joints is None else self._joints.copy()
                grasp = deepcopy(self._grasp)
                grasp_batch = deepcopy(self._grasp_batch)
                attached_collision_points = self._attached_collision_points.copy()
                attached_collision_points_local = (
                    self._attached_collision_points.copy())
                sequence = self._state_sequence
                now_ros = rospy.Time.now()
                tracking_age = (now_ros - self._last_tracking_stamp).to_sec()
                map_age = (now_ros - self._last_map_stamp).to_sec()
            if any(item is None for item in (
                    program, snapshot, sdf, joints, registry, map_status)):
                raise RuntimeError("closed-loop input generation is incomplete")
            if (tracked is None or not tracked.all_valid
                    or any(not item.valid for item in tracked.keypoints)):
                raise RuntimeError("dual-view keypoint tracking is lost")
            if (tracking_age < 0.0 or map_age < 0.0
                    or tracking_age > self._tracking_max_age
                    or map_age > self._map_max_age):
                raise RuntimeError("tracking_or_map_generation_is_stale")
            if (not sdf.valid or not snapshot.valid or not program.approved
                    or program.snapshot_id != snapshot.snapshot_id):
                raise RuntimeError("program_snapshot_map_binding_is_invalid")
            if (map_status.state != SafeMappingStatus.READY
                    or not map_status.map_query_allowed
                    or not map_status.planning_safe
                    or map_status.map_generation_uuid
                    != sdf.map_generation_uuid):
                raise RuntimeError("safe_map_generation_is_not_ready")
            if (compute_program_sha256(program.program_directory)
                    != program.program_sha256):
                raise RuntimeError("approved program hash changed")
            keypoints = np.asarray([[item.position.x, item.position.y, item.position.z]
                                    for item in tracked.keypoints], dtype=float)
            groups = np.asarray([item.rigid_group_id for item in tracked.keypoints], dtype=int)
            if len(keypoints) != len(snapshot.keypoints.keypoints):
                raise RuntimeError("tracked keypoint layout changed")
            previous_held_group = self._held_group_id
            attached = self._synchronize_attachment(
                registry, program, snapshot)
            if previous_held_group > 0 and attached is None:
                stages = [index + 1 for index, value in enumerate(
                    program.grasp_keypoints)
                    if int(value) >= 0
                    and int(snapshot.keypoints.keypoints[
                        int(value)].rigid_group_id) == previous_held_group]
                eligible_stages = [value for value in stages
                                   if value <= self._stage]
                if not eligible_stages:
                    raise RuntimeError("detached_object_has_no_grasp_stage")
                self._stage = max(eligible_stages)
                if self._coordinator is not None:
                    self._coordinator.enter_stage(self._stage)
                    self._grasp_attempt = self._coordinator.grasp_attempt
                self._backtracks += 1
                self._success_ticks = 0
                self._grasp = self._grasp_batch = None
                self._attached_grasp = None
                self._rebuild_and_wait("verified_detachment_backtrack")
                self._reason = "detachment_backtrack_stage_{}".format(
                    self._stage)
                return
            matrix = self._ik.forward(joints)
            position, quaternion = self._planner.modules.transform_utils.mat2pose(matrix)
            ee = np.r_[position, quaternion]
            with self._lock:
                if self._backtrack_if_needed(ee, keypoints):
                    return
            shape = (sdf.size_x, sdf.size_y, sdf.size_z)
            sdf_lower = np.asarray([sdf.bounds_min.x, sdf.bounds_min.y,
                                    sdf.bounds_min.z], dtype=float)
            sdf_upper = np.asarray([sdf.bounds_max.x, sdf.bounds_max.y,
                                    sdf.bounds_max.z], dtype=float)
            for solver_config in (self._planner.subgoal_config,
                                  self._planner.path_config):
                if (not np.allclose(sdf_lower, solver_config["bounds_min"],
                                    atol=1e-7, rtol=0.0)
                        or not np.allclose(
                            sdf_upper, solver_config["bounds_max"],
                            atol=1e-7, rtol=0.0)):
                    raise RuntimeError(
                        "SDF and official solver workspace axes differ")
            expected_shape = tuple(len(axis) for axis in inclusive_grid_axes(
                sdf_lower,sdf_upper,sdf.resolution_m))
            if tuple(shape) != expected_shape:
                raise RuntimeError(
                    "SDF shape does not match endpoint-inclusive solver axes")
            sdf_voxels = np.asarray(sdf.distances_m, dtype=float).reshape(shape)
            # Official ReKep rigidly transforms this cloud with candidate EE
            # poses.  It must therefore contain only geometry rigidly attached
            # to the EE, never links from the rest of the arm.
            collision_points, _radii = self._collision_sampler.samples(
                joints, links=("link6", "gripper_base"))
            if self._held_keypoint >= 0:
                if not len(attached_collision_points):
                    raise RuntimeError("held object collision cloud is unavailable")
                attached_transform = self._ik.link_transforms(joints)['gripper_base']
                attached_collision_points = (
                    attached_collision_points @ attached_transform[:3,:3].T
                    + attached_transform[:3,3])
                collision_points = np.vstack(
                    (collision_points, attached_collision_points))
            grasp_index = int(program.grasp_keypoints[self._stage - 1])
            grasp_target = None
            if grasp_index >= 0:
                if grasp is not None and grasp.map_generation_uuid != sdf.map_generation_uuid:
                    with self._lock:
                        self._grasp = self._grasp_batch = None
                    grasp = grasp_batch = None
                if grasp is None:
                    if (grasp_batch_matches_binding(
                            grasp_batch, program.session_id, program.program_sha256,
                            snapshot.snapshot_id, sdf.map_generation_uuid,
                            self._stage, self._grasp_attempt)
                            and grasp_batch.status != 'safe_anygrasp_candidates_available'):
                        self._record_grasp_inference_failure(self._stage, grasp_batch.status)
                    self._request_grasp_target(program, snapshot, sdf, sequence)
                    self._publish_status()
                    return
                grasp_target = self._prepare_grasp_target(
                    grasp, registry, map_status, program, snapshot, keypoints, grasp_index)
            else:
                self._contact_policy = None
                self._approach_opening_m = None
            contact_policy = self._contact_policy
            layout = hashlib.sha256(json.dumps(
                [(int(item.id), int(item.rigid_group_id))
                 for item in snapshot.keypoints.keypoints],
                separators=(",", ":")).encode("utf-8")).hexdigest()
            generation = PlanningGeneration(
                snapshot.snapshot_id, layout, sdf.map_generation_uuid,
                program.program_sha256, sequence)
            self._fresh_motion_inputs()
            backend = self._backend()
            opening = (contact_policy.opening_m if contact_policy is not None
                       else self._gripper_opening_m)
            def valid_joint(q):
                if not backend.self_clear(q,opening):
                    return False
                try:
                    audit_joint_path([q],self._collision_sampler,sdf,
                        minimum_clearance_m=.01,
                        contact_policy=contact_policy,
                        attached_points_local=attached_collision_points_local
                            if self._held_keypoint >= 0 else None)
                    return True
                except TrajectoryAuditError:
                    return False
            request = RealtimePlanningRequest(
                generation=generation, program_directory=program.program_directory,
                stage=self._stage, ee_pose=ee, joint_positions=joints,
                keypoints=keypoints, rigid_group_ids=groups,
                held_keypoint=self._held_keypoint, sdf_voxels=sdf_voxels,
                collision_points=collision_points,
                grasping_cost_fn=self._grasp_cost,
                is_grasp_stage=grasp_index >= 0,
                grasp_target_pose=grasp_target,
                joint_validity_fn=valid_joint,
                joint_fallback_fn=lambda start,goal,check,budget,seed: backend.plan(
                    self._ik._lower,self._ik._upper,start,goal,check,budget,seed,opening_m=opening),
                solver_call_timeouts=(30.,20.))
            context = {'identity': (program.program_sha256,snapshot.snapshot_id,layout,
                self._stage, self._grasp_target_key if grasp_index >= 0 else None,
                tuple(sorted((o.object_uuid,int(o.rigid_group_id),int(o.state))
                                        for o in registry.objects))),
                'joints':joints.copy(),'keypoints':keypoints.copy()}
            worker = self._planning_worker
            if worker.process is None:
                state = self._trajectory_client.get_state()
                if state in (GoalStatus.PENDING,GoalStatus.ACTIVE,GoalStatus.PREEMPTING):
                    self._reason = 'waiting_for_previous_prefix'
                    return
                worker.start(context['identity'],context,
                             lambda: solve_with_history(self._planner,request))
                self._reason = 'planning_worker_running_no_motion'
                self._publish_status()
                return
            if worker.key != context['identity']:
                worker.cancel()
                self._reason = 'planning_identity_changed_replan'
                return
            completed = worker.poll()
            if completed is None:
                self._reason = 'planning_worker_running_no_motion'
                self._publish_status()
                return
            validate_returned_context(worker.context,context)
            result,history,targets = completed
            if result.generation.map_uuid == generation.map_uuid:
                restore_history(self._planner,request,history,targets)
            else:
                self._planner.invalidate()
            # A result is a candidate until checked against this latest map/state.
            connector = np.linspace(joints,result.joint_path[0],3)
            actual_path = np.vstack([connector,result.joint_path[1:]])
            self._audit_executable(actual_path,sdf,keypoints,contact_policy)
            subgoals,paths = load_official_stage_constraints(program.program_directory,
                self._stage,len(keypoints),self._grasp_cost)
            subgoals,paths = grasp_target_constraints(subgoals,paths,grasp_target)
            actual_poses = []
            for q in actual_path:
                xyz,quat = self._planner.modules.transform_utils.mat2pose(self._ik.forward(q))
                actual_poses.append(np.r_[xyz,quat])
            self._planner._check_constraints(ee,np.vstack([ee[:3],keypoints]),
                self._planner._movable_mask(groups,self._held_keypoint),
                result.semantic_subgoal_pose,np.asarray(actual_poses),subgoals,paths)
            result = replace(result,joint_path=actual_path,cartesian_path=np.asarray(actual_poses),
                             generation=generation)
            endpoint_links = None
            # Only audited target fingertip contacts are allowed near the grasp TCP.
            audit_joint_path(
                result.joint_path, self._collision_sampler, sdf,
                minimum_clearance_m=float(rospy.get_param(
                    "~minimum_clearance_m", 0.01)),
                final_waypoint_links=endpoint_links,
                contact_policy=contact_policy,
                attached_points_local=(
                    attached_collision_points_local
                    if self._held_keypoint >= 0 else None))
            prefix = self._prefix(
                result.joint_path, joints, sdf.map_generation_uuid)
            self._audit_executable([p.positions for p in prefix.points],sdf,keypoints,contact_policy)
            # Use measured position AND orientation, with tighter grasp tolerances.
            satisfied = pose_reached(
                ee, result.target_pose,
                min(.003,self._goal_position_tolerance) if grasp_index >= 0
                    else self._goal_position_tolerance,
                min(.03,self._goal_rotation_tolerance) if grasp_index >= 0
                    else self._goal_rotation_tolerance)
            event_will_trigger = bool(
                satisfied
                and self._success_ticks + 1 >= self._success_ticks_required)
            authorized = bool(self._mode == "autonomous"
                              and self._allow_commands
                              and not event_will_trigger)
            horizon = ReKepHorizon(
                header=sdf.header, session_id=program.session_id,
                program_sha256=program.program_sha256,
                snapshot_id=snapshot.snapshot_id,
                map_generation_uuid=sdf.map_generation_uuid,
                state_sequence=sequence, stage_index=self._stage,
                semantic_subgoal_pose=_pose_message(
                    result.semantic_subgoal_pose),
                target_pose=_pose_message(result.target_pose),
                predicted_cartesian_path=PoseArray(
                    header=sdf.header,
                    poses=[_pose_message(value) for value in result.cartesian_path]),
                authorized_prefix=prefix, predicted_horizon_s=0.5,
                authorized_duration_s=0.1,
                planning_latency_s=result.planning_latency_s,
                subgoal_constraint_values=list(result.subgoal_values),
                maximum_path_constraint_values=list(result.path_max_values),
                from_scratch=result.subgoal_from_scratch or result.path_from_scratch,
                valid=True, authorized=authorized,
                status=("authorized_short_horizon" if authorized else
                        "stage_event_pending" if event_will_trigger else
                        "shadow_validated"))
            if grasp_index >= 0 and self._mode == 'autonomous' and self._allow_commands:
                self._ensure_grasp_open(grasp)
            self._horizon_pub.publish(horizon)
            if authorized:
                state = self._trajectory_client.get_state()
                if state in (GoalStatus.PENDING, GoalStatus.ACTIVE,
                             GoalStatus.PREEMPTING, GoalStatus.RECALLING):
                    self._reason = "previous_short_horizon_still_active"
                elif state in (GoalStatus.ABORTED, GoalStatus.REJECTED):
                    raise RuntimeError(
                        "previous_short_horizon_failed:{}".format(state))
                else:
                    self._assert_motion_authorized()
                    current_joints,current_grid = self._fresh_motion_inputs()
                    if (current_grid.map_generation_uuid != sdf.map_generation_uuid
                            or np.max(np.abs(current_joints-joints)) > .01):
                        raise RuntimeError('state_changed_before_prefix_dispatch')
                    current_points = np.array([[k.position.x,k.position.y,k.position.z]
                                               for k in self._tracked.keypoints])
                    self._audit_executable([p.positions for p in prefix.points],
                                           current_grid,current_points,self._contact_policy)
                    smooth = smooth_pchip_trajectory(JOINT_NAMES,result.joint_path,joints,
                        maximum_velocity_rad_s=self._joint_velocity,
                        maximum_acceleration_rad_s2=.10,maximum_jerk_rad_s3=.50)
                    self._audit_executable(smooth.positions,current_grid,current_points,self._contact_policy)
                    # Same guarded 100 ms executor as local grasp/withdrawal;
                    # no single long trajectory is sent to the controller.
                    self._send_local_trajectory(smooth,current_grid,current_points)
            event_due = False
            with self._lock:
                self._last_plan_latency = result.planning_latency_s
                self._planning_rate.mark(time.monotonic())
                self._last_plan_time = time.monotonic()
                # The paper reports roughly one second for the first global
                # solve; only warm-start replans are subject to the realtime
                # deadline gate.
                cold_start = bool(
                    result.subgoal_from_scratch or result.path_from_scratch)
                self._deadline_misses = (
                    0 if cold_start else self._deadline_misses + 1
                    if result.planning_latency_s > self._deadline else 0)
                if self._deadline_misses >= self._maximum_deadline_misses:
                    self._paused = True
                    self._safe_hold("planning_deadline_missed_consecutively")
                    return
                self._success_ticks = self._success_ticks + 1 if satisfied else 0
                if self._success_ticks >= self._success_ticks_required:
                    event_due = True
                else:
                    self._reason = "short_horizon_planned"
                self._tick_count += 1
                self._publish_status()
            if event_due:
                if grasp_index >= 0:
                    self._validate_grasp_binding(
                        grasp, registry, map_status, program, snapshot,
                        self._stage, grasp_index)
                    candidate_matrix = _grasp_matrix(grasp)
                    candidate_tcp = np.asarray([
                        grasp.tcp_position.x, grasp.tcp_position.y,
                        grasp.tcp_position.z], dtype=float)
                    candidate_is_local = bool(
                        np.all(np.isfinite(candidate_tcp))
                        and np.linalg.norm(
                            candidate_tcp - keypoints[grasp_index]) <= 0.10
                        and np.linalg.norm(
                            candidate_matrix[:3, 3]
                            - result.semantic_subgoal_pose[:3]) <= 0.10)
                    if not candidate_is_local:
                        with self._lock:
                            self._record_grasp_inference_failure(
                                self._stage,
                                "candidate_invalid_after_replanning")
                            self._publish_status()
                        return
                completed = self._stage_event(
                    program, snapshot, registry, map_status, sdf, joints,
                    keypoints, grasp, result.semantic_subgoal_pose)
                with self._lock:
                    if completed:
                        if self._stage >= program.num_stages:
                            self._active = False
                            self._task_finished = (True, "all_stages_complete")
                            self._safe_hold("task_complete")
                        else:
                            self._stage += 1
                            if self._coordinator is not None:
                                self._coordinator.enter_stage(self._stage)
                                self._grasp_attempt = (
                                    self._coordinator.grasp_attempt)
                            self._success_ticks = 0
                            self._grasp = self._grasp_batch = None
                            self._reason = "autonomous_stage_transition"
                    self._publish_status()
        except Exception as exc:
            with self._lock:
                self._paused = True
                self._safe_hold("{}:{}".format(type(exc).__name__, exc))
                self._publish_status()
        finally:
            self._planning_lock.release()

    def _publish_status(self):
        now = rospy.Time.now()
        monotonic_now = time.monotonic()
        tracking_age = ((now - self._last_tracking_stamp).to_sec()
                        if self._last_tracking_stamp != rospy.Time(0)
                        else float("inf"))
        map_age = ((now - self._last_map_stamp).to_sec()
                   if self._last_map_stamp != rospy.Time(0)
                   else float("inf"))
        state = (ClosedLoopStatus.DISARMED if not self._armed else
                 ClosedLoopStatus.PAUSED if self._paused else
                 ClosedLoopStatus.EXECUTING if self._active else
                 ClosedLoopStatus.ARMED)
        self._status_pub.publish(ClosedLoopStatus(
            header=Header(stamp=rospy.Time.now(), frame_id=self._base_frame),
            state=state, state_name={0: "DISARMED", 3: "ARMED", 5: "EXECUTING",
                                    6: "PAUSED"}.get(state, "STATE"),
            session_id="" if self._program is None else self._program.session_id,
            stage_index=self._stage, grasp_attempt=self._grasp_attempt,
            tick=self._tick_count,
            tracking_rate_hz=self._tracking_rate.rate(
                monotonic_now, self._tracking_max_age),
            mapping_rate_hz=self._mapping_rate.rate(
                monotonic_now, self._map_max_age),
            planning_rate_hz=self._planning_rate.rate(
                monotonic_now, max(2.0 / self._rate_hz, self._deadline)),
            tracking_age_s=max(0.0, tracking_age),
            map_age_s=max(0.0, map_age),
            planning_latency_s=self._last_plan_latency,
            deadline_miss_count=self._deadline_misses,
            backtrack_count=self._backtracks,
            hardware_commands_allowed=self._armed and self._allow_commands,
            reason=self._reason))


if __name__ == "__main__":
    rospy.init_node("rekpiper_closed_loop")
    ClosedLoopNode()
    rospy.spin()
