#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import unittest
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import rospy


def load_node_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "closed_loop_node.py"
    spec = importlib.util.spec_from_file_location("closed_loop_node_tested", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Publisher:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class ClosedLoopConstructionTest(unittest.TestCase):
    def test_arrival_event_closes_without_a_second_approach_or_open_command(self):
        module = load_node_module()
        node = module.ClosedLoopNode.__new__(module.ClosedLoopNode)
        node._stage = 1
        node._mode, node._allow_commands = 'autonomous', True
        node._grasp_failures = {}
        events = []
        node._validate_grasp_binding = MagicMock(return_value=SimpleNamespace(
            object_uuid='cube', rigid_group_id=1))
        node._require_grasp_arrival = MagicMock(side_effect=lambda *args: events.append('arrival'))
        node._begin_grasp = lambda uuid: (events.append('begin') or SimpleNamespace(success=True))
        node._send_gripper_goal = MagicMock(side_effect=lambda command, *args:
            (events.append(command) or SimpleNamespace(stable_contact=True, final_opening_m=.05)))
        node._fresh_motion_inputs = lambda: (np.zeros(6), None)
        node._make_contact_policy = MagicMock()
        node._ik = MagicMock()
        node._owned_cloud = MagicMock()
        node._execute_grasp_approach = MagicMock()
        node._verification_probe = lambda *args: events.append('probe')
        node._wait_for_lifecycle_evidence = MagicMock()
        node._confirm_attachment = lambda uuid: SimpleNamespace(success=True)
        node._wait_for_object_state = lambda *args: SimpleNamespace(rigid_group_id=1)
        node._rebuild_and_wait = MagicMock()
        candidate = module.GraspCandidate(predicted_width_m=.05)
        candidate.grasp_pose.orientation.w = 1.
        program = SimpleNamespace(grasp_keypoints=[0], release_keypoints=[-1])
        args = (program, None, None, None, None, np.zeros(6), np.zeros((1, 3)),
                candidate, np.array([0., 0., 0., 0., 0., 0., 1.]))
        self.assertTrue(node._stage_event(*args))
        self.assertEqual(events, ['arrival', 'begin', 'arrival', module.CommandGripperGoal.CLOSE, 'probe'])
        node._execute_grasp_approach.assert_not_called()
        node._send_gripper_goal.reset_mock()
        node._require_grasp_arrival.side_effect = RuntimeError('grasp_tcp_not_reached_before_close')
        with self.assertRaisesRegex(RuntimeError, 'tcp_not_reached'):
            node._stage_event(*args)
        node._send_gripper_goal.assert_not_called()

    def test_closed_gripper_feedback_prevents_trajectory_dispatch(self):
        module = load_node_module()
        node = module.ClosedLoopNode.__new__(module.ClosedLoopNode)
        node._program = SimpleNamespace(program_sha256='program')
        node._snapshot = SimpleNamespace(snapshot_id='snapshot')
        node._tracked = SimpleNamespace(keypoints=[])
        node._contact_policy = None
        node._held_group_id = 0
        node._release_probe = False
        node._approach_opening_m, node._gripper_opening_m = .060, .030
        node._assert_motion_authorized = MagicMock()
        node._trajectory_client = MagicMock()
        with self.assertRaisesRegex(RuntimeError, 'not_open_during_approach'):
            node._send_local_trajectory(
                SimpleNamespace(positions=np.zeros((2, 6)), duration_s=1.), None, np.empty((0, 3)))
        node._trajectory_client.send_goal.assert_not_called()

    def test_target_request_has_no_trajectory_or_motion_authority(self):
        module = load_node_module()
        node = module.ClosedLoopNode.__new__(module.ClosedLoopNode)
        node._stage = 1
        node._planning_worker = MagicMock()
        node._horizon_pub = _Publisher()
        node._request_grasp_target(
            SimpleNamespace(session_id='session', program_sha256='program'),
            SimpleNamespace(snapshot_id='snapshot'),
            SimpleNamespace(header=module.Header(), map_generation_uuid='map'), 4)
        request = node._horizon_pub.messages[0]
        self.assertEqual(request.status, 'grasp_target_pending')
        self.assertTrue(request.valid)
        self.assertFalse(request.authorized)
        self.assertEqual(request.authorized_prefix.points, [])
        self.assertEqual(request.stage_index, 1)
        node._planning_worker.cancel.assert_called_once_with()

    def test_selected_target_is_frozen_for_one_attempt(self):
        module = load_node_module()
        node = module.ClosedLoopNode.__new__(module.ClosedLoopNode)
        node._lock = threading.RLock()
        node._grasp = None
        first = module.GraspCandidate(candidate_id='first', stage_index=1, grasp_attempt=1)
        node._grasp_cb(first)
        node._grasp_cb(module.GraspCandidate(candidate_id='second', stage_index=1, grasp_attempt=1))
        self.assertEqual(node._grasp.candidate_id, 'first')
        node._grasp_cb(module.GraspCandidate(candidate_id='retry', stage_index=1, grasp_attempt=2))
        self.assertEqual(node._grasp.candidate_id, 'retry')

    def test_preopen_is_confirmed_once_and_feedback_loss_stops_approach(self):
        module = load_node_module()
        node = module.ClosedLoopNode.__new__(module.ClosedLoopNode)
        node._grasp_target_key = ('target', 1)
        node._preopened_grasp_key = None
        node._gripper_opening_m = .060
        node._send_gripper_goal = MagicMock(return_value=SimpleNamespace(
            success=True, final_opening_m=.060))
        node._make_contact_policy = MagicMock()
        candidate = SimpleNamespace(object_uuid='cube', suggested_preopen_width_m=.060)
        node._ensure_grasp_open(candidate)
        node._ensure_grasp_open(candidate)
        node._send_gripper_goal.assert_called_once_with(module.CommandGripperGoal.OPEN, 'cube', .060, 1.)
        node._gripper_opening_m = .040
        with self.assertRaisesRegex(RuntimeError, 'not_open_during_approach'):
            node._ensure_grasp_open(candidate)

    def test_close_requires_measured_position_orientation_and_open_feedback(self):
        module = load_node_module()
        node = module.ClosedLoopNode.__new__(module.ClosedLoopNode)
        node._grasp_target_key = node._preopened_grasp_key = ('target', 1)
        node._approach_opening_m = node._gripper_opening_m = .060
        node._goal_position_tolerance = .01
        node._goal_rotation_tolerance = .10
        from scipy.spatial.transform import Rotation
        transforms = SimpleNamespace(mat2pose=lambda m: (m[:3, 3], Rotation.from_matrix(m[:3, :3]).as_quat()))
        node._planner = SimpleNamespace(modules=SimpleNamespace(transform_utils=transforms))
        node._fresh_motion_inputs = lambda: (np.zeros(6), None)
        node._ik = MagicMock()
        candidate = module.GraspCandidate()
        candidate.grasp_pose.orientation.w = 1.
        target = np.array([0., 0., 0., 0., 0., 0., 1.])
        for matrix in [np.eye(4), module._ee_matrix([0., 0., 0., 0., 0., .05, np.sqrt(1-.05**2)])]:
            if np.array_equal(matrix, np.eye(4)):
                matrix[0, 3] = .004
            node._ik.forward.return_value = matrix
            with self.assertRaisesRegex(RuntimeError, 'tcp_not_reached'):
                node._require_grasp_arrival(candidate, target)
        node._ik.forward.return_value = np.eye(4)
        node._require_grasp_arrival(candidate, target)
        node._gripper_opening_m = .040
        with self.assertRaisesRegex(RuntimeError, 'not_open_during_approach'):
            node._require_grasp_arrival(candidate, target)

    def test_zero_offset_grasp_probe_uses_tool_axis_and_keeps_ik_gate(self):
        module = load_node_module()
        node = module.ClosedLoopNode.__new__(module.ClosedLoopNode)
        pose = module.Pose()
        pose.orientation.y = np.sqrt(.5)
        pose.orientation.w = np.sqrt(.5)
        candidate = MagicMock(grasp_pose=pose, pregrasp_pose=pose)
        current = module._pose_matrix(pose)
        joints = np.zeros(6)
        node._fresh_motion_inputs = lambda: (joints, None)
        node._release_probe = False
        node._ik = MagicMock()
        node._ik.forward.return_value = current
        node._ik.validate_pose_sequence.return_value = {'valid': False}
        with self.assertRaisesRegex(RuntimeError, 'attachment_probe_ik_failed'):
            node._verification_probe(candidate, None, joints)
        poses, seed = node._ik.validate_pose_sequence.call_args[0]
        np.testing.assert_allclose(poses[0], current)
        np.testing.assert_allclose(poses[-1][:3, 3], [-.02, 0., 0.], atol=1e-12)
        np.testing.assert_allclose(seed, joints)

    def test_fake_ros_constructs_and_publishes_initial_status(self):
        module = load_node_module()

        def parameter(name, default=None):
            required = {
                "~subgoal_solver": {
                    "bounds_min": [-0.2, -0.6, -0.07],
                    "bounds_max": [0.8, 0.6, 0.8]},
                "~path_solver": {
                    "bounds_min": [-0.2, -0.6, -0.07],
                    "bounds_max": [0.8, 0.6, 0.8]},
                "/robot_description": "<robot name='fake'/>",
            }
            return required.get(name, default)

        server = MagicMock()
        with patch.object(module.rospy, "get_param", side_effect=parameter), \
                patch.object(module.rospy, "has_param", return_value=True), \
                patch.object(module.rospy, "Publisher", side_effect=_Publisher), \
                patch.object(module.rospy, "Subscriber", return_value=MagicMock()), \
                patch.object(module.rospy, "ServiceProxy", return_value=MagicMock()), \
                patch.object(module.rospy, "Service", return_value=MagicMock()), \
                patch.object(module.rospy, "Timer", return_value=MagicMock()), \
                patch.object(module.rospy.Time, "now",
                             return_value=rospy.Time(1.0)), \
                patch.object(module.tf2_ros, "Buffer", return_value=MagicMock()), \
                patch.object(module.tf2_ros, "TransformListener",
                             return_value=MagicMock()), \
                patch.object(module.actionlib, "SimpleActionClient",
                             return_value=MagicMock()), \
                patch.object(module.actionlib, "SimpleActionServer",
                             return_value=server), \
                patch.object(module.PiperURDFIKSolver, "from_urdf_xml",
                             return_value=MagicMock()), \
                patch.object(module, "PiperCollisionSampler",
                             return_value=MagicMock(
                                 sampled_links=("link1", "link6",
                                                "gripper_base"))), \
                patch.object(module, "PersistentReKepPlanner",
                             return_value=MagicMock()):
            node = module.ClosedLoopNode()

        server.start.assert_called_once_with()
        self.assertEqual(len(node._status_pub.messages), 1)
        status = node._status_pub.messages[0]
        self.assertEqual(status.stage_index, 1)
        self.assertEqual(status.grasp_attempt, 0)
        self.assertEqual(status.reason, "initialized_disarmed")


if __name__ == "__main__":
    unittest.main()
