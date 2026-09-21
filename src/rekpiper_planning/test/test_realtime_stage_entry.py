#!/usr/bin/env python3

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from rekpiper_planning.realtime_planner import (
    PersistentReKepPlanner, PlanningGeneration, RealtimePlanningRequest,
    RealtimePlanningError, grasp_execution_pose, grasp_target_constraints)


class _Transforms:
    @staticmethod
    def mat2pose(matrix):
        return matrix[:3, 3], np.array([0., 0., 0., 1.])

    @staticmethod
    def pose2mat(value):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(value[0])
        return pose


class _Utils:
    @staticmethod
    def get_linear_interpolation_steps(_start, _end, _position, _rotation):
        return 4

    @staticmethod
    def spline_interpolate_poses(controls, count):
        controls = np.asarray(controls)
        return np.asarray([
            controls[0], controls[1],
            0.5 * (controls[1] + controls[-1]), controls[-1]])[:count]

    @staticmethod
    def linear_interpolate_poses(start, end, count):
        return np.linspace(start, end, count)


def modules():
    return SimpleNamespace(transform_utils=_Transforms(), utils=_Utils())


class _Solver:
    def __init__(self, value):
        self.last_opt_result = value


class RealtimeStageEntryTest(unittest.TestCase):
    def test_detector_target_is_passed_to_path_solver_without_semantic_solve_or_retreat(self):
        planner = PersistentReKepPlanner.__new__(PersistentReKepPlanner)
        planner.modules = modules()
        planner._bind = planner._enter_stage = MagicMock()
        planner.continuity_guard_enabled = True
        planner.maximum_warm_target_jump_m = .20
        planner._last_target = {}
        planner.path_config = {'constraint_tolerance': .0001}
        planner._transformed = lambda pose, ee, points, movable: np.vstack([pose[:3], points[1:]])
        planner._official_spline_path = lambda controls: controls
        planner._joint_path = lambda poses, joints: np.c_[poses[:, :3], np.zeros((len(poses), 3))]
        planner.ik_solver = SimpleNamespace(forward=lambda q: _Transforms.pose2mat([q[:3], q[3:]]))
        target = np.array([.35, .02, .20, 0., 0., 0., 1.])
        subgoal_solver = MagicMock(last_opt_result=None)
        subgoal_solver.solve.side_effect = AssertionError('must not move detector target')
        path_solver = MagicMock(last_opt_result=None)
        path_solver.solve.return_value = (target[None, :], {})
        planner._solver_pair = lambda stage, joints: (subgoal_solver, path_solver)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath('metadata.json').write_text(json.dumps({'instruction': 'grasp the cube'}))
            constraint = ('def stage1_subgoal_constraint1(end_effector, keypoints):\n'
                          '    return np.linalg.norm(end_effector - keypoints[0])\n')
            root.joinpath('program.py').write_text(
                'num_stages = 1\ngrasp_keypoints = [0]\nrelease_keypoints = [-1]\n' + constraint)
            root.joinpath('stage1_subgoal_constraints.txt').write_text(constraint)
            request = RealtimePlanningRequest(
                PlanningGeneration('snapshot', 'layout', 'map', 'hash', 1),
                directory, 1, np.array([.2, 0., .3, 0., 0., 0., 1.]),
                np.zeros(6), np.array([[.37, .02, .20]]), np.array([1]), -1,
                np.zeros((2, 2, 2)), np.zeros((1, 3)), lambda index: 1.,
                is_grasp_stage=True, grasp_target_pose=target)
            result = planner.solve(request)
            planner._joint_path = lambda poses, joints: np.c_[
                poses[:, :3] + np.array([.01, 0., 0.]), np.zeros((len(poses), 3))]
            with self.assertRaisesRegex(RealtimePlanningError, 'does_not_reach_grasp_tcp'):
                planner.solve(request)
        subgoal_solver.solve.assert_not_called()
        np.testing.assert_array_equal(path_solver.solve.call_args[0][1], target)
        np.testing.assert_array_equal(result.target_pose, target)
        np.testing.assert_array_equal(result.semantic_subgoal_pose, target)
        np.testing.assert_array_equal(result.cartesian_path[-1], target)
        self.assertEqual(result.diagnostics['subgoal_source'], 'bound_anygrasp_tcp')
        self.assertEqual(result.subgoal_values, (0.,))

    def test_detector_constraint_does_not_change_other_stages_or_accept_bad_pose(self):
        functions = [lambda ee, keypoints: 7.]
        paths = [lambda ee, keypoints: 3.]
        self.assertEqual(grasp_target_constraints(functions, paths, None), (functions, paths))
        for target in (np.zeros(7), np.array([0, 0, 0, 0, 0, 0, np.nan])):
            with self.assertRaises(RealtimePlanningError):
                grasp_target_constraints(functions, [], target)
        with self.assertRaisesRegex(RealtimePlanningError, 'invalid detector grasp stage'):
            grasp_target_constraints(functions, paths, np.array([0, 0, 0, 0, 0, 0, 1]))

    def test_grasp_request_without_detector_pose_fails_before_solving(self):
        request = RealtimePlanningRequest(
            PlanningGeneration('snapshot', 'layout', 'map', 'hash', 1),
            '/unused', 1, np.array([0., 0., .3, 0., 0., 0., 1.]),
            np.zeros(6), np.zeros((1, 3)), np.ones(1), -1,
            np.zeros((2, 2, 2)), np.zeros((1, 3)), lambda index: 1.,
            is_grasp_stage=True)
        with self.assertRaisesRegex(RealtimePlanningError, 'requires a bound detector'):
            PersistentReKepPlanner._validate(request)

    def test_grasp_target_retreats_five_centimeters_along_ee_x(self):
        core = modules()
        semantic = np.asarray([0.4, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0])
        execution = grasp_execution_pose(
            semantic, core.transform_utils, 0.10)
        np.testing.assert_allclose(
            execution[:3], [0.35, 0.1, 0.3], atol=1e-9)
        np.testing.assert_array_equal(execution[3:], semantic[3:])

    def test_official_spline_then_safety_resampling(self):
        core = modules()
        planner = PersistentReKepPlanner.__new__(PersistentReKepPlanner)
        planner.modules = core
        planner.maximum_position_step_m = 0.005
        planner.maximum_rotation_step_rad = np.deg2rad(1.0)
        planner.official_interpolate_position_step_m = 0.05
        planner.official_interpolate_rotation_step_rad = 0.34
        controls = np.asarray([
            [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
            [0.04, 0.03, 0.22, 0.0, 0.0, 0.0, 1.0],
            [0.10, 0.00, 0.25, 0.0, 0.0, 0.0, 1.0],
        ])
        count = core.utils.get_linear_interpolation_steps(
            controls[0], controls[-1], 0.05, 0.34)
        official = core.utils.spline_interpolate_poses(controls, count)
        actual = planner._official_spline_path(controls)
        expected = planner._safety_resample(np.asarray(official))
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(
            actual[[0, -1]], controls[[0, -1]], atol=1e-15)

    def test_reentering_a_stage_restores_global_first_iteration(self):
        planner = PersistentReKepPlanner.__new__(PersistentReKepPlanner)
        stage_one = (_Solver(object()), _Solver(object()))
        planner._solvers = {1: stage_one}
        planner._last_target = {1: object()}
        planner._active_stage = 2
        planner._enter_stage(1)
        self.assertIsNone(stage_one[0].last_opt_result)
        self.assertIsNone(stage_one[1].last_opt_result)
        self.assertNotIn(1, planner._last_target)
        self.assertEqual(planner._active_stage, 1)

    def test_same_stage_preserves_warm_start(self):
        marker = object()
        stage_one = (_Solver(marker), _Solver(marker))
        planner = PersistentReKepPlanner.__new__(PersistentReKepPlanner)
        planner._solvers = {1: stage_one}
        planner._last_target = {}
        planner._active_stage = 1
        planner._enter_stage(1)
        self.assertIs(stage_one[0].last_opt_result, marker)

    def test_map_uuid_change_invalidates_same_stage_warm_start(self):
        planner = PersistentReKepPlanner.__new__(PersistentReKepPlanner)
        planner._solvers = {1: (_Solver(object()), _Solver(object()))}
        planner._last_target = {1: object()}
        planner._active_stage = 1
        planner._binding = ("program", "layout", "map-a")
        planner._bind(PlanningGeneration(
            "snapshot", "layout", "map-b", "program", 1))
        self.assertEqual(planner._solvers, {})
        self.assertIsNone(planner._active_stage)
        self.assertEqual(planner._binding, ("program", "layout", "map-b"))


if __name__ == "__main__":
    unittest.main()
