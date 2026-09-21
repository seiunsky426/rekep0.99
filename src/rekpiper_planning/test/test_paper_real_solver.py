#!/usr/bin/env python3

import unittest
from types import SimpleNamespace
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from rekpiper_planning.paper_real_solver import (
    PaperRealWeights, endpoint_collision_mask, robot_table_penetration_cost,
    table_penetration_cost, PaperRealPathSolver, PaperRealSubgoalSolver)
from rekpiper_planning.solver_acceptance import (
    REQUIRED_SCENARIOS, SolverAcceptanceError, calibrate_replay_report,
    approve_candidate, load_workspace_table_height, validate_acceptance)
from rekpiper_planning.upstream import EXPECTED_OFFICIAL_COMMIT


class _Utilities:
    @staticmethod
    def batch_transform_points(points, transforms):
        return np.asarray([
            points @ transform[:3, :3].T + transform[:3, 3]
            for transform in transforms])


class _IKResult:
    success = True
    cspace_position = np.zeros(7)


class _IK:
    def solve(self, _pose, max_iterations, initial_joint_pos):
        self.last_seed = np.asarray(initial_joint_pos)
        return _IKResult()


def trial(scenario, jitter=0.01, latency=0.05, success=True):
    return {
        "scenario": scenario,
        "success": success,
        "collision_failures": 0,
        "unknown_space_passes": 0,
        "maximum_constraint_violation": 0.0,
        "planning_latency_s": latency,
        "warm": True,
        "path_jitter_m": jitter,
        "map_uuid_first_solve_from_scratch": scenario == "map_uuid_change",
    }


class PaperRealSolverTest(unittest.TestCase):
    def test_cold_solvers_actually_run_bounded_local_search(self):
        # Cheap quadratic isolates optimizer routing from robot geometry.
        def pose2mat(value):
            matrix = np.eye(4)
            matrix[:3, 3] = value[0]
            return matrix

        utils = SimpleNamespace(
            normalize_vars=lambda x, b: 2 * (x - np.asarray(b)[:, 0]) /
                (np.asarray(b)[:, 1] - np.asarray(b)[:, 0]) - 1,
            unnormalize_vars=lambda x, b: (x + 1) / 2 *
                (np.asarray(b)[:, 1] - np.asarray(b)[:, 0]) + np.asarray(b)[:, 0],
            transform_keypoints=lambda _t, k, _m: k,
            get_linear_interpolation_steps=lambda *_: 3,
            linear_interpolate_poses=np.linspace)
        transforms = SimpleNamespace(
            pose2mat=pose2mat, quat2euler=lambda _: np.zeros(3),
            euler2quat=lambda _: np.array([0., 0., 0., 1.]),
            convert_pose_euler2quat=lambda p: np.c_[p[:, :3],
                np.tile([0., 0., 0., 1.], (len(p), 1))])
        modules = SimpleNamespace(utils=utils, transform_utils=transforms)
        config = dict(bounds_min=np.full(3, -1.), bounds_max=np.ones(3),
                      sampling_maxfun=100, max_collision_points=10,
                      minimizer_options={'maxiter': 10}, constraint_tolerance=1.e-4,
                      opt_pos_step_size=.2, opt_rot_step_size=.5,
                      opt_interpolate_pos_step_size=.05,
                      opt_interpolate_rot_step_size=.1)
        pose = np.array([0., 0., 0., 0., 0., 0., 1.])
        for is_path in (False, True):
            with self.subTest(path_solver=is_path):
                def objective(x, *args, **kwargs):
                    cost = float(np.sum((x - .25) ** 2))
                    if kwargs.get('return_debug_dict'):
                        result = (cost, {'total_cost': cost})
                        return result + (np.eye(4)[None],) if is_path else result
                    return cost

                name = '_path_objective' if is_path else '_subgoal_objective'
                with patch('rekpiper_planning.paper_real_solver.' + name, objective), \
                        patch('scipy.optimize._dual_annealing.minimize', wraps=minimize) as local:
                    arguments = (config, None, np.zeros(7), modules, PaperRealWeights(1., 1., 1.))
                    if is_path:
                        solver = PaperRealPathSolver(*arguments, 0., lambda _: np.zeros((1, 3)))
                        _, debug = solver.solve(pose, pose, np.zeros((1, 3)), [True], [],
                            np.zeros((2, 2, 2)), np.zeros((1, 3)), np.zeros(7), from_scratch=True)
                    else:
                        solver = PaperRealSubgoalSolver(*arguments)
                        _, debug = solver.solve(pose, np.zeros((1, 3)), [True], [], [],
                            np.zeros((2, 2, 2)), np.zeros((1, 3)), False, np.zeros(7), from_scratch=True)
                    self.assertGreater(local.call_count, 0)
                    self.assertLess(debug['total_cost'], 1.e-8)
                    for call in local.call_args_list:
                        self.assertEqual(call.kwargs['method'], 'SLSQP')
                        self.assertEqual(call.kwargs['bounds'], [(-1., 1.)] * 6)

    def test_zero_exemption_checks_short_path_including_endpoints(self):
        poses = np.repeat(np.eye(4)[None], 5, axis=0)
        poses[:, 0, 3] = np.linspace(0., .075, 5)
        end = np.asarray([.075, 0., 0., 0., 0., 0.])
        self.assertFalse(endpoint_collision_mask(poses, np.zeros(6), end, .05).any())
        self.assertTrue(endpoint_collision_mask(poses, np.zeros(6), end, 0.).all())

    def test_endpoint_exemption_is_metric_not_sample_index(self):
        poses = np.repeat(np.eye(4)[None], 7, axis=0)
        poses[:, 0, 3] = [0.0, 0.02, 0.049, 0.051, 0.151, 0.18, 0.20]
        start = np.zeros(6)
        end = np.asarray([0.20, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(
            endpoint_collision_mask(poses, start, end, 0.05),
            [False, False, False, True, False, False, False])

    def test_table_cost_checks_every_pose_and_attached_point(self):
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[1, 2, 3] = -0.02
        points = np.asarray([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]])
        self.assertAlmostEqual(
            table_penetration_cost(poses, points, 0.0, _Utilities()),
            0.05)

    def test_full_arm_table_cost_checks_every_dense_pose(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        calls = []

        def samples(joints):
            calls.append(np.asarray(joints))
            return np.asarray([[0.0, 0.0, -0.01],
                               [0.0, 0.0, 0.02]])

        cost, feasible = robot_table_penetration_cost(
            poses, _IK(), np.zeros(7), samples, 0.0)
        self.assertAlmostEqual(cost, 0.03)
        self.assertEqual(len(calls), 3)
        self.assertTrue(np.all(feasible))

    def test_weight_schema_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            PaperRealWeights.from_mapping({
                "subgoal_consistency": -1,
                "path_consistency": 1,
                "table_clearance": 1,
            })

    def test_autonomous_table_height_requires_accepted_workspace(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "workspace.yaml"
            path.write_text(
                "status: MEASURED\nprecision_operation_allowed: false\n"
                "table_height_m: 0.01\n", encoding="utf-8")
            self.assertEqual(load_workspace_table_height(path)[0], 0.01)
            with self.assertRaisesRegex(
                    SolverAcceptanceError, "not accepted"):
                load_workspace_table_height(path, require_accepted=True)

    def test_replay_calibration_selects_only_safe_candidate(self):
        baseline = [trial(name, jitter=0.02) for name in REQUIRED_SCENARIOS]
        candidate = [trial(name, jitter=0.01) for name in REQUIRED_SCENARIOS]
        report = {
            "schema_version": 1,
            "official_commit": EXPECTED_OFFICIAL_COMMIT,
            "constraint_tolerance": 0.0001,
            "inputs": {"workspace_sha256": "w"},
            "baseline_trials": baseline,
            "candidates": [{
                "weights": {
                    "subgoal_consistency": 1.0,
                    "path_consistency": 1.0,
                    "table_clearance": 1.0,
                },
                "trials": candidate,
            }],
        }
        selected = calibrate_replay_report(report)
        self.assertFalse(selected["approved"])
        self.assertEqual(selected["profile"], "paper_real")
        with self.assertRaisesRegex(SolverAcceptanceError, "not operator-approved"):
            validate_acceptance(selected, require_approved=True)
        with self.assertRaisesRegex(SolverAcceptanceError, "input hashes"):
            approve_candidate(selected, "operator")


if __name__ == "__main__":
    unittest.main()
