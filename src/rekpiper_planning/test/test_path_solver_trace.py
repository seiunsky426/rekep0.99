#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.optimize import minimize, dual_annealing

from rekpiper_planning.path_solver_trace import PathSolverTrace, control_poses
from rekpiper_planning.realtime_planner import PersistentReKepPlanner


CALLS = []


def objective(x, bounds, start, end, return_debug_dict=False):
    CALLS.append(np.asarray(x).copy())
    value = float(np.sum(np.asarray(x) ** 2))
    debug = {'total_cost': value, 'path_length_cost': value,
             'reset_reg_cost': 7., 'ik_pos_error': np.array([np.nan])}
    return (value, debug) if return_debug_dict else value


class Solver:
    def solve(self, global_search=False):
        bounds = np.tile([[-1., 1.]], (6, 1))
        args = (bounds, np.zeros(6), np.ones(6))
        if global_search:
            result = dual_annealing(objective, bounds, args=args, x0=np.full(6, .4),
                                    maxfun=12, seed=7, no_local_search=True)
        else:
            result = minimize(fun=objective, x0=np.full(6, .4), args=args,
                              bounds=bounds, method='SLSQP')
        return result.x, objective(result.x, *args, return_debug_dict=True)[1]


class PathSolverTraceTest(unittest.TestCase):
    def test_trace_preserves_result_and_objective_call_count_for_both_optimizers(self):
        original = Solver.solve.__globals__['objective']
        for global_search in (False, True):
            with self.subTest(global_search=global_search), tempfile.TemporaryDirectory() as root:
                CALLS.clear()
                expected, _ = Solver().solve(global_search)
                expected_calls = len(CALLS)
                CALLS.clear()
                trace = PathSolverTrace(root, {'solver_profile': 'official_exact'})
                actual, debug = trace.solve(Solver(), global_search)
                trace.record_path(control_poses=np.zeros((3, 7)))
                trace.finish('planner_returned')
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(len(CALLS), expected_calls)
                self.assertEqual(trace.evaluations, expected_calls)
                self.assertIs(Solver.solve.__globals__['objective'], original)
                events = [json.loads(line) for line in (trace.directory / 'events.jsonl').read_text().splitlines()]
                evaluations = [e for e in events if e['event'] == 'evaluation']
                self.assertEqual(len(evaluations), expected_calls)
                self.assertTrue(evaluations[-1]['final_debug_evaluation'])
                self.assertIsNone(evaluations[0]['costs']['ik_pos_error'][0])
                best = [e['best_cost'] for e in evaluations]
                self.assertTrue(np.all(np.diff(best) <= 0))
                self.assertTrue(any(e['event'] == 'initial_controls' for e in events))
                self.assertEqual(evaluations[0]['control_poses'][0][:3], [0., 0., 0.])
                self.assertEqual(evaluations[0]['control_poses'][-1][:3], [1., 1., 1.])
                pointer = json.loads((Path(root) / 'latest.json').read_text())
                self.assertEqual(Path(pointer['directory']), trace.directory)

    def test_incomplete_run_is_not_advertised_as_finished_and_events_are_flushed(self):
        with tempfile.TemporaryDirectory() as root:
            trace = PathSolverTrace(root, {})
            trace.event('test_progress', index=1)
            self.assertIn('test_progress', (trace.directory / 'events.jsonl').read_text())
            self.assertEqual(json.loads((trace.directory / 'summary.json').read_text())['status'], 'incomplete')
            self.assertFalse((Path(root) / 'latest.json').exists())
            trace.finish('planner_failed', 'timeout')
            self.assertTrue(trace.events.closed)

    def test_planner_failure_finalizes_trace_and_preserves_error(self):
        with tempfile.TemporaryDirectory() as root:
            planner = PersistentReKepPlanner.__new__(PersistentReKepPlanner)
            traces = []

            def fail(_request):
                planner._path_trace = PathSolverTrace(root, {})
                traces.append(planner._path_trace)
                raise TimeoutError('test timeout')

            planner._solve = fail
            with self.assertRaisesRegex(TimeoutError, 'test timeout'):
                planner.solve(None)
            summary = json.loads((traces[0].directory / 'summary.json').read_text())
            self.assertEqual(summary['status'], 'planner_failed')
            self.assertFalse(summary['motion_allowed'])
            self.assertIn('TimeoutError', summary['error'])
            self.assertIsNone(planner._path_trace)

    def test_control_variables_are_unnormalized_to_pose_with_unit_quaternion(self):
        bounds = np.array([[-2., 2.]] * 3 + [[-np.pi, np.pi]] * 3)
        poses = control_poses(np.full(6, .5), bounds, np.zeros(6), np.ones(6))
        np.testing.assert_allclose(poses[1, :3], [1., 1., 1.])
        np.testing.assert_allclose(np.linalg.norm(poses[:, 3:], axis=1), 1.)


if __name__ == '__main__':
    unittest.main()
