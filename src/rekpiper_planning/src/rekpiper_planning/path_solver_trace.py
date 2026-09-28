"""Per-solve, streaming PathSolver evidence without changing upstream globals."""

import csv
import json
import math
import os
from pathlib import Path
import time
from types import FunctionType
import uuid

import numpy as np
from scipy.spatial.transform import Rotation


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(json_value(value), ensure_ascii=False,
                                    allow_nan=False, indent=2), encoding='utf-8')
    os.replace(str(temporary), str(path))


def default_trace_directory():
    root = os.environ.get('REKPIPER_DATA_ROOT')
    if root is None:
        workspace = next(parent for parent in Path(__file__).resolve().parents
                         if (parent / '.catkin_workspace').exists())
        root = str(workspace / 'runtime' / 'data')
    return str(Path(root) / 'path_solver')


def control_poses(variables, bounds, start_euler, end_euler):
    bounds = np.asarray(bounds)
    raw = (np.asarray(variables) + 1.) * .5 * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    euler = np.vstack([start_euler, raw.reshape(-1, 6), end_euler])
    return np.c_[euler[:, :3], Rotation.from_euler('xyz', euler[:, 3:]).as_quat()]


class PathSolverTrace:
    """Every completed objective evaluation is flushed, including rejected trials.

    A killed worker leaves events/costs and an explicitly incomplete summary.
    The latest pointer is updated only after finalization. Logging errors are
    propagated so a run cannot silently claim to have complete evidence.
    """

    def __init__(self, root, metadata):
        self.root = Path(root).expanduser()
        self.directory = self.root / (time.strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:12])
        self.directory.mkdir(parents=True)
        self.started = time.monotonic()
        self.evaluations = 0
        self.best = math.inf
        self.summary = dict(schema_version=1, status='incomplete',
                            motion_allowed=False, **metadata)
        write_json(self.directory / 'summary.json', self.summary)
        self.events = (self.directory / 'events.jsonl').open('w', encoding='utf-8', buffering=1)
        self.cost_file = (self.directory / 'costs.csv').open('w', newline='', encoding='utf-8', buffering=1)
        self.costs = csv.writer(self.cost_file)
        self.costs.writerow(['evaluation', 'elapsed_s', 'total_cost', 'best_cost',
                             'collision_cost', 'path_length_cost', 'ik_cost',
                             'reset_reg_cost', 'path_constraint_cost', 'joint_limit_cost',
                             'table_clearance_cost', 'robot_table_clearance_cost',
                             'previous_solution_cost', 'final_debug_evaluation'])
        self.event('start', **metadata)

    def event(self, kind, **values):
        self.events.write(json.dumps(json_value(dict(
            event=kind, elapsed_s=time.monotonic() - self.started, **values)),
            ensure_ascii=False, allow_nan=False) + '\n')

    def solve(self, solver, *args, **kwargs):
        # Clone only solve's global lookup table. Neither the pinned source nor
        # process-wide modules/other solver instances are monkey-patched.
        method = solver.solve.__func__
        namespace = dict(method.__globals__)
        name = '_path_objective' if '_path_objective' in namespace else 'objective'
        objective = namespace[name]

        def observed(variables, *auxiliary, **options):
            final = bool(options.get('return_debug_dict', False))
            options['return_debug_dict'] = True
            result = objective(variables, *auxiliary, **options)
            total, debug = result[:2]
            self.evaluations += 1
            if np.isfinite(total):
                self.best = min(self.best, float(total))
            controls = control_poses(variables, *auxiliary[:3])
            self.event('evaluation', index=self.evaluations,
                       normalized_variables=variables, control_poses=controls,
                       costs=debug, best_cost=self.best, final_debug_evaluation=final)
            self.costs.writerow([self.evaluations, time.monotonic() - self.started,
                                json_value(float(total)), json_value(self.best)] +
                               [json_value(debug.get(key, 0.)) for key in (
                                   'collision_cost', 'path_length_cost', 'ik_cost',
                                   'reset_reg_cost', 'path_constraint_cost', 'joint_limit_cost',
                                   'table_clearance_cost', 'robot_table_clearance_cost',
                                   'previous_solution_cost')] + [int(final)])
            if final:
                self.summary['control_poses'] = controls
                self.summary['objective_debug'] = debug
            return result if final else total

        def optimizer(original, label):
            def run(*positional, **options):
                initial = options.get('x0')
                if initial is None and label == 'minimize' and len(positional) > 1:
                    initial = positional[1]
                auxiliary = options['args']
                self.event('initial_controls', optimizer=label,
                           normalized_variables=initial,
                           control_poses=control_poses(initial, *auxiliary[:3]),
                           variable_bounds=auxiliary[0])
                result = original(*positional, **options)
                self.event('optimizer_result', optimizer=label, success=result.success,
                           message=str(result.message), normalized_variables=result.x,
                           total_cost=result.fun, evaluations=result.nfev)
                self.summary['optimizer'] = dict(success=bool(result.success),
                                                 message=str(result.message))
                return result
            return run

        namespace[name] = observed
        for key in ('minimize', 'dual_annealing'):
            namespace[key] = optimizer(namespace[key], key)
        instrumented = FunctionType(method.__code__, namespace, method.__name__,
                                    method.__defaults__, method.__closure__)
        instrumented.__kwdefaults__ = method.__kwdefaults__
        return instrumented(solver, *args, **kwargs)

    def record_path(self, **arrays):
        self.summary.update(arrays)
        self.event('path', **arrays)
        write_json(self.directory / 'summary.json', self.summary)

    def finish(self, status, error=''):
        try:
            self.summary.update(status=status, error=error, evaluations=self.evaluations,
                                elapsed_s=time.monotonic() - self.started)
            self.event('finish', status=status, error=error)
            write_json(self.directory / 'summary.json', self.summary)
            write_json(self.root / 'latest.json', {'directory': str(self.directory.resolve())})
        finally:
            self.events.close()
            self.cost_file.close()
