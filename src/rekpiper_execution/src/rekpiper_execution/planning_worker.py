"""One cancellable planning process. The child has no execution responsibilities."""
import multiprocessing
import threading
import time


def _synchronized(method):
    def call(self,*args,**kwargs):
        with self._lock:
            return method(self,*args,**kwargs)
    return call


def _run(connection, function):
    try:
        connection.send(('ok', function()))
    except Exception as exc:
        connection.send(('error', type(exc).__name__+': '+str(exc)))
    finally:
        connection.close()


def solve_with_history(planner, request):
    result = planner.solve(request)
    history = {stage: [solver.last_opt_result for solver in pair]
               for stage,pair in planner._solvers.items()}
    return result, history, planner._last_target


def restore_history(planner, request, history, targets):
    planner._bind(request.generation)
    planner._enter_stage(request.stage)
    for stage, results in history.items():
        for solver, value in zip(planner._solver_pair(stage,request.joint_positions),results):
            solver.last_opt_result = value
    planner._last_target = targets


class PlanningWorker:
    def __init__(self, timeout_s=60.):
        if not 0 < timeout_s <= 60.:
            raise ValueError('planning timeout must be in (0,60]')
        self.timeout_s = timeout_s
        self._lock = threading.RLock()
        self.process = self.connection = None
        self.key = self.context = None

    @_synchronized
    def start(self, key, context, function):
        if self.process is not None:
            raise RuntimeError('planning worker already active')
        # Linux-only workspace; fork inherits pure numerical planner inputs.
        # No ROS calls or GPU inference are permitted in function.
        ctx = multiprocessing.get_context('fork')
        self.connection, child = ctx.Pipe(duplex=False)
        self.key, self.context = key, context
        self.started = time.monotonic()
        self.process = ctx.Process(target=_run, args=(child,function), daemon=True)
        self.process.start()
        child.close()

    @_synchronized
    def poll(self):
        if self.process is None:
            return None
        if self.connection.poll():
            try:
                kind, value = self.connection.recv()
            except EOFError:
                kind, value = 'error', 'planning_worker_exited_without_result'
            self.cancel()
            if kind != 'ok':
                raise RuntimeError(value)
            return value
        if time.monotonic()-self.started > self.timeout_s:
            self.cancel()
            raise RuntimeError('planning_worker_timeout_60s')
        if not self.process.is_alive():
            self.cancel()
            raise RuntimeError('planning_worker_died')
        return None

    @_synchronized
    def cancel(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=.2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=.2)
            self.connection.close()
        self.process = self.connection = None


def validate_returned_context(old, new, position_tolerance=.005, joint_tolerance=.01):
    """Keep identity exact; small measured drift still requires full reaudit."""
    import numpy as np
    if old['identity'] != new['identity']:
        raise RuntimeError('planning_identity_changed')
    for key, limit in [('joints',joint_tolerance),('keypoints',position_tolerance)]:
        a,b = np.asarray(old[key]),np.asarray(new[key])
        if a.shape != b.shape or not np.all(np.isfinite(b)) or np.max(np.abs(a-b)) > limit:
            raise RuntimeError('planning_'+key+'_changed')
