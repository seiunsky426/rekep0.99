"""Native OMPL/MoveIt bridge. All callbacks are local immutable-scene checks."""
import ctypes
import os
from pathlib import Path
import time
import numpy as np


class MotionBackendError(RuntimeError):
    pass


class MotionBackend:
    def __init__(self, robot_xml):
        candidates = [Path(p)/'lib/librekpiper_motion_backend.so'
                      for p in os.environ.get('CMAKE_PREFIX_PATH', '').split(':') if p]
        library = next((p for p in candidates if p.is_file()), None)
        if library is None:
            raise MotionBackendError('native_motion_backend_not_built')
        self.lib = ctypes.CDLL(str(library))
        self.ptr = ctypes.POINTER(ctypes.c_double)
        self.callback = ctypes.CFUNCTYPE(ctypes.c_int, self.ptr)
        self.lib.rekpiper_rrt.argtypes = [self.ptr]*4 + [self.callback,
            ctypes.c_double, ctypes.c_uint, self.ptr, ctypes.c_int]
        self.lib.rekpiper_rrt.restype = ctypes.c_int
        self.lib.rekpiper_collision_create.argtypes = [ctypes.c_char_p]
        self.lib.rekpiper_collision_create.restype = ctypes.c_void_p
        self.lib.rekpiper_self_clear.argtypes = [ctypes.c_void_p, self.ptr, ctypes.c_double]
        self.lib.rekpiper_collision_destroy.argtypes = [ctypes.c_void_p]
        self.handle = self.lib.rekpiper_collision_create(robot_xml.encode())
        if not self.handle:
            raise MotionBackendError('self_collision_model_invalid')

    def self_clear(self, joints, opening_m=.070):
        q = np.ascontiguousarray(joints, dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            return False
        return bool(self.lib.rekpiper_self_clear(self.handle, q.ctypes.data_as(self.ptr), opening_m))

    def plan(self, lower, upper, start, goal, valid_state, timeout_s=2., seed=0,
             opening_m=.070):
        arrays = [np.ascontiguousarray(x, dtype=np.float64) for x in (lower,upper,start,goal)]
        if any(x.shape != (6,) or not np.all(np.isfinite(x)) for x in arrays):
            raise ValueError('RRT requires six finite joints')
        if (not 0 < timeout_s <= 2. or np.any(arrays[0] >= arrays[1])
                or any(np.any(q < arrays[0]) or np.any(q > arrays[1]) for q in arrays[2:])):
            raise ValueError('RRT bounds or budget invalid')
        deadline = time.monotonic()+timeout_s
        errors = []
        def check(pointer):
            if time.monotonic() >= deadline:
                return 0
            try:
                q = np.ctypeslib.as_array(pointer, shape=(6,)).copy()
                return int(self.self_clear(q,opening_m) and bool(valid_state(q)))
            except Exception as exc:
                if not errors:
                    errors.append(str(exc))
                return 0
        callback = self.callback(check)
        output = np.empty((10000,6), dtype=np.float64)
        count = self.lib.rekpiper_rrt(
            *[a.ctypes.data_as(self.ptr) for a in arrays], callback, timeout_s,
            seed, output.ctypes.data_as(self.ptr), len(output))
        if errors:
            raise MotionBackendError('state_check_failed:'+errors[0])
        if count <= 0:
            raise MotionBackendError('rrt_no_exact_path_or_timeout:{}'.format(count))
        # Native EdgeCheck already checked every edge at this exact spacing.
        # Materialize those checked samples; execution separately reaudits a
        # fresh map and the time-parameterized trajectory, not just endpoints.
        path = [output[0].copy()]
        for a,b in zip(output[:count-1],output[1:count]):
            n = max(2,int(np.ceil(np.sum(np.abs(b-a))/.005))+1)
            path.extend(np.linspace(a,b,n)[1:])
        path = np.asarray(path)
        if time.monotonic() >= deadline:
            raise MotionBackendError('returned_path_materialization_timeout')
        if (not np.all(np.isfinite(path)) or np.any(path<arrays[0]) or np.any(path>arrays[1])
                or not np.allclose(path[0],arrays[2],rtol=0,atol=1e-9)
                or not np.allclose(path[-1],arrays[3],rtol=0,atol=1e-9)):
            raise MotionBackendError('returned_path_invalid_or_inexact')
        return path

    def close(self):
        if self.handle:
            self.lib.rekpiper_collision_destroy(self.handle)
            self.handle = None
