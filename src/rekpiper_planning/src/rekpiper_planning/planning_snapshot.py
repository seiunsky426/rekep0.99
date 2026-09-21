"""Immutable observation data for one planning batch; never motion authority."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class PlanningSDFSnapshot:
    def __init__(self, grid):
        if (grid.header.frame_id != 'base_link' or not grid.map_generation_uuid
                or grid.header.stamp.to_sec() <= 0.0
                or grid.status != 'stepwise_snapshot_for_planning_preview'
                or grid.valid):
            raise ValueError('expected a timestamped stepwise observation snapshot')
        self.generation_uuid = str(grid.map_generation_uuid)
        self.stamp_s = grid.header.stamp.to_sec()
        self.resolution_m = float(grid.resolution_m)
        self.bounds_min = np.array([grid.bounds_min.x, grid.bounds_min.y, grid.bounds_min.z])
        self.bounds_max = np.array([grid.bounds_max.x, grid.bounds_max.y, grid.bounds_max.z])
        shape = np.array([grid.size_x, grid.size_y, grid.size_z], dtype=int)
        if (not np.isfinite(self.bounds_min).all() or not np.isfinite(self.bounds_max).all()
                or np.any(self.bounds_max <= self.bounds_min)
                or not np.isfinite(self.resolution_m) or self.resolution_m <= 0
                or np.any(shape < 2)):
            raise ValueError('invalid snapshot grid geometry')
        expected = np.ceil((self.bounds_max - self.bounds_min) / self.resolution_m - 1e-5).astype(int) + 1
        if not np.array_equal(shape, expected):
            raise ValueError('snapshot shape differs from endpoint-inclusive axes')
        distances = np.asarray(grid.distances_m, dtype=np.float32).copy()
        observed = (np.frombuffer(grid.observed, dtype=np.uint8)
                    if isinstance(grid.observed, (bytes, bytearray))
                    else np.asarray(grid.observed))
        if (distances.size != np.prod(shape) or observed.size != distances.size
                or not np.isfinite(distances).all() or not np.isin(observed, [0, 1]).all()):
            raise ValueError('incomplete or invalid snapshot arrays')
        self.observed = observed.astype(bool).reshape(tuple(shape))
        self.sdf_voxels = distances.reshape(tuple(shape))
        self.sdf_voxels[~self.observed] = np.maximum(self.sdf_voxels[~self.observed], 1.0)
        self.axes = tuple(np.linspace(lo, hi, int(n)) for lo, hi, n in
                          zip(self.bounds_min, self.bounds_max, shape))
        for array in (self.bounds_min, self.bounds_max, self.sdf_voxels, self.observed):
            array.setflags(write=False)
        self._distance = RegularGridInterpolator(
            self.axes, self.sdf_voxels, bounds_error=False, fill_value=1.0)
        self._observed = RegularGridInterpolator(
            self.axes, self.observed.astype(float), bounds_error=False, fill_value=0.0)

    def query(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('query points must be finite Nx3 base-frame coordinates')
        distances = self._distance(points)
        known = self._observed(points) >= 1.0 - 1e-7
        distances[~known] = np.maximum(distances[~known], 1.0)
        return distances, known
