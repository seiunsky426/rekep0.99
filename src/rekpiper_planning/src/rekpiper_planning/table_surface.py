"""Measured tabletop plane shared by depth correction and collision geometry."""
import numpy as np
from scipy.spatial import ConvexHull


def plane_parameters(model):
    normal = np.asarray(model['table_plane_normal'], dtype=float)
    offset = float(model['table_plane_offset_m'])
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or not np.isfinite(offset) or not np.isclose(np.linalg.norm(normal), 1.)
            or normal[2] < .9):
        raise ValueError('invalid_table_plane')
    return normal, offset


def fit_table_plane(points, threshold_m=.006):
    """Spatially balanced RANSAC, followed by inlier-only plane refinement."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.isfinite(points).all():
        raise ValueError('insufficient_finite_table_points')
    cells = np.floor(points[:, :2]/.025).astype(int)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    balanced = np.array([np.median(points[inverse == i], axis=0)
                         for i in range(inverse.max()+1) if np.sum(inverse == i) >= 5])
    if len(balanced) < 30 or np.any(np.ptp(balanced[:, :2], axis=0) < .2):
        raise ValueError('insufficient_table_spatial_coverage')
    rng = np.random.RandomState(7)
    best, score = None, 0
    for _ in range(500):
        a, b, c = balanced[rng.choice(len(balanced), 3, replace=False)]
        n = np.cross(b-a, c-a)
        if np.linalg.norm(n) < 1e-8:
            continue
        n /= np.linalg.norm(n)
        if abs(n[2]) < .95:
            continue
        inside = np.abs(balanced @ n-a @ n) < threshold_m
        if inside.sum() > score:
            best, score = inside, inside.sum()
    if best is None or score < .5*len(balanced):
        raise ValueError('table_plane_consensus_failed')
    for _ in range(4):
        center = balanced[best].mean(axis=0)
        n = np.linalg.svd(balanced[best]-center, full_matrices=False)[2][-1]
        if n[2] < 0:
            n = -n
        d = -float(center @ n)
        best = np.abs(balanced @ n+d) < threshold_m
    # Remove elevated objects before averaging cells; a mixed-cell median
    # alone can still bias the plane toward one-sided foreground outliers.
    for _ in range(2):
        accepted = np.abs(points @ n+d) < threshold_m
        means = np.array([points[accepted & (inverse == i)].mean(axis=0)
                          for i in range(inverse.max()+1)
                          if np.sum(accepted & (inverse == i)) >= 5])
        center = means.mean(axis=0)
        n = np.linalg.svd(means-center, full_matrices=False)[2][-1]
        if n[2] < 0:
            n = -n
        d = -float(center @ n)
    return dict(table_plane_normal=n.tolist(), table_plane_offset_m=d,
                fit_cells=len(balanced), inlier_cells=int(best.sum()),
                fit_threshold_m=threshold_m)


def table_footprint(points):
    """Convex support polygon of confirmed empty-table observations, in XY."""
    xy = np.asarray(points)[:, :2]
    return xy[ConvexHull(xy).vertices].tolist()


def inside_table_footprint(points, model):
    polygon = np.asarray(model['table_footprint_xy_m'])
    hull = ConvexHull(polygon)
    return np.all(np.asarray(points)[:, :2] @ hull.equations[:, :2].T
                  + hull.equations[:, 2] <= 1e-8, axis=1)


def table_pixel_mask(frame, model, protected=None, maximum_residual_m=.008):
    """Only bright neutral, plane-supported interior pixels may be replaced.

    Dilated object/robot masks and RGB/depth edges are never projected. Missing
    depth remains missing. A wider residual is only for an explicit empty-table
    calibration, never an automatic foreground-removal rule.
    """
    import cv2
    from .offline_observation import rgbd_points, depth_metres
    points, rgb, (v, u) = rgbd_points(frame)
    n, d = plane_parameters(model)
    neutral = ((rgb.max(axis=1).astype(float)-rgb.min(axis=1)) < 45)
    neutral &= rgb.min(axis=1) > 65
    selected = neutral & (np.abs(points @ n+d) <= maximum_residual_m)
    selected &= inside_table_footprint(points, model)
    mask = np.zeros(frame['depth'].shape, np.uint8)
    mask[v[selected], u[selected]] = 1
    protect = np.zeros(mask.shape, np.uint8) if protected is None else np.asarray(protected, np.uint8)
    depth = depth_metres(frame)
    # Detect edges without converting invalid pixels into geometry.
    edges = cv2.Canny(frame['rgb'], 45, 100) > 0
    for axis in (0, 1):
        first = [slice(None), slice(None)]; second = first.copy()
        first[axis] = slice(None, -1); second[axis] = slice(1, None)
        valid_pair = (depth[tuple(first)] > 0) & (depth[tuple(second)] > 0)
        change = (np.abs(np.diff(depth, axis=axis)) > .018) & valid_pair
        edges[tuple(first)] |= change
        edges[tuple(second)] |= change
    excluded = cv2.dilate((edges | (protect > 0)).astype(np.uint8), np.ones((7, 7), np.uint8))
    return (mask > 0) & (excluded == 0)


def project_table_depth(frame, model, mask):
    """Intersect selected pixel rays with the plane; preserve every other pixel."""
    from .offline_observation import depth_metres
    n, d = plane_parameters(model)
    depth = depth_metres(frame)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != depth.shape or np.any(mask & (~np.isfinite(depth) | (depth <= 0))):
        raise ValueError('table_mask_includes_invalid_depth')
    v, u = np.nonzero(mask)
    k = np.asarray(frame['K']).reshape(3, 3)
    t = np.asarray(frame['base_from_camera'])
    rays = np.column_stack(((u-k[0, 2])/k[0, 0], (v-k[1, 2])/k[1, 1], np.ones(len(u))))
    denominator = (rays @ t[:3, :3].T) @ n
    if np.any(np.abs(denominator) < 1e-6):
        raise ValueError('table_ray_parallel_to_plane')
    z = -(t[:3, 3] @ n+d)/denominator
    if np.any((z <= .15) | (z >= 1.5)):
        raise ValueError('table_ray_intersection_out_of_depth_range')
    depth[v, u] = z
    result = dict(frame, depth=depth, depth_encoding=np.asarray('32FC1'))
    return result


def table_prism(model, thickness_m=.10):
    """Finite tilted prism; top vertices lie exactly on the shared plane."""
    n, d = plane_parameters(model)
    xy = np.asarray(model['table_footprint_xy_m'])
    top = np.column_stack((xy, -(xy @ n[:2]+d)/n[2]))
    bottom = top.copy(); bottom[:, 2] -= thickness_m/n[2]
    size = len(top)
    faces = []
    for i in range(1, size-1):
        faces.extend([[0, i, i+1], [size, size+i+1, size+i]])
    for i in range(size):
        j = (i+1) % size
        faces.extend([[i, size+i, size+j], [i, size+j, j]])
    return np.vstack((top, bottom)), np.asarray(faces, dtype=int)
