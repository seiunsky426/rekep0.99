"""Reject unsupported depth samples before projection, without using RGB color."""
import numpy as np
from scipy.ndimage import binary_dilation, minimum_filter, uniform_filter


def free_space_conflicts(points_base, other_frame, margin_m):
    """Flag points lying clearly in another view's measured free ray segment.

    Use the nearest valid depth in a 3x3 neighborhood and require five valid
    pixels. Occluded points, missing depth and points outside the image remain
    undecided and are kept. This is not a nearest-cloud or center-distance gate.
    """
    points = np.asarray(points_base, dtype=float)
    depth = np.asarray(other_frame['depth'], dtype=float)
    transform = np.asarray(other_frame['base_from_camera'])
    k = np.asarray(other_frame['K']).reshape(3, 3)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or depth.ndim != 2 or not np.isfinite(margin_m) or margin_m <= 0):
        raise ValueError('invalid_free_space_conflict_input')
    camera = (points-transform[:3, 3]) @ transform[:3, :3]
    safe_z = np.where(camera[:, 2] > .15, camera[:, 2], 1.)
    uv = np.rint((camera @ k.T)[:, :2]/safe_z[:, None]).astype(int)
    inside = ((camera[:, 2] > .15) & (uv[:, 0] >= 1) & (uv[:, 0] < depth.shape[1]-1)
              & (uv[:, 1] >= 1) & (uv[:, 1] < depth.shape[0]-1))
    valid = np.isfinite(depth) & (depth > .15) & (depth < 1.5)
    nearest = minimum_filter(np.where(valid, depth, np.inf), size=3,
                             mode='constant', cval=np.inf)
    count = uniform_filter(valid.astype(float), size=3, mode='constant')*9
    indices = np.flatnonzero(inside)
    u, v = uv[indices].T
    conflict = np.zeros(len(points), bool)
    conflict[indices] = (count[v, u] >= 4.9) & (camera[indices, 2] < nearest[v, u]-margin_m)
    return conflict


def depth_support_mask(depth_m, jump_m=.025, support_m=.008,
                       minimum_neighbors=2, edge_radius=0):
    """Keep locally supported depths away from foreground/background jumps.

    Invalid values stay invalid. A jump removes its two adjacent pixels; an
    optional dilation expands that band. Removed pixels are unknown, not free.
    The thresholds are metric distances in the camera's native depth image.
    """
    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2 or not (jump_m > support_m > 0) or not 0 <= minimum_neighbors <= 8 or edge_radius < 0:
        raise ValueError('invalid_depth_quality_parameters')
    valid = np.isfinite(depth) & (depth > .15) & (depth < 1.5)
    edges = np.zeros(depth.shape, bool)
    for axis in (0, 1):
        a = [slice(None), slice(None)]; b = a.copy()
        a[axis] = slice(None, -1); b[axis] = slice(1, None)
        a, b = tuple(a), tuple(b)
        jump = valid[a] & valid[b] & (np.abs(depth[a]-depth[b]) > jump_m)
        edges[a] |= jump; edges[b] |= jump
    if edge_radius:
        edges = binary_dilation(edges, iterations=int(edge_radius))
    padded = np.pad(depth, 1, constant_values=np.nan)
    support = np.zeros(depth.shape, np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dx or dy):
                continue
            neighbor = padded[1+dy:1+dy+depth.shape[0], 1+dx:1+dx+depth.shape[1]]
            support += np.isfinite(neighbor) & (neighbor > .15) & (neighbor < 1.5) & (np.abs(neighbor-depth) <= support_m)
    return valid & ~edges & (support >= minimum_neighbors)
