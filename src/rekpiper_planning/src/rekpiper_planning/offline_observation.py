"""Depth-supported collision grids for hypothetical offline planning only."""
import numpy as np
from scipy.ndimage import distance_transform_edt


def depth_metres(frame):
    depth = np.asarray(frame['depth'], dtype=np.float32).copy()
    encoding = str(np.asarray(frame['depth_encoding']).item())
    if encoding in ('16UC1', 'mono16'):
        depth *= .001
    elif encoding != '32FC1':
        raise ValueError('unsupported_depth_encoding')
    return depth


def rgbd_points(frame):
    """Return base-frame points, original RGB bytes and matching pixel indices."""
    depth = depth_metres(frame)
    rgb = np.asarray(frame['rgb'])
    k = np.asarray(frame['K']).reshape(3, 3)
    transform = np.asarray(frame['base_from_camera'])
    if (rgb.shape != depth.shape + (3,) or rgb.dtype != np.uint8
            or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)):
        raise ValueError('invalid_rgbd_geometry')
    v, u = np.nonzero(np.isfinite(depth) & (depth > .15) & (depth < 1.5))
    z = depth[v, u]
    points = np.column_stack(((u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z))
    return points @ transform[:3, :3].T + transform[:3, 3], rgb[v, u], (v, u)


def build_depth_collision_grid(frames, bounds_min, bounds_max, resolution_m,
                               vacated_masks=None, table_model=None,
                               table_masks=None):
    """Fuse visible free space; unknown/occluded voxels remain blocked.

    Vacated masks explicitly identify hypothetical removed robot/object
    surfaces. Only the observed surface band is vacated, never the unseen
    volume behind it. Another view's static surface takes precedence.
    """
    if set(frames) not in ({'rs1'}, {'rs1', 'rs3'}):
        raise ValueError('rs1_camera_frame_required')
    stamps = [float(f[k]) for f in frames.values() for k in ('rgb_stamp_s', 'depth_stamp_s')]
    if not np.isfinite(stamps).all() or max(stamps)-min(stamps) > .025:
        raise ValueError('snapshot_camera_timestamps_inconsistent')
    lower, upper = np.asarray(bounds_min, dtype=float), np.asarray(bounds_max, dtype=float)
    resolution = float(resolution_m)
    if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
            or np.any(upper <= lower) or not np.isfinite(resolution) or resolution <= 0):
        raise ValueError('invalid_grid_bounds')
    shape = np.ceil((upper-lower)/resolution-1e-5).astype(int)+1
    axes = [np.linspace(a, b, int(n)) for a, b, n in zip(lower, upper, shape)]
    coordinates = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    free = np.zeros(len(coordinates), dtype=bool)
    occupied = free.copy()
    vacated = free.copy()
    band = np.sqrt(3.)*resolution/2
    table_height = None
    if table_model is not None:
        from .table_surface import plane_parameters, inside_table_footprint
        normal, offset = plane_parameters(table_model)
        table_height = coordinates @ normal+offset
        table_inside = inside_table_footprint(coordinates, table_model)
    for name, frame in frames.items():
        rgbd_points(frame)  # Validate units, intrinsics and rigid transform.
        depth = depth_metres(frame)
        k = np.asarray(frame['K']).reshape(3, 3)
        t = np.asarray(frame['base_from_camera'])
        camera = (coordinates-t[:3, 3]) @ t[:3, :3]
        z = camera[:, 2]
        valid_z = z > .15
        safe_z = np.where(valid_z, z, 1.)
        u = np.rint(camera[:, 0]*k[0, 0]/safe_z+k[0, 2]).astype(int)
        v = np.rint(camera[:, 1]*k[1, 1]/safe_z+k[1, 2]).astype(int)
        inside = valid_z & (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
        indices = np.flatnonzero(inside)
        measured = depth[v[inside], u[inside]]
        valid = np.isfinite(measured) & (measured > .15) & (measured < 1.5)
        indices, measured = indices[valid], measured[valid]
        delta = measured-z[indices]
        surface = np.abs(delta) <= band
        removed = np.zeros(len(indices), dtype=bool)
        if vacated_masks is not None:
            mask = np.asarray(vacated_masks[name], dtype=bool)
            if mask.shape != depth.shape:
                raise ValueError('vacated_mask_shape_mismatch')
            removed = mask[v[indices], u[indices]]
        if table_model is not None and table_masks is not None:
            table_mask = np.asarray(table_masks[name], dtype=bool)
            if table_mask.shape != depth.shape:
                raise ValueError('table_mask_shape_mismatch')
            on_table = table_mask[v[indices], u[indices]] & table_inside[indices] & ~removed
            # A confirmed planar pixel uses the same zero crossing as the
            # collision mesh, rather than a second thick noisy surface band.
            heights = table_height[indices[on_table]]
            surface[on_table] = ((heights <= 0)
                                 & (heights >= -float(table_model.get('thickness_m', .1))))
            free[indices[on_table & (table_height[indices] > 0)]] = True
        free[indices[delta > band]] = True
        occupied[indices[surface & ~removed]] = True
        vacated[indices[surface & removed]] = True
    if table_model is not None:
        # The finite, measured table slab is known solid. Never clear any
        # unobserved volume above it or beyond its support polygon.
        occupied |= table_inside & (table_height <= 0) & (table_height >= -float(table_model.get('thickness_m', .1)))
    known = free | occupied | vacated
    free = (free | vacated) & ~occupied
    free = free.reshape(tuple(shape))
    occupied = occupied.reshape(tuple(shape))
    # Distance to unknown space is also a clearance limit, not just surfaces.
    distances = -distance_transform_edt(free, sampling=resolution)
    distances[occupied] = distance_transform_edt(occupied, sampling=resolution)[occupied]
    known = known.reshape(tuple(shape))
    distances[~known] = 1.0
    result = dict(bounds_min=lower, bounds_max=upper, resolution_m=resolution,
                distances_m=distances.astype(np.float32), observed=known,
                occupied=occupied, assumed_vacated=vacated.reshape(tuple(shape)) & ~occupied,
                snapshot_stamp_s=min(stamps), motion_allowed=False)
    if table_model is not None:
        result.update(table_plane_normal=normal, table_plane_offset_m=offset,
                      table_footprint_xy_m=np.asarray(table_model['table_footprint_xy_m']))
    return result
