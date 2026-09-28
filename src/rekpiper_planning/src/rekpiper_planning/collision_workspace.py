"""Explicit environment-audit scope; robot self-collision is independent."""
import numpy as np


def workspace_mask(points, workspace, radii=0.0):
    """Include samples whose bounding spheres intersect the declared workspace.

    Positive radius expands the half-space tests, conservatively retaining
    robot geometry that crosses a workspace boundary. No unobserved space is
    labeled free by this function.
    """
    points = np.asarray(points, dtype=float)
    radius = np.broadcast_to(np.asarray(radii, dtype=float), (len(points),))
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('invalid_workspace_points')
    if not np.isfinite(radius).all() or np.any(radius < 0):
        raise ValueError('invalid_workspace_radii')
    normal = np.asarray(workspace['table_plane_normal'], dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1., atol=1e-5):
        raise ValueError('workspace_table_normal_must_be_unit')
    return ((points[:, 0]+radius >= workspace['x_min_m'])
            & (points[:, 0]-radius <= workspace['x_max_m'])
            & (points[:, 1]+radius >= workspace['y_min_m'])
            & (points[:, 1]-radius <= workspace['y_max_m'])
            & (points @ normal+workspace['table_plane_offset_m']+radius
               >= -workspace['below_table_tolerance_m']))


def named_region_mask(points, regions):
    """Only explicitly named, bounded scene regions may be excluded."""
    points = np.asarray(points, dtype=float)
    excluded = np.zeros(len(points), dtype=bool)
    for region in regions:
        lo = np.asarray(region['min_m'], dtype=float)
        hi = np.asarray(region['max_m'], dtype=float)
        if not region.get('name') or lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
            raise ValueError('invalid_named_exclusion_region')
        excluded |= np.all((points >= lo) & (points <= hi), axis=1)
    return excluded
