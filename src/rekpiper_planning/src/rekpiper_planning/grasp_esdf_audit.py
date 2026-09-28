"""Piper geometry against RS1 ESDF; target contact never exempts the table."""
import numpy as np
from scipy.spatial import cKDTree

from .collision_workspace import workspace_mask, named_region_mask
from .table_surface import plane_parameters, inside_table_footprint


def query_grid(points, radii, grid):
    lo, hi = np.asarray(grid['bounds_min']), np.asarray(grid['bounds_max'])
    shape = np.asarray(grid['observed'].shape)
    indices = np.rint((points-lo)*(shape-1)/(hi-lo)).astype(int)
    inside = np.all((points >= lo) & (points <= hi), axis=1)
    inside &= np.all((indices >= 0) & (indices < shape), axis=1)
    known = np.zeros(len(points), bool)
    clearance = np.full(len(points), -np.inf)
    selected = indices[inside]
    known[inside] = grid['observed'][tuple(selected.T)]
    clearance[inside] = -grid['distances_m'][tuple(selected.T)]-radii[inside]
    known &= np.isfinite(clearance)
    return clearance, inside & known


class GraspESDFAudit:
    def __init__(self, sampler, grid, target_removed_grid, target_points,
                 table_model=None, workspace=None, ignored_regions=(), clearance_m=.01):
        self.sampler = sampler
        self.grid, self.target_removed_grid = grid, target_removed_grid
        self.target = cKDTree(np.asarray(target_points))
        self.table, self.workspace = table_model, workspace
        self.ignored_regions, self.clearance_m = ignored_regions, clearance_m
        # Only the end effector is rigidly transformed for pre-IK checks.
        zero = np.zeros(6)
        self.reference_tcp = sampler.kinematics.forward(zero)
        self.reference_q = zero
        self._pose_samples = {}

    def pose_samples(self, tcp, opening):
        if opening not in self._pose_samples:
            points, radii, labels = self.sampler.contact_samples(self.reference_q, opening)
            selected = np.isin(labels, ['gripper_base', 'link7', 'link8'])
            self._pose_samples[opening] = points[selected], radii[selected], labels[selected]
        points, radii, labels = self._pose_samples[opening]
        transform = tcp @ np.linalg.inv(self.reference_tcp)
        return points @ transform[:3, :3].T+transform[:3, 3], radii, labels

    def check(self, points, radii, labels, tcp, opening, contact=False,
              grid=None, target_points=None, support=None):
        points, radii, labels = np.asarray(points), np.asarray(radii), np.asarray(labels)
        if (points.ndim != 2 or points.shape[1] != 3 or radii.shape != (len(points),)
                or labels.shape != (len(points),) or not np.isfinite(points).all()
                or not np.isfinite(radii).all() or np.any(radii < 0)):
            raise ValueError('invalid_collision_samples')
        selected = labels != 'base_link'
        if self.workspace is not None:
            selected &= workspace_mask(points, self.workspace, radii)
            selected &= ~named_region_mask(points, self.ignored_regions)
        points, radii, labels = points[selected], radii[selected], labels[selected]
        if not len(points):
            return dict(passed=True, checked_samples=0, minimum_clearance_m=None)
        query = points.copy()
        if self.workspace is not None:
            query[:, 0] = np.clip(query[:, 0], self.workspace['x_min_m'], self.workspace['x_max_m'])
            query[:, 1] = np.clip(query[:, 1], self.workspace['y_min_m'], self.workspace['y_max_m'])
        active_grid = self.grid if grid is None else grid
        clearance, known = query_grid(query, radii, active_grid)
        allowed = np.zeros(len(points), bool)
        if contact:
            local = (points-tcp[:3, 3]) @ tcp[:3, :3]
            # Inner pads near the URDF TCP, not the complete finger links.
            pads = (np.isin(labels, ['link7', 'link8'])
                    & (np.abs(local[:, 2]) <= .012)
                    & (np.abs(local[:, 0]) <= .015)
                    & (np.abs(np.abs(local[:, 1])-opening/2) <= radii+.003))
            target = self.target if target_points is None else cKDTree(target_points)
            near = target.query(points)[0] <= radii+.003
            without_target, without_known = query_grid(query, radii, self.target_removed_grid)
            # Both maps must be observed. Removal of the named target must
            # explain the collision; another obstacle can never be exempted.
            allowed = pads & near & known & without_known & (without_target >= self.clearance_m)
        if support is not None:
            center, radius, top = support
            allowed |= ((labels == 'attached_object') & known
                        & (np.linalg.norm(points[:, :2]-np.asarray(center)[:2], axis=1) <= radius)
                        & (points[:, 2] >= top) & (points[:, 2] <= top+.003))
        moved_target_bad = np.zeros(len(points), bool)
        if target_points is not None:
            # After pickup/release the original target has been vacated in
            # ESDF. Its new bounding volume must still block non-pad contact.
            target_points = np.asarray(target_points)
            overlap = np.all((points+radii[:, None] >= target_points.min(axis=0))
                             & (points-radii[:, None] <= target_points.max(axis=0)), axis=1)
            moved_target_bad = overlap & (labels != 'attached_object') & ~allowed
        table_bad = np.zeros(len(points), bool)
        if self.table is not None:
            normal, offset = plane_parameters(self.table)
            table_clearance = points @ normal+offset-radii
            on_table = inside_table_footprint(points, self.table)
            table_bad = on_table & (table_clearance < self.clearance_m)
        bad = (~known) | ((clearance < self.clearance_m) & ~allowed) | table_bad | moved_target_bad
        outside = np.any((query < active_grid['bounds_min']) | (query > active_grid['bounds_max']), axis=1)
        occupied = known & (clearance+radii < 0)
        counts = {str(link): int(np.sum(bad & (labels == link))) for link in np.unique(labels)}
        indices = np.flatnonzero(bad)
        return dict(passed=bool(not bad.any()), checked_samples=len(points),
                    unknown_count=int((~known).sum()), table_collision_count=int(table_bad.sum()),
                    outside_count=int(outside.sum()), occupied_sample_count=int(occupied.sum()),
                    known_clearance_failure_count=int((known & (clearance < self.clearance_m) & ~allowed).sum()),
                    moved_target_collision_count=int(moved_target_bad.sum()),
                    collision_count=int(bad.sum()), contact_samples=int(allowed.sum()),
                    minimum_clearance_m=float(np.min(clearance[known])) if known.any() else None,
                    failed_links={k: v for k, v in counts.items() if v},
                    first_bad_point=points[indices[0]].tolist() if len(indices) else None)

    def pose(self, tcp, opening, contact=False):
        return self.check(*self.pose_samples(tcp, opening), tcp, opening, contact)

    def joints(self, q, opening, contact=False, grid=None, target_points=None):
        # Every link is recomputed by FK at this joint state.
        return self.check(*self.sampler.contact_samples(q, opening),
                          self.sampler.kinematics.forward(q), opening, contact, grid, target_points)

    def grasp_sweep(self, tcp, opening, approach_m=.06, step_m=.002):
        count = 0
        for offset in np.linspace(approach_m, 0, int(np.ceil(approach_m/step_m))+1):
            pose = tcp.copy()
            pose[:3, 3] -= offset*tcp[:3, 2]
            result = self.pose(pose, .07, contact=False)
            count += 1
            if not result['passed']:
                return dict(passed=False, phase='approach', offset_m=float(offset), samples=count, audit=result)
        for width in np.linspace(.07, opening, max(2, int(np.ceil(abs(.07-opening)/.001))+1)):
            result = self.pose(tcp, float(width), contact=True)
            count += 1
            if not result['passed']:
                return dict(passed=False, phase='closing', opening_m=float(width), samples=count, audit=result)
        return dict(passed=True, samples=count, approach_m=approach_m, step_m=step_m)
