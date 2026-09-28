"""Observed-surface checks with optional horizontal side-grasp restrictions."""
import numpy as np
from scipy.spatial import cKDTree


def observed_table_plane(scene, target_mask, target_center):
    """Fit a local, nearly horizontal background plane; never invent table height."""
    points = np.asarray(scene)[~np.asarray(target_mask, dtype=bool)]
    points = points[np.linalg.norm(points[:, :2] - target_center[:2], axis=1) < .15]
    if len(points) < 100:
        raise ValueError('horizontal_grasp_table_points_missing')
    rng = np.random.default_rng(13)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(500):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1e-8 or abs(normal[2]) / length < np.cos(np.deg2rad(10)):
            continue
        inside = np.abs((points-a) @ (normal/length)) < .003
        if inside.sum() > best.sum():
            best = inside
    if best.sum() < max(100, .2*len(points)):
        raise ValueError('horizontal_grasp_table_plane_unreliable')
    inliers = points[best]
    if np.min(np.ptp(inliers[:, :2], axis=0)) < .08:
        raise ValueError('horizontal_grasp_table_extent_insufficient')
    center = inliers.mean(axis=0)
    _, _, axes = np.linalg.svd(inliers-center, full_matrices=False)
    normal = axes[-1] if axes[-1, 2] > 0 else -axes[-1]
    offset = float(-normal @ center)
    rms = float(np.sqrt(np.mean((inliers @ normal+offset)**2)))
    if normal[2] < np.cos(np.deg2rad(10)) or rms > .003:
        raise ValueError('horizontal_grasp_table_plane_unreliable')
    return normal, offset, rms


class HorizontalGraspPolicy:
    def __init__(self, scene_points, target_mask, camera_origin, gripper,
                 maximum_horizontal_angle_deg=10., contact_tolerance_m=.005,
                 table_clearance_m=.003, restrict_horizontal=True):
        self.scene = np.asarray(scene_points, dtype=float)
        target_mask = np.asarray(target_mask, dtype=bool)
        if (self.scene.ndim != 2 or self.scene.shape[1] != 3
                or target_mask.shape != (len(self.scene),)
                or not np.all(np.isfinite(self.scene)) or target_mask.sum() < 120):
            raise ValueError('horizontal_grasp_observed_target_missing')
        if not 0 < maximum_horizontal_angle_deg <= 30:
            raise ValueError('invalid_horizontal_grasp_angle')
        self.angle_deg = float(maximum_horizontal_angle_deg)
        self.restrict_horizontal = bool(restrict_horizontal)
        self.contact_tolerance_m = float(contact_tolerance_m)
        self.table_clearance_m = float(table_clearance_m)
        self.gripper = gripper
        self.target = self.scene[target_mask]
        self.center = np.median(self.target, axis=0)
        self.normal, self.offset, self.plane_rms_m = observed_table_plane(
            self.scene, target_mask, self.center)
        self.tree = cKDTree(self.target)
        distances, indices = self.tree.query(self.target, k=20)
        neighbors = self.target[indices]
        centered = neighbors-neighbors.mean(axis=1, keepdims=True)
        values, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', centered, centered)/20)
        self.normals = vectors[:, :, 0]
        self.normal_valid = ((values[:, 0] / np.maximum(values[:, 1], 1e-12) < .25)
                             & (distances[:, -1] < .025) & (values[:, 1] > 1e-8))
        if not self.restrict_horizontal:
            self.region_mask = target_mask.copy()
            self.region_tree = self.tree
            self.closing_axes = []
            self.approach_directions = []
            return
        heights = self.target @ self.normal + self.offset
        lateral = (self.normal_valid & (np.abs(self.normals[:, 2]) < .5)
                   & (heights >= gripper.finger_height_m/2 + table_clearance_m))
        if lateral.sum() < 120:
            raise ValueError('horizontal_grasp_visible_side_surface_insufficient')
        self.region_mask = np.zeros(len(self.scene), dtype=bool)
        self.region_mask[np.flatnonzero(target_mask)[lateral]] = True
        self.region_tree = cKDTree(self.target[lateral])
        # Estimate up to two distinct visible face-normal axes. No hidden face synthesis.
        normals = self.normals[lateral].copy()
        normals[:, 2] = 0.
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        angles = np.arctan2(normals[:, 1], normals[:, 0]) % np.pi
        counts, edges = np.histogram(angles, bins=18, range=(0., np.pi))
        self.closing_axes = []
        for index in np.argsort(-counts):
            if counts[index] < max(10, .15*counts.max()):
                break
            angle = (edges[index]+edges[index+1])/2
            axis = np.array([np.cos(angle), np.sin(angle), 0.])
            if any(abs(axis @ previous) > np.cos(np.deg2rad(40)) for previous in self.closing_axes):
                continue
            local = normals[np.abs(normals @ axis) > np.cos(np.deg2rad(15))]
            local[local @ axis < 0] *= -1
            axis = local.mean(axis=0)
            axis /= np.linalg.norm(axis)
            self.closing_axes.append(axis)
            if len(self.closing_axes) == 2:
                break
        self.approach_directions = []
        for axis in self.closing_axes:
            direction = np.cross(axis, [0., 0., 1.])
            if direction @ (self.center-np.asarray(camera_origin)) < 0:
                direction = -direction
            self.approach_directions.append(direction)

    def directions_in_camera(self, base_from_camera):
        if not self.restrict_horizontal:
            return [None]
        rotation = np.asarray(base_from_camera)[:3, :3]
        return [rotation.T @ direction for direction in self.approach_directions]

    def audit(self, candidate):
        """Geometry only: returned valid never replaces SDK, IK, or full path audits."""
        approach, closing = candidate.approach_axis_base, candidate.closing_axis_base
        angles = np.degrees(np.arcsin(np.clip(np.abs([approach[2], closing[2]]), 0., 1.)))
        contacts = candidate.tcp_position + np.array([[-1.], [1.]])*candidate.predicted_width_m/2*closing
        distances, indices = self.tree.query(contacts)
        normal_angles = np.degrees(np.arccos(np.clip(np.abs(self.normals[indices] @ closing), 0., 1.)))
        origin = candidate.tcp_position-candidate.insertion_depth_m*approach
        region_distance = float(self.region_tree.query(origin)[0])
        open_contacts = candidate.tcp_position + np.array([[-1.], [1.]])*candidate.suggested_preopen_width_m/2*closing
        open_distance = self.tree.query(open_contacts)[0]
        # End effector geometry includes the palm and both fingers at pre-open width.
        points = self.gripper.points(candidate)
        if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
                or not np.all(np.isfinite(points))):
            raise ValueError('piper_gripper_geometry_missing')
        clearance = float(np.min(points @ self.normal+self.offset))
        contact_heights = contacts @ self.normal+self.offset
        # The two observed contacts must straddle the target along the closing axis.
        projected = (self.target-candidate.tcp_position) @ closing
        near_slice = np.abs((self.target-candidate.tcp_position) @ approach) < .01
        height_axis = np.array([0., 0., 1.]) if self.restrict_horizontal else np.cross(approach, closing)
        near_slice &= np.abs((self.target-candidate.tcp_position) @ height_axis) < .01
        bracketed = (np.any(near_slice) and np.min(projected[near_slice]) < -.005
                     and np.max(projected[near_slice]) > .005)
        checks = dict(
            piper_width=bool(candidate.width_ok),
            piper_insertion_depth=0 < candidate.insertion_depth_m <= self.gripper.usable_depth_m,
            visible_surface_region=region_distance <= self.contact_tolerance_m,
            contacts_on_observed_target=bool(np.all(distances <= self.contact_tolerance_m)),
            closing_normal_alignment=bool(np.all(self.normal_valid[indices]) and np.all(normal_angles <= 20.)),
            contacts_on_opposite_sides=bool(bracketed),
            open_contacts_outside_target=bool(np.all(open_distance >= .003)),
            contacts_above_table=bool(np.all(contact_heights >= self.table_clearance_m)),
            piper_gripper_above_table=clearance >= self.table_clearance_m)
        if self.restrict_horizontal:
            checks.update(horizontal_approach=angles[0] <= self.angle_deg,
                          horizontal_closing=angles[1] <= self.angle_deg,
                          no_upward_approach=approach[2] <= 1e-6)
        checks = {name: bool(value) for name, value in checks.items()}
        return dict(valid=all(checks.values()), checks=checks,
                    rejection_reasons=[name for name, ok in checks.items() if not ok],
                    horizontal_angles_deg=angles.tolist(), contact_distances_m=distances.tolist(),
                    normal_alignment_deg=normal_angles.tolist(), region_distance_m=region_distance,
                    contact_heights_m=contact_heights.tolist(), gripper_table_clearance_m=clearance)
