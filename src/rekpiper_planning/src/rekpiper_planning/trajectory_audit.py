"""Full Piper joint-path auditing against the live immutable ESDF grid."""

from __future__ import annotations

import numpy as np
from .continuous_ik import densify_joint_path


class TrajectoryAuditError(RuntimeError):
    pass


def sample_sdf_nearest(points, bounds_min, resolution_m, distances, observed,
                       bounds_max=None):
    points = np.asarray(points, dtype=float)
    lower = np.asarray(bounds_min, dtype=float)
    grid = np.asarray(distances, dtype=float)
    known = np.asarray(observed, dtype=bool)
    if bounds_max is None:
        indices = np.rint(
            (points - lower) / float(resolution_m)).astype(np.int64)
    else:
        upper = np.asarray(bounds_max, dtype=float)
        if upper.shape != (3,) or np.any(upper <= lower):
            raise ValueError("SDF bounds must be ordered XYZ vectors")
        scale = (np.asarray(grid.shape, dtype=float) - 1.0) / (upper - lower)
        indices = np.rint((points - lower) * scale).astype(np.int64)
    inside = np.all((indices >= 0) & (indices < np.asarray(grid.shape)), axis=1)
    values = np.zeros(len(points), dtype=float)  # outside/unknown is occupied
    selected = indices[inside]
    if len(selected):
        sampled_known = known[selected[:, 0], selected[:, 1], selected[:, 2]]
        sampled = grid[selected[:, 0], selected[:, 1], selected[:, 2]]
        values[np.flatnonzero(inside)] = np.where(sampled_known, sampled, 0.0)
    return values, inside


def audit_joint_path(joint_path, collision_sampler, sdf_grid,
                     minimum_clearance_m=0.01, links=None,
                     final_waypoint_links=None,
                     attached_points_local=None,
                     attached_frame="gripper_base", contact_policy=None):
    joints = np.asarray(joint_path, dtype=float)
    if joints.ndim != 2 or joints.shape[1] < 6 or not len(joints):
        raise TrajectoryAuditError("joint path must contain finite Piper waypoints")
    try:
        joints=densify_joint_path(joints)
    except ValueError as exc:
        raise TrajectoryAuditError(str(exc)) from exc
    shape = (int(sdf_grid.size_x), int(sdf_grid.size_y), int(sdf_grid.size_z))
    expected = int(np.prod(shape))
    if (not sdf_grid.valid or not sdf_grid.map_generation_uuid
            or len(sdf_grid.distances_m) != expected
            or len(sdf_grid.observed) != expected):
        raise TrajectoryAuditError("SDFGrid is incomplete")
    distances = np.asarray(sdf_grid.distances_m, dtype=float).reshape(shape)
    observed_values = (np.frombuffer(sdf_grid.observed, dtype=np.uint8)
                       if isinstance(sdf_grid.observed, (bytes, bytearray))
                       else np.asarray(sdf_grid.observed, dtype=np.uint8))
    if not np.isin(observed_values, [0, 1]).all():
        raise TrajectoryAuditError("SDFGrid observed flags are not binary")
    observed = observed_values.astype(bool).reshape(shape)
    bounds_min = [sdf_grid.bounds_min.x, sdf_grid.bounds_min.y,
                  sdf_grid.bounds_min.z]
    bounds_max = [sdf_grid.bounds_max.x, sdf_grid.bounds_max.y,
                  sdf_grid.bounds_max.z]
    minimum = float("inf")
    attached = (None if attached_points_local is None else
                np.asarray(attached_points_local, dtype=float).reshape(-1, 3))
    for index, waypoint in enumerate(joints):
        if not np.all(np.isfinite(waypoint)):
            raise TrajectoryAuditError("joint path contains NaN or Inf")
        selected_links = (final_waypoint_links
                          if (index == len(joints) - 1
                              and final_waypoint_links is not None)
                          else links)
        if contact_policy is None:
            points, radii = collision_sampler.samples(waypoint[:6], links=selected_links)
            labels = None
        else:
            points,radii,labels = collision_sampler.contact_samples(
                waypoint[:6],contact_policy.opening_m)
        if attached is not None and len(attached):
            transforms = collision_sampler.kinematics.link_transforms(
                waypoint[:6])
            transform = transforms.get(str(attached_frame))
            if transform is None:
                raise TrajectoryAuditError(
                    "attached collision frame is absent from Piper FK")
            attached_world = (attached @ transform[:3, :3].T
                              + transform[:3, 3])
            from scipy.spatial import cKDTree
            # A carried object is also an obstacle to the rest of the robot.
            arm_links=tuple(name for name in collision_sampler.sampled_links
                            if name != 'gripper_base')
            if arm_links:
                arm,arm_radii=collision_sampler.samples(waypoint[:6],links=arm_links)
                if np.any(cKDTree(attached_world).query(arm)[0] < arm_radii):
                    raise TrajectoryAuditError('attached_object_robot_self_collision')
            points = np.vstack((points, attached_world))
            # The stored cloud is already inflated by 10 mm.
            radii = np.r_[radii, np.zeros(len(attached_world), dtype=float)]
            if labels is not None:
                labels = np.r_[labels,['attached_object']*len(attached_world)]
        values, inside = sample_sdf_nearest(
            points, bounds_min, sdf_grid.resolution_m, distances, observed,
            bounds_max=bounds_max)
        # Official ReKep/nvblox convention is free < 0, surface == 0,
        # occupied > 0.  Clearance is therefore the negated SDF value.
        clearance = -values - radii
        minimum = min(minimum, float(np.min(clearance)))
        rejected = clearance < float(minimum_clearance_m)
        if contact_policy is not None:
            # Exceptions cannot turn unknown/out-of-grid space into free space.
            scale = (np.asarray(shape)-1)/(np.asarray(bounds_max)-np.asarray(bounds_min))
            indices = np.rint((points-bounds_min)*scale).astype(int)
            known = np.zeros(len(points),dtype=bool)
            valid_indices = indices[inside]
            known[inside] = observed[valid_indices[:,0],valid_indices[:,1],valid_indices[:,2]]
            allowed = contact_policy.allowed(waypoint[:6],points,radii,labels)
            rejected |= contact_policy.target_collision(waypoint[:6],points,radii,labels)
            rejected &= ~(np.asarray(allowed,dtype=bool) & known)
        if not np.all(inside) or np.any(rejected):
            raise TrajectoryAuditError(
                "ESDF rejected joint waypoint {} (clearance {:.4f} m)".format(
                    index, float(np.min(clearance))))
    return {"valid": True, "minimum_clearance_m": minimum,
            "waypoint_count": len(joints), "esdf_checked": True}
