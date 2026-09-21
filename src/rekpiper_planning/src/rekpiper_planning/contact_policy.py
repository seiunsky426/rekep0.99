"""Shared finger-pad/target exceptions and supported-release geometry checks."""
import numpy as np
from scipy.spatial import cKDTree, ConvexHull


class TargetContactPolicy:
    def __init__(self, ik, candidate, target_points, scene_points, finger_names,
                 opening_m, grasp_matrix, contact_radius_m=.005, other_object_points=None):
        self.ik, self.opening_m = ik, float(opening_m)
        self.rigid_group_id = int(getattr(candidate,'rigid_group_id',0))
        self.object_uuid = str(getattr(candidate,'object_uuid',''))
        self.target = cKDTree(np.asarray(target_points))
        scene = np.asarray(scene_points)
        owned = self.target.query(scene)[0] <= .005
        if other_object_points is not None and len(other_object_points):
            owned &= (self.target.query(scene)[0]+.002
                      < cKDTree(np.asarray(other_object_points)).query(scene)[0])
        if np.count_nonzero(owned) < 120 or not np.any(~owned):
            raise ValueError('contact_scene_ownership_unavailable')
        self.other = cKDTree(scene[~owned])
        self.fingers = set(finger_names)
        self.contacts = np.array([[p.x,p.y,p.z] for p in candidate.contact_points])
        if self.contacts.shape != (2,3) or not np.all(np.isfinite(self.contacts)):
            raise ValueError('two_bound_finger_contacts_required')
        if np.max(self.target.query(self.contacts)[0]) > contact_radius_m:
            raise ValueError('contact_not_on_target_mask_cloud')
        self.grasp_matrix = grasp_matrix
        self.contact_radius_m = contact_radius_m
        self.probe_frame = None
        self.target_points = np.asarray(target_points)

    def refresh_scene(self,target_points,scene_points,other_object_points):
        target=cKDTree(np.asarray(target_points)); scene=np.asarray(scene_points)
        distance=target.query(scene)[0]; owned=distance<=.005
        if len(other_object_points):
            owned &= distance+.002 < cKDTree(np.asarray(other_object_points)).query(scene)[0]
        if np.count_nonzero(owned)<120 or not np.any(~owned):
            raise ValueError('fresh_contact_ownership_unavailable')
        self.target=target
        self.other=cKDTree(scene[~owned])

    def begin_attachment_probe(self, gripper_matrix, target_points):
        """Hypothesized attachment is only for collision checking, not lifecycle success."""
        self.probe_frame = np.asarray(gripper_matrix)
        inverse = np.linalg.inv(self.probe_frame)
        self.probe_points_local = np.asarray(target_points) @ inverse[:3,:3].T+inverse[:3,3]

    def target_collision(self,q,points,radii,labels):
        target = self.target
        if self.probe_frame is not None:
            matrix = self.ik.forward(q)
            target = cKDTree(self.probe_points_local @ matrix[:3,:3].T+matrix[:3,3])
        return ((target.query(points)[0] < radii+.01) & (labels != 'attached_object'))

    def allowed(self, q, points, radii, labels):
        # Only the terminal 5 mm neighborhood may intentionally make contact.
        position, rotation = self.ik._pose_errors(self.ik.forward(q),self.grasp_matrix)
        if self.probe_frame is None and (position > .005 or rotation > .10):
            return np.zeros(len(points),dtype=bool)
        contacts = self.contacts
        if self.probe_frame is not None:
            transform = self.ik.forward(q) @ np.linalg.inv(self.probe_frame)
            contacts = contacts @ transform[:3,:3].T+transform[:3,3]
        pad_distance = cKDTree(contacts).query(points)[0]
        allowed = (np.isin(labels,list(self.fingers))
                & (pad_distance <= self.contact_radius_m+radii)
                & (self.other.query(points)[0] > radii+.01))
        if self.probe_frame is None:
            allowed &= self.target.query(points)[0] <= self.contact_radius_m+radii
        else:
            # Carried geometry overlapping its own old static surface is not
            # permission to touch any other object or unknown map voxel.
            allowed |= ((labels == 'attached_object')
                & (self.target.query(points)[0] <= radii+.005)
                & (self.other.query(points)[0] > radii+.01))
        return allowed


def validate_support_geometry(object_points, support_points, uncertainty_m,
                              maximum_gap_m=.003, support_margin_m=.002):
    """No drop release: support must contain the object's footprint with margin."""
    obj, support = np.asarray(object_points),np.asarray(support_points)
    if (obj.ndim != 2 or support.ndim != 2 or obj.shape[1] != 3 or support.shape[1] != 3
            or min(len(obj),len(support)) < 30 or not np.all(np.isfinite(obj))
            or not np.all(np.isfinite(support)) or not 0 <= uncertainty_m <= .003):
        raise ValueError('placement_geometry_not_precise_or_complete')
    top = np.quantile(support[:,2],.95)
    face = support[support[:,2] >= top-.003]
    hull = ConvexHull(face[:,:2])
    footprint = obj[:,:2]
    signed = footprint @ hull.equations[:,:2].T+hull.equations[:,2]
    if np.any(signed > -(support_margin_m+uncertainty_m)):
        raise ValueError('object_footprint_outside_support')
    gap = float(np.min(obj[:,2])-top)
    if gap < -uncertainty_m or gap+uncertainty_m > maximum_gap_m:
        raise ValueError('K10_K9_height_conflicts_with_supported_release:gap_m={}'.format(gap))
    return {'support_gap_m':gap,'supported':True}


def validate_preopen(result, requested_m, tolerance_m=.003):
    if (result is None or not result.success or not np.isfinite(result.final_opening_m)
            or abs(float(result.final_opening_m)-requested_m) > tolerance_m):
        raise RuntimeError('gripper_preopen_feedback_mismatch')
