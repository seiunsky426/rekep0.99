"""Offline direction and exact URDF-mesh/table-plane checks; no robot clients."""
import numpy as np
from scipy.spatial import cKDTree

from .grasp_geometry import anygrasp_to_piper_pose
from .piper_gripper import PiperGripperGeometry
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver


def load_gripper(xml):
    ik = PiperURDFIKSolver.from_urdf_xml(xml, 'base_link', 'rekep_tcp',
                                       ['joint{}'.format(i) for i in range(1, 7)])
    return PiperGripperGeometry(xml, ik)


def direction_queries(normal, mode):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    if mode == 'top':
        return [-normal], 15.
    if mode != 'side':
        raise ValueError('unknown_direction_mode')
    first = np.array([1., 0., 0.])
    first -= normal * (first @ normal)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    return [np.cos(a)*first + np.sin(a)*second for a in np.deg2rad(np.arange(0, 360, 30))], 15.


def direction_checks(approach, closing, normal, mode):
    downward = float(np.rad2deg(np.arccos(np.clip(-np.dot(approach, normal), -1, 1))))
    horizontal = float(np.rad2deg(np.arcsin(np.clip(abs(np.dot(approach, normal)), 0, 1))))
    closing_angle = float(np.rad2deg(np.arcsin(np.clip(abs(np.dot(closing, normal)), 0, 1))))
    checks = dict(approach_direction=(downward <= 15.+1e-7 if mode == 'top' else horizontal <= 10.+1e-7),
                  closing_parallel_to_table=closing_angle <= (15. if mode == 'top' else 10.)+1e-7)
    return checks, dict(downward_deg=downward, horizontal_deg=horizontal, closing_deg=closing_angle)


class DirectionalAudit:
    def __init__(self, target, normal, offset, gripper):
        self.target = np.asarray(target)
        self.normal = np.asarray(normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)
        self.offset = float(offset)
        self.gripper = gripper
        self.tree = cKDTree(self.target)
        distances, indices = self.tree.query(self.target, k=min(20, len(self.target)))
        local = self.target[indices]
        local -= local.mean(axis=1, keepdims=True)
        values, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', local, local)/local.shape[1])
        self.normals = vectors[:, :, 0]
        self.normal_valid = ((values[:, 0]/np.maximum(values[:, 1], 1e-12) < .25)
                             & (distances[:, -1] < .025) & (values[:, 1] > 1e-8))

    def audit(self, grasp, arm_from_reference, mode, identifier):
        candidate = anygrasp_to_piper_pose(
            grasp, arm_from_reference, pregrasp_distance_m=.06,
            physical_opening_m=self.gripper.maximum_opening_m, candidate_id=identifier)
        checks, angles = direction_checks(candidate.approach_axis_base,
                                            candidate.closing_axis_base, self.normal, mode)
        checks.update(width=bool(candidate.width_ok),
                      insertion_depth=0 < candidate.insertion_depth_m <= self.gripper.usable_depth_m)
        clearances = {}
        if checks['width']:
            # Plane distance is affine in approach translation and prismatic jaw
            # opening. Checking endpoint vertices exactly bounds both sweeps.
            for state, pose, width in (
                    ('preopen_start', candidate.pregrasp_pose, candidate.suggested_preopen_width_m),
                    ('preopen_goal', candidate.grasp_pose, candidate.suggested_preopen_width_m),
                    ('closed_goal', candidate.grasp_pose, candidate.predicted_width_m)):
                for name, mesh in self.gripper.meshes(pose, width):
                    clearances[state+'/'+name] = float(np.min(mesh.vertices @ self.normal+self.offset))
        minimum = min(clearances.values()) if clearances else None
        checks['piper_table_clearance'] = minimum is not None and minimum >= .003
        closing = candidate.closing_axis_base
        contacts = candidate.tcp_position + np.array([[-1.], [1.]])*candidate.predicted_width_m/2*closing
        distances, indices = self.tree.query(contacts)
        normal_angles = np.rad2deg(np.arccos(np.clip(np.abs(self.normals[indices] @ closing), 0, 1)))
        relative = self.target-candidate.tcp_position
        near = np.abs(relative @ candidate.approach_axis_base) < .01
        near &= np.abs(relative @ np.cross(candidate.approach_axis_base, closing)) < .01
        projected = relative[near] @ closing
        contact_checks = dict(
            two_observed_contacts=bool(np.all(distances <= .005)),
            opposing_surface_normals=bool(np.all(self.normal_valid[indices]) and np.all(normal_angles <= 20.)),
            target_between_fingers=bool(len(projected) and np.min(projected) < -.005 and np.max(projected) > .005))
        return dict(geometry_pass=all(checks.values()), contact_supported=all(contact_checks.values()),
                    checks=checks, contact_checks=contact_checks,
                    rejection_reasons=[n for n, ok in checks.items() if not ok],
                    contact_reasons=[n for n, ok in contact_checks.items() if not ok],
                    angles_deg=angles, minimum_table_clearance_m=minimum,
                    part_clearances_m=clearances, contact_distances_m=distances.tolist(),
                    contact_normal_angles_deg=normal_angles.tolist(),
                    piper_pose_arm_base=candidate.grasp_pose.tolist(),
                    piper_pregrasp_arm_base=candidate.pregrasp_pose.tolist(),
                    preopen_width_m=candidate.suggested_preopen_width_m,
                    contact_points_arm_base=contacts.tolist(),
                    whole_arm_ik_checked=False, scene_collision_checked=False,
                    physical_grasp_verified=False, planning_authorized=False)
