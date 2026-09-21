"""Grasp dimensions and display geometry from the installed Piper collision URDF."""
import numpy as np
import trimesh
from urdf_parser_py.urdf import URDF

from rekpiper_planning.piper_collision_sampling import _origin_matrix, _resolve_mesh


class PiperGripperGeometry:
    def __init__(self, robot_xml, ik):
        robot = URDF.from_xml_string(robot_xml)
        transforms = ik.link_transforms(np.zeros(len(ik.joint_names)))
        self.tcp_from_palm = np.linalg.inv(ik.forward(np.zeros(len(ik.joint_names)))) @ transforms['gripper_base']
        fingers = [joint for joint in robot.joints
                   if joint.parent == 'gripper_base' and joint.type == 'prismatic']
        if len(fingers) != 2:
            raise ValueError('piper_two_finger_urdf_required')
        self.maximum_opening_m = sum(float(j.limit.upper-j.limit.lower) for j in fingers)
        if not 0 < self.maximum_opening_m <= .070001:
            raise ValueError('piper_urdf_opening_exceeds_driver_contract')
        self.parts = []
        for name, joint in [('gripper_base', None)] + [(j.child, j) for j in fingers]:
            link = robot.link_map[name]
            meshes = []
            for collision in link.collisions:
                geometry = collision.geometry
                if hasattr(geometry, 'filename'):
                    mesh = trimesh.load(str(_resolve_mesh(geometry.filename)), force='mesh', process=False)
                    if geometry.scale is not None:
                        mesh.apply_scale(np.asarray(geometry.scale))
                elif hasattr(geometry, 'size'):
                    mesh = trimesh.creation.box(extents=geometry.size)
                else:
                    raise ValueError('unsupported_piper_gripper_geometry')
                mesh.apply_transform(_origin_matrix(collision.origin))
                meshes.append(mesh)
            if not meshes:
                raise ValueError('piper_gripper_collision_mesh_missing')
            mesh = trimesh.util.concatenate(meshes)
            self.parts.append((name, mesh, joint))
        closed = self.meshes(np.eye(4), 0.)
        finger_points = np.vstack([mesh.vertices for _, mesh in closed[1:]])
        self.finger_height_m = float(np.ptp(finger_points[:, 0]))
        self.finger_depth_m = float(np.ptp(finger_points[:, 2]))
        self.usable_depth_m = min(self.finger_depth_m, float(-np.max(closed[0][1].vertices[:, 2])))
        self.tcp_offset_m = float(np.linalg.norm(self.tcp_from_palm[:3, 3]))
        if min(self.finger_height_m, self.usable_depth_m) <= 0:
            raise ValueError('piper_gripper_tcp_geometry_inconsistent')

    def meshes(self, tcp_pose, opening_m):
        if not 0 <= opening_m <= self.maximum_opening_m:
            raise ValueError('piper_gripper_opening_outside_urdf')
        result = []
        for name, source, joint in self.parts:
            transform = np.asarray(tcp_pose) @ self.tcp_from_palm
            if joint is not None:
                position = opening_m/2 if joint.limit.upper > 0 else -opening_m/2
                if not joint.limit.lower-1e-9 <= position <= joint.limit.upper+1e-9:
                    raise ValueError('piper_finger_opening_outside_urdf')
                motion = np.eye(4)
                motion[:3, 3] = np.asarray(joint.axis)*position
                transform = transform @ _origin_matrix(joint.origin) @ motion
            mesh = source.copy()
            mesh.apply_transform(transform)
            result.append((name, mesh))
        return result

    def points(self, candidate):
        return np.vstack([mesh.vertices for _, mesh in self.meshes(
            candidate.grasp_pose, candidate.suggested_preopen_width_m)])
