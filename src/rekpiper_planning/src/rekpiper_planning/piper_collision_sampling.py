"""Deterministic URDF collision-volume samples for full-arm ESDF audits."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.spatial.transform import Rotation
from urdf_parser_py.urdf import URDF


class CollisionSamplingError(ValueError):
    pass


def _origin_matrix(origin) -> np.ndarray:
    result = np.eye(4, dtype=float)
    if origin is None:
        return result
    result[:3, 3] = np.asarray(origin.xyz or [0.0, 0.0, 0.0], dtype=float)
    result[:3, :3] = Rotation.from_euler(
        "xyz", np.asarray(origin.rpy or [0.0, 0.0, 0.0], dtype=float)).as_matrix()
    return result


def _resolve_mesh(filename: str, mesh_root: str = "") -> Path:
    value = str(filename)
    if value.startswith("package://"):
        package, relative = value[len("package://"):].split("/", 1)
        import rospkg
        return Path(rospkg.RosPack().get_path(package)) / relative
    if value.startswith("file://"):
        return Path(value[len("file://"):])
    path = Path(value)
    if path.is_absolute():
        return path
    if mesh_root:
        return Path(mesh_root) / path
    return path


class PiperCollisionSampler:
    """Voxelize each modeled chain link once, then transform it per joint state."""

    def __init__(self, robot_description_xml: str, kinematics,
                 voxel_size_m: float = 0.020,
                 maximum_points_per_link: int = 300,
                 mesh_root: str = ""):
        import trimesh

        self.kinematics = kinematics
        self.voxel_size_m = float(voxel_size_m)
        self.sample_radius_m = float(np.sqrt(3.0) * self.voxel_size_m / 2.0)
        self.maximum_points_per_link = int(maximum_points_per_link)
        if (not robot_description_xml or self.voxel_size_m <= 0.0
                or self.maximum_points_per_link <= 0):
            raise CollisionSamplingError("collision sampler configuration is invalid")
        robot = URDF.from_xml_string(str(robot_description_xml))
        modeled_links = set(kinematics.link_transforms(
            np.zeros(len(kinematics.joint_names))))

        def voxelize_link(link):
            chunks = []
            collisions = list(getattr(link, "collisions", None) or [])
            if not collisions and getattr(link, "collision", None) is not None:
                collisions = [link.collision]
            for collision in collisions:
                geometry = collision.geometry
                if hasattr(geometry, "filename"):
                    path = _resolve_mesh(geometry.filename, mesh_root)
                    if not path.is_file():
                        raise CollisionSamplingError(
                            "collision mesh is missing: {}".format(path))
                    mesh = trimesh.load(str(path), force="mesh", process=False)
                    scale = getattr(geometry, "scale", None)
                    if scale is not None:
                        mesh.apply_scale(np.asarray(scale, dtype=float))
                elif hasattr(geometry, "size"):
                    mesh = trimesh.creation.box(
                        extents=np.asarray(geometry.size, dtype=float))
                elif hasattr(geometry, "length") and hasattr(geometry, "radius"):
                    mesh = trimesh.creation.cylinder(
                        radius=float(geometry.radius),
                        height=float(geometry.length), sections=24)
                elif hasattr(geometry, "radius"):
                    mesh = trimesh.creation.icosphere(
                        subdivisions=2, radius=float(geometry.radius))
                else:
                    raise CollisionSamplingError(
                        "unsupported URDF collision geometry on {}".format(
                            link.name))
                mesh.apply_transform(_origin_matrix(collision.origin))
                try:
                    values = np.asarray(
                        mesh.voxelized(self.voxel_size_m).fill().points,
                        dtype=float)
                except Exception as exc:
                    raise CollisionSamplingError(
                        "failed to voxelize collision link {}: {}".format(
                            link.name, exc)) from exc
                if len(values):
                    chunks.append(values)
            return chunks

        self._local_samples = {}
        for link in robot.links:
            if link.name not in modeled_links:
                continue
            chunks = voxelize_link(link)
            if chunks:
                samples = np.vstack(chunks)
                if len(samples) > self.maximum_points_per_link:
                    indices = np.linspace(
                        0, len(samples) - 1, self.maximum_points_per_link,
                        dtype=np.int64)
                    samples = samples[indices]
                self._local_samples[link.name] = samples

        # The official solver treats its collision cloud as rigidly attached
        # to the end effector. Represent the two Piper fingers by a conservative
        # swept volume over their complete ±35 mm prismatic ranges, expressed
        # in gripper_base coordinates. This covers every measured opening while
        # preserving the official rigid-cloud collision interface.
        tip = str(kinematics.tip_frame)
        anchor = ("gripper_base" if "gripper_base" in self._local_samples
                  else tip)
        if anchor in self._local_samples:
            self._palm_samples = self._local_samples[anchor].copy()
            self._finger_samples = []
            self._gripper_anchor = anchor
            swept = [self._local_samples[anchor]]
            link_by_name = {link.name: link for link in robot.links}
            for joint in robot.joints:
                if joint.parent != anchor or joint.type != "prismatic":
                    continue
                child_chunks = voxelize_link(link_by_name[joint.child])
                if not child_chunks:
                    continue
                child_points = np.vstack(child_chunks)
                lower = float(joint.limit.lower)
                upper = float(joint.limit.upper)
                axis = np.asarray(joint.axis, dtype=float)
                origin = _origin_matrix(joint.origin)
                self._finger_samples.append((joint.child, child_points, origin, axis, lower, upper))
                for position in np.linspace(lower, upper, 5):
                    motion = np.eye(4)
                    motion[:3, 3] = axis * position
                    matrix = origin @ motion
                    swept.append(
                        child_points @ matrix[:3, :3].T + matrix[:3, 3])
            combined = np.vstack(swept)
            if len(combined) > self.maximum_points_per_link:
                indices = np.linspace(
                    0, len(combined) - 1, self.maximum_points_per_link,
                    dtype=np.int64)
                combined = combined[indices]
            self._local_samples[anchor] = combined
        if not self._local_samples:
            raise CollisionSamplingError("URDF chain has no collision samples")

    @property
    def sampled_links(self):
        return tuple(sorted(self._local_samples))

    def contact_samples(self, joints, opening_m):
        """Measured-opening geometry with finger identity, never a swept palm exemption."""
        if not 0 <= opening_m <= .070 or not hasattr(self,'_finger_samples'):
            raise CollisionSamplingError('contact gripper geometry unavailable')
        transforms = self.kinematics.link_transforms(joints)
        points, labels = [], []
        for link, local in self._local_samples.items():
            if link == self._gripper_anchor:
                local = self._palm_samples
            matrix = transforms[link]
            points.extend(local @ matrix[:3,:3].T+matrix[:3,3])
            labels.extend([link]*len(local))
        anchor = transforms[self._gripper_anchor]
        for name,local,origin,axis,lower,upper in self._finger_samples:
            position = opening_m/2 if upper > 0 else -opening_m/2
            if not lower-1e-9 <= position <= upper+1e-9:
                raise CollisionSamplingError('measured opening outside finger limits')
            motion = np.eye(4)
            motion[:3,3] = axis*position
            matrix = anchor @ origin @ motion
            points.extend(local @ matrix[:3,:3].T+matrix[:3,3])
            labels.extend([name]*len(local))
        values = np.asarray(points)
        radius = np.sqrt(3.)*self.voxel_size_m/2
        return values,np.full(len(values),radius),np.asarray(labels)

    def samples(self, joints, links: Optional[Iterable[str]] = None
                ) -> tuple[np.ndarray, np.ndarray]:
        transforms = self.kinematics.link_transforms(joints)
        selected = (set(self._local_samples) if links is None
                    else set(str(link) for link in links))
        unknown = selected.difference(self._local_samples)
        if unknown:
            raise CollisionSamplingError(
                "collision samples unavailable for links: {}".format(
                    ", ".join(sorted(unknown))))
        world = []
        for link, local in self._local_samples.items():
            if link not in selected:
                continue
            matrix = transforms.get(link)
            if matrix is None:
                raise CollisionSamplingError(
                    "missing FK transform for collision link {}".format(link))
            world.append(local @ matrix[:3, :3].T + matrix[:3, 3])
        if not world:
            raise CollisionSamplingError("no collision links were selected")
        points = np.vstack(world)
        radii = np.full(len(points), self.sample_radius_m, dtype=float)
        return points, radii
