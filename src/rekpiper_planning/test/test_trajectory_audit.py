#!/usr/bin/env python3

from types import SimpleNamespace
import io
import unittest

import numpy as np
from geometry_msgs.msg import Point
from rekpiper_msgs.msg import SDFGrid

from rekpiper_planning.trajectory_audit import (
    TrajectoryAuditError, audit_joint_path)


class _Sampler:
    sampled_links=('arm','gripper_base')
    def __init__(self):
        self.links = []
        self.kinematics = self

    @staticmethod
    def link_transforms(_joints):
        return {"gripper_base": np.eye(4)}

    def samples(self, _joints, links=None):
        self.links.append(links)
        return np.asarray([[0.5, 0.5, 0.5]]), np.asarray([0.01])


def grid(distance):
    values = [float(distance)] * 8
    return SimpleNamespace(
        valid=True, map_generation_uuid="generation",
        size_x=2, size_y=2, size_z=2,
        distances_m=values, observed=[True] * 8,
        bounds_min=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        bounds_max=SimpleNamespace(x=1.0, y=1.0, z=1.0),
        resolution_m=1.0)


class TrajectoryAuditTest(unittest.TestCase):
    def test_contact_exception_cannot_open_unknown_space(self):
        sampler=_Sampler()
        sampler.contact_samples=lambda q,width:(np.array([[.5,.5,.5]]),np.array([.01]),np.array(['finger']))
        policy=SimpleNamespace(opening_m=.04,
            allowed=lambda *args:np.array([True]),target_collision=lambda *args:np.array([True]))
        known=grid(0.)
        self.assertTrue(audit_joint_path(np.zeros((2,6)),sampler,known,contact_policy=policy)['valid'])
        known.observed=[0]*8
        with self.assertRaises(TrajectoryAuditError):
            audit_joint_path(np.zeros((2,6)),sampler,known,contact_policy=policy)

    def test_dense_postprocess_path_reaudit_detects_interior_collision(self):
        sampler=_Sampler()
        def samples(q,links=None):
            return np.array([[q[0],.5,.5]]),np.array([.01])
        sampler.samples=samples
        state=grid(-.1); state.size_x=3
        state.distances_m=[-.1]*4+[0.]*4+[-.1]*4; state.observed=[1]*12
        endpoints=np.zeros((2,6)); endpoints[1,0]=1.
        # Even sparse input must be densified by the common auditor.
        with self.assertRaises(TrajectoryAuditError):
            audit_joint_path(endpoints,sampler,state)
        with self.assertRaises(TrajectoryAuditError):
            audit_joint_path(np.linspace(endpoints[0],endpoints[1],11),sampler,state)

    def test_ros_uint8_observed_roundtrip_preserves_unknown_rejection(self):
        message = SDFGrid(valid=True, map_generation_uuid="generation",
            size_x=2, size_y=2, size_z=2, resolution_m=1.,
            distances_m=[-.1] * 8, observed=[1] * 8,
            bounds_min=Point(), bounds_max=Point(1., 1., 1.))
        wire = io.BytesIO()
        message.serialize(wire)
        decoded = SDFGrid().deserialize(wire.getvalue())
        self.assertIsInstance(decoded.observed, bytes)
        self.assertTrue(audit_joint_path(np.zeros((2, 6)), _Sampler(), decoded)["valid"])
        decoded.observed = bytes([0] * 8)
        with self.assertRaises(TrajectoryAuditError):
            audit_joint_path(np.zeros((2, 6)), _Sampler(), decoded)
        decoded.observed = bytes([2] * 8)
        with self.assertRaises(TrajectoryAuditError):
            audit_joint_path(np.zeros((2, 6)), _Sampler(), decoded)
        decoded.observed = bytes([1] * 8)
        decoded.valid = False
        with self.assertRaises(TrajectoryAuditError):
            audit_joint_path(np.zeros((2, 6)), _Sampler(), decoded)

    def test_negative_free_space_is_accepted(self):
        result = audit_joint_path(
            np.zeros((2, 6)), _Sampler(), grid(-0.10),
            minimum_clearance_m=0.02)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["minimum_clearance_m"], 0.09)

    def test_surface_or_occupied_space_is_rejected(self):
        for distance in (0.0, 0.10):
            with self.subTest(distance=distance):
                with self.assertRaises(TrajectoryAuditError):
                    audit_joint_path(np.zeros((2, 6)), _Sampler(),
                                     grid(distance))

    def test_contact_link_exception_applies_only_to_final_waypoint(self):
        sampler = _Sampler()
        audit_joint_path(
            np.zeros((2, 6)), sampler, grid(-0.10),
            links=("all",), final_waypoint_links=("body",))
        self.assertEqual(sampler.links, [("all",), ("body",)])

    def test_inflated_attached_cloud_is_also_fail_closed(self):
        attached = np.asarray([[0.5, 0.5, 0.5]])
        with self.assertRaises(TrajectoryAuditError):
            audit_joint_path(
                np.zeros((2, 6)), _Sampler(), grid(0.0),
                attached_points_local=attached)


if __name__ == "__main__":
    unittest.main()
