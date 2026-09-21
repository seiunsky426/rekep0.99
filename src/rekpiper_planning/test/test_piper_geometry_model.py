#!/usr/bin/env python3
"""Integration checks for the exact Piper URDF and collision meshes."""

from pathlib import Path
import unittest

import numpy as np
from urdf_parser_py.urdf import URDF

from rekpiper_planning.piper_collision_sampling import PiperCollisionSampler
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver


JOINT_NAMES = ["joint{}".format(index) for index in range(1, 7)]
SOURCE_ROOT = Path(__file__).resolve().parents[2]
URDF_PATH = SOURCE_ROOT / "piper_description" / "urdf" / \
    "piper_description.urdf"


class PiperGeometryModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xml = URDF_PATH.read_text(encoding="utf-8")
        cls.urdf = URDF.from_xml_string(cls.xml)
        cls.solver = PiperURDFIKSolver.from_urdf_xml(
            cls.xml, "base_link", "rekep_tcp", JOINT_NAMES,
            position_tolerance=1e-4, orientation_tolerance=1e-3)

    def test_tcp_and_gripper_geometry_match_the_runtime_contract(self):
        joints = {joint.name: joint for joint in self.urdf.joints}
        tcp = joints["gripper_base_to_rekep_tcp"]
        self.assertEqual(tcp.type, "fixed")
        np.testing.assert_allclose(tcp.origin.xyz, [0.0, 0.0, 0.1358])
        self.assertAlmostEqual(joints["joint7"].limit.upper, 0.035)
        self.assertAlmostEqual(joints["joint8"].limit.lower, -0.035)

    def test_real_urdf_fk_ik_round_trips_and_rejects_unreachable_pose(self):
        cases = [
            [-0.162350527, 1.114497447, -1.033741063,
             -0.887115952, 1.143923698, 0.023893557],
            [0.2, 0.8, -1.0, 0.3, 0.7, -0.4],
            [-0.5, 1.5, -1.8, -0.6, 0.4, 0.8],
        ]
        perturbation = np.asarray([0.03, -0.02, 0.04, -0.03, 0.02, -0.04])
        for values in cases:
            joints = np.asarray(values)
            target = self.solver.forward(joints)
            seed = np.clip(
                joints + perturbation, self.solver._lower, self.solver._upper)
            result = self.solver.solve(
                target, max_iterations=400, initial_joint_pos=seed,
                position_tolerance=1e-4, orientation_tolerance=1e-3)
            self.assertTrue(result.success)
            self.assertLessEqual(result.position_error, 1e-4)
            self.assertLessEqual(result.rotation_error, 1e-3)
            self.assertTrue(np.all(result.cspace_position[:-1]
                                   >= self.solver._lower))
            self.assertTrue(np.all(result.cspace_position[:-1]
                                   <= self.solver._upper))

        unreachable = np.eye(4)
        unreachable[:3, 3] = [4.0, 0.0, 0.0]
        result = self.solver.solve(
            unreachable, max_iterations=400, initial_joint_pos=cases[0],
            position_tolerance=1e-4, orientation_tolerance=1e-3)
        self.assertFalse(result.success)

    def test_collision_samples_cover_every_rigid_arm_link_and_swept_gripper(self):
        sampler = PiperCollisionSampler(
            self.xml, self.solver, voxel_size_m=0.020,
            maximum_points_per_link=300)
        expected = {
            "base_link", "link1", "link2", "link3", "link4", "link5",
            "link6", "gripper_base"}
        self.assertEqual(set(sampler.sampled_links), expected)
        self.assertTrue(all(len(sampler._local_samples[name]) > 0
                            for name in expected))
        points, radii = sampler.samples(np.zeros(6))
        self.assertEqual(points.shape[1], 3)
        self.assertEqual(points.shape[0], radii.shape[0])
        self.assertTrue(np.all(np.isfinite(points)))
        self.assertTrue(np.all(radii > 0.0))


if __name__ == "__main__":
    unittest.main()
