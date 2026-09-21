#!/usr/bin/env python3
"""离线测试 Piper 本地 URDF IK 与官方 ReKep 的接口契约。"""

import unittest

import numpy as np

from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver


_PLANAR_URDF = """<robot name="planar_two_joint">
  <link name="base"/><link name="link1"/><link name="tip"/>
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="tip"/>
    <origin xyz="1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
</robot>"""


class PiperURDFIKTest(unittest.TestCase):
    def setUp(self):
        self.solver = PiperURDFIKSolver.from_urdf_xml(
            _PLANAR_URDF, "base", "tip", ["joint1", "joint2"],
            position_tolerance=1e-3, orientation_tolerance=1e-3)

    def test_forward_kinematics_uses_urdf_chain(self):
        pose = self.solver.forward([np.pi / 2.0, 0.0])
        np.testing.assert_allclose(pose[:3, 3], [0.0, 1.0, 0.0], atol=1e-6)

    def test_result_matches_official_solver_fields(self):
        target = np.eye(4)
        target[:3, 3] = [0.0, 1.0, 0.0]
        target[:3, :3] = [[0.0, -1.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]]
        result = self.solver.solve(
            target, max_iterations=100, initial_joint_pos=[0.0, 0.0])
        self.assertTrue(result.success)
        self.assertLess(result.position_error, 1e-3)
        self.assertEqual(result.cspace_position.shape, (3,))
        self.assertTrue(hasattr(result, "num_descents"))

    def test_sequence_validation_fails_closed(self):
        unreachable = np.eye(4)
        unreachable[:3, 3] = [4.0, 0.0, 0.0]
        report = self.solver.validate_pose_sequence(
            [unreachable], initial_joint_pos=[0.0, 0.0])
        self.assertFalse(report["valid"])
        self.assertEqual(report["reason"], "ik_failed")


if __name__ == "__main__":
    unittest.main()
