#!/usr/bin/env python3
"""Offline coverage for deterministic full-arm collision samples."""

import unittest

import numpy as np

from rekpiper_planning.piper_collision_sampling import PiperCollisionSampler


_URDF = """<robot name="collision_test">
  <link name="base">
    <collision><geometry><box size="0.10 0.10 0.10"/></geometry></collision>
  </link>
  <link name="arm">
    <collision>
      <origin xyz="0.10 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="0.02" length="0.20"/></geometry>
    </collision>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="arm"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
</robot>"""


class _Kinematics:
    joint_names = ("joint1",)
    tip_frame = "arm"

    @staticmethod
    def link_transforms(joints):
        angle = float(np.asarray(joints)[0])
        rotation = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        arm = np.eye(4)
        arm[:3, :3] = rotation
        return {"base": np.eye(4), "arm": arm}


class PiperCollisionSamplingTest(unittest.TestCase):
    def test_samples_are_deterministic_and_follow_link_fk(self):
        sampler = PiperCollisionSampler(
            _URDF, _Kinematics(), voxel_size_m=0.02,
            maximum_points_per_link=100)
        first, radii = sampler.samples([0.0])
        repeated, repeated_radii = sampler.samples([0.0])
        rotated, _ = sampler.samples([np.pi / 2.0])
        np.testing.assert_array_equal(first, repeated)
        np.testing.assert_array_equal(radii, repeated_radii)
        self.assertEqual(set(sampler.sampled_links), {"arm", "base"})
        self.assertTrue(np.all(radii > 0.0))
        self.assertFalse(np.array_equal(first, rotated))


if __name__ == "__main__":
    unittest.main()
