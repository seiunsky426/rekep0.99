import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from rekpiper_grasp.anygrasp_adapter import CameraGrasp
from rekpiper_grasp.grasp_geometry import anygrasp_to_piper_pose
from rekpiper_grasp.horizontal_grasp import HorizontalGraspPolicy
from rekpiper_grasp.piper_gripper import PiperGripperGeometry
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver


class HorizontalGraspTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x, y = np.meshgrid(np.linspace(.36, .64, 40), np.linspace(-.14, .14, 40))
        table = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        x, z = np.meshgrid(np.linspace(.485, .515, 21), np.linspace(.025, .075, 31))
        sides = [np.c_[x.ravel(), np.full(x.size, y), z.ravel()] for y in (-.02, .02)]
        y, z = np.meshgrid(np.linspace(-.02, .02, 25), np.linspace(.025, .075, 31))
        front = np.c_[np.full(y.size, .485), y.ravel(), z.ravel()]
        cls.target = np.vstack(sides+[front])
        cls.scene = np.vstack([cls.target, table])
        cls.mask = np.arange(len(cls.scene)) < len(cls.target)
        cls.gripper = SimpleNamespace(finger_height_m=.02, usable_depth_m=.06,
            points=lambda c: c.tcp_position+np.array([[-.06, -.035, -.01], [0., .035, .01]]))
        cls.policy = HorizontalGraspPolicy(cls.scene, cls.mask, [.2, -.3, .3], cls.gripper)

    def candidate(self, position=(.485, 0., .05), rotation=None, width=.04, depth=.015):
        return anygrasp_to_piper_pose(CameraGrasp(np.array(position),
            np.eye(3) if rotation is None else rotation, width, depth, .9, 'rs1'), np.eye(4))

    def test_horizontal_contacts_and_open_space_pass(self):
        result = self.policy.audit(self.candidate())
        self.assertTrue(result['valid'], result)
        self.assertGreater(self.policy.region_mask.sum(), 120)
        self.assertFalse(np.any(self.policy.region_mask[~self.mask]))

    def test_roll_is_checked_even_with_horizontal_approach(self):
        rotation = Rotation.from_euler('x', 90, degrees=True).as_matrix()
        result = self.policy.audit(self.candidate(rotation=rotation))
        self.assertTrue(result['checks']['horizontal_approach'])
        self.assertFalse(result['checks']['horizontal_closing'])

    def test_upward_and_top_down_grasps_rejected(self):
        for angle in (-5, 90):
            result = self.policy.audit(self.candidate(
                rotation=Rotation.from_euler('y', angle, degrees=True).as_matrix()))
            self.assertFalse(result['valid'])
            key = 'no_upward_approach' if angle == -5 else 'horizontal_approach'
            self.assertFalse(result['checks'][key])

    def test_contacts_must_be_observed_and_on_opposite_sides(self):
        result = self.policy.audit(self.candidate(position=(.485, .06, .05)))
        self.assertFalse(result['checks']['contacts_on_observed_target'])
        self.assertFalse(result['checks']['contacts_on_opposite_sides'])

    def test_palm_penetration_rejected_with_contacts_above_table(self):
        self.gripper.points = lambda c: np.array([[.45, 0., -.001]])
        try:
            result = self.policy.audit(self.candidate())
            self.assertTrue(result['checks']['contacts_above_table'])
            self.assertFalse(result['checks']['piper_gripper_above_table'])
        finally:
            self.gripper.points = lambda c: c.tcp_position+np.array([[-.06, -.035, -.01], [0., .035, .01]])

    def test_no_table_does_not_fall_back_to_zero(self):
        with self.assertRaisesRegex(ValueError, 'table_points_missing'):
            HorizontalGraspPolicy(self.target, np.ones(len(self.target), dtype=bool), [.2, 0., .3], self.gripper)

    def test_unrestricted_top_down_keeps_contact_and_table_checks(self):
        x, y = np.meshgrid(np.linspace(.485, .515, 21), np.linspace(-.02, .02, 25))
        top = np.c_[x.ravel(), y.ravel(), np.full(x.size, .075)]
        target = np.vstack([self.target, top])
        scene = np.vstack([target, self.scene[~self.mask]])
        mask = np.arange(len(scene)) < len(target)
        policy = HorizontalGraspPolicy(scene, mask, [.2, -.3, .3], self.gripper,
                                       restrict_horizontal=False)
        self.assertEqual(policy.directions_in_camera(np.eye(4)), [None])
        np.testing.assert_array_equal(policy.region_mask, mask)
        rotation = Rotation.from_euler('y', 90, degrees=True).as_matrix()
        result = policy.audit(self.candidate(position=(.5, 0., .075), rotation=rotation))
        self.assertTrue(result['valid'], result)
        self.assertNotIn('horizontal_approach', result['checks'])
        invalid = policy.audit(self.candidate(position=(.5, .08, .075), rotation=rotation))
        self.assertFalse(invalid['checks']['contacts_on_observed_target'])

    def test_unrestricted_does_not_require_visible_side_surfaces(self):
        x, y = np.meshgrid(np.linspace(.485, .515, 21), np.linspace(-.02, .02, 25))
        top = np.c_[x.ravel(), y.ravel(), np.full(x.size, .075)]
        scene = np.vstack([top, self.scene[~self.mask]])
        mask = np.arange(len(scene)) < len(top)
        policy = HorizontalGraspPolicy(scene, mask, [.2, -.3, .3], self.gripper,
                                       restrict_horizontal=False)
        self.assertEqual(policy.region_mask.sum(), len(top))
        with self.assertRaisesRegex(ValueError, 'visible_side_surface_insufficient'):
            HorizontalGraspPolicy(scene, mask, [.2, -.3, .3], self.gripper)

    def test_model_depth_limit_rejects_long_insertion(self):
        result = self.policy.audit(self.candidate(depth=.08))
        self.assertFalse(result['checks']['piper_insertion_depth'])

    def test_tilted_camera_keeps_horizontal_world_directions(self):
        rotation = Rotation.from_euler('xyz', [35, -20, 60], degrees=True).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        for base, camera in zip(self.policy.approach_directions, self.policy.directions_in_camera(transform)):
            np.testing.assert_allclose(rotation @ camera, base, atol=1e-12)
            self.assertAlmostEqual(float(base[2]), 0.)
            self.assertGreater(np.linalg.norm(camera-[0., 0., 1.]), .1)


class PiperGripperGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        xml = (root/'src/piper_description/urdf/piper_description.urdf').read_text()
        ik = PiperURDFIKSolver.from_urdf_xml(xml, 'base_link', 'rekep_tcp',
                                           ['joint{}'.format(i) for i in range(1, 7)])
        cls.geometry = PiperGripperGeometry(xml, ik)

    def test_urdf_dimensions_and_symmetric_opening(self):
        geometry = self.geometry
        self.assertAlmostEqual(geometry.maximum_opening_m, .07)
        self.assertAlmostEqual(geometry.tcp_offset_m, .1358)
        self.assertGreater(geometry.finger_depth_m, .07)
        self.assertGreater(geometry.finger_height_m, .05)
        closed = geometry.meshes(np.eye(4), 0.)
        opened = geometry.meshes(np.eye(4), .06)
        for i, sign in ((1, -1), (2, 1)):
            delta = opened[i][1].vertices.mean(0)-closed[i][1].vertices.mean(0)
            np.testing.assert_allclose(delta, [0., sign*.03, 0.], atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'opening_outside_urdf'):
            geometry.meshes(np.eye(4), .08)

    def test_gripper_vertices_follow_tcp_rotation_and_translation(self):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('xyz', [20, 30, 40], degrees=True).as_matrix()
        transform[:3, 3] = [.4, -.2, .3]
        local = self.geometry.meshes(np.eye(4), .04)
        placed = self.geometry.meshes(transform, .04)
        for (_, a), (_, b) in zip(local, placed):
            np.testing.assert_allclose(b.vertices, a.vertices @ transform[:3, :3].T+transform[:3, 3], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
