#!/usr/bin/env python3

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from rekpiper_planning.path_preview import (
    display_indices, live_joint_seed, matrix_pose, pose_matrices, preview_ik)
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver


class PathPreviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        xml = (Path(__file__).resolve().parents[2] /
               'piper_description/urdf/piper_description.urdf').read_text()
        cls.ik = PiperURDFIKSolver.from_urdf_xml(
            xml, 'base_link', 'rekep_tcp', ['joint{}'.format(i) for i in range(1, 7)])
        cls.seed = np.array([0., .5, -.5, 0., .2, 0.])

    def test_display_thinning_preserves_endpoints(self):
        indices = display_indices(1001, 35)
        self.assertEqual(len(indices), 35)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 1000)
        np.testing.assert_array_equal(display_indices(4, 35), range(4))

    def test_live_feedback_reorders_by_name_and_rejects_bad_data(self):
        names, positions = self.ik.joint_names, self.seed.tolist()
        actual = live_joint_seed(names[::-1], positions[::-1], 10., 10.1, .1, self.ik)
        np.testing.assert_array_equal(actual, self.seed)
        cases = [(names, positions, 0., 10., .1),
                 (names, positions, 9., 10., .1),
                 (names, positions, 11., 10., .1),
                 (names, positions, 10., 10., 1.),
                 (names[:-1], positions[:-1], 10., 10., .1),
                 (names + [names[0]], positions + [0.], 10., 10., .1),
                 (names, [float('nan')] * 6, 10., 10., .1),
                 (names, [99.] * 6, 10., 10., .1)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                live_joint_seed(*args, self.ik)

    def test_current_pose_ik_and_changed_target_report_angles(self):
        for offset in (0., .025):
            desired = self.seed.copy()
            desired[0] += offset
            pose = matrix_pose(self.ik.forward(desired))
            report = preview_ik(self.ik, [pose], self.seed, sequence=False)
            self.assertTrue(report['valid'])
            self.assertFalse(report['motion_allowed'])
            self.assertFalse(report['collision_checked'])
            self.assertEqual(report['preview_kind'], 'endpoint_only')
            actual = report['joint_positions'][-1]
            np.testing.assert_allclose(self.ik.forward(actual), self.ik.forward(desired), atol=.02)
            np.testing.assert_allclose(report['joint_delta_rad'][-1], actual - self.seed)

    def test_path_ik_uses_every_dense_pose_and_does_not_accept_failed_suffix(self):
        pose = matrix_pose(self.ik.forward(self.seed))
        path = np.tile(pose, (121, 1))
        with patch('rekpiper_planning.path_preview.validate_continuous_ik',
                   return_value=dict(valid=False, reason='ik_failed', steps=[], failed_pose_index=17)) as validate:
            report = preview_ik(self.ik, path, self.seed)
        self.assertEqual(validate.call_args[0][1].shape, (121, 4, 4))
        self.assertFalse(report['valid'])
        self.assertEqual(report['failed_pose_index'], 17)
        self.assertEqual(report['joint_positions'].shape, (1, 6))

    def test_unreachable_target_remains_failed(self):
        pose = np.array([9., 9., 9., 0., 0., 0., 1.])
        report = preview_ik(self.ik, [pose], self.seed, sequence=False)
        self.assertFalse(report['valid'])
        self.assertEqual(len(report['steps']), 0)

    def test_bad_pose_rejected(self):
        for pose in (np.zeros(7), [0., 0., float('nan'), 0., 0., 0., 1.]):
            with self.assertRaises(ValueError):
                pose_matrices([pose])


if __name__ == '__main__':
    unittest.main()
