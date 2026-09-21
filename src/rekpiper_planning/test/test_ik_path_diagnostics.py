import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from rekpiper_planning.continuous_ik import validate_continuous_ik
from rekpiper_planning.ik_path_diagnostics import COLORS, diagnose_path, joint_metrics


class DiagnosticIK:
    _lower = np.full(6, -3.)
    _upper = np.full(6, 3.)
    position_tolerance = .01
    orientation_tolerance = .1

    def __init__(self):
        self.seeds = []

    def forward(self, q):
        p = np.eye(4); p[0, 3] = q[0]
        return p

    def _pose_errors(self, a, b):
        return abs(a[0, 3]-b[0, 3]), 0.

    def solve(self, p, initial_joint_pos, max_iterations):
        self.seeds.append(np.asarray(initial_joint_pos).copy())
        q = np.zeros(7); q[0] = p[0, 3]
        success = q[0] <= .4
        return SimpleNamespace(success=success, cspace_position=q, num_descents=1,
                               position_error=0. if success else 1., rotation_error=0.)

    def validate_pose_sequence(self, poses, initial, trace=None):
        return validate_continuous_ik(self, poses, initial, trace=trace)


class IKDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.ik = DiagnosticIK()
        self.poses = np.array([self.ik.forward([x]) for x in (.1, .8, .2)])
        self.report = diagnose_path(self.ik, self.poses, np.zeros(6))

    def test_disconnected_but_independently_valid_point_cannot_be_green(self):
        rows = self.report['waypoints']
        self.assertEqual([r['continuity_status'] for r in rows], ['accepted', 'failed', 'not_reached'])
        self.assertTrue(rows[2]['independent']['ik_success'])
        self.assertEqual(rows[2]['display_status'], 'not_reached')
        self.assertIsNone(rows[2]['primary_attempt_id'])
        np.testing.assert_array_equal(self.ik.seeds[-3:], np.zeros((3, 6)))

    def test_bounded_limit_margin_is_not_wrapped(self):
        result = joint_metrics(self.ik, [3.1, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(result['minimum_joint_margin_rad'], -.1)
        self.assertFalse(result['within_joint_limits'])

    def test_rviz_uses_continuation_status_instead_of_independent_success(self):
        script = Path(__file__).resolve().parents[1]/'scripts/m8_offline_preview_node.py'
        spec = importlib.util.spec_from_file_location('ik_preview', script)
        preview = importlib.util.module_from_spec(spec); spec.loader.exec_module(preview)
        poses7 = np.array([r['target_pose7'] for r in self.report['waypoints']])
        with patch.object(preview.rospy.Time, 'now', return_value=preview.rospy.Time(0)):
            markers = preview._continuous_ik_markers(poses7, self.report)
        points = [m for m in markers if m.ns == 'continuous_ik_waypoints']
        for marker, status in zip(points, ['accepted', 'ik_residual_exceeded', 'not_reached']):
            np.testing.assert_allclose([marker.color.r, marker.color.g, marker.color.b], COLORS[status])
        np.testing.assert_array_equal([[p.x, p.y, p.z] for p in markers[0].points], poses7[:, :3])


if __name__ == '__main__':
    unittest.main()
