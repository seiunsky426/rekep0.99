import unittest
from unittest.mock import Mock

import numpy as np

from rekpiper_grasp.anygrasp_adapter import CameraGrasp, AnyGraspAdapter
from rekpiper_grasp.grasp_geometry import anygrasp_to_piper_pose


class GraspGeometryTest(unittest.TestCase):
    def test_sdk_collision_is_enabled_by_default_and_can_be_replaced(self):
        adapter = AnyGraspAdapter.__new__(AnyGraspAdapter)
        adapter._detector = Mock()
        adapter._detector.get_grasp.return_value = None
        points, mask = np.ones((1,3), np.float32), np.ones(1, bool)
        adapter.infer(points, mask, 'rs1')
        self.assertTrue(adapter._detector.get_grasp.call_args[0][1]['collision_detection'])
        adapter.infer(points, mask, 'rs1', collision_detection=False)
        self.assertFalse(adapter._detector.get_grasp.call_args[0][1]['collision_detection'])

    def test_none_limit_keeps_all_candidates_and_default_stays_fifty(self):
        grasp = Mock(rotation_matrix=np.eye(3), translation=np.zeros(3),
                     width=.03, depth=.02, score=.8)
        ranked = [grasp]*84
        detector = Mock()
        detector.get_grasp.return_value.nms.return_value.sort_by_score.return_value = ranked
        adapter = AnyGraspAdapter.__new__(AnyGraspAdapter)
        adapter._detector = detector
        points, mask = np.ones((1, 3), np.float32), np.ones(1, bool)
        self.assertEqual(len(adapter.infer(points, mask, 'rs1')), 50)
        self.assertEqual(len(adapter.infer(points, mask, 'fused', max_candidates=None)), 84)
        self.assertEqual(len(adapter.infer(points, mask, 'rs3', max_candidates=3)), 3)

    def test_default_targets_grasp_without_retreat(self):
        grasp = CameraGrasp(np.array([.4, .1, .2]), np.eye(3), .04, .02, .9, 'fused_rs1_rs3')
        candidate = anygrasp_to_piper_pose(grasp, np.eye(4))
        np.testing.assert_allclose(candidate.grasp_pose[:3, 3], [.42, .1, .2])
        np.testing.assert_allclose(candidate.pregrasp_pose, candidate.grasp_pose)
        np.testing.assert_allclose(candidate.grasp_pose[:3, 2], candidate.approach_axis_base)
        self.assertFalse(candidate.planning_authorized)

    def test_explicit_offset_preserves_grasp_pose_and_axis(self):
        grasp = CameraGrasp(np.array([.4, .1, .2]), np.eye(3), .04, .02, .9, 'rs1')
        direct = anygrasp_to_piper_pose(grasp, np.eye(4))
        offset = anygrasp_to_piper_pose(grasp, np.eye(4), pregrasp_distance_m=.08)
        np.testing.assert_allclose(offset.grasp_pose, direct.grasp_pose)
        np.testing.assert_allclose(offset.pregrasp_pose[:3, 3], [.34, .1, .2])


if __name__ == '__main__':
    unittest.main()
