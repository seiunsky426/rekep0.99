import unittest

import numpy as np

from rekpiper_grasp.anygrasp_adapter import CameraGrasp
from rekpiper_grasp.grasp_geometry import anygrasp_to_piper_pose


class GraspGeometryTest(unittest.TestCase):
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
