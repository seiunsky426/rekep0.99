import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/m8_offline_preview_node.py'
spec = importlib.util.spec_from_file_location('m8_preview', SCRIPT)
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


class GraspPreviewTest(unittest.TestCase):
    def test_tip_spacing_and_open_approach_axis(self):
        points = preview._gripper_outline(np.eye(4), .07)
        self.assertEqual(points.shape, (4, 3))
        np.testing.assert_allclose((points[0] + points[3]) / 2, [0, 0, 0])
        np.testing.assert_allclose(points[3] - points[0], [0, .07, 0])
        np.testing.assert_allclose(points[0] - points[1], [0, 0, .04])
        np.testing.assert_allclose(points[3] - points[2], [0, 0, .04])

    def test_pose_rotation_and_translation(self):
        pose = np.eye(4)
        pose[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        pose[:3, 3] = [.5, -.1, .02]
        points = preview._gripper_outline(pose, .05)
        np.testing.assert_allclose((points[0] + points[3]) / 2, pose[:3, 3])
        np.testing.assert_allclose(points[3] - points[0], .05 * pose[:3, 1])
        np.testing.assert_allclose(points[0] - points[1], .04 * pose[:3, 2])

    def test_invalid_input_rejected(self):
        for width in (0, -.01, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                preview._gripper_outline(np.eye(4), width)
        with self.assertRaises(ValueError):
            preview._gripper_outline(np.eye(3), .05)


if __name__ == '__main__':
    unittest.main()
