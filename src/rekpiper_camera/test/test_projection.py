#!/usr/bin/env python3
"""Pure geometry regression tests; no camera or ROS master required."""

from types import SimpleNamespace
import unittest

import numpy as np

from rekpiper_camera import (
    CameraIntrinsics,
    depth_to_xyz,
    transform_points,
    transform_to_matrix,
)


RGBD_NODE = (
    __import__("importlib.util").util.spec_from_file_location(
        "rgbd_projection_node",
        str(__import__("pathlib").Path(__file__).parents[1]
            / "scripts" / "rgbd_projection_node.py")))
_RGBD_MODULE = __import__("importlib.util").util.module_from_spec(RGBD_NODE)
RGBD_NODE.loader.exec_module(_RGBD_MODULE)


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.intrinsics = CameraIntrinsics(2.0, 2.0, 1.0, 1.0, 3, 3)

    def test_principal_point_projects_to_optical_axis(self):
        depth = np.full((3, 3), 1000, dtype=np.uint16)
        xyz, valid = depth_to_xyz(depth, self.intrinsics, 0.001, 0.1, 2.0)
        np.testing.assert_allclose(xyz[1, 1], [0.0, 0.0, 1.0], atol=1e-7)
        self.assertTrue(valid.all())

    def test_pixel_projection_uses_camera_info_values(self):
        depth = np.full((3, 3), 1000, dtype=np.uint16)
        xyz, _ = depth_to_xyz(depth, self.intrinsics, 0.001, 0.1, 2.0)
        np.testing.assert_allclose(xyz[0, 0], [-0.5, -0.5, 1.0], atol=1e-7)
        np.testing.assert_allclose(xyz[2, 2], [0.5, 0.5, 1.0], atol=1e-7)

    def test_invalid_depth_is_nan_and_masked(self):
        depth = np.full((3, 3), 1000, dtype=np.uint16)
        depth[0, 0] = 0
        depth[2, 2] = 3000
        xyz, valid = depth_to_xyz(depth, self.intrinsics, 0.001, 0.1, 2.0)
        self.assertFalse(valid[0, 0])
        self.assertFalse(valid[2, 2])
        self.assertTrue(np.isnan(xyz[0, 0]).all())
        self.assertTrue(np.isnan(xyz[2, 2]).all())

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            depth_to_xyz(np.ones((2, 3)), self.intrinsics, 0.001)

    def test_rigid_transform(self):
        transform = SimpleNamespace(
            translation=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        matrix = transform_to_matrix(transform)
        points = np.array([[[0.0, 0.0, 1.0], [np.nan, np.nan, np.nan]]])
        result = transform_points(points, matrix)
        np.testing.assert_allclose(result[0, 0], [1.0, 2.0, 4.0], atol=1e-7)
        self.assertTrue(np.isnan(result[0, 1]).all())

    def test_colored_cloud_packs_bgr_as_rviz_rgb(self):
        header = SimpleNamespace(stamp=SimpleNamespace(), frame_id="base_link")
        xyz = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        bgr = np.array([[[0x33, 0x22, 0x11]]], dtype=np.uint8)
        cloud = _RGBD_MODULE.RGBDProjectionNode._organized_rgb_cloud(
            header, xyz, bgr)
        self.assertEqual([field.name for field in cloud.fields],
                         ["x", "y", "z", "rgb"])
        self.assertEqual(cloud.point_step, 16)
        record = np.frombuffer(cloud.data, dtype=np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4"),
        ]))
        self.assertEqual(int(record["rgb"].view(np.uint32)[0]), 0x112233)

if __name__ == "__main__":
    unittest.main()
