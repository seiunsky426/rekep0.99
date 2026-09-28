"""Offline tests for color targets, fusion labels and native grasp frames."""
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from rekpiper_grasp.blue_cloud_comparison import (
    transform_points, select_blue, prepare_case, voxel_average,
    native_pose_record, gripper_segments, view_label)


class BlueCloudComparisonTest(unittest.TestCase):
    def test_color_mask_depth_correspondence_and_small_component(self):
        hsv = np.zeros((40, 60, 3), np.uint8)
        hsv[:, :, 2] = 180
        hsv[5:30, 10:35] = [95, 200, 180]
        hsv[32:36, 45:49] = [95, 200, 180]
        frame = dict(rgb=cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB),
                     depth=np.full((40, 60), 500, np.uint16), depth_encoding='16UC1',
                     K=np.array([[600, 0, 30], [0, 600, 20], [0, 0, 1]]), base_from_camera=np.eye(4))
        frame['depth'][15, 20] = 0
        selected = select_blue(frame, [-1, -1, 0], [1, 1, 1], np.eye(4))
        self.assertEqual(selected['report']['status'], 'ready')
        self.assertFalse(selected['mask'][15, 20])
        self.assertFalse(selected['mask'][33, 46])
        self.assertEqual(int(selected['mask'].sum()), len(selected['target']))
        np.testing.assert_allclose(selected['target'][:, 2], .5)
        np.testing.assert_array_equal(selected['target_rgb'], frame['rgb'][selected['mask']])

    def test_empty_color_target_records_failure(self):
        frame = dict(rgb=np.full((20, 20, 3), 255, np.uint8),
                     depth=np.full((20, 20), .5, np.float32), depth_encoding='32FC1',
                     K=np.array([[100, 0, 10], [0, 100, 10], [0, 0, 1]]), base_from_camera=np.eye(4))
        selected = select_blue(frame, [-1, -1, 0], [1, 1, 1], np.eye(4))
        self.assertEqual(selected['report']['status'], 'insufficient_blue_depth_points')
        self.assertEqual(len(selected['target']), 0)

    def test_nonidentity_arm_camera_transform_roundtrip_and_native_pose(self):
        matrix = np.array([[0., -1, 0, .2], [1, 0, 0, -.1], [0, 0, 1, .6], [0, 0, 0, 1]])
        points = np.array([[.1, .2, .3], [-.1, .1, .2]])
        np.testing.assert_allclose(transform_points(transform_points(points, matrix), np.linalg.inv(matrix)), points)
        grasp = SimpleNamespace(rotation=np.eye(3), translation=points[0], width_m=.03, depth_m=.02, score=.1)
        row = native_pose_record(grasp, matrix, 'test')
        pose = np.asarray(row['pose_arm_base'])
        np.testing.assert_allclose(pose[:3, 3], transform_points(points[:1], matrix)[0])
        # No Piper TCP insertion-depth shift is added to the native origin.
        np.testing.assert_allclose(pose[:3, :3], matrix[:3, :3])
        tips = gripper_segments(row)[[1, 3]]
        local_tips = transform_points(tips, np.linalg.inv(pose))
        np.testing.assert_allclose(local_tips[:, 0], .02)
        np.testing.assert_allclose(local_tips[:, 1], [-.017, .017])

    def test_voxel_provenance_and_target_background_exclusion(self):
        def selected(target, background):
            return dict(target=np.array(target), target_rgb=np.full((len(target), 3), 150),
                        background=np.array(background), background_rgb=np.full((len(background), 3), 80))
        sources = {'rs1': selected([[.1002, 0, .2]], [[.1005, 0, .2], [.2, 0, .2]]),
                   'rs3': selected([[.1008, 0, .2], [.15, 0, .2]], [[.3, 0, .2]])}
        data = prepare_case(sources, ('rs1', 'rs3'), np.eye(4))
        mask = data['region_mask']
        self.assertEqual(int(mask.sum()), 2)
        self.assertEqual(set(data['source_bits'][mask]), {2, 3})
        self.assertEqual(int((~mask).sum()), 2)
        self.assertEqual(set(data['source_bits'][~mask]), {1, 2})
        for size in (.002, .003):
            target = set(map(tuple, np.floor(data['points_arm'][mask]/size).astype(int)))
            background = set(map(tuple, np.floor(data['points_arm'][~mask]/size).astype(int)))
            self.assertFalse(target & background)
        np.testing.assert_allclose(data['points_arm'], data['points_camera'])

    def test_background_sampling_is_bounded_and_reproducible(self):
        background = np.c_[np.arange(31000)*.004, np.zeros(31000), np.ones(31000)]
        selected = dict(target=np.array([[0., 0, .2]]), target_rgb=np.ones((1, 3)),
                        background=background, background_rgb=np.zeros((31000, 3)))
        first = prepare_case({'rs1': selected}, ('rs1',), np.eye(4))
        second = prepare_case({'rs1': selected}, ('rs1',), np.eye(4))
        self.assertEqual(int((~first['region_mask']).sum()), 30000)
        np.testing.assert_array_equal(first['points_arm'], second['points_arm'])

    def test_zero_candidates_have_explicit_display_label(self):
        report = dict(status='zero_candidates', candidates=[])
        self.assertIn('NO CANDIDATES', view_label('rs3', report, 5))
        report['error'] = 'insufficient_blue_depth_points'
        self.assertIn(report['error'], view_label('rs3', report, 1))


if __name__ == '__main__':
    unittest.main()
