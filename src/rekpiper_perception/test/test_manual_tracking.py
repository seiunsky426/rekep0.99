#!/usr/bin/env python3
"""Live-tracking regressions: delayed references, identity gating and loss."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch
import cv2

from rekpiper_perception.keypoint_tracking import (
    nearest_reference_frames, tensor_reference_descriptor, tensor_feature_observation)
from rekpiper_perception.object_registry import visually_tracked
from rekpiper_perception.snapshot_contract import groups_in_seed_bounds
from rekpiper_perception.tracking_geometry import mask_patch_weights


class ManualTrackingTest(unittest.TestCase):
    def test_thin_mask_survives_patch_pooling(self):
        mask = np.zeros((160, 160), np.uint8)
        mask[1:6, 1:50] = 1
        self.assertFalse(cv2.resize(mask, (16, 16),
                                   interpolation=cv2.INTER_NEAREST).any())
        weights = mask_patch_weights(mask, 16)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertEqual(np.count_nonzero(weights), 5)
        with self.assertRaisesRegex(ValueError, "mask is empty"):
            mask_patch_weights(np.zeros_like(mask), 16)

    def test_delayed_sam_uses_capture_time_and_rejects_missing_reference(self):
        def frame(seconds):
            return (SimpleNamespace(to_nsec=lambda: int(seconds * 1e9)), None, None)
        buffers = {'rs1': [frame(10), frame(17)], 'rs3': [frame(10.02), frame(17)]}
        selected = nearest_reference_frames(buffers, 10_000_000_000, .15)
        self.assertEqual(selected['rs1'][0].to_nsec(), 10_000_000_000)
        self.assertIsNone(nearest_reference_frames(buffers, 11_000_000_000, .15))

    def test_masks_prevent_identity_switch_and_occlusion_becoming_an_observation(self):
        features = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]])
        points = np.array([[[.1, 0., 0.], [.5, 0., 0.], [.8, 0., 0.]]])
        reference = tensor_reference_descriptor([features], [points], [.1, 0., 0.], .02)
        result = tensor_feature_observation(reference, [features], [points],
            masks=[np.array([[True, False, False]])])
        np.testing.assert_allclose(result.point, [.1, 0., 0.])
        self.assertIsNone(tensor_feature_observation(reference, [features], [points],
            masks=[np.zeros((1, 3), bool)]))

    def test_two_camera_top_k_rejects_spatial_outlier(self):
        features = torch.tensor([[[1., 0.], [1., .01], [1., .02]]])
        points = np.array([[[.1, 0., 0.], [.101, 0., 0.], [2., 2., 2.]]])
        result = tensor_feature_observation(torch.tensor([1., 0.]),
            [features, features], [points, points], top_k=6)
        self.assertLess(np.linalg.norm(result.point - [.1, 0., 0.]), .002)

    def test_visual_lifecycle_requires_fresh_evidence(self):
        self.assertTrue(visually_tracked([1, 3], [.1, .1], .6))
        self.assertFalse(visually_tracked([1, 3], [.8, .1], .6))
        self.assertFalse(visually_tracked([3, 3], [.1, .1], .6))
        self.assertTrue(visually_tracked([3, 1], [.1, .1], .6))

    def test_seed_selection_preserves_existing_group_ids(self):
        objects = [SimpleNamespace(rigid_group_id=g,
            surface_medoid=SimpleNamespace(x=x, y=0., z=.03)) for g, x in [(2, .4), (8, .1)]]
        self.assertEqual(groups_in_seed_bounds(objects, [.27, -.25, -.08], [.75, .45, .2]), {2})


if __name__ == '__main__':
    unittest.main()
