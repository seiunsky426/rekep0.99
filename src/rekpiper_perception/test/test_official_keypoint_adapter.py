#!/usr/bin/env python3
"""Regression tests for the adapter around the pristine official proposer."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from rekpiper_perception.official_keypoint_adapter import (
    DINOV2_NUM_REGISTER_TOKENS,
    OfficialKeypointProposerAdapter,
    validate_reg4_checkpoint,
)


class _FakeOfficialProposer:
    def __init__(self, config):
        self.config = config
        self.model = torch.hub.load("ignored", "dinov2_vits14")

    @staticmethod
    def _project_keypoints_to_img(
            rgb, candidate_pixels, candidate_groups, masks, features):
        del candidate_groups, masks, features
        projected = rgb.copy()
        for index, pixel in enumerate(candidate_pixels, start=1):
            projected[pixel[0], pixel[1], 0] = index
        return projected

    def get_keypoints(self, rgb, points, masks):
        if masks.ndim != 2:
            raise AssertionError("adapter did not convert NxHxW to labels")
        pixels = np.asarray([[1, 1], [4, 5], [20, 21]], dtype=np.int32)
        groups = np.asarray([0, 1, 2], dtype=np.int32)
        annotated = self._project_keypoints_to_img(
            rgb, pixels, groups, masks, np.zeros((3, 3)))
        return points[pixels[:, 0], pixels[:, 1]], annotated


class OfficialKeypointAdapterTest(unittest.TestCase):
    def test_reg4_checkpoint_contract_rejects_plain_and_wrong_shapes(self):
        with self.assertRaisesRegex(RuntimeError, "not the pinned reg4"):
            validate_reg4_checkpoint({"cls_token": torch.zeros(1, 1, 384)})
        with self.assertRaisesRegex(RuntimeError, "not the pinned reg4"):
            validate_reg4_checkpoint({"register_tokens": torch.zeros(1, 2, 384)})
        validate_reg4_checkpoint({"register_tokens": torch.zeros(1, 4, 384)})

    def test_local_model_injection_nhw_masks_and_projection_capture(self):
        fake_model = torch.nn.Identity()
        fake_model.num_register_tokens = DINOV2_NUM_REGISTER_TOKENS
        with patch(
                "rekpiper_perception.official_keypoint_adapter.load_local_dinov2",
                return_value=fake_model), patch(
                "rekpiper_perception.official_keypoint_adapter._load_official_class",
                return_value=_FakeOfficialProposer):
            proposer = OfficialKeypointProposerAdapter(
                "/pristine/official", "/pinned/dinov2", "/pinned/model.pth",
                {"device": "cpu"})

        rgb = np.zeros((28, 28, 3), dtype=np.uint8)
        points = np.zeros((28, 28, 3), dtype=np.float32)
        rows, cols = np.indices((28, 28))
        points[..., 0] = cols * 0.005
        points[..., 1] = rows * 0.005
        points[..., 2] = 0.4
        masks = np.zeros((2, 28, 28), dtype=bool)
        masks[0, 2:12, 2:12] = True
        masks[1, 16:26, 16:26] = True

        result = proposer.propose(rgb, points, masks)
        self.assertIs(proposer._proposer.model, fake_model)
        self.assertEqual(proposer._proposer.model.num_register_tokens, 4)
        self.assertEqual(result.points.shape, (2, 3))
        np.testing.assert_array_equal(result.pixels_rc, [[4, 5], [20, 21]])
        np.testing.assert_array_equal(result.rigid_group_ids, [0, 1])
        self.assertEqual(result.annotated_rgb.shape, rgb.shape)
        self.assertEqual(result.annotated_rgb[1, 1, 0], 0)
        self.assertEqual(result.annotated_rgb[4, 5, 0], 1)
        self.assertEqual(result.annotated_rgb[20, 21, 0], 2)

    def test_overlapping_binary_instances_fail_closed(self):
        with patch(
                "rekpiper_perception.official_keypoint_adapter.load_local_dinov2",
                return_value=torch.nn.Identity()), patch(
                "rekpiper_perception.official_keypoint_adapter._load_official_class",
                return_value=_FakeOfficialProposer):
            proposer = OfficialKeypointProposerAdapter(
                "/official", "/dino", "/weights", {"device": "cpu"})
        masks = np.zeros((2, 28, 28), dtype=bool)
        masks[0, 2:12, 2:12] = True
        masks[1, 8:18, 8:18] = True
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            proposer.propose(
                np.zeros((28, 28, 3), dtype=np.uint8),
                np.zeros((28, 28, 3), dtype=np.float32), masks)

    def test_invalid_shapes_fail_closed(self):
        with patch(
                "rekpiper_perception.official_keypoint_adapter.load_local_dinov2",
                return_value=torch.nn.Identity()), patch(
                "rekpiper_perception.official_keypoint_adapter._load_official_class",
                return_value=_FakeOfficialProposer):
            proposer = OfficialKeypointProposerAdapter(
                "/official", "/dino", "/weights", {"device": "cpu"})
        with self.assertRaisesRegex(ValueError, "NxHxW"):
            proposer.propose(
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.zeros((8, 8, 3), dtype=np.float32),
                np.zeros((2, 7, 8), dtype=bool))


if __name__ == "__main__":
    unittest.main()
