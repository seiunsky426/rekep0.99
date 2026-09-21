#!/usr/bin/env python3

import unittest
import numpy as np

from rekpiper_calibration.camera_validation import CameraValidationMetrics


class CameraValidationMetricsTest(unittest.TestCase):
    def test_valid_camera_sequence_passes(self):
        metrics = CameraValidationMetrics()
        depth = np.full((4, 6), 500, dtype=np.uint16)
        k = [600, 0, 3, 0, 601, 2, 0, 0, 1]
        for index in range(30):
            stamp = 10.0 + index / 30.0
            metrics.add_frame(
                stamp, stamp + 0.002, (4, 6), depth, k,
                "rs1_color_optical_frame")
        report = metrics.summary()
        self.assertTrue(report["passed"])
        self.assertEqual(report["frames"], 30)
        self.assertAlmostEqual(report["valid_depth_fraction"]["mean"], 1.0)
        self.assertEqual(report["frame_id"], "rs1_color_optical_frame")

    def test_invalid_depth_and_skew_fail(self):
        metrics = CameraValidationMetrics()
        depth = np.zeros((3, 3), dtype=np.uint16)
        metrics.add_frame(
            1.0, 1.2, (3, 3), depth,
            [600, 0, 1, 0, 600, 1, 0, 0, 1],
            "rs3_color_optical_frame")
        report = metrics.summary()
        self.assertFalse(report["passed"])
        self.assertIn("timestamp_skew", report["failure_reasons"])
        self.assertIn("insufficient_valid_depth", report["failure_reasons"])

    def test_changed_frame_id_fails(self):
        metrics = CameraValidationMetrics()
        depth = np.full((3, 3), 500, dtype=np.uint16)
        k = [600, 0, 1, 0, 600, 1, 0, 0, 1]
        metrics.add_frame(1.0, 1.0, (3, 3), depth, k, "rs1_color_optical_frame")
        metrics.add_frame(2.0, 2.0, (3, 3), depth, k, "wrong_frame")
        self.assertIn(
            "frame_id_invalid_or_changed", metrics.summary()["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
