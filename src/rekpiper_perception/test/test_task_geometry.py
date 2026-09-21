#!/usr/bin/env python3

import unittest

import numpy as np

from rekpiper_perception.task_geometry import (
    capture_follows_task_trigger, clean_organized_mask, estimate_instance_geometry,
    organized_workspace_mask, workspace_mask_area_ratio)


class TaskGeometryTest(unittest.TestCase):
    def test_new_task_rejects_queued_pretrigger_rgb_or_depth(self):
        trigger = 1789388297886640800
        self.assertFalse(capture_follows_task_trigger(trigger-52000000, trigger-52000000, trigger))
        self.assertFalse(capture_follows_task_trigger(trigger+1, trigger-1, trigger))
        self.assertFalse(capture_follows_task_trigger(trigger-1, trigger+1, trigger))
        self.assertTrue(capture_follows_task_trigger(trigger, trigger+1, trigger))
        self.assertFalse(capture_follows_task_trigger(0, 0, 0))

    def test_workspace_and_instance_geometry_use_metric_points(self):
        rows, cols = np.mgrid[:30, :40]
        points = np.dstack((cols * 0.005, rows * 0.005,
                            np.full_like(rows, 0.4, dtype=float)))
        workspace = organized_workspace_mask(
            points, [0.02, 0.02, 0.3], [0.18, 0.13, 0.5])
        mask = np.zeros((30, 40), dtype=bool)
        mask[8:22, 10:30] = True
        cleaned = clean_organized_mask(points, mask, erosion_px=1)
        estimate = estimate_instance_geometry(points, cleaned)
        self.assertTrue(np.all(np.isfinite(estimate.surface_medoid)))
        self.assertGreater(estimate.geometry_valid_pixels, 100)
        self.assertGreater(workspace_mask_area_ratio(mask, workspace), 0.0)


if __name__ == "__main__":
    unittest.main()
