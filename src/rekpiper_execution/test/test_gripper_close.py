#!/usr/bin/env python3

import unittest

from rekpiper_execution.gripper_close import (
    ContactDetector,
    stepped_close_targets,
)


class GripperCloseTest(unittest.TestCase):
    def test_profile_uses_half_millimetre_steps_and_width_minus_three(self):
        targets = stepped_close_targets(0.057, 0.054)
        self.assertAlmostEqual(targets[-1], 0.051)
        self.assertGreaterEqual(len(targets), 12)
        for before, after in zip(targets[:-1], targets[1:]):
            self.assertLessEqual(before - after, 0.0005 + 1e-12)

    def test_stationary_obstructed_feedback_is_contact(self):
        detector = ContactDetector()
        decision = None
        for index in range(31):
            decision = detector.update(
                index * 0.02, 0.0541, 0.0525, 0.054,
                effort=1.0, empty_closed_opening_p99_m=0.003,
                empty_effort_p95=0.5, minimum_effort_margin=0.2,
                require_effort=True)
        self.assertTrue(decision.contact)

    def test_motion_or_missing_obstruction_is_not_contact(self):
        moving = ContactDetector()
        for index in range(31):
            decision = moving.update(
                index * 0.02, 0.057 - index * 0.0002,
                0.052, 0.054)
        self.assertFalse(decision.contact)

    def test_effort_only_and_mechanical_jam_are_not_contact(self):
        effort_only = ContactDetector()
        for index in range(31):
            decision = effort_only.update(
                index * 0.02, 0.002, 0.0, 0.054,
                effort=4.0, empty_closed_opening_p99_m=0.003,
                empty_effort_p95=0.5, minimum_effort_margin=0.2,
                require_effort=True)
        self.assertFalse(decision.contact)
        jam = ContactDetector()
        for index in range(31):
            decision = jam.update(
                index * 0.02, 0.020, 0.018, 0.054,
                effort=4.0, empty_closed_opening_p99_m=0.003,
                empty_effort_p95=0.5, minimum_effort_margin=0.2,
                require_effort=True)
        self.assertFalse(decision.contact)

    def test_encoder_visual_fallback_does_not_require_effort(self):
        detector = ContactDetector()
        for index in range(31):
            decision = detector.update(
                index * 0.02, 0.0541, 0.0525, 0.054,
                effort=0.0, empty_closed_opening_p99_m=0.003,
                empty_effort_p95=0.5, minimum_effort_margin=0.2,
                require_effort=False)
        self.assertTrue(decision.contact)
        unblocked = ContactDetector()
        for index in range(31):
            decision = unblocked.update(
                index * 0.02, 0.0524, 0.052, 0.054)
        self.assertFalse(decision.contact)

    def test_contact_window_tolerates_normal_feedback_jitter(self):
        for period in (0.017, 0.019, 0.021):
            detector = ContactDetector()
            decision = None
            index = 0
            while index * period < 0.65:
                decision = detector.update(
                    index * period, 0.0541, 0.0525, 0.054,
                    effort=1.0, empty_closed_opening_p99_m=0.003,
                    empty_effort_p95=0.5, minimum_effort_margin=0.2,
                    require_effort=True)
                index += 1
            self.assertTrue(decision.contact, "period={}".format(period))


if __name__ == "__main__":
    unittest.main()
