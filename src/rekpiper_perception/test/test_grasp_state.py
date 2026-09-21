#!/usr/bin/env python3

import unittest

from rekpiper_perception.grasp_state import (
    GraspEvidence, GraspState, GraspStateMachine, GripperBaseline)


def evidence(**updates):
    values = dict(
        gripper_opening_m=0.012,
        gripper_effort=1.0,
        vision_tracks_safe=True,
        object_in_gripper_region=True,
        tcp_motion_translation_m=0.0,
        tcp_motion_rotation_deg=0.0,
        object_tcp_span_m=0.003,
        object_world_span_m=0.003,
    )
    values.update(updates)
    return GraspEvidence(**values)


class GraspStateTest(unittest.TestCase):
    def setUp(self):
        self.machine = GraspStateMachine(GripperBaseline(0.003, 0.5))

    def test_attach_requires_gripper_and_verification_motion(self):
        self.machine.request_grasp()
        self.assertEqual(self.machine.update(evidence()), GraspState.CLOSING)
        self.assertEqual(self.machine.update(evidence()), GraspState.ATTACH_CANDIDATE)
        self.assertEqual(self.machine.update(evidence()), GraspState.ATTACH_VERIFYING)
        self.assertEqual(self.machine.update(evidence()), GraspState.ATTACH_VERIFYING)
        self.assertEqual(self.machine.update(evidence(tcp_motion_translation_m=0.021)),
                         GraspState.ATTACH_VERIFYING)
        self.assertEqual(self.machine.reason,
                         "rigid_tcp_attachment_evidence_ready")
        self.assertEqual(self.machine.confirm_attachment(),
                         GraspState.ATTACHED)

    def test_empty_close_does_not_attach(self):
        self.machine.request_grasp()
        self.machine.update(evidence())
        self.assertEqual(self.machine.update(evidence(
            gripper_opening_m=0.002, gripper_effort=0.2,
            object_in_gripper_region=False)), GraspState.CLOSING)

    def test_large_verification_motion_is_ambiguous(self):
        self.machine.request_grasp()
        self.machine.update(evidence())
        self.machine.update(evidence())
        self.machine.update(evidence())
        self.assertEqual(self.machine.update(evidence(
            tcp_motion_translation_m=0.08)), GraspState.AMBIGUOUS)

    def test_release_requires_world_stability_and_separation(self):
        self.machine.state = GraspState.ATTACHED
        self.machine.request_release()
        self.machine.update(evidence())
        self.machine.update(evidence(gripper_opening_m=0.02))
        stable = evidence(
            gripper_opening_m=0.02, object_tcp_span_m=0.02,
            object_world_span_m=0.003)
        for _ in range(9):
            self.assertEqual(
                self.machine.update(stable), GraspState.RELEASE_VERIFYING)
        self.assertEqual(
            self.machine.update(stable), GraspState.RELEASE_VERIFYING)
        self.assertEqual(self.machine.confirm_release(),
                         GraspState.FREE_TRACKED)
        self.assertEqual(self.machine.reason, "release_verified_stable")

    def test_release_stability_counter_resets_on_motion(self):
        machine = GraspStateMachine(
            GripperBaseline(0.003, 0.5),
            minimum_release_stable_updates=2)
        machine.state = GraspState.ATTACHED
        machine.request_release()
        machine.update(evidence())
        machine.update(evidence(gripper_opening_m=0.02))
        stable = evidence(
            gripper_opening_m=0.02, object_tcp_span_m=0.02,
            object_world_span_m=0.003)
        self.assertEqual(machine.update(stable), GraspState.RELEASE_VERIFYING)
        self.assertEqual(machine.update(evidence(
            gripper_opening_m=0.02, object_tcp_span_m=0.02,
            object_world_span_m=0.020)), GraspState.RELEASE_VERIFYING)
        self.assertEqual(machine.update(stable), GraspState.RELEASE_VERIFYING)
        self.assertEqual(machine.update(stable), GraspState.RELEASE_VERIFYING)
        self.assertEqual(machine.confirm_release(), GraspState.FREE_TRACKED)

    def test_encoder_visual_mode_is_explicit_and_still_requires_motion(self):
        machine = GraspStateMachine(
            GripperBaseline(0.003, 0.5),
            contact_evidence_mode="encoder_visual")
        machine.request_grasp()
        machine.update(evidence(gripper_effort=0.0))
        self.assertEqual(machine.update(evidence(gripper_effort=0.0)),
                         GraspState.ATTACH_CANDIDATE)
        machine.update(evidence(gripper_effort=0.0))
        self.assertEqual(machine.update(evidence(gripper_effort=0.0)),
                         GraspState.ATTACH_VERIFYING)
        with self.assertRaisesRegex(ValueError, "not ready"):
            machine.confirm_attachment()

    def test_encoder_visual_with_occluded_vision_is_ambiguous(self):
        machine = GraspStateMachine(
            GripperBaseline(0.003, 0.5),
            contact_evidence_mode="encoder_visual")
        machine.request_grasp()
        self.assertEqual(machine.update(evidence(
            gripper_effort=0.0, vision_tracks_safe=False)),
            GraspState.AMBIGUOUS)

    def test_cancel_requires_verified_open_and_free_object(self):
        self.machine.request_grasp()
        with self.assertRaisesRegex(ValueError, "open"):
            self.machine.cancel_grasp(False, True)
        self.assertEqual(self.machine.cancel_grasp(True, False),
                         GraspState.AMBIGUOUS)

    def test_effort_alone_cannot_create_contact_candidate(self):
        self.machine.request_grasp()
        self.machine.update(evidence())
        self.assertEqual(self.machine.update(evidence(
            gripper_opening_m=0.002, gripper_effort=10.0,
            object_in_gripper_region=False)), GraspState.CLOSING)

    def test_mechanical_jam_cannot_be_confirmed_without_object_motion(self):
        self.machine.request_grasp()
        self.machine.update(evidence())
        self.machine.update(evidence())
        self.machine.update(evidence())
        # The TCP moved, but a jammed gripper did not carry the object, so its
        # position in the TCP frame changed by the full probe displacement.
        self.assertEqual(self.machine.update(evidence(
            tcp_motion_translation_m=0.021,
            object_tcp_span_m=0.021)), GraspState.ATTACH_VERIFYING)
        with self.assertRaisesRegex(ValueError, "not ready"):
            self.machine.confirm_attachment()

    def test_visual_slip_after_attachment_becomes_ambiguous(self):
        self.machine.state = GraspState.ATTACHED
        self.assertEqual(self.machine.update(evidence(
            object_tcp_span_m=0.020)), GraspState.AMBIGUOUS)


if __name__ == "__main__":
    unittest.main()
