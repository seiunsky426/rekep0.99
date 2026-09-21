#!/usr/bin/env python3

from types import SimpleNamespace
import unittest

from rekpiper_grasp.selection import (
    GraspBinding, horizon_requests_grasp, inference_request_key,
    select_keypoint_candidate)


def candidate(identifier, x, attempt=0, safe=True, stage=2):
    return SimpleNamespace(
        candidate_id=identifier, candidate_origin="native_anygrasp",
        perception_valid=safe, planning_safe=safe, ik_ok=safe,
        robot_collision_free=safe, esdf_clear=safe,
        contact_membership_ok=safe, finger_collision_free=safe,
        approach_clear=safe, rejection_reasons=[], rigid_group_id=4,
        object_uuid="object", session_id="session", program_sha256="program",
        snapshot_id="snapshot", map_generation_uuid="map", stage_index=stage,
        grasp_attempt=attempt, tcp_position=SimpleNamespace(x=x, y=0.0, z=0.0),
        network_score=0.7)


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.binding = GraspBinding(
            4, "object", "session", "program", "snapshot", "map", 2, 1)

    def test_closest_audited_candidate_is_selected(self):
        selected = select_keypoint_candidate(
            [candidate("far", 0.08, attempt=1),
             candidate("near", 0.02, attempt=1)],
            [0.0, 0.0, 0.0], self.binding)
        self.assertEqual(selected.candidate_id, "near")

    def test_non_grasp_or_stale_attempt_has_no_candidate(self):
        self.assertIsNone(select_keypoint_candidate(
            [candidate("old", 0.02, attempt=0)], [0.0, 0.0, 0.0],
            self.binding))

    def test_distance_precedes_audited_joint_motion_cost(self):
        near=candidate('near',.01,attempt=1); near.joint_motion_cost=4.
        smooth=candidate('smooth',.05,attempt=1); smooth.joint_motion_cost=.2
        selected=select_keypoint_candidate([near,smooth],[0.,0.,0.],self.binding)
        self.assertEqual(selected.candidate_id,'near')
        self.assertIsNone(select_keypoint_candidate(
            [candidate("wrong-stage", 0.02, attempt=1, stage=3)],
            [0.0, 0.0, 0.0], self.binding))

    def test_inference_key_changes_only_for_a_new_bound_attempt(self):
        first = inference_request_key(self.binding)
        duplicate = inference_request_key(GraspBinding(
            4, "object", "session", "program", "snapshot", "map", 2, 1))
        retry = inference_request_key(GraspBinding(
            4, "object", "session", "program", "snapshot", "map", 2, 2))
        self.assertEqual(first, duplicate)
        self.assertNotEqual(first, retry)

    def test_horizon_requests_inference_before_planning_without_motion_authority(self):
        horizon = SimpleNamespace(
            valid=True, authorized=False, status="grasp_target_pending", session_id="session",
            program_sha256="program", snapshot_id="snapshot",
            map_generation_uuid="map", stage_index=2)
        self.assertTrue(horizon_requests_grasp(horizon, self.binding))
        for status in ('shadow_validated', 'stage_event_pending', 'authorized_short_horizon'):
            horizon.status = status
            self.assertFalse(horizon_requests_grasp(horizon, self.binding))
        horizon.status = "grasp_target_pending"
        horizon.authorized = True
        self.assertFalse(horizon_requests_grasp(horizon, self.binding))
        horizon.authorized = False
        horizon.map_generation_uuid = "old-map"
        self.assertFalse(horizon_requests_grasp(horizon, self.binding))

    def test_unsafe_closest_candidate_is_never_selected(self):
        selected = select_keypoint_candidate(
            [candidate('unsafe', .001, attempt=1, safe=False),
             candidate('safe', .04, attempt=1)], [0., 0., 0.], self.binding)
        self.assertEqual(selected.candidate_id, 'safe')


if __name__ == "__main__":
    unittest.main()
