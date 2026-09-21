"""Evidence-gated grasp lifecycle that never commands the robot."""

from dataclasses import dataclass
from enum import IntEnum


class GraspState(IntEnum):
    FREE_TRACKED = 1
    GRASP_REQUESTED = 2
    CLOSING = 3
    ATTACH_CANDIDATE = 4
    ATTACH_VERIFYING = 5
    ATTACHED = 6
    RELEASE_REQUESTED = 7
    OPENING = 8
    RELEASE_VERIFYING = 9
    LOST = 10
    AMBIGUOUS = 11
    FAULT = 12


@dataclass(frozen=True)
class GraspEvidence:
    gripper_opening_m: float
    gripper_effort: float
    vision_tracks_safe: bool
    object_in_gripper_region: bool
    tcp_motion_translation_m: float = 0.0
    tcp_motion_rotation_deg: float = 0.0
    object_tcp_span_m: float = float("inf")
    object_world_span_m: float = float("inf")
    gripper_stable_duration_s: float = 0.0
    gripper_opening_span_m: float = float("inf")
    gripper_closure_from_start_m: float = 0.0


@dataclass(frozen=True)
class GripperBaseline:
    empty_closed_opening_p99_m: float
    empty_effort_p95: float
    minimum_opening_margin_m: float = 0.002
    minimum_effort_margin: float = 0.2


class GraspStateMachine:
    """Transition only on explicit intent and corroborating physical evidence."""

    def __init__(self, baseline, tcp_span_limit_m=0.008,
                 minimum_tcp_translation_m=0.020,
                 minimum_tcp_rotation_deg=8.0,
                 maximum_verification_translation_m=0.060,
                 maximum_verification_rotation_deg=20.0,
                 released_world_span_m=0.006,
                 minimum_release_stable_updates=10,
                 contact_evidence_mode="encoder_effort_vision"):
        self.baseline = baseline
        self.tcp_span_limit_m = float(tcp_span_limit_m)
        self.minimum_tcp_translation_m = float(minimum_tcp_translation_m)
        self.minimum_tcp_rotation_deg = float(minimum_tcp_rotation_deg)
        self.maximum_verification_translation_m = float(
            maximum_verification_translation_m)
        self.maximum_verification_rotation_deg = float(
            maximum_verification_rotation_deg)
        self.released_world_span_m = float(released_world_span_m)
        self.minimum_release_stable_updates = int(
            minimum_release_stable_updates)
        self.contact_evidence_mode = str(contact_evidence_mode)
        if self.contact_evidence_mode not in (
                "encoder_effort_vision", "encoder_visual"):
            raise ValueError("invalid contact_evidence_mode")
        if self.minimum_release_stable_updates <= 0:
            raise ValueError("minimum_release_stable_updates must be positive")
        self._release_stable_updates = 0
        self._attachment_verified = False
        self._release_verified = False
        self.state = GraspState.FREE_TRACKED
        self.reason = "initialized"

    def _set(self, state, reason):
        self.state = GraspState(state)
        self.reason = str(reason)
        return self.state

    def request_grasp(self):
        if self.state != GraspState.FREE_TRACKED:
            raise ValueError("grasp can only begin from FREE_TRACKED")
        self._attachment_verified = False
        return self._set(GraspState.GRASP_REQUESTED, "grasp_requested")

    def request_release(self):
        if self.state != GraspState.ATTACHED:
            raise ValueError("release can only begin from ATTACHED")
        self._release_stable_updates = 0
        self._release_verified = False
        return self._set(GraspState.RELEASE_REQUESTED, "release_requested")

    def confirm_attachment(self):
        if (self.state != GraspState.ATTACH_VERIFYING
                or not self._attachment_verified):
            raise ValueError("rigid attachment evidence is not ready")
        return self._set(GraspState.ATTACHED,
                         "rigid_tcp_attachment_verified")

    def confirm_release(self):
        if (self.state != GraspState.RELEASE_VERIFYING
                or not self._release_verified):
            raise ValueError("release evidence is not ready")
        return self._set(GraspState.FREE_TRACKED,
                         "release_verified_stable")

    def cancel_grasp(self, gripper_open, vision_confirms_free):
        allowed = (GraspState.GRASP_REQUESTED, GraspState.CLOSING,
                   GraspState.ATTACH_CANDIDATE,
                   GraspState.ATTACH_VERIFYING)
        if self.state not in allowed:
            raise ValueError("grasp cancellation is not permitted from current state")
        if not gripper_open:
            raise ValueError("gripper must be verified open before cancellation")
        if not vision_confirms_free:
            return self._set(GraspState.AMBIGUOUS,
                             "cancel_grasp_visual_state_ambiguous")
        self._attachment_verified = False
        return self._set(GraspState.FREE_TRACKED,
                         "grasp_cancelled_object_verified_free")

    def fault(self, reason):
        return self._set(GraspState.FAULT, reason)

    def lost(self, reason="vision_lost"):
        return self._set(GraspState.LOST, reason)

    def resolve(self, state, note):
        allowed = (GraspState.FREE_TRACKED, GraspState.ATTACHED, GraspState.FAULT)
        if GraspState(state) not in allowed:
            raise ValueError("operator resolution target is not permitted")
        return self._set(state, "operator:" + str(note))

    def update(self, evidence):
        if self.state in (GraspState.FAULT, GraspState.LOST, GraspState.AMBIGUOUS):
            return self.state
        if not evidence.vision_tracks_safe and self.state not in (
                GraspState.ATTACHED, GraspState.RELEASE_REQUESTED,
                GraspState.OPENING, GraspState.RELEASE_VERIFYING):
            if (self.contact_evidence_mode == "encoder_visual"
                    and self.state in (
                        GraspState.GRASP_REQUESTED, GraspState.CLOSING,
                        GraspState.ATTACH_CANDIDATE,
                        GraspState.ATTACH_VERIFYING)):
                return self._set(
                    GraspState.AMBIGUOUS,
                    "encoder_visual_mode_requires_visible_attachment")
            return self._set(GraspState.LOST, "object_tracking_not_safe")

        if self.state == GraspState.GRASP_REQUESTED:
            return self._set(GraspState.CLOSING, "observing_gripper_close")
        if self.state == GraspState.CLOSING:
            opening_ok = evidence.gripper_opening_m >= (
                self.baseline.empty_closed_opening_p99_m
                + self.baseline.minimum_opening_margin_m)
            effort_ok = evidence.gripper_effort >= (
                self.baseline.empty_effort_p95 + self.baseline.minimum_effort_margin)
            if (opening_ok
                    and (effort_ok
                         or self.contact_evidence_mode == "encoder_visual")
                    and evidence.object_in_gripper_region):
                return self._set(GraspState.ATTACH_CANDIDATE,
                                 "gripper_and_vision_candidate")
            return self.state
        if self.state == GraspState.ATTACH_CANDIDATE:
            return self._set(GraspState.ATTACH_VERIFYING, "awaiting_tcp_motion")
        if self.state == GraspState.ATTACH_VERIFYING:
            if (evidence.tcp_motion_translation_m > self.maximum_verification_translation_m
                    or evidence.tcp_motion_rotation_deg > self.maximum_verification_rotation_deg):
                return self._set(GraspState.AMBIGUOUS,
                                 "attachment_verification_motion_exceeded")
            moved = (evidence.tcp_motion_translation_m >= self.minimum_tcp_translation_m
                     or evidence.tcp_motion_rotation_deg >= self.minimum_tcp_rotation_deg)
            if moved and evidence.object_tcp_span_m <= self.tcp_span_limit_m:
                self._attachment_verified = True
                self.reason = "rigid_tcp_attachment_evidence_ready"
            return self.state
        if self.state == GraspState.ATTACHED:
            if evidence.vision_tracks_safe and evidence.object_tcp_span_m > 0.015:
                return self._set(GraspState.AMBIGUOUS, "possible_object_slip")
            return self.state
        if self.state == GraspState.RELEASE_REQUESTED:
            return self._set(GraspState.OPENING, "observing_gripper_open")
        if self.state == GraspState.OPENING:
            if evidence.gripper_opening_m > self.baseline.empty_closed_opening_p99_m + 0.010:
                return self._set(GraspState.RELEASE_VERIFYING, "gripper_open")
            return self.state
        if self.state == GraspState.RELEASE_VERIFYING:
            if (evidence.vision_tracks_safe
                    and evidence.object_world_span_m <= self.released_world_span_m
                    and evidence.object_tcp_span_m > self.tcp_span_limit_m):
                self._release_stable_updates += 1
                if (self._release_stable_updates
                        >= self.minimum_release_stable_updates):
                    self._release_verified = True
                    self.reason = "release_evidence_ready"
            else:
                self._release_stable_updates = 0
        return self.state
