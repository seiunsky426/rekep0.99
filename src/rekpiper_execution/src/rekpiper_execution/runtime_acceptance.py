"""Validation of measured 10 Hz mapping / 20 Hz command evidence."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml


class RuntimeAcceptanceError(ValueError):
    pass


def validate_evidence(evidence, expected_profile=None):
    failures = []
    schema = int(evidence.get("schema_version", 1))
    if expected_profile is not None and evidence.get("profile") != expected_profile:
        failures.append("profile_mismatch")
    if schema >= 2:
        if evidence.get("profile") not in ("software", "hardware_hold"):
            failures.append("profile_invalid")
        if not bool(evidence.get("continuous", False)):
            failures.append("stream_not_continuous")
        digest = str(evidence.get("raw_event_log_sha256", ""))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            failures.append("raw_event_log_hash_invalid")
    if float(evidence.get("duration_s", 0.0)) < 600.0:
        failures.append("duration_lt_600s")
    sdf = evidence.get("sdf", {})
    trajectory = evidence.get("trajectory", {})
    planning = evidence.get("planning", {})
    if float(sdf.get("actual_rate_hz", 0.0)) < 9.5:
        failures.append("sdf_rate_lt_9_5hz")
    if float(sdf.get("latency_p99_s", float("inf"))) > 0.100:
        failures.append("sdf_latency_p99_gt_100ms")
    if float(trajectory.get("actual_rate_hz", 0.0)) < 19.0:
        failures.append("trajectory_rate_lt_19hz")
    if float(trajectory.get("period_p99_s", float("inf"))) > 0.050:
        failures.append("trajectory_period_p99_gt_50ms")
    if float(planning.get("warm_latency_p99_s", float("inf"))) > 0.100:
        failures.append("planning_warm_p99_gt_100ms")
    if schema >= 2:
        if int(sdf.get("diagnostic_samples", 0)) < 5700:
            failures.append("sdf_samples_lt_5700")
        if float(sdf.get("maximum_gap_s", float("inf"))) > 0.200:
            failures.append("sdf_gap_gt_200ms")
        if int(trajectory.get("command_samples", 0)) < 11400:
            failures.append("trajectory_samples_lt_11400")
        if float(trajectory.get("maximum_gap_s", float("inf"))) > 0.100:
            failures.append("trajectory_gap_gt_100ms")
        if int(planning.get("warm_samples", 0)) < 5700:
            failures.append("planning_samples_lt_5700")
        if float(planning.get("maximum_gap_s", float("inf"))) > 0.200:
            failures.append("planning_gap_gt_200ms")
        minimum_raw_events = (
            int(sdf.get("diagnostic_samples", 0))
            + int(trajectory.get("command_samples", 0))
            + int(planning.get("warm_samples", 0)))
        if int(evidence.get("raw_event_count", 0)) < minimum_raw_events:
            failures.append("raw_event_coverage_inconsistent")
        if evidence.get("profile") == "software":
            contract = evidence.get("software_contract", {})
            replay_hash = str(contract.get("replay_dataset_sha256", "")).lower()
            program_hash = str(contract.get("program_sha256", "")).lower()
            if (not bool(contract.get("piper_driver_absent", False))
                    or not bool(contract.get("can_disabled_verified", False))
                    or len(replay_hash) != 64
                    or any(c not in "0123456789abcdef" for c in replay_hash)
                    or len(program_hash) != 64
                    or any(c not in "0123456789abcdef" for c in program_hash)):
                failures.append("software_no_can_contract_missing")
        if evidence.get("profile") == "hardware_hold":
            hold = evidence.get("hardware_hold", {})
            bootstrap_hash = str(hold.get(
                "hardware_acceptance_bundle_sha256", "")).lower()
            if (int(hold.get("feedback_samples", 0)) < 11400
                    or int(hold.get("command_samples", 0)) < 11400
                    or not bool(hold.get("gripper_motion_forbidden", False))
                    or not bool(hold.get("operator_present", False))
                    or not bool(hold.get("physical_estop_confirmed", False))
                    or not str(hold.get("robot_id", "")).strip()
                    or not str(hold.get("piper_identity", "")).strip()
                    or len(bootstrap_hash) != 64
                    or any(c not in "0123456789abcdef"
                           for c in bootstrap_hash)
                    or float(hold.get("maximum_error_rad", float("inf")))
                    > float(hold.get("tolerance_rad", 0.0))):
                failures.append("hardware_hold_contract_failed")
    if failures:
        raise RuntimeAcceptanceError(
            "runtime evidence failed: " + ",".join(failures))
    return True


def approve_evidence(evidence, operator):
    validate_evidence(evidence)
    name = str(operator).strip()
    if not name:
        raise RuntimeAcceptanceError("operator name is required")
    encoded = json.dumps(
        evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": 1,
        "approved": True,
        "evidence_sha256": hashlib.sha256(encoded).hexdigest(),
        "evidence": evidence,
        "approval": {
            "operator": name,
            "approved_utc": datetime.now(timezone.utc).isoformat(),
        },
    }


def load_runtime_acceptance(path):
    try:
        payload = yaml.safe_load(
            Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeAcceptanceError(
            "runtime acceptance is unavailable") from exc
    if (not isinstance(payload, dict) or payload.get("schema_version") != 1
            or not bool(payload.get("approved", False))):
        raise RuntimeAcceptanceError("runtime performance is not approved")
    validate_evidence(payload.get("evidence", {}))
    encoded = json.dumps(
        payload.get("evidence", {}), sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    if payload.get("evidence_sha256") != hashlib.sha256(encoded).hexdigest():
        raise RuntimeAcceptanceError("runtime evidence hash mismatch")
    if not str(payload.get("approval", {}).get("operator", "")).strip():
        raise RuntimeAcceptanceError("runtime approval operator is missing")
    return payload
