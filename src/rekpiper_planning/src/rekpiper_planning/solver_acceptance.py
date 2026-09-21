"""Offline calibration and fail-closed acceptance for paper-real weights."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from .paper_real_solver import PaperRealWeights
from .upstream import EXPECTED_OFFICIAL_COMMIT


REQUIRED_SCENARIOS = frozenset({
    "static", "perception_noise", "moving_target", "obstacle_change",
    "near_table", "unreachable", "map_uuid_change",
})
REQUIRED_INPUT_HASHES = frozenset({
    "source_tree_sha256", "dataset_sha256", "workspace_sha256",
    "robot_description_sha256", "solver_config_sha256",
})


class SolverAcceptanceError(ValueError):
    pass


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def paper_real_source_sha256():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(root.glob("*.py")):
        digest.update(source.name.encode("utf-8") + b"\0")
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _validate_input_hashes(inputs):
    if not isinstance(inputs, dict):
        raise SolverAcceptanceError("paper-real input hashes are missing")
    missing = sorted(REQUIRED_INPUT_HASHES - set(inputs))
    invalid = sorted(name for name in REQUIRED_INPUT_HASHES.intersection(inputs)
                     if len(str(inputs[name])) != 64
                     or any(character not in "0123456789abcdef"
                            for character in str(inputs[name]).lower()))
    if missing or invalid:
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if invalid:
            detail.append("invalid=" + ",".join(invalid))
        raise SolverAcceptanceError(
            "paper-real input hashes are incomplete: " + ";".join(detail))


def load_workspace_table_height(path, require_accepted=False):
    source = Path(path).expanduser().resolve()
    try:
        values = yaml.safe_load(source.read_text(encoding="utf-8"))
        height = float(values["table_height_m"])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise SolverAcceptanceError(
            "accepted workspace table_height_m is unavailable") from exc
    if not np.isfinite(height):
        raise SolverAcceptanceError("workspace table height is not finite")
    if require_accepted and (
            values.get("status") != "ACCEPTED"
            or not bool(values.get("precision_operation_allowed", False))):
        raise SolverAcceptanceError(
            "workspace is not accepted for precision operation")
    return height, sha256_file(source)


def aggregate_trials(trials):
    values = list(trials)
    if not values:
        raise SolverAcceptanceError("candidate contains no replay trials")
    scenarios = {str(item.get("scenario", "")) for item in values}
    missing = sorted(REQUIRED_SCENARIOS - scenarios)
    if missing:
        raise SolverAcceptanceError(
            "replay scenarios missing: " + ",".join(missing))
    warm = [float(item["planning_latency_s"]) for item in values
            if bool(item.get("warm", False))]
    static_jitter = [float(item["path_jitter_m"]) for item in values
                     if item.get("scenario") == "static"]
    if not warm or not static_jitter:
        raise SolverAcceptanceError(
            "warm latency and static jitter evidence are required")
    return {
        "trial_count": len(values),
        "success_rate": float(np.mean([bool(item["success"])
                                       for item in values])),
        "collision_failures": int(sum(int(item["collision_failures"])
                                      for item in values)),
        "unknown_space_passes": int(sum(int(item["unknown_space_passes"])
                                        for item in values)),
        "maximum_constraint_violation": float(max(
            float(item["maximum_constraint_violation"]) for item in values)),
        "warm_latency_p99_s": float(np.percentile(warm, 99)),
        "static_path_jitter_m": float(np.mean(static_jitter)),
        "map_uuid_first_solve_from_scratch": bool(all(
            bool(item.get("map_uuid_first_solve_from_scratch", False))
            for item in values if item.get("scenario") == "map_uuid_change")),
    }


def candidate_is_eligible(metrics, baseline, constraint_tolerance):
    reasons = []
    if metrics["collision_failures"] != 0:
        reasons.append("collision_failure")
    if metrics["unknown_space_passes"] != 0:
        reasons.append("unknown_space_passed")
    if metrics["maximum_constraint_violation"] > float(constraint_tolerance):
        reasons.append("constraint_violation")
    if metrics["warm_latency_p99_s"] > 0.100:
        reasons.append("warm_latency_p99")
    if metrics["success_rate"] < baseline["success_rate"]:
        reasons.append("success_rate_regression")
    if metrics["static_path_jitter_m"] >= baseline["static_path_jitter_m"]:
        reasons.append("jitter_not_improved")
    if not metrics["map_uuid_first_solve_from_scratch"]:
        reasons.append("map_uuid_warm_start_retained")
    return not reasons, reasons


def calibrate_replay_report(report):
    if report.get("schema_version") != 1:
        raise SolverAcceptanceError("replay report schema_version must be 1")
    if report.get("official_commit") != EXPECTED_OFFICIAL_COMMIT:
        raise SolverAcceptanceError("replay report uses the wrong ReKep commit")
    baseline = aggregate_trials(report.get("baseline_trials", []))
    tolerance = float(report.get("constraint_tolerance", 0.0001))
    eligible = []
    rejected = []
    for index, candidate in enumerate(report.get("candidates", [])):
        weights = PaperRealWeights.from_mapping(candidate.get("weights", {}))
        metrics = aggregate_trials(candidate.get("trials", []))
        ok, reasons = candidate_is_eligible(metrics, baseline, tolerance)
        value = {
            "index": index,
            "weights": {
                "subgoal_consistency": weights.subgoal_consistency,
                "path_consistency": weights.path_consistency,
                "table_clearance": weights.table_clearance,
            },
            "metrics": metrics,
            "reasons": reasons,
        }
        (eligible if ok else rejected).append(value)
    if not eligible:
        raise SolverAcceptanceError("no paper-real weight candidate passed")
    eligible.sort(key=lambda item: (
        item["metrics"]["static_path_jitter_m"],
        item["metrics"]["warm_latency_p99_s"],
        -item["metrics"]["success_rate"], item["index"]))
    selected = eligible[0]
    return {
        "schema_version": 1,
        "profile": "paper_real",
        "approved": False,
        "official_commit": EXPECTED_OFFICIAL_COMMIT,
        "inputs": dict(report.get("inputs", {})),
        "input_paths": dict(report.get("input_paths", {})),
        "replay_report_sha256": sha256_json(report),
        "constraint_tolerance": tolerance,
        "baseline_metrics": baseline,
        "weights": selected["weights"],
        "metrics": selected["metrics"],
        "rejected_candidates": rejected,
    }


def validate_acceptance(payload, expected_inputs=None, require_approved=True):
    if payload.get("schema_version") != 1 or payload.get("profile") != "paper_real":
        raise SolverAcceptanceError("paper-real acceptance schema/profile mismatch")
    if payload.get("official_commit") != EXPECTED_OFFICIAL_COMMIT:
        raise SolverAcceptanceError("paper-real acceptance commit mismatch")
    weights = PaperRealWeights.from_mapping(payload.get("weights", {}))
    if require_approved and not bool(payload.get("approved", False)):
        raise SolverAcceptanceError("paper-real weights are not operator-approved")
    if expected_inputs:
        actual = payload.get("inputs", {})
        mismatched = [name for name, value in expected_inputs.items()
                      if actual.get(name) != value]
        if mismatched:
            raise SolverAcceptanceError(
                "paper-real input hash mismatch: " + ",".join(mismatched))
    if bool(payload.get("approved", False)):
        approval = payload.get("approval", {})
        candidate_digest = str(approval.get("candidate_sha256", ""))
        unsigned = copy_mapping(payload)
        unsigned["approved"] = False
        unsigned.pop("approval", None)
        if candidate_digest != sha256_json(unsigned):
            raise SolverAcceptanceError(
                "approved paper-real candidate hash mismatch")
        _validate_input_hashes(payload.get("inputs"))
        dataset_path = payload.get("input_paths", {}).get("dataset")
        if not dataset_path:
            raise SolverAcceptanceError(
                "approved replay dataset path is missing")
        try:
            actual_dataset_hash = sha256_file(dataset_path)
        except OSError as exc:
            raise SolverAcceptanceError(
                "approved replay dataset is unavailable") from exc
        if actual_dataset_hash != payload["inputs"]["dataset_sha256"]:
            raise SolverAcceptanceError(
                "approved replay dataset hash changed")
        metrics = payload.get("metrics", {})
        baseline = payload.get("baseline_metrics", {})
        ok, reasons = candidate_is_eligible(
            metrics, baseline, payload.get("constraint_tolerance", 0.0001))
        if not ok:
            raise SolverAcceptanceError(
                "approved paper-real metrics are invalid: " + ",".join(reasons))
        if not str(approval.get("operator", "")).strip():
            raise SolverAcceptanceError("paper-real approval operator is missing")
    return weights


def load_acceptance(path, expected_inputs=None, require_approved=True):
    try:
        payload = yaml.safe_load(
            Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SolverAcceptanceError("cannot load paper-real acceptance") from exc
    if not isinstance(payload, dict):
        raise SolverAcceptanceError("paper-real acceptance must be a mapping")
    weights = validate_acceptance(
        payload, expected_inputs=expected_inputs,
        require_approved=require_approved)
    return payload, weights


def approve_candidate(candidate, operator):
    payload = copy_mapping(candidate)
    validate_acceptance(payload, require_approved=False)
    name = str(operator).strip()
    if not name:
        raise SolverAcceptanceError("operator name is required")
    payload["approved"] = True
    payload["approval"] = {
        "operator": name,
        "approved_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha256": sha256_json(candidate),
    }
    validate_acceptance(payload, require_approved=True)
    return payload


def copy_mapping(value):
    return json.loads(json.dumps(value))
