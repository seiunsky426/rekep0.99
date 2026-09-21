"""Canonical Ed25519 artifacts and aggregate release-bundle validation."""

from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)
import yaml


SCHEMA_VERSION = 2
REQUIRED_RELEASE_ARTIFACTS = frozenset({
    "camera_extrinsics_rs1", "camera_extrinsics_rs3", "workspace",
    "gripper_baseline", "safe_map", "paper_real_solver",
    "runtime_performance_software", "runtime_performance_hardware",
})
REQUIRED_HARDWARE_ACCEPTANCE_ARTIFACTS = frozenset(
    REQUIRED_RELEASE_ARTIFACTS - {"runtime_performance_hardware"})
EXPECTED_OFFICIAL_REKEP_COMMIT = "63c43fdba60354980258beaeb8a7d48e088e1e3e"


class AcceptanceError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AcceptanceError("artifact is not canonical-JSON compatible") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _load_private_key(path: str) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(
            Path(path).read_bytes(), password=None, backend=default_backend())
    except (OSError, ValueError, TypeError) as exc:
        raise AcceptanceError("cannot load Ed25519 private key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise AcceptanceError("private key is not Ed25519")
    return key


def _load_public_key(path: str) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(
            Path(path).read_bytes(), backend=default_backend())
    except (OSError, ValueError, TypeError) as exc:
        raise AcceptanceError("cannot load Ed25519 public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise AcceptanceError("public key is not Ed25519")
    return key


def key_id_for_public_key(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return sha256_bytes(raw)[:16]


def _signed_portion(document: Mapping[str, Any]) -> Dict[str, Any]:
    value = deepcopy(dict(document))
    approval = value.get("approval")
    if not isinstance(approval, dict):
        raise AcceptanceError("artifact approval mapping is missing")
    approval.pop("signature_ed25519_b64", None)
    return value


def _resolve_evidence(root: Path, supplied: str) -> Path:
    relative = Path(str(supplied))
    if relative.is_absolute():
        raise AcceptanceError("evidence paths must be relative to the artifact")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise AcceptanceError("evidence path escapes artifact directory") from exc
    return resolved


def create_signed_artifact(
        artifact_type: str, artifact_id: str, payload: Mapping[str, Any],
        evidence: Iterable[Mapping[str, str]], operator: str,
        private_key_path: str, created_utc: Optional[str] = None) -> Dict[str, Any]:
    name = str(operator).strip()
    if not name:
        raise AcceptanceError("operator is required")
    private_key = _load_private_key(private_key_path)
    public_key = private_key.public_key()
    payload_copy = deepcopy(dict(payload))
    evidence_copy = [deepcopy(dict(item)) for item in evidence]
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": str(artifact_type).strip(),
        "artifact_id": str(artifact_id).strip(),
        "payload": payload_copy,
        "payload_sha256": sha256_bytes(canonical_bytes(payload_copy)),
        "evidence": evidence_copy,
        "approval": {
            "operator": name,
            "approved_utc": created_utc or datetime.now(timezone.utc).isoformat(),
            "key_id": key_id_for_public_key(public_key),
            "algorithm": "Ed25519",
        },
    }
    if not document["artifact_type"] or not document["artifact_id"]:
        raise AcceptanceError("artifact type and id are required")
    signature = private_key.sign(canonical_bytes(_signed_portion(document)))
    document["approval"]["signature_ed25519_b64"] = base64.b64encode(
        signature).decode("ascii")
    return document


def verify_signed_artifact(
        document: Mapping[str, Any], public_key_path: str,
        artifact_root: Optional[str] = None,
        expected_type: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(document, Mapping) or document.get("schema_version") != SCHEMA_VERSION:
        raise AcceptanceError("signed artifact schema_version must be 2")
    if expected_type is not None and document.get("artifact_type") != expected_type:
        raise AcceptanceError("signed artifact type mismatch")
    payload = document.get("payload")
    if not isinstance(payload, Mapping):
        raise AcceptanceError("signed artifact payload must be a mapping")
    if document.get("payload_sha256") != sha256_bytes(canonical_bytes(payload)):
        raise AcceptanceError("signed artifact payload hash mismatch")
    approval = document.get("approval")
    if not isinstance(approval, Mapping) or approval.get("algorithm") != "Ed25519":
        raise AcceptanceError("signed artifact approval is invalid")
    public_key = _load_public_key(public_key_path)
    if approval.get("key_id") != key_id_for_public_key(public_key):
        raise AcceptanceError("signed artifact key id mismatch")
    try:
        signature = base64.b64decode(
            str(approval.get("signature_ed25519_b64", "")), validate=True)
        public_key.verify(signature, canonical_bytes(_signed_portion(document)))
    except (ValueError, InvalidSignature) as exc:
        raise AcceptanceError("signed artifact signature is invalid") from exc
    if not str(approval.get("operator", "")).strip():
        raise AcceptanceError("signed artifact operator is missing")
    evidence = document.get("evidence", [])
    if not isinstance(evidence, list):
        raise AcceptanceError("signed artifact evidence must be a list")
    root = Path(artifact_root).resolve() if artifact_root else None
    roles = set()
    for item in evidence:
        if not isinstance(item, Mapping):
            raise AcceptanceError("signed artifact evidence entry is invalid")
        role = str(item.get("role", "")).strip()
        if not role or role in roles:
            raise AcceptanceError("evidence roles must be non-empty and unique")
        roles.add(role)
        digest = str(item.get("sha256", "")).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise AcceptanceError("evidence sha256 is invalid")
        if root is not None:
            path = _resolve_evidence(root, str(item.get("path", "")))
            if not path.is_file() or sha256_file(str(path)) != digest:
                raise AcceptanceError("evidence file is missing or changed: " + role)
    return deepcopy(dict(document))


def load_signed_artifact(
        path: str, public_key_path: str,
        expected_type: Optional[str] = None) -> Dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AcceptanceError("cannot load signed artifact") from exc
    return verify_signed_artifact(
        document, public_key_path, artifact_root=str(source.parent),
        expected_type=expected_type)


def _load_minimum_counter(path: str) -> int:
    try:
        value = int(Path(path).read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise AcceptanceError("release counter state is unavailable") from exc
    if value < 0:
        raise AcceptanceError("release counter state is invalid")
    return value


def validate_release_bundle(
        bundle_path: str, public_key_path: str, minimum_counter_path: str,
        expected_robot_id: Optional[str] = None,
        required_artifacts: Optional[Iterable[str]] = None,
        source_root: Optional[str] = None) -> Dict[str, Any]:
    bundle_source = Path(bundle_path).expanduser().resolve()
    bundle_type = os.environ.get(
        "REKPIPER_ACCEPTANCE_BUNDLE_TYPE", "release_bundle")
    if bundle_type not in ("release_bundle", "hardware_acceptance_bundle"):
        raise AcceptanceError("acceptance bundle type is invalid")
    if required_artifacts is None:
        required_artifacts = (
            REQUIRED_RELEASE_ARTIFACTS if bundle_type == "release_bundle"
            else REQUIRED_HARDWARE_ACCEPTANCE_ARTIFACTS)
    bundle = load_signed_artifact(
        str(bundle_source), public_key_path, expected_type=bundle_type)
    payload = bundle["payload"]
    counter = int(payload.get("release_counter", -1))
    installed_counter = _load_minimum_counter(minimum_counter_path)
    if counter != installed_counter:
        # Requiring equality prevents a higher, not-yet-installed bundle from
        # running once and then being rolled back to another still-acceptable
        # counter.  Replacement is a two-person/offline operation completed by
        # atomically installing the exact new counter.
        raise AcceptanceError(
            "release bundle counter is not the installed counter")
    if expected_robot_id is not None and payload.get("robot_id") != expected_robot_id:
        raise AcceptanceError("release bundle robot identity mismatch")
    if payload.get("official_rekep_commit") != EXPECTED_OFFICIAL_REKEP_COMMIT:
        raise AcceptanceError("release bundle official ReKep commit mismatch")
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        raise AcceptanceError("release bundle bindings mapping is missing")
    for name in ("source_manifest_sha256", "urdf_sha256",
                 "system_config_sha256", "runtime_fingerprint_sha256"):
        digest = str(bindings.get(name, "")).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise AcceptanceError("release bundle binding is invalid: " + name)
    bundle_evidence = {
        item.get("role"): item.get("sha256") for item in bundle.get("evidence", [])
        if isinstance(item, Mapping)}
    for role in ("source_manifest", "urdf", "system_config",
                 "runtime_fingerprint"):
        if bindings.get(role + "_sha256") != bundle_evidence.get(role):
            raise AcceptanceError("release bundle evidence binding mismatch: " + role)
    monitored_source_files = {}
    source_manifest_entry = next(
        (item for item in bundle.get("evidence", [])
         if isinstance(item, Mapping) and item.get("role") == "source_manifest"),
        None)
    active_source_root = source_root
    if active_source_root is None:
        active_source_root = os.environ.get("REKPIPER_ROOT", "")
    if active_source_root:
        manifest_path = _resolve_evidence(
            bundle_source.parent, str(source_manifest_entry.get("path", "")))
        try:
            source_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, AttributeError) as exc:
            raise AcceptanceError("source manifest is invalid") from exc
        if not isinstance(source_manifest, Mapping):
            raise AcceptanceError("source manifest must be a mapping")
        files = source_manifest.get("files")
        unsigned = {"schema_version": source_manifest.get("schema_version"),
                    "files": files}
        if (source_manifest.get("schema_version") != 1
                or not isinstance(files, Mapping)
                or source_manifest.get("tree_sha256") != sha256_bytes(
                    canonical_bytes(unsigned))):
            raise AcceptanceError("source manifest self-hash mismatch")
        source_base = Path(active_source_root).expanduser().resolve()
        for relative, expected in files.items():
            path = _resolve_evidence(source_base, str(relative))
            if not path.is_file() or sha256_file(str(path)) != expected:
                raise AcceptanceError("runtime source differs from release: " + str(relative))
            monitored_source_files[str(path)] = expected
    entries = payload.get("artifacts")
    if not isinstance(entries, Mapping):
        raise AcceptanceError("release bundle artifacts mapping is missing")
    missing = sorted(set(required_artifacts) - set(entries))
    if missing:
        raise AcceptanceError("release bundle artifacts missing: " + ",".join(missing))
    verified = {}
    monitored = {
        str(bundle_source): sha256_file(str(bundle_source)),
        str(Path(public_key_path).expanduser().resolve()): sha256_file(
            str(Path(public_key_path).expanduser().resolve())),
        str(Path(minimum_counter_path).expanduser().resolve()): sha256_file(
            str(Path(minimum_counter_path).expanduser().resolve())),
    }
    monitored.update(monitored_source_files)
    for artifact_type in required_artifacts:
        entry = entries[artifact_type]
        if not isinstance(entry, Mapping):
            raise AcceptanceError("release bundle artifact entry is invalid")
        artifact_path = _resolve_evidence(
            bundle_source.parent, str(entry.get("path", "")))
        if not artifact_path.is_file():
            raise AcceptanceError("release artifact is missing: " + artifact_type)
        if sha256_file(str(artifact_path)) != str(entry.get("sha256", "")):
            raise AcceptanceError("release artifact file hash mismatch: " + artifact_type)
        artifact = load_signed_artifact(
            str(artifact_path), public_key_path, expected_type=artifact_type)
        artifact_evidence = {
            item.get("role"): item for item in artifact.get("evidence", [])
            if isinstance(item, Mapping)}
        artifact_payload = artifact["payload"]
        if artifact_type.startswith("runtime_performance_"):
            for role in ("summary", "raw_event_log"):
                if role not in artifact_evidence:
                    raise AcceptanceError(
                        "runtime artifact evidence is missing: " + role)
            summary_path = _resolve_evidence(
                artifact_path.parent, artifact_evidence["summary"]["path"])
            try:
                summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                raise AcceptanceError("runtime summary cannot be loaded") from exc
            if canonical_bytes(summary) != canonical_bytes(artifact_payload):
                raise AcceptanceError("runtime signed payload and summary differ")
            if artifact_payload.get("raw_event_log_sha256") != \
                    artifact_evidence["raw_event_log"].get("sha256"):
                raise AcceptanceError("runtime raw-event binding mismatch")
            if (artifact_type == "runtime_performance_software"
                    and artifact_payload.get("software_contract", {}).get(
                        "replay_dataset_sha256")
                    != artifact_evidence.get("replay_dataset", {}).get(
                        "sha256")):
                raise AcceptanceError("software replay dataset binding mismatch")
            if (artifact_type == "runtime_performance_hardware"
                    and artifact_payload.get("hardware_hold", {}).get(
                        "hardware_acceptance_bundle_sha256")
                    != artifact_evidence.get(
                        "hardware_acceptance_bundle", {}).get("sha256")):
                raise AcceptanceError(
                    "hardware performance bootstrap binding mismatch")
            if artifact_type == "runtime_performance_hardware":
                bootstrap_path = _resolve_evidence(
                    artifact_path.parent, artifact_evidence[
                        "hardware_acceptance_bundle"]["path"])
                bootstrap = load_signed_artifact(
                    str(bootstrap_path), public_key_path,
                    expected_type="hardware_acceptance_bundle")
                bootstrap_payload = bootstrap["payload"]
                for name in ("robot_id", "piper_identity", "camera_serials",
                             "official_rekep_commit", "bindings"):
                    if bootstrap_payload.get(name) != payload.get(name):
                        raise AcceptanceError(
                            "hardware performance environment drift: " + name)
                bootstrap_entries = bootstrap_payload.get("artifacts", {})
                for name in REQUIRED_HARDWARE_ACCEPTANCE_ARTIFACTS:
                    if (bootstrap_entries.get(name, {}).get("sha256")
                            != entries.get(name, {}).get("sha256")):
                        raise AcceptanceError(
                            "hardware performance artifact drift: " + name)
        elif artifact_type.startswith("camera_extrinsics_"):
            if (artifact_payload.get("status") != "ACCEPTED"
                    or artifact_payload.get("publish_tf_allowed") is not True
                    or artifact_payload.get("precision_operation_allowed") is not True
                    or artifact_payload.get("validation_report_sha256")
                    != artifact_evidence.get("accepted_report", {}).get("sha256")
                    or artifact_payload.get("dataset_sha256")
                    != artifact_evidence.get("calibration_dataset", {}).get("sha256")):
                raise AcceptanceError("camera calibration evidence chain is invalid")
            camera_name = artifact_type.rsplit("_", 1)[-1]
            report_path = _resolve_evidence(
                artifact_path.parent,
                artifact_evidence.get("accepted_report", {}).get("path", ""))
            dataset_path = _resolve_evidence(
                artifact_path.parent,
                artifact_evidence.get("calibration_dataset", {}).get(
                    "path", ""))
            try:
                report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
                dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                raise AcceptanceError(
                    "camera calibration evidence cannot be loaded") from exc
            validation = report.get(
                "validation", {}) if isinstance(report, Mapping) else {}
            if not isinstance(validation, Mapping):
                validation = {}
            checks = validation.get(
                "checks", {})
            if not isinstance(checks, Mapping):
                checks = {}
            report_results = report.get(
                "results", {}) if isinstance(report, Mapping) else {}
            if not isinstance(report_results, Mapping):
                report_results = {}
            camera_serials = report.get(
                "camera_serials", {}) if isinstance(report, Mapping) else {}
            if not isinstance(camera_serials, Mapping):
                camera_serials = {}
            dataset_cameras = dataset.get(
                "cameras", {}) if isinstance(dataset, Mapping) else {}
            if not isinstance(dataset_cameras, Mapping):
                dataset_cameras = {}
            try:
                report_metrics_pass = bool(
                    float(validation.get("reprojection_rmse_px", float("inf")))
                    <= 0.8
                    and float(validation.get(
                        "dual_camera_3d_median_m", float("inf"))) <= 0.005
                    and float(validation.get(
                        "dual_camera_3d_p95_m", float("inf"))) <= 0.008
                    and len(validation.get("paired_sample_ids", [])) >= 6
                    and int(checks.get(
                        "minimum_optimization_poses_per_camera", 0)) >= 18
                    and int(checks.get(
                        "minimum_validation_poses_per_camera", 0)) >= 6)
            except (TypeError, ValueError):
                report_metrics_pass = False
            serial = str(artifact_payload.get("serial", "")).lstrip("_")
            if (not isinstance(report, Mapping)
                    or report.get("schema_version") != 1
                    or report.get("status") != "ACCEPTED"
                    or report.get("dataset_sha256")
                    != artifact_evidence["calibration_dataset"].get("sha256")
                    or str(camera_serials.get(
                        camera_name, "")).lstrip("_") != serial
                    or (report_results.get(camera_name, {})
                        if isinstance(report_results.get(camera_name, {}), Mapping)
                        else {}).get("base_T_link")
                    != artifact_payload.get("base_T_link")
                    or not report_metrics_pass
                    or checks.get("duplicates_absent") is not True
                    or checks.get("splits_disjoint") is not True
                    or checks.get("pose_diversity_passed") is not True
                    or not isinstance(dataset, Mapping)
                    or dataset.get("schema_version") != 2
                    or str((dataset_cameras.get(camera_name, {})
                        if isinstance(dataset_cameras.get(camera_name, {}), Mapping)
                        else {}).get(
                        "serial", "")).lstrip("_") != serial):
                raise AcceptanceError(
                    "camera calibration report/dataset content is invalid")
        elif artifact_type == "gripper_baseline":
            if (not bool(artifact_payload.get("calibrated", False))
                    or artifact_payload.get("dataset_sha256")
                    != artifact_evidence.get("calibration_dataset", {}).get(
                        "sha256")):
                raise AcceptanceError("gripper baseline is not calibrated")
            dataset_path = _resolve_evidence(
                artifact_path.parent,
                artifact_evidence.get("calibration_dataset", {}).get(
                    "path", ""))
            try:
                dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                raise AcceptanceError(
                    "gripper calibration dataset cannot be loaded") from exc
            counts = dataset.get("counts", {}) if isinstance(dataset, Mapping) else {}
            thresholds = dataset.get(
                "measured_thresholds", {}) if isinstance(dataset, Mapping) else {}
            try:
                counts_pass = bool(
                    int(counts.get("empty_cycles", 0)) >= 20
                    and int(counts.get("rigid_grasps", 0)) >= 10
                    and int(counts.get("empty_or_slip_trials", 0)) >= 5)
            except (TypeError, ValueError):
                counts_pass = False
            if (not isinstance(dataset, Mapping)
                    or dataset.get("schema_version") != 1
                    or dataset.get("status") != "ACCEPTED"
                    or dataset.get("piper_identity")
                    != artifact_payload.get("piper_identity")
                    or not counts_pass
                    or thresholds.get("empty_closed_opening_p99_m")
                    != artifact_payload.get("empty_closed_opening_p99_m")
                    or thresholds.get("empty_effort_p95")
                    != artifact_payload.get("empty_effort_p95")):
                raise AcceptanceError(
                    "gripper calibration dataset content is invalid")
        elif artifact_type == "safe_map":
            report_roles = (
                "e3_robot_mask_accepted",
                "e4_multicamera_tracking_accepted",
                "grasp_release_slip_accepted",
                "tsdf_clear_and_residuals_accepted",
                "dual_camera_independent_3d_accepted",
            )
            report_hashes = artifact_payload.get("reports_sha256", {})
            if (not bool(artifact_payload.get("accepted", False))
                    or any(not bool(artifact_payload.get(role, False))
                           for role in report_roles)
                    or any(report_hashes.get(role)
                           != artifact_evidence.get(role, {}).get("sha256")
                           for role in report_roles)):
                raise AcceptanceError("safe-map artifact is not accepted")
            for role in report_roles:
                report_path = _resolve_evidence(
                    artifact_path.parent,
                    artifact_evidence.get(role, {}).get("path", ""))
                try:
                    report = yaml.safe_load(
                        report_path.read_text(encoding="utf-8"))
                except (OSError, yaml.YAMLError) as exc:
                    raise AcceptanceError(
                        "safe-map report cannot be loaded: " + role) from exc
                if (not isinstance(report, Mapping)
                        or report.get("schema_version") != 1
                        or report.get("report_type") != role
                        or report.get("passed") is not True
                        or report.get("robot_id") != payload.get("robot_id")):
                    raise AcceptanceError(
                        "safe-map report content is invalid: " + role)
        elif artifact_type == "workspace":
            if (artifact_payload.get("status") != "ACCEPTED"
                    or artifact_payload.get("precision_operation_allowed") is not True
                    or artifact_payload.get("measurement_report_sha256")
                    != artifact_evidence.get("measurement_report", {}).get(
                        "sha256")):
                raise AcceptanceError("workspace artifact is not accepted")
            measurement_path = _resolve_evidence(
                artifact_path.parent,
                artifact_evidence.get("measurement_report", {}).get(
                    "path", ""))
            try:
                measurement = yaml.safe_load(
                    measurement_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                raise AcceptanceError(
                    "workspace measurement report cannot be loaded") from exc
            if (not isinstance(measurement, Mapping)
                    or measurement.get("schema_version") != 1
                    or measurement.get("status") != "ACCEPTED"
                    or measurement.get("robot_id") != payload.get("robot_id")
                    or measurement.get("frame_id")
                    != artifact_payload.get("frame_id")
                    or measurement.get("workspace_bounds_min")
                    != artifact_payload.get("workspace_bounds_min")
                    or measurement.get("workspace_bounds_max")
                    != artifact_payload.get("workspace_bounds_max")
                    or measurement.get("table_height_m")
                    != artifact_payload.get("table_height_m")):
                raise AcceptanceError(
                    "workspace measurement report content is invalid")
        elif artifact_type == "paper_real_solver":
            replay = artifact_evidence.get("replay_report", {})
            replay_path = _resolve_evidence(
                artifact_path.parent, str(replay.get("path", "")))
            try:
                replay_payload = yaml.safe_load(
                    replay_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                raise AcceptanceError("solver replay report cannot be loaded") from exc
            if artifact_payload.get("replay_report_sha256") != sha256_bytes(
                    canonical_bytes(replay_payload)):
                raise AcceptanceError("solver replay report binding mismatch")
            if artifact_payload.get("inputs", {}).get("dataset_sha256") != \
                    artifact_evidence.get("replay_dataset", {}).get("sha256"):
                raise AcceptanceError("solver replay dataset binding mismatch")
        verified[artifact_type] = artifact
        monitored[str(artifact_path)] = sha256_file(str(artifact_path))
        for evidence in artifact.get("evidence", []):
            evidence_path = _resolve_evidence(
                artifact_path.parent, str(evidence.get("path", "")))
            monitored[str(evidence_path)] = sha256_file(str(evidence_path))
    base_artifacts = REQUIRED_HARDWARE_ACCEPTANCE_ARTIFACTS
    if set(base_artifacts).issubset(verified):
        workspace_hash = verified["workspace"]["payload_sha256"]
        if verified["safe_map"]["payload"].get(
                "workspace_payload_sha256") != workspace_hash:
            raise AcceptanceError("safe-map is not bound to the signed workspace")
        serials = payload.get("camera_serials", {})
        for name in ("rs1", "rs3"):
            if (not str(serials.get(name, "")).lstrip("_")
                    or str(verified["camera_extrinsics_" + name]["payload"].get(
                        "serial", "")).lstrip("_")
                    != str(serials[name]).lstrip("_")):
                raise AcceptanceError("release camera serial mismatch: " + name)
        piper_identity = str(payload.get("piper_identity", "")).strip()
        if (not piper_identity
                or verified["gripper_baseline"]["payload"].get(
                    "piper_identity") != piper_identity):
            raise AcceptanceError("gripper baseline Piper identity mismatch")
        if "runtime_performance_hardware" in verified:
            hold = verified["runtime_performance_hardware"]["payload"].get(
                "hardware_hold", {})
            if (hold.get("robot_id") != payload.get("robot_id")
                    or hold.get("piper_identity") != piper_identity):
                raise AcceptanceError(
                    "hardware performance robot identity mismatch")
    result = deepcopy(bundle)
    result["verified_artifacts"] = verified
    result["_validated_files"] = monitored
    result["_validated_stats"] = {
        path: (Path(path).stat().st_dev, Path(path).stat().st_ino,
               Path(path).stat().st_size, Path(path).stat().st_mtime_ns,
               Path(path).stat().st_ctime_ns)
        for path in monitored}
    return result


def assert_release_unchanged(release: Mapping[str, Any]) -> None:
    files = release.get("_validated_files")
    stats = release.get("_validated_stats")
    if (not isinstance(files, Mapping) or not files
            or not isinstance(stats, Mapping) or set(stats) != set(files)):
        raise AcceptanceError("release validation file state is missing")
    for supplied in files:
        path = Path(str(supplied))
        try:
            current = (path.stat().st_dev, path.stat().st_ino,
                       path.stat().st_size, path.stat().st_mtime_ns,
                       path.stat().st_ctime_ns)
        except OSError:
            current = None
        if current != tuple(stats[supplied]):
            raise AcceptanceError("release input changed after validation: " + str(path))


def validate_program_approval(
        manifest_path: str, public_key_path: str, session_id: str,
        snapshot_id: str, program_sha256: str,
        official_prompt_sha256: str,
        local_safety_contract_version: str,
        expected_snapshot_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Validate the signed, immutable provenance for one executable program."""
    manifest_source = Path(manifest_path).expanduser().resolve()
    public_source = Path(public_key_path).expanduser().resolve()
    document = load_signed_artifact(
        str(manifest_source), str(public_source), expected_type="rekep_program")
    payload = document["payload"]
    expected = {
        "session_id": session_id,
        "snapshot_id": snapshot_id,
        "program_sha256": program_sha256,
        "official_prompt_sha256": official_prompt_sha256,
        "local_safety_contract_version": local_safety_contract_version,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise AcceptanceError("program approval {} mismatch".format(name))
    evidence_by_role = {
        item.get("role"): item for item in document.get("evidence", [])
        if isinstance(item, Mapping)}
    snapshot_sha256 = str(payload.get("snapshot_sha256", "")).lower()
    if (len(snapshot_sha256) != 64
            or any(c not in "0123456789abcdef" for c in snapshot_sha256)
            or snapshot_sha256 != evidence_by_role.get(
                "scene_snapshot_rosmsg", {}).get("sha256")
            or (expected_snapshot_sha256 is not None
                and snapshot_sha256 != expected_snapshot_sha256)):
        raise AcceptanceError("program approval scene snapshot mismatch")
    for name in ("annotated_image_sha256", "raw_response_sha256",
                 "effective_prompt_sha256"):
        value = str(payload.get(name, "")).lower()
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise AcceptanceError("program approval {} is invalid".format(name))
    monitored = {
        str(manifest_source): sha256_file(str(manifest_source)),
        str(public_source): sha256_file(str(public_source)),
    }
    for evidence in document.get("evidence", []):
        path = _resolve_evidence(
            manifest_source.parent, str(evidence.get("path", "")))
        monitored[str(path)] = sha256_file(str(path))
    document["_validated_files"] = monitored
    document["_validated_stats"] = {
        path: (Path(path).stat().st_dev, Path(path).stat().st_ino,
               Path(path).stat().st_size, Path(path).stat().st_mtime_ns,
               Path(path).stat().st_ctime_ns)
        for path in monitored}
    return document
