#!/usr/bin/env python3
"""Create an autonomous runtime gate from a ten-minute evidence report."""

import argparse
import json
from pathlib import Path
import sys

import yaml

from rekpiper_acceptance import create_signed_artifact, sha256_file

from rekpiper_execution.runtime_acceptance import (
    RuntimeAcceptanceError, approve_evidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence")
    parser.add_argument("output")
    parser.add_argument("--operator", required=True)
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--profile", choices=("software", "hardware_hold"),
                        required=True)
    parser.add_argument("--replay-dataset", default="")
    parser.add_argument("--hardware-acceptance-bundle", default="")
    args = parser.parse_args()
    try:
        evidence = yaml.safe_load(
            Path(args.evidence).read_text(encoding="utf-8"))
        if evidence.get("profile") != args.profile:
            raise RuntimeAcceptanceError("runtime profile mismatch")
        approve_evidence(evidence, args.operator)
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            raise RuntimeAcceptanceError("approval output already exists")
        evidence_path = Path(args.evidence).expanduser().resolve()
        if output.parent != evidence_path.parent:
            raise RuntimeAcceptanceError(
                "approval, summary and raw log must share one immutable directory")
        raw_path = evidence_path.parent / str(evidence.get("raw_event_log", ""))
        if not raw_path.is_file() or sha256_file(str(raw_path)) != \
                evidence.get("raw_event_log_sha256"):
            raise RuntimeAcceptanceError("raw runtime event log hash mismatch")
        artifact_type = "runtime_performance_" + (
            "software" if args.profile == "software" else "hardware")
        artifact_evidence = [
            {"role": "summary", "path": evidence_path.name,
             "sha256": sha256_file(str(evidence_path))},
            {"role": "raw_event_log", "path": raw_path.name,
             "sha256": sha256_file(str(raw_path))},
        ]
        if args.profile == "software":
            replay_path = Path(args.replay_dataset).expanduser().resolve()
            if (replay_path.parent != output.parent
                    or sha256_file(str(replay_path)) != evidence.get(
                        "software_contract", {}).get(
                            "replay_dataset_sha256")):
                raise RuntimeAcceptanceError(
                    "software replay dataset hash/path mismatch")
            artifact_evidence.append({
                "role": "replay_dataset", "path": replay_path.name,
                "sha256": sha256_file(str(replay_path))})
        else:
            bootstrap = Path(
                args.hardware_acceptance_bundle).expanduser().resolve()
            if (bootstrap.parent != output.parent
                    or sha256_file(str(bootstrap)) != evidence.get(
                        "hardware_hold", {}).get(
                            "hardware_acceptance_bundle_sha256")):
                raise RuntimeAcceptanceError(
                    "hardware acceptance bundle hash/path mismatch")
            artifact_evidence.append({
                "role": "hardware_acceptance_bundle", "path": bootstrap.name,
                "sha256": sha256_file(str(bootstrap))})
        artifact = create_signed_artifact(
            artifact_type, args.profile + "-runtime", evidence,
            artifact_evidence,
            args.operator, args.private_key)
        output.write_text(yaml.safe_dump(artifact, sort_keys=False), encoding="utf-8")
        print(json.dumps({"success": True, "output": args.output}, sort_keys=True))
        return 0
    except (OSError, yaml.YAMLError, RuntimeAcceptanceError, ValueError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
