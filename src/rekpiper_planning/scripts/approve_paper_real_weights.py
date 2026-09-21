#!/usr/bin/env python3
"""Explicitly approve an offline-calibrated paper-real solver profile."""

import argparse
import json
from pathlib import Path
import sys

import yaml

from rekpiper_acceptance import create_signed_artifact, sha256_file

from rekpiper_planning.solver_acceptance import (
    SolverAcceptanceError, approve_candidate, sha256_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("output")
    parser.add_argument("--operator", required=True)
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--replay-report", required=True)
    args = parser.parse_args()
    try:
        candidate = yaml.safe_load(
            Path(args.candidate).read_text(encoding="utf-8"))
        accepted = approve_candidate(candidate, args.operator)
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            raise SolverAcceptanceError("approval output already exists")
        candidate_path = Path(args.candidate).expanduser().resolve()
        if candidate_path.parent != output.parent:
            raise SolverAcceptanceError(
                "candidate must be copied beside the signed output")
        replay_path = Path(args.replay_report).expanduser().resolve()
        if replay_path.parent != output.parent:
            raise SolverAcceptanceError(
                "replay report must be copied beside the signed output")
        replay_report = yaml.safe_load(replay_path.read_text(encoding="utf-8"))
        if sha256_json(replay_report) != accepted.get("replay_report_sha256"):
            raise SolverAcceptanceError("replay report hash mismatch")
        dataset_path = Path(accepted.get(
            "input_paths", {}).get("dataset", "")).expanduser().resolve()
        if dataset_path.parent != output.parent:
            raise SolverAcceptanceError(
                "replay dataset must be copied beside the signed output")
        if sha256_file(str(dataset_path)) != accepted.get(
                "inputs", {}).get("dataset_sha256"):
            raise SolverAcceptanceError("replay dataset hash mismatch")
        artifact = create_signed_artifact(
            "paper_real_solver", "paper-real-" + accepted["official_commit"][:12],
            accepted, [{"role": "candidate", "path": candidate_path.name,
                        "sha256": sha256_file(str(candidate_path))},
                       {"role": "replay_report", "path": replay_path.name,
                        "sha256": sha256_file(str(replay_path))},
                       {"role": "replay_dataset", "path": dataset_path.name,
                        "sha256": sha256_file(str(dataset_path))}],
            args.operator, args.private_key)
        output.write_text(yaml.safe_dump(artifact, sort_keys=False), encoding="utf-8")
        print(json.dumps({"success": True, "output": args.output}, sort_keys=True))
        return 0
    except (OSError, yaml.YAMLError, SolverAcceptanceError, ValueError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
