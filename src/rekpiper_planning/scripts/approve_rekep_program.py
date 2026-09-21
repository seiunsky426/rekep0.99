#!/usr/bin/env python3
"""Offline Ed25519 approval of a reviewed ReKep program directory."""

import argparse
import json
from pathlib import Path

import yaml

from rekpiper_acceptance import create_signed_artifact, sha256_file
from rekpiper_planning.official_program import (
    LOCAL_SAFETY_CONTRACT_VERSION, OFFICIAL_PROMPT_SHA256,
    compute_program_sha256)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("program_directory")
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--operator", required=True)
    args = parser.parse_args()
    root = Path(args.program_directory).expanduser().resolve()
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    raw_name = audit.get("raw_response_file", "gpt4o_raw_response.txt")
    if raw_name not in ("vlm_raw_response.txt", "gpt4o_raw_response.txt"):
        raise SystemExit("unsupported raw response filename")
    required = [
        "audit.json", "annotated_keypoints.png", "prompt.txt",
        raw_name, "metadata.json", "program.py",
        "scene_snapshot.rosmsg",
    ]
    required.extend(sorted(path.name for path in root.glob(
        "stage*_constraints.txt")))
    for name in required:
        if not (root / name).is_file():
            raise SystemExit("program evidence is missing: " + name)
    program_hash = compute_program_sha256(str(root))
    if audit.get("program_sha256") != program_hash:
        raise SystemExit("audit and executable program hash mismatch")
    prompt_hash = sha256_file(str(root / "prompt.txt"))
    raw_hash = sha256_file(str(root / raw_name))
    image_hash = sha256_file(str(root / "annotated_keypoints.png"))
    snapshot_hash = sha256_file(str(root / "scene_snapshot.rosmsg"))
    if audit.get("prompt_sha256") != prompt_hash:
        raise SystemExit("effective prompt hash mismatch")
    if audit.get("raw_response_sha256") != raw_hash:
        raise SystemExit("raw model response hash mismatch")
    if audit.get("query_image_sha256") != image_hash:
        raise SystemExit("annotated image hash mismatch")
    if audit.get("snapshot_sha256") != snapshot_hash:
        raise SystemExit("scene snapshot hash mismatch")
    payload = {
        "session_id": str(audit["session_id"]),
        "snapshot_id": str(audit["snapshot_id"]),
        "snapshot_sha256": snapshot_hash,
        "program_sha256": program_hash,
        "official_prompt_sha256": OFFICIAL_PROMPT_SHA256,
        "local_safety_contract_version": LOCAL_SAFETY_CONTRACT_VERSION,
        "effective_prompt_sha256": prompt_hash,
        "raw_response_sha256": raw_hash,
        "annotated_image_sha256": image_hash,
        "model": str(audit.get("vlm_model", "")),
    }
    evidence = [{
        "role": name.replace(".", "_").replace("*", "_")[:80],
        "path": name,
        "sha256": sha256_file(str(root / name)),
    } for name in required]
    artifact = create_signed_artifact(
        "rekep_program", str(audit["session_id"]), payload, evidence,
        args.operator, args.private_key)
    output = root / "program_approval.yaml"
    if output.exists():
        raise SystemExit("program approval already exists; create a new session")
    output.write_text(
        yaml.safe_dump(artifact, sort_keys=False), encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
