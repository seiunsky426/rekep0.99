#!/usr/bin/env python3
"""Sign a prepared payload and local evidence files as a schema-v2 artifact."""

import argparse
import json
from pathlib import Path
import sys

import yaml

from rekpiper_acceptance import create_signed_artifact, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("payload")
    parser.add_argument("output")
    parser.add_argument("--artifact-type", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--evidence", action="append", default=[],
                        help="ROLE=relative/path; file must be beside output")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit("refusing to overwrite signed artifact")
    payload = yaml.safe_load(Path(args.payload).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("payload must be a mapping")
    evidence = []
    for value in args.evidence:
        role, supplied = value.split("=", 1)
        source = (output.parent / supplied).resolve()
        source.relative_to(output.parent)
        evidence.append({"role": role, "path": supplied,
                         "sha256": sha256_file(str(source))})
    document = create_signed_artifact(
        args.artifact_type, args.artifact_id, payload, evidence,
        args.operator, args.private_key)
    output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(json.dumps({"success": True, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
