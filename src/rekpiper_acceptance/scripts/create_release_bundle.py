#!/usr/bin/env python3
"""Create the final signed site release from already signed artifacts."""

import argparse
from pathlib import Path

import yaml

from rekpiper_acceptance import (
    REQUIRED_HARDWARE_ACCEPTANCE_ARTIFACTS, REQUIRED_RELEASE_ARTIFACTS,
    create_signed_artifact, load_signed_artifact,
    sha256_file)
from rekpiper_acceptance.signed_artifact import EXPECTED_OFFICIAL_REKEP_COMMIT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--artifact", action="append", default=[],
                        help="TYPE=relative/path under the output directory")
    parser.add_argument("--release-counter", required=True, type=int)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--piper-identity", required=True)
    parser.add_argument("--camera-serial", action="append", default=[],
                        help="rs1=SERIAL and rs3=SERIAL")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--system-config", required=True)
    parser.add_argument("--runtime-fingerprint", required=True)
    parser.add_argument("--public-key", required=True)
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--bundle-type", choices=(
        "release_bundle", "hardware_acceptance_bundle"),
        default="release_bundle")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit("refusing to overwrite release bundle")
    if args.release_counter < 0:
        raise SystemExit("release counter must be nonnegative")
    root = output.parent
    artifacts = {}
    for supplied in args.artifact:
        kind, relative = supplied.split("=", 1)
        path = (root / relative).resolve()
        path.relative_to(root)
        load_signed_artifact(str(path), args.public_key, expected_type=kind)
        artifacts[kind] = {"path": relative, "sha256": sha256_file(str(path))}
    required = (REQUIRED_RELEASE_ARTIFACTS
                if args.bundle_type == "release_bundle"
                else REQUIRED_HARDWARE_ACCEPTANCE_ARTIFACTS)
    missing = sorted(required - set(artifacts))
    if missing:
        raise SystemExit("release artifacts missing: " + ",".join(missing))
    camera_serials = dict(value.split("=", 1) for value in args.camera_serial)
    if (set(camera_serials) != {"rs1", "rs3"}
            or any(not value.strip().lstrip("_")
                   for value in camera_serials.values())):
        raise SystemExit("both rs1 and rs3 camera serials are required")
    evidence = []
    bindings = {}
    sources = {
        "source_manifest": args.source_manifest,
        "urdf": args.urdf,
        "system_config": args.system_config,
        "runtime_fingerprint": args.runtime_fingerprint,
    }
    for role, relative in sources.items():
        path = (root / relative).resolve()
        path.relative_to(root)
        digest = sha256_file(str(path))
        evidence.append({"role": role, "path": relative, "sha256": digest})
        bindings[role + "_sha256"] = digest
    payload = {
        "release_counter": args.release_counter,
        "robot_id": args.robot_id,
        "piper_identity": args.piper_identity,
        "camera_serials": camera_serials,
        "official_rekep_commit": EXPECTED_OFFICIAL_REKEP_COMMIT,
        "bindings": bindings,
        "artifacts": artifacts,
    }
    document = create_signed_artifact(
        args.bundle_type, "{}-{}".format(
            args.robot_id, args.release_counter), payload, evidence,
        args.operator, args.private_key)
    output.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
