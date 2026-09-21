#!/usr/bin/env python3
"""Offline dependency/model/release preflight; never initializes ROS or CAN."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from rekpiper_acceptance import AcceptanceError, validate_release_bundle
from rekpiper_execution.runtime_acceptance import (
    RuntimeAcceptanceError, validate_evidence)
from rekpiper_planning.solver_acceptance import (
    SolverAcceptanceError, validate_acceptance)


def model_errors(path):
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return ["model_manifest_missing"]
    errors = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            errors.append("model_manifest_invalid_line")
            continue
        expected, relative = parts
        asset = (source.parent / relative.strip()).resolve()
        try:
            asset.relative_to(source.parent)
        except ValueError:
            errors.append("model_path_escape:" + relative)
            continue
        if not asset.is_file():
            errors.append("missing_model:" + relative)
        elif hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
            errors.append("model_hash_mismatch:" + relative)
    return errors


def runtime_errors(source_root, vendor_root):
    vendor = Path(vendor_root).expanduser().resolve()
    roots = [
        vendor / "segment_anything", vendor / "curobo/src",
        vendor / "nvblox_torch/src", vendor / "Cutie", vendor / "dinov2",
        vendor / "MinkowskiEngine/build/lib.linux-x86_64-3.8",
        vendor / "graspnetAPI",
        vendor / "anygrasp_sdk/pointnet2/build/lib.linux-x86_64-3.8",
    ]
    sys.path[:0] = [str(path) for path in roots]
    required = (
        "torch", "open3d", "scipy", "numba", "segment_anything",
        "dinov2", "cutie", "curobo", "nvblox_torch", "MinkowskiEngine",
        "graspnetAPI")
    errors = ["missing_python_runtime:" + name for name in required
              if importlib.util.find_spec(name) is None]
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("torch_cuda_unavailable")
    except ImportError:
        pass
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("shadow", "autonomous"), required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--vendor-root", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--release-bundle", default="")
    parser.add_argument("--public-key", default="")
    parser.add_argument("--minimum-counter", default="")
    parser.add_argument("--robot-id", default="piper-rekpiper")
    args = parser.parse_args()
    errors = model_errors(args.model_manifest)
    errors.extend(runtime_errors(args.source_root, args.vendor_root))
    if args.mode == "autonomous":
        try:
            release = validate_release_bundle(
                args.release_bundle, args.public_key, args.minimum_counter,
                args.robot_id)
            artifacts = release["verified_artifacts"]
            validate_acceptance(
                artifacts["paper_real_solver"]["payload"],
                require_approved=True)
            validate_evidence(
                artifacts["runtime_performance_software"]["payload"],
                expected_profile="software")
            if os.environ.get(
                    "REKPIPER_ACCEPTANCE_BUNDLE_TYPE",
                    "release_bundle") == "release_bundle":
                validate_evidence(
                    artifacts["runtime_performance_hardware"]["payload"],
                    expected_profile="hardware_hold")
        except (AcceptanceError, RuntimeAcceptanceError,
                SolverAcceptanceError, KeyError, TypeError, ValueError) as exc:
            errors.append("signed_release_rejected:" + str(exc))
    print(json.dumps({"ready": not errors, "mode": args.mode,
                      "errors": errors}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
