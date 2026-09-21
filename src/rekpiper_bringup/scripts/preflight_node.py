#!/usr/bin/env python3
"""Fail closed before realtime or autonomous components can become ready."""

import hashlib
from pathlib import Path
import json
import importlib.util
import os
import sys

import yaml

import rospy
from std_msgs.msg import String

from rekpiper_acceptance import AcceptanceError, validate_release_bundle
from rekpiper_execution.runtime_acceptance import (
    RuntimeAcceptanceError, validate_evidence)
from rekpiper_planning.solver_acceptance import (
    SolverAcceptanceError, validate_acceptance)


def _check_manifest(path):
    source = Path(path).resolve()
    if not source.is_file():
        return ["model_manifest_missing"]
    root = source.parent
    errors = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            errors.append("model_manifest_invalid_line")
            continue
        expected, name = parts
        asset = root / name.strip()
        if not asset.is_file():
            errors.append("missing_model:" + name.strip())
        elif hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
            errors.append("model_hash_mismatch:" + name.strip())
    return errors


def _check_runtime_imports(source_root, vendor_root):
    vendor = Path(vendor_root).resolve()
    for relative in (
            "segment_anything", "curobo/src", "nvblox_torch/src", "Cutie",
            "dinov2"):
        source = str(vendor / relative)
        if source not in sys.path:
            sys.path.insert(0, source)
    for source in (
            vendor / "MinkowskiEngine" / "build" / "lib.linux-x86_64-3.8",
            vendor / "graspnetAPI",
            vendor / "anygrasp_sdk" / "pointnet2" / "build" /
            "lib.linux-x86_64-3.8"):
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    required = (
        "torch", "open3d", "scipy", "numba", "segment_anything",
        "dinov2", "cutie", "curobo", "nvblox_torch",
        "MinkowskiEngine", "graspnetAPI")
    return ["missing_python_runtime:" + name for name in required
            if importlib.util.find_spec(name) is None]


def main():
    rospy.init_node("rekpiper_preflight")
    mode = str(rospy.get_param("~mode", "shadow"))
    if mode not in ("shadow", "autonomous"):
        raise rospy.ROSInitException("mode must be shadow or autonomous")
    manifest = Path(rospy.get_param("~model_manifest")).resolve()
    errors = _check_manifest(manifest)
    if mode in ("shadow", "autonomous"):
        errors.extend(_check_runtime_imports(
            rospy.get_param("~source_root"),
            rospy.get_param("~vendor_root")))
        try:
            import torch
        except ImportError:
            torch = None
        if torch is None or not torch.cuda.is_available():
            errors.append("torch_cuda_unavailable")
    if mode == "autonomous":
        try:
            bundle_path = rospy.get_param("~release_bundle", "")
            release = validate_release_bundle(
                bundle_path,
                rospy.get_param("~acceptance_public_key", ""),
                rospy.get_param("~minimum_release_counter", ""),
                rospy.get_param("~robot_id", ""))
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
            rospy.set_param(
                "/rekpiper/acceptance/release_counter",
                int(release["payload"]["release_counter"]))
            rospy.set_param(
                "/rekpiper/acceptance/release_bundle_sha256",
                hashlib.sha256(Path(bundle_path).read_bytes()).hexdigest())
        except (AcceptanceError, RuntimeAcceptanceError,
                SolverAcceptanceError, OSError, KeyError, TypeError,
                ValueError) as exc:
            errors.append("signed_release_rejected:" + str(exc))
    publisher = rospy.Publisher(
        "/rekpiper/preflight", String, queue_size=1, latch=True)
    payload = {"mode": mode, "ready": not errors, "errors": errors}
    publisher.publish(String(data=json.dumps(payload, sort_keys=True)))
    if errors:
        raise rospy.ROSInitException("preflight failed: " + ",".join(errors))
    rospy.loginfo("Rekpiper %s preflight passed", mode)
    rospy.spin()


if __name__ == "__main__":
    main()
