#!/usr/bin/env python3
"""Audit a Git checkout or source archive without requiring Git metadata."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import yaml


REQUIRED = (
    ".catkin_workspace", ".gitignore", "README.md", "docs", "requirements.in",
    "requirements.lock",
    "setup.bash", "src", "third_party", "tools",
)


def has_git_metadata(root):
    """Return true only for plausible checkout/worktree metadata."""
    marker = root / ".git"
    if marker.is_file():
        try:
            return marker.read_text(encoding="utf-8").lstrip().startswith("gitdir:")
        except OSError:
            return False
    return marker.is_dir() and (marker / "HEAD").is_file()


def audit(root):
    errors = ["missing:" + name for name in REQUIRED
              if not (root / name).exists()]
    lock = root / "third_party" / "UPSTREAM.lock.yaml"
    if not (root / "third_party" / "VENDOR.lock.yaml").is_file():
        errors.append("vendor_lock_missing")
    try:
        manifest = yaml.safe_load(lock.read_text(encoding="utf-8"))
        if manifest["rekep"]["commit"] != \
                "63c43fdba60354980258beaeb8a7d48e088e1e3e":
            errors.append("official_commit_mismatch")
        official = root / "third_party" / "ReKep"
        for name, expected in manifest["rekep"]["files"].items():
            source = official / name
            if not source.is_file():
                errors.append("official_file_missing:" + name)
            elif hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                errors.append("official_hash_mismatch:" + name)
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        errors.append("upstream_lock_invalid")
    source_space = root / "src"
    nested_manifests = sorted((source_space / "src").glob("*/package.xml"))
    if nested_manifests:
        errors.append("nested_source_space_not_allowed")
    package_manifests = sorted(source_space.glob("*/package.xml"))
    if not package_manifests:
        errors.append("ros_packages_missing")
    package_index = {
        path.parent.name: path.parent for path in package_manifests
    }
    reference = re.compile(r"\$\(find\s+([^)]+)\)([^\"'<\s]*)")
    launch_files = sorted(
        launch
        for package_path in package_index.values()
        for launch in package_path.rglob("*.launch")
    )
    for launch in launch_files:
        try:
            text = launch.read_text(encoding="utf-8")
        except OSError:
            errors.append("launch_unreadable:" + str(launch.relative_to(root)))
            continue
        for package, suffix in reference.findall(text):
            if package not in package_index:
                # System ROS packages are resolved by rosdep, not this archive.
                continue
            if "$" in suffix:
                continue
            target = (package_index[package] / suffix.lstrip("/")).resolve()
            try:
                target.relative_to(root.resolve())
            except ValueError:
                errors.append("launch_path_escape:" + str(launch.relative_to(root)))
                continue
            if suffix and not target.exists():
                errors.append(
                    "launch_path_missing:{}:{}".format(
                        launch.relative_to(root), suffix))
    # Runtime bytecode caches are intentionally ignored.  They can appear as
    # soon as an archive is tested and are not part of the delivery contract;
    # requiring an entirely cache-free working tree made the audit depend on
    # test execution order.  Release packaging is responsible for excluding
    # them from the produced archive.
    return {"clean": not errors, "errors": errors,
            "source_archive": not has_git_metadata(root)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    report = audit(Path(args.source_root).resolve())
    print(json.dumps(report, sort_keys=True))
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
