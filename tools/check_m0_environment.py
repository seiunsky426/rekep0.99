#!/usr/bin/env python3
"""Verify M0.1 paths, locked packages, ROS imports, and a CUDA calculation.

Run after sourcing the workspace setup.bash. This check starts no ROS nodes
and does not validate vendor models, calibration, or hardware acceptance.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import site
import subprocess
import sys

from packaging.specifiers import SpecifierSet

from verify_runtime import CORE_MODULES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime = Path(os.environ.get("REKPIPER_RUNTIME_ROOT", root / "runtime"))
    errors = []
    report = {
        "scope": "M0.1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "python": sys.version,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "user_site_enabled": site.ENABLE_USER_SITE,
        "paths": {}, "locks": {}, "packages": {}, "imports": {},
    }
    if sys.version_info[:2] != (3, 8):
        errors.append("python_not_3_8")
    if os.environ.get("ROS_DISTRO") != "noetic":
        errors.append("ros_not_noetic")
    report["ros_distro"] = os.environ.get("ROS_DISTRO")
    report["os_release"] = Path("/etc/os-release").read_text()
    if 'VERSION_ID="20.04"' not in report["os_release"]:
        errors.append("os_not_ubuntu_20_04")
    if Path(sys.prefix).resolve() != runtime.resolve():
        errors.append("python_outside_configured_runtime")
    if site.ENABLE_USER_SITE:
        errors.append("user_site_enabled")
    if os.environ.get("REKPIPER_ROOT") != str(root):
        errors.append("wrong_workspace_environment")
    for name in (
            "REKPIPER_ROOT", "REKPIPER_RUNTIME_ROOT", "REKPIPER_MODEL_ROOT",
            "REKPIPER_VENDOR_ROOT", "REKPIPER_DATA_ROOT",
            "REKPIPER_SITE_CONFIG_ROOT", "REKPIPER_TRUST_ROOT",
            "REKEP_OFFICIAL_ROOT", "CUDA_HOME"):
        value = os.environ.get(name, "")
        exists = bool(value) and Path(value).is_dir()
        report["paths"][name] = {
            "value": value, "exists": exists,
            "resolved": str(Path(value).resolve()) if value else "",
        }
        if not exists:
            errors.append("missing_directory:" + name)
    for filename in ("requirements.lock", "requirements.platform.lock",
                     "requirements.models.lock"):
        lock = root / filename
        if not lock.is_file():
            errors.append("missing_lock:" + filename)
            continue
        report["locks"][filename] = hashlib.sha256(lock.read_bytes()).hexdigest()
        for name, expected in re.findall(
                r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", lock.read_text(), re.M):
            try:
                actual = importlib.metadata.version(name)
                matches = actual in SpecifierSet("==" + expected)
            except importlib.metadata.PackageNotFoundError:
                actual, matches = None, False
            report["packages"][name] = {
                "expected": expected, "actual": actual, "matches": matches}
            if not matches:
                errors.append("locked_version_mismatch:" + name)
    modules = list(CORE_MODULES.values()) + [
        "torch", "torchvision", "rospy", "rospkg", "catkin_pkg", "em",
        "cv_bridge", "tf2_ros", "piper_sdk"]
    for name in modules:
        try:
            module = importlib.import_module(name)
            origin = str(Path(module.__file__).resolve())
            report["imports"][name] = origin
            if name in CORE_MODULES.values() or name in ("torch", "torchvision"):
                if runtime.resolve() not in Path(origin).parents:
                    errors.append("import_outside_runtime:" + name)
        except Exception as exc:
            errors.append("import_failed:{}:{}".format(name, exc))
    try:
        import rospkg
        packages = rospkg.RosPack()
        report["ros_packages"] = {
            manifest.parent.name: packages.get_path(manifest.parent.name)
            for manifest in sorted((root / "src").glob("*/package.xml"))}
        for name, path in report["ros_packages"].items():
            if Path(path).resolve() != root / "src" / name:
                errors.append("wrong_ros_package:" + name)
    except Exception as exc:
        errors.append("ros_package_resolution_failed:" + str(exc))
    for tool in ("nvcc", "ninja", "catkin_make"):
        report[tool] = shutil.which(tool)
        if not report[tool]:
            errors.append("tool_not_on_path:" + tool)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable to this process")
        matrix = torch.ones((16, 16), device="cuda")
        value = (matrix @ matrix).sum().item()
        torch.cuda.synchronize()
        report["cuda"] = {
            "torch": torch.__version__, "compiled_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0), "matmul_sum": value}
        if torch.version.cuda != "12.1":
            errors.append("torch_cuda_not_12_1")
        if value != 4096.0:
            errors.append("cuda_calculation_failed")
    except Exception as exc:
        errors.append("cuda_check_failed:" + str(exc))
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    report["pip_check"] = result.stdout.strip()
    if result.returncode:
        errors.append("pip_dependency_check_failed")
    report.update(errors=errors, m0_1_complete=not errors)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
