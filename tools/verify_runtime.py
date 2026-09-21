#!/usr/bin/env python3
"""Verify locked core imports and external vendor/model provenance."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import yaml


CORE_MODULES = {
    "cryptography": "cryptography", "numba": "numba", "numpy": "numpy",
    "open3d": "open3d", "opencv-contrib-python": "cv2", "PyYAML": "yaml",
    "requests": "requests", "scikit-learn": "sklearn", "scipy": "scipy",
    "trimesh": "trimesh",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_commit(root):
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return ""
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = root / ".git" / value[5:]
        return ref.read_text(encoding="utf-8").strip() if ref.is_file() else ""
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--vendor-lock", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    errors, packages, vendors = [], {}, {}
    for distribution, module_name in CORE_MODULES.items():
        try:
            module = importlib.import_module(module_name)
            packages[distribution] = {
                "version": importlib.metadata.version(distribution),
                "origin": str(Path(module.__file__).resolve()),
            }
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            errors.append("core_import_failed:{}:{}".format(distribution, exc))
    lock_path = Path(args.vendor_lock).expanduser().resolve()
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    if lock.get("schema_version") != 1 or not bool(lock.get("approved", False)):
        errors.append("vendor_lock_not_site_approved")
    vendor_root = Path(args.vendor_root).expanduser().resolve()
    model_root = Path(args.model_root).expanduser().resolve()
    for name, spec in (lock.get("vendors") or {}).items():
        root = (vendor_root / str(spec.get("relative_path", ""))).resolve()
        try:
            root.relative_to(vendor_root)
        except ValueError:
            errors.append("vendor_path_escape:" + name)
            continue
        expected_commit = str(spec.get("commit", ""))
        actual_commit = tree_commit(root)
        if (len(expected_commit) != 40 or actual_commit != expected_commit):
            errors.append("vendor_commit_mismatch:" + name)
        assets = {}
        imported = {}
        if root.is_dir():
            sys.path.insert(0, str(root))
        for module_name in spec.get("required_imports", []):
            try:
                module = importlib.import_module(str(module_name))
                imported[str(module_name)] = str(
                    Path(module.__file__).resolve())
            except (ImportError, OSError, TypeError) as exc:
                errors.append("vendor_import_failed:{}:{}:{}".format(
                    name, module_name, exc))
        for relative, expected_hash in (spec.get("assets") or {}).items():
            candidates = [root / relative, model_root / relative]
            source = next((value for value in candidates if value.is_file()), None)
            actual = sha256(source) if source else ""
            assets[relative] = actual
            if actual != expected_hash:
                errors.append("vendor_asset_mismatch:{}:{}".format(name, relative))
        vendors[name] = {"root": str(root), "commit": actual_commit,
                         "imports": imported, "assets": assets}
    report = {
        "schema_version": 1,
        "python": sys.version,
        "python_abi": "cp{}{}".format(sys.version_info.major, sys.version_info.minor),
        "platform": platform.platform(),
        "packages": packages,
        "vendors": vendors,
        "vendor_lock_sha256": sha256(lock_path),
        "errors": errors,
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["runtime_fingerprint_sha256"] = hashlib.sha256(
        encoded.encode("utf-8")).hexdigest()
    output = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        target = Path(args.output).expanduser().resolve()
        if target.exists():
            raise SystemExit("refusing to overwrite runtime fingerprint")
        target.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
