#!/usr/bin/env python3
"""Verify the local M0 asset inventory; this is not a signed site approval."""

import argparse
import hashlib
import json
import os
from pathlib import Path


REQUIRED_ASSETS = {
    "models/sam_vit_h_4b8939.pth",
    "models/dinov2_vits14_reg4_pretrain.pth",
    "models/cutie-base-mega.pth",
    "vendor/anygrasp_sdk/grasp_detection/checkpoint_detection.tar",
    "vendor/anygrasp_sdk/grasp_detection/gsnet.so",
    "vendor/nvblox_torch/src/nvblox_torch/bin/libpy_nvblox.so",
    "native/nvblox/lib/libnvblox_lib.so",
    "vendor/MinkowskiEngine/build/lib.linux-x86_64-3.8/MinkowskiEngineBackend/_C.cpython-38-x86_64-linux-gnu.so",
    "vendor/anygrasp_sdk/pointnet2/build/lib.linux-x86_64-3.8/pointnet2/_ext.cpython-38-x86_64-linux-gnu.so",
    "lib/python3.8/site-packages/grasp_nms.cpython-38-x86_64-linux-gnu.so",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime = Path(os.environ.get("REKPIPER_RUNTIME_ROOT", root / "runtime"))
    manifest = runtime / "site-config/M0_ASSETS.lock.json"
    inventory = json.loads(manifest.read_text())
    entries = inventory["files"]
    errors = ["required_asset_not_locked:" + name
              for name in sorted(REQUIRED_ASSETS - set(entries))]
    for name, expected in entries.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append("invalid_relative_path:" + name)
            continue
        path = runtime / relative
        if not path.is_file():
            errors.append("missing:" + name)
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            errors.append("hash_mismatch:" + name)
    report = {"scope": "M0.2 local provenance", "manifest": str(manifest),
              "checked_files": len(entries), "errors": errors,
              "passed": not errors, "hardware_execution_authorized": False}
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
