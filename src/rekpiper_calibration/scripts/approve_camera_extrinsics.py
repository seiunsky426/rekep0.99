#!/usr/bin/env python3
"""Export accepted fixed-camera extrinsics after explicit operator review."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import yaml

from rekpiper_calibration.eye_to_hand import (
    CAMERAS, EyeToHandError, approve_dual_report)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("output_directory")
    parser.add_argument("--operator", required=True)
    args = parser.parse_args()
    try:
        output = Path(args.output_directory).expanduser().resolve()
        if "active_fixed_camera_extrinsics" in output.parts:
            raise EyeToHandError(
                "approval tool never overwrites the active extrinsics directory")
        if output.exists():
            raise EyeToHandError("approval output directory already exists")
        report = yaml.safe_load(
            Path(args.report).expanduser().read_text(encoding="utf-8"))
        accepted = approve_dual_report(report, args.operator)
        output.mkdir(parents=True, exist_ok=False)
        accepted_report_path = output / "accepted_report.yaml"
        accepted_report = deepcopy(report)
        accepted_report["status"] = "ACCEPTED"
        accepted_report["approval"] = {
            "operator": str(args.operator).strip(),
            "approved_utc": datetime.now(timezone.utc).isoformat(),
        }
        accepted_report_path.write_text(
            yaml.safe_dump(accepted_report, sort_keys=False), encoding="utf-8")
        accepted_report_sha = sha256_file(str(accepted_report_path))
        dataset_source = Path(report["dataset_path"]).expanduser().resolve()
        dataset_copy = output / ("calibration_dataset" + dataset_source.suffix)
        shutil.copy2(str(dataset_source), str(dataset_copy))
        if sha256_file(str(dataset_copy)) != report["dataset_sha256"]:
            raise EyeToHandError("copied calibration dataset hash mismatch")
        for name in CAMERAS:
            accepted[name]["validation_report"] = accepted_report_path.name
            accepted[name]["validation_report_sha256"] = accepted_report_sha
            artifact = deepcopy(accepted[name])
            artifact["status"] = "OPERATOR_ACCEPTED"
            artifact["approval"] = {
                "operator": str(args.operator).strip(),
                "approved_utc": datetime.now(timezone.utc).isoformat(),
            }
            artifact["evidence"] = {
                "accepted_report": accepted_report_path.name,
                "accepted_report_sha256": accepted_report_sha,
                "calibration_dataset": dataset_copy.name,
                "calibration_dataset_sha256": sha256_file(str(dataset_copy)),
            }
            (output / "{}_extrinsics.yaml".format(name)).write_text(
                yaml.safe_dump(artifact, sort_keys=False), encoding="utf-8")
        print(json.dumps({"success": True, "output": str(output)},
                         sort_keys=True))
        return 0
    except (OSError, ValueError, yaml.YAMLError, EyeToHandError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
