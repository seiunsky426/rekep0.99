#!/usr/bin/env python3
"""Solve a complete dual-camera schema-v2 dataset without activating TF."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from rekpiper_calibration.checkerboard_calibration import matrix_to_quaternion_xyzw
from rekpiper_calibration.eye_to_hand import (
    CAMERAS, EyeToHandError, solve_dual_dataset)


def candidate(camera, report_path, report):
    matrix = np.asarray(report["results"][camera]["base_T_link"], dtype=float)
    quaternion = matrix_to_quaternion_xyzw(matrix)
    return {
        "schema_version": 1,
        "status": "PENDING_REVIEW",
        "publish_tf_allowed": False,
        "precision_operation_allowed": False,
        "base_absolute_accuracy": "UNVERIFIED",
        "logical_name": camera,
        "role": "fixed_eye_to_hand",
        "parent_frame": "base_link",
        "child_frame": camera + "_link",
        "base_T_link": matrix.tolist(),
        "translation_m": dict(zip(("x", "y", "z"), matrix[:3, 3].tolist())),
        "rotation_xyzw": dict(zip(("x", "y", "z", "w"), quaternion.tolist())),
        "validation_report": str(report_path),
        "dataset_sha256": report["dataset_sha256"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output_directory")
    args = parser.parse_args()
    try:
        output = Path(args.output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        report = solve_dual_dataset(args.dataset)
        report_path = output / "calibration_report.yaml"
        report_path.write_text(
            yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
        for name in CAMERAS:
            (output / "{}_extrinsics_candidate.yaml".format(name)).write_text(
                yaml.safe_dump(candidate(name, report_path, report),
                               sort_keys=False), encoding="utf-8")
        print(json.dumps({"success": True, "report": str(report_path)},
                         sort_keys=True))
        return 0
    except (OSError, ValueError, yaml.YAMLError, EyeToHandError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
