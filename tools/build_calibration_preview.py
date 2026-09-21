#!/usr/bin/env python3
"""Build 20 display-only IK candidates from archived hand-eye robot poses."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import yaml

from rekpiper_calibration.robot_world_handeye import check_pose_diversity
from rekpiper_calibration.trajectory_preview import JOINT_NAMES, validate_preview
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True,
                        help="Archived eye-to-hand YAML; repeat for another camera")
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    urdf = Path(__file__).resolve().parents[1] / "src/piper_description/urdf/piper_description.urdf"
    solver = PiperURDFIKSolver.from_urdf_xml(
        urdf.read_text(), "base_link", "link6", JOINT_NAMES, 1e-4, 1e-3)
    dataset_path = Path(args.validation_dataset).resolve()
    dataset = yaml.safe_load(dataset_path.read_text())
    holdout = [item for item in dataset["samples"] if item["split"] == "validation"]
    seeds = []
    for item in holdout:
        state = item["joint_states_single"]
        mapping = dict(zip(state["names"], state["positions_rad"]))
        seeds.append(np.asarray([mapping[name] for name in JOINT_NAMES]))
    if not seeds:
        raise ValueError("validation joint feedback required for IK initialization")
    seed = seeds[0]
    poses, rejected, source_records = [], [], []
    for source_name in args.source:
        path = Path(source_name).resolve()
        source = yaml.safe_load(path.read_text())
        source_records.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for sample in source["samples"]:
            if len(poses) == 20:
                break
            target = np.asarray(sample["base_to_tool_matrix_4x4"], dtype=float)
            candidates = [solver.solve(target, 600, q, orientation_weight=0.2)
                          for q in [seed] + seeds]
            good = [result for result in candidates if result.success]
            if not good:
                rejected.append({"source": str(path), "index": sample["index"], "reason": "IK_tolerance"})
                continue
            result = min(good, key=lambda value: np.linalg.norm(value.cspace_position[:6]-seed))
            q = result.cspace_position[:6]
            actual = solver.forward(q)
            existing = [np.asarray(item["base_T_link6"]) for item in holdout + poses]
            # Keep candidates apart from the six holdout poses and one another.
            if any(np.linalg.norm(actual[:3, 3]-other[:3, 3]) < 0.010
                   and solver._pose_errors(actual, other)[1] < np.deg2rad(5)
                   for other in existing):
                rejected.append({"source": str(path), "index": sample["index"], "reason": "near_existing_pose"})
                continue
            poses.append({
                "id": len(poses)+1, "split": "optimization",
                "source": str(path), "source_sample_index": sample["index"],
                "positions_rad": q.tolist(), "base_T_link6": actual.tolist(),
                "source_target_position_error_m": result.position_error,
                "source_target_rotation_error_rad": result.rotation_error,
            })
            seed = q
    plan = {
        "schema_version": 1, "status": "PREVIEW_ONLY", "hardware_execution_allowed": False,
        "base_frame": "base_link", "tip_frame": "link6", "joint_names": JOINT_NAMES,
        "urdf_sha256": hashlib.sha256(urdf.read_text().encode()).hexdigest(),
        "sources": source_records, "validation_dataset": str(dataset_path),
        "validation_dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "checks_pending": ["joint_path_collision", "board_and_mount_geometry", "dual_camera_visibility",
                           "site_motion_acceptance", "live_start_connection", "controller_timing"],
        "poses": poses, "rejected": rejected,
    }
    validate_preview(plan, solver)
    plan["pose_diversity"] = check_pose_diversity(
        [pose["base_T_link6"] for pose in poses], [0.05, 0.05, 0.03], 20.0, 20)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        yaml.safe_dump(plan, stream, sort_keys=False, allow_unicode=True)
    print("20 PREVIEW_ONLY candidates written to", args.output)
    print("Rejected:", len(rejected), "Diversity:", plan["pose_diversity"])


if __name__ == "__main__":
    main()
