#!/usr/bin/env python3
"""Sort taught poses into height rows and interpolate a display-only 20-point path."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import yaml

from rekpiper_calibration.robot_world_handeye import pose_span_metrics
from rekpiper_calibration.trajectory_preview import JOINT_NAMES, playback_samples, validate_preview
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver


def ordered_rows(waypoints, row_height_m):
    """Anchor each height row at its lowest point; scan +Y to -Y within it."""
    remaining = sorted(waypoints, key=lambda p: (p["translation_m"][2], p["id"]))
    rows = []
    while remaining:
        ceiling = remaining[0]["translation_m"][2] + row_height_m
        row = [p for p in remaining if p["translation_m"][2] <= ceiling]
        remaining = remaining[len(row):]
        rows.append(sorted(row, key=lambda p: (-p["translation_m"][1], p["id"])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waypoints", required=True)
    parser.add_argument("--calibration-source", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = Path(args.waypoints).resolve()
    source_bytes = source.read_bytes()
    data = yaml.safe_load(source_bytes)
    points = data["waypoints"]
    if data["joint_names"] != JOINT_NAMES or not 2 <= len(points) <= 20:
        raise ValueError("requires 2..20 taught poses in joint1..joint6 order")
    urdf = Path(__file__).resolve().parents[1] / "src/piper_description/urdf/piper_description.urdf"
    xml = urdf.read_text()
    solver = PiperURDFIKSolver.from_urdf_xml(xml, "base_link", "link6", JOINT_NAMES)
    for p in points:
        q = np.asarray(p["positions_rad"])
        if (q.shape != (6,) or not np.all(np.isfinite(q))
                or np.any(q < solver._lower) or np.any(q > solver._upper)
                or not np.allclose(solver.forward(q), p["base_T_link6"], atol=1e-7, rtol=0)
                or not np.allclose(np.asarray(p["base_T_link6"])[:3, 3], p["translation_m"], atol=1e-7)):
            raise ValueError("invalid taught pose {}".format(p["id"]))
    rows = ordered_rows(points, 0.020)
    ordered = [p for row in rows for p in row]
    joints = np.asarray([p["positions_rad"] for p in ordered])
    # Subdivide the segment with the largest remaining single-joint change.
    distances = np.max(np.abs(np.diff(joints, axis=0)), axis=1)
    divisions = np.ones(len(distances), dtype=int)
    for _ in range(20-len(points)):
        divisions[np.argmax(distances/divisions)] += 1
    poses = []

    def append(q, kind, source_ids, fraction):
        index = len(poses)+1
        poses.append({
            "id": index, "kind": kind, "source_waypoint_ids": source_ids,
            "interpolation_fraction": float(fraction),
            "label": "{}:{}{}".format(index, "T" if kind == "taught" else "I",
                                      source_ids[0] if kind == "taught" else ""),
            "positions_rad": q.tolist(), "positions_deg": np.rad2deg(q).tolist(),
            "base_T_link6": solver.forward(q).tolist(),
            "camera_evidence": "recorded_at_teaching" if kind == "taught" else "UNVERIFIED",
        })

    append(joints[0], "taught", [ordered[0]["id"]], 0)
    for i, count in enumerate(divisions):
        for step in range(1, int(count)):
            u = step/float(count)
            append((1-u)*joints[i]+u*joints[i+1], "interpolated",
                   [ordered[i]["id"], ordered[i+1]["id"]], u)
        append(joints[i+1], "taught", [ordered[i+1]["id"]], 0)
    sources = []
    for name in args.calibration_source:
        path = Path(name).resolve()
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    plan = {
        "schema_version": 1, "status": "PREVIEW_ONLY", "hardware_execution_allowed": False,
        "base_frame": "base_link", "tip_frame": "link6", "joint_names": JOINT_NAMES,
        "urdf_sha256": hashlib.sha256(xml.encode()).hexdigest(), "sources": sources,
        "teaching_source": str(source), "teaching_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "ordering": {"row_height_m": 0.020, "rows_low_to_high": [[p["id"] for p in row] for row in rows],
                     "within_row": "descending base_link Y (+Y left, -Y right)",
                     "height_reference": "link6 origin; height may vary within a row"},
        "interpolation": "affine joint subdivision; quintic joint playback between the 20 stops",
        "taught_count": len(points), "interpolated_count": 20-len(points),
        "checks_pending": ["interpolated_pose_camera_visibility", "joint_path_collision",
                           "full_arm_and_mount_geometry", "live_start_connection",
                           "site_motion_acceptance", "controller_time_parameterization",
                           "actual_calibration_image_capture"],
        "poses": poses,
    }
    q20 = validate_preview(plan, solver)
    dense = np.asarray([q for _, _, q in playback_samples(q20)])
    if np.any(dense < solver._lower-1e-12) or np.any(dense > solver._upper+1e-12):
        raise ValueError("interpolated path exceeds joint limits")
    plan["diagnostics"] = {
        "sample_count": len(dense), "joint_limits_pass": True,
        "max_adjacent_joint_change_deg": float(np.rad2deg(np.max(np.abs(np.diff(q20, axis=0))))),
        "minimum_joint_limit_margin_deg": float(np.rad2deg(np.min(np.minimum(
            dense-solver._lower, solver._upper-dense)))),
        "pose_span": pose_span_metrics([p["base_T_link6"] for p in poses]),
    }
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "teaching_snapshot.yaml").write_bytes(source_bytes)
    (out / "plan.yaml").write_text(yaml.safe_dump(plan, sort_keys=False, allow_unicode=True))
    with (out / "points.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["preview_id", "kind", "source_ids", "fraction", "x_mm", "y_mm", "z_mm"] + JOINT_NAMES)
        for p in poses:
            writer.writerow([p["id"], p["kind"], "-".join(map(str, p["source_waypoint_ids"])),
                             p["interpolation_fraction"]] +
                            (np.asarray(p["base_T_link6"])[:3, 3]*1000).tolist() + p["positions_deg"])
    print(yaml.safe_dump({"output": str(out), "ordering": plan["ordering"],
                          "diagnostics": plan["diagnostics"]}, sort_keys=False))


if __name__ == "__main__":
    main()
