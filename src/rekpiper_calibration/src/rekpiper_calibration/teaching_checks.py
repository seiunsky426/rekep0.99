"""Stopped-pose checks shared by the persistent teaching recorder and tests."""

import numpy as np


def check_teaching_window(joints, statuses, camera_statuses, now_s):
    if len(joints) < 50:
        raise ValueError("waiting_for_one_second_joint_window")
    stamps = np.asarray([row["stamp_s"] for row in joints], dtype=float)
    values = np.asarray([row["positions_rad"] for row in joints], dtype=float)
    if (values.shape != (len(joints), 6) or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(stamps)) or not np.isfinite(now_s)):
        raise ValueError("invalid_joint_feedback")
    if np.any(np.diff(stamps) <= 0) or np.max(np.diff(stamps)) > 0.10:
        raise ValueError("joint_feedback_gap_or_repeated_stamp")
    if stamps[-1]-stamps[0] < 0.90:
        raise ValueError("waiting_for_one_second_joint_window")
    if not -.05 <= now_s-stamps[-1] <= .25:
        raise ValueError("joint_feedback_stale")
    span = np.ptp(values, axis=0)
    if np.max(span) > .002:
        raise ValueError("robot_not_stationary")
    for rows, timeout, label in ((statuses, .25, "robot_status"),
                                  (camera_statuses, .30, "camera_status")):
        if len(rows) < 3 or not -.05 <= now_s-rows[-1]["received_ros_s"] <= timeout:
            raise ValueError(label+"_stale_or_missing")
        times = np.asarray([r["received_ros_s"] for r in rows], dtype=float)
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0):
            raise ValueError(label+"_time_invalid")
        if times[-1]-times[0] < .7 or np.max(np.diff(times)) > timeout:
            raise ValueError(label+"_window_incomplete")
    if any(not row["valid"] for row in statuses):
        raise ValueError("robot_not_in_normal_teaching_mode")
    for row in camera_statuses:
        data = row["status"]
        quality = data.get("quality", {})
        if (data.get("state") != "READY_TO_CAPTURE"
                or set(data.get("active_cameras", [])) != {"rs1", "rs3"}
                or any(name not in quality or quality[name].get("reason")
                       for name in ("rs1", "rs3"))):
            raise ValueError("cameras_not_ready: " + str(data.get("reason") or quality))
    median = np.median(values, axis=0)
    index = int(np.argmin(np.linalg.norm(values-median, axis=1)))
    return {"representative_index": index, "joint_span_rad": span.tolist(),
            "sample_count": len(joints), "duration_s": float(stamps[-1]-stamps[0])}
