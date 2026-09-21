"""Immutable, explicitly stepped operator-supervised calibration sessions."""

import hashlib
from pathlib import Path

import numpy as np
import yaml

from rekpiper_calibration.trajectory_preview import JOINT_NAMES, validate_preview
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver
from rekpiper_execution.trajectory import normalize_feedback_to_joint_limits


DEFAULT_LIMITS = {
    "driver_speed_percent": 5, "velocity_deg_s": 3.0,
    "acceleration_deg_s2": 6.0, "jerk_deg_s3": 12.0,
    "tip_speed_m_s": 0.010, "rate_hz": 50.0,
}

GO_ZERO_PURPOSE = "GO_ZERO_RECOVERY"
GO_ZERO_BOUNDARY_TOLERANCE_RAD = np.deg2rad(3.0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def solver_from_file(path):
    return PiperURDFIKSolver.from_urdf_xml(
        Path(path).read_text(), "base_link", "link6", JOINT_NAMES)


def check_limits(limits):
    if set(limits) != set(DEFAULT_LIMITS):
        raise ValueError("all six speed settings are required")
    for key, maximum in DEFAULT_LIMITS.items():
        value = limits[key]
        if not np.isfinite(value) or not 0 < value <= maximum:
            raise ValueError("invalid or excessive limit: " + key)
    if int(limits["driver_speed_percent"]) != limits["driver_speed_percent"]:
        raise ValueError("driver percentage must be an integer")
    if limits["rate_hz"] != 50.0:
        raise ValueError("50 Hz command loop required")


def quintic(u):
    u = np.clip(u, 0.0, 1.0)
    return 10*u**3 - 15*u**4 + 6*u**5


def positions(segment, elapsed):
    q0, q1 = np.asarray(segment["start_rad"]), np.asarray(segment["goal_rad"])
    return q0 + quintic(elapsed/segment["duration_s"])*(q1-q0)


def make_segment(name, start, goal, solver, limits):
    check_limits(limits)
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    for q in (start, goal):
        if (q.shape != (6,) or not np.all(np.isfinite(q))
                or np.any(q < solver._lower) or np.any(q > solver._upper)):
            raise ValueError("invalid joint position for " + name)
    change = float(np.max(np.abs(goal-start)))
    duration = max(2.0, 1.875*change/np.deg2rad(limits["velocity_deg_s"]),
                   np.sqrt(5.774*change/np.deg2rad(limits["acceleration_deg_s2"])),
                   np.cbrt(60*change/np.deg2rad(limits["jerk_deg_s3"])))
    normalized = np.linspace(0, 1, 501)
    xyz = np.array([solver.forward(start+quintic(u)*(goal-start))[:3, 3]
                    for u in normalized])
    # A 25% timing margin around the sampled Cartesian speed estimate.
    duration = max(duration, 1.25*np.max(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
                   *500/limits["tip_speed_m_s"])
    if np.any(xyz < [0.0, -0.4, 0.05]) or np.any(xyz > [0.7, 0.4, 0.7]):
        raise ValueError("link6 sampled path outside XYZ workspace: " + name)
    return {"name": name, "start_rad": start.tolist(), "goal_rad": goal.tolist(),
            "duration_s": float(duration), "dwell_s": 1.0,
            "sampled_minimum_xyz_m": xyz.min(axis=0).tolist(),
            "sampled_maximum_xyz_m": xyz.max(axis=0).tolist()}


def build_go_zero_session(start, solver, limits):
    """Build one slow, monotonic segment from measured feedback to six-axis zero.

    The installed arm reports a small mechanical/model zero disagreement on
    joint2 and joint3.  This recovery permits at most three degrees outside
    those two nominal URDF boundaries, and every commanded sample moves toward
    the legal zero boundary.
    """
    check_limits(limits)
    start = validate_go_zero_feedback(start, solver)

    goal = np.zeros(6, dtype=float)
    change = float(np.max(np.abs(start)))
    duration = max(
        2.0,
        1.875*change/np.deg2rad(limits["velocity_deg_s"]),
        np.sqrt(5.774*change/np.deg2rad(limits["acceleration_deg_s2"])),
        np.cbrt(60*change/np.deg2rad(limits["jerk_deg_s3"])))
    normalized = np.linspace(0, 1, 501)
    q = np.asarray([start+quintic(u)*(goal-start) for u in normalized])
    xyz = np.asarray([solver.forward(value)[:3, 3] for value in q])
    duration = max(
        duration,
        1.25*np.max(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
        * 500/limits["tip_speed_m_s"])
    if np.any(xyz < [0.0, -0.4, 0.05]) or np.any(xyz > [0.7, 0.4, 0.7]):
        raise ValueError("go-zero link6 sampled path is outside the reviewed XYZ workspace")
    return [{
        "name": "go_zero",
        "start_rad": start.tolist(),
        "goal_rad": goal.tolist(),
        "duration_s": float(duration),
        "dwell_s": 1.0,
        "sampled_minimum_xyz_m": xyz.min(axis=0).tolist(),
        "sampled_maximum_xyz_m": xyz.max(axis=0).tolist(),
    }]


def validate_go_zero_feedback(values, solver):
    """Accept nominal joints plus the measured joint2/joint3 zero offset."""
    start = np.asarray(values, dtype=float)
    if start.shape != (6,) or not np.all(np.isfinite(start)):
        raise ValueError("go-zero feedback is invalid")
    lower_error = np.maximum(solver._lower-start, 0.0)
    upper_error = np.maximum(start-solver._upper, 0.0)
    outside = np.maximum(lower_error, upper_error)
    permitted = np.zeros(6, dtype=float)
    permitted[1:3] = GO_ZERO_BOUNDARY_TOLERANCE_RAD
    if np.any(outside > permitted+1e-9):
        raise ValueError("go-zero start exceeds the three-degree joint2/joint3 recovery allowance")
    return start


def build_session(plan, start, solver, limits):
    q = validate_preview(plan, solver)
    start = np.asarray(start, dtype=float)
    delta = q[0]-start
    size = float(np.max(np.abs(delta)))
    if size < 1e-6:
        raise ValueError("already at first point; choose a reviewed nonzero trial direction")
    trial = start + delta*min(1.0, np.deg2rad(1)/size)
    result = [make_segment("trial", start, trial, solver, limits)]
    previous = trial
    for i, goal in enumerate(q):
        result.append(make_segment("point_{:02d}".format(i+1), previous, goal, solver, limits))
        previous = goal
    return result


class MotionSession:
    """Validate a prepared local session and its bound code/plan files."""

    def __init__(self, session_path, urdf_path):
        self.path = Path(session_path).resolve()
        self.data = yaml.safe_load(self.path.read_text())
        self.solver = solver_from_file(urdf_path)
        self.urdf_path = Path(urdf_path).resolve()
        if (self.data.get("schema_version") != 1
                or self.data.get("status") != "OPERATOR_SUPERVISED"
                or self.data.get("hardware_execution_allowed") is not True):
            raise ValueError("operator-supervised motion session required")
        check_limits(self.data["limits"])
        self.purpose = self.data.get("purpose", "CALIBRATION_20_POSE")
        expected_boundary_tolerance = (
            GO_ZERO_BOUNDARY_TOLERANCE_RAD
            if self.purpose == GO_ZERO_PURPOSE else 0.01)
        if not np.isclose(self.data.get("feedback_boundary_tolerance_rad", 0.01),
                          expected_boundary_tolerance, atol=1e-12, rtol=0):
            raise ValueError("fixed project feedback boundary tolerance required")
        raw = self.data.get("raw_start_rad", self.data["start_rad"])
        if self.purpose == GO_ZERO_PURPOSE:
            if not np.allclose(raw, self.data["start_rad"], atol=1e-10, rtol=0):
                raise ValueError("go-zero session must retain raw measured start")
            build_go_zero_session(raw, self.solver, self.data["limits"])
        elif not np.allclose(normalize_feedback_to_joint_limits(raw, .01),
                             self.data["start_rad"], atol=1e-10, rtol=0):
            raise ValueError("start normalization differs from the existing project boundary rule")
        if self.data["urdf_sha256"] != digest(urdf_path):
            raise ValueError("URDF changed")
        plan_path = self.path.parent / "plan.yaml"
        if digest(plan_path) != self.data["plan_sha256"]:
            raise ValueError("preview changed")
        plan = yaml.safe_load(plan_path.read_text())
        expected = (build_go_zero_session(
            self.data["start_rad"], self.solver, self.data["limits"])
            if self.purpose == GO_ZERO_PURPOSE else build_session(
                plan, self.data["start_rad"], self.solver,
                self.data["limits"]))
        if self.data["segments"] != expected:
            raise ValueError("motion sequence differs from regenerated session")
        self.files = {self.path: digest(self.path), plan_path: digest(plan_path),
                      self.urdf_path: digest(urdf_path)}
        root = Path(__file__).resolve().parents[4]
        for relative, expected_hash in self.data["code_sha256"].items():
            path = (root / relative).resolve()
            path.relative_to(root)
            if digest(path) != expected_hash:
                raise ValueError("motion code changed: " + relative)
            self.files[path] = expected_hash
        required_code = {
            "src/rekpiper_calibration/src/rekpiper_calibration/calibration_motion.py",
            "src/rekpiper_calibration/scripts/calibration_motion_node.py",
        }
        if not required_code <= set(self.data["code_sha256"]):
            raise ValueError("motion code bindings missing")

    @property
    def requires_camera(self):
        return self.purpose != GO_ZERO_PURPOSE

    def normalize_feedback(self, values):
        values = np.asarray(values, dtype=float)
        if self.purpose == GO_ZERO_PURPOSE:
            return validate_go_zero_feedback(values, self.solver)
        return normalize_feedback_to_joint_limits(values, .01)

    def require_ready(self):
        for path, expected in self.files.items():
            if digest(path) != expected:
                raise ValueError("prepared session file changed: " + str(path))


class StepSequence:
    """No skipping, concurrent motion, or automatic restart after a stop."""

    def __init__(self, segments):
        self.segments = segments
        self.cursor = 0
        self.state = "CONNECTED_DISABLED"

    def enable(self):
        if self.state != "CONNECTED_DISABLED":
            raise ValueError("enable is allowed once per process/session")
        self.state = "HOLDING"

    def begin(self, name, actual):
        if self.state != "HOLDING" or self.cursor >= len(self.segments):
            raise ValueError("not ready for another segment")
        segment = self.segments[self.cursor]
        if name != segment["name"]:
            raise ValueError("expected " + segment["name"])
        actual = np.asarray(actual, dtype=float)
        if (actual.shape != (6,) or not np.all(np.isfinite(actual))
                or np.max(np.abs(actual-segment["start_rad"])) > np.deg2rad(.2)):
            raise ValueError("actual start differs by more than 0.2 degrees")
        self.state = "MOVING"
        return segment

    def finish(self):
        if self.state != "MOVING":
            raise ValueError("no segment in progress")
        self.cursor += 1
        self.state = "HOLDING"

    def stop(self):
        self.state = "STOPPED"
