"""Hardware-free checks of timing, session identity and step/stop gates."""

from copy import deepcopy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import yaml

from rekpiper_calibration.calibration_motion import (
    DEFAULT_LIMITS, MotionSession, StepSequence, build_go_zero_session,
    build_session, check_limits, digest, make_segment, positions,
    solver_from_file)
from rekpiper_calibration.trajectory_preview import JOINT_NAMES


class MotionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[3]
        cls.urdf = cls.root/"src/piper_description/urdf/piper_description.urdf"
        cls.solver = solver_from_file(cls.urdf)
        cls.start = np.array([0, .9, -.5, 0, .85, 0])
        cls.plan = {"schema_version": 1, "status": "PREVIEW_ONLY", "hardware_execution_allowed": False,
                    "joint_names": JOINT_NAMES, "base_frame": "base_link", "tip_frame": "link6", "poses": []}
        for angle in np.linspace(.2, -.2, 20):
            q = cls.start.copy()
            q[0] = angle
            cls.plan["poses"].append({"positions_rad": q.tolist(), "base_T_link6": cls.solver.forward(q).tolist()})
        cls.segments = build_session(cls.plan, cls.start, cls.solver, DEFAULT_LIMITS)

    def test_timing_respects_speed_acceleration_jerk_and_tip_speed(self):
        s = make_segment("test", self.start, self.start+[.3, .1, -.1, .15, -.1, .2], self.solver, DEFAULT_LIMITS)
        t = np.linspace(0, s["duration_s"], 2001)
        q = np.array([positions(s, u) for u in t])
        dt = t[1]-t[0]
        for order, limit in [(1, 3), (2, 6), (3, 12)]:
            self.assertLessEqual(np.max(np.abs(np.diff(q, n=order, axis=0)/dt**order)), np.deg2rad(limit)*1.001)
        xyz = np.array([self.solver.forward(v)[:3, 3] for v in q])
        self.assertLessEqual(np.max(np.linalg.norm(np.diff(xyz, axis=0), axis=1)/dt), .010)
        np.testing.assert_allclose(q[0], self.start)
        np.testing.assert_allclose(q[-1], s["goal_rad"])
        np.testing.assert_allclose(positions(s, 2*s["duration_s"]), s["goal_rad"])

    def test_invalid_speeds_and_joint_limits_rejected(self):
        for key in DEFAULT_LIMITS:
            for value in (0, float("nan"), DEFAULT_LIMITS[key]*2):
                with self.assertRaises(ValueError):
                    check_limits(dict(DEFAULT_LIMITS, **{key: value}))
        bad = self.start.copy(); bad[4] = 1.23
        with self.assertRaises(ValueError):
            make_segment("bad", self.start, bad, self.solver, DEFAULT_LIMITS)

    def test_go_zero_allows_only_small_joint2_joint3_boundary_recovery(self):
        start = np.deg2rad([0.53, -2.49, 2.42, 2.63, 25.27, -1.23])
        limits = dict(DEFAULT_LIMITS, velocity_deg_s=1.0,
                      tip_speed_m_s=0.005)
        segment = build_go_zero_session(start, self.solver, limits)[0]
        np.testing.assert_allclose(segment["start_rad"], start)
        np.testing.assert_allclose(segment["goal_rad"], np.zeros(6))
        self.assertGreater(segment["duration_s"], 40.0)
        for bad in (np.deg2rad([0, -3.1, 0, 0, 0, 0]),
                    np.asarray([2.2, 0, 0, 0, 0, 0])):
            with self.assertRaises(ValueError):
                build_go_zero_session(bad, self.solver, limits)

    def test_sequence_requires_hold_trial_order_start_and_latches_stop(self):
        state = StepSequence(self.segments)
        with self.assertRaises(ValueError):
            state.begin("trial", self.start)
        state.enable()
        with self.assertRaises(ValueError):
            state.begin("point_01", self.start)
        with self.assertRaises(ValueError):
            state.begin("trial", self.start+.01)
        first = state.begin("trial", self.start)
        with self.assertRaises(ValueError):
            state.begin("trial", self.start)
        self.assertLessEqual(np.rad2deg(np.max(np.abs(np.array(first["goal_rad"])-self.start))), 1.000001)
        state.finish()
        self.assertEqual(state.cursor, 1)
        state.stop()
        with self.assertRaises(ValueError):
            state.enable()
        with self.assertRaises(ValueError):
            state.begin("point_01", first["goal_rad"])

    def session_files(self, root):
        (root/"plan.yaml").write_text(yaml.safe_dump(self.plan))
        code = ["src/rekpiper_calibration/src/rekpiper_calibration/calibration_motion.py",
                "src/rekpiper_calibration/scripts/calibration_motion_node.py"]
        data = {"schema_version": 1, "status": "OPERATOR_SUPERVISED", "hardware_execution_allowed": True,
                "limits": DEFAULT_LIMITS, "start_rad": self.start.tolist(), "segments": self.segments,
                "urdf_sha256": digest(self.urdf), "plan_sha256": digest(root/"plan.yaml"),
                "code_sha256": {p: digest(self.root/p) for p in code}}
        path = root/"session.yaml"
        path.write_text(yaml.safe_dump(data))
        return path

    def test_prepared_session_is_ready_and_changed_plan_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); path = self.session_files(root)
            s = MotionSession(path, self.urdf)
            s.require_ready()
            (root/"plan.yaml").write_text("{}")
            with self.assertRaises(ValueError):
                MotionSession(path, self.urdf)

    def test_loaded_session_detects_bound_file_mutation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); path = self.session_files(root)
            s = MotionSession(path, self.urdf)
            (root/"plan.yaml").write_text("changed")
            with self.assertRaises(ValueError):
                s.require_ready()

    def test_disabled_node_never_constructs_sdk(self):
        path = self.root/"src/rekpiper_calibration/scripts/calibration_motion_node.py"
        spec = importlib.util.spec_from_file_location("test_calibration_motion_node", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with patch.object(module, "MotionSession"), patch.object(module.rospy, "get_param", return_value=False), \
                patch.object(module, "C_PiperInterface_V2") as sdk:
            with self.assertRaises(ValueError):
                module.CalibrationMotion()
            sdk.assert_not_called()

    def test_invalid_session_never_opens_can_even_when_execute_true(self):
        path = self.root/"src/rekpiper_calibration/scripts/calibration_motion_node.py"
        spec = importlib.util.spec_from_file_location("test_unapproved_motion", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with patch.object(module, "MotionSession") as session, \
                patch.object(module.rospy, "get_param", return_value=True), \
                patch.object(module, "C_PiperInterface_V2") as sdk, \
                patch.object(module.can.interface, "Bus") as bus:
            session.return_value.require_ready.side_effect = ValueError("invalid session")
            with self.assertRaises(ValueError):
                module.CalibrationMotion()
            sdk.assert_not_called()
            bus.assert_not_called()

    def test_pre_enable_disable_waits_for_all_six_disabled(self):
        path = self.root/"src/rekpiper_calibration/scripts/calibration_motion_node.py"
        spec = importlib.util.spec_from_file_location("test_pre_enable_disable", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        node = module.CalibrationMotion.__new__(module.CalibrationMotion)
        node.driver = MagicMock()
        node._motor_enable_states = MagicMock(side_effect=[
            [True, True, True, True, True, True],
            [False, False, False, False, False, False],
        ])
        with patch.object(module.rospy, "is_shutdown", return_value=False), \
                patch.object(module.time, "sleep"):
            states = node._disable_motors()
        self.assertEqual(states, [False]*6)
        self.assertEqual(node.driver.DisableArm.call_count, 2)

    def test_go_zero_runner_holds_until_operator_requests_disable(self):
        path = self.root/"tools/run_go_zero.py"
        source = path.read_text()
        hold = source.index("六关节零位已确认；机械臂保持使能")
        wait = source.index("input(", hold)
        disable = source.index(
            'ServiceProxy("/calibration_motion/disable"', wait)
        self.assertLess(hold, wait)
        self.assertLess(wait, disable)

    def test_enable_acks_all_motors_before_mode_and_position_hold(self):
        path = self.root/"src/rekpiper_calibration/scripts/calibration_motion_node.py"
        spec = importlib.util.spec_from_file_location("test_staged_enable", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        node = module.CalibrationMotion.__new__(module.CalibrationMotion)
        node.driver = MagicMock()
        node.session = MagicMock()
        node.session.data = {"limits": {"driver_speed_percent": 5}}
        node.sequence = MagicMock()
        node.sequence.state = "PREPARED_DISABLED"
        node.reference = np.deg2rad([3, -2, 2, 0, 25, -1])
        node.desired = node.reference.copy()
        node.lock = threading.RLock()
        node.stop_event = threading.Event()
        node.commanded = False
        node.event = MagicMock()
        normal = SimpleNamespace(ctrl_mode=0, mode_feed=1)
        can_move_j = SimpleNamespace(ctrl_mode=1, mode_feed=1)
        calls = {"feedback": 0}

        def feedback(*_args, **_kwargs):
            calls["feedback"] += 1
            if calls["feedback"] == 1:
                return node.reference.copy(), normal, False
            if calls["feedback"] <= 11:
                return node.reference.copy(), normal, True
            return node.reference.copy(), can_move_j, True

        node.feedback = MagicMock(side_effect=feedback)
        node._motor_enable_states = MagicMock(return_value=[False]*6)
        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        with patch.object(module.rospy, "is_shutdown", return_value=False), \
                patch.object(module.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(module.time, "sleep", side_effect=sleep):
            response = node.enable(None)

        self.assertTrue(response.success, response.message)
        methods = [call[0] for call in node.driver.method_calls]
        first_mode = methods.index("MotionCtrl_2")
        self.assertEqual(methods[:first_mode], ["EnablePiper"]*10)
        self.assertEqual(methods[first_mode:first_mode+2],
                         ["MotionCtrl_2", "JointCtrl"])


if __name__ == "__main__":
    unittest.main()
