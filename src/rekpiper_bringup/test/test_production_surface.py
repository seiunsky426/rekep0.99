#!/usr/bin/env python3

from pathlib import Path
import importlib.util
import math
from types import SimpleNamespace
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import yaml
import numpy as np


class ProductionSurfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = Path(__file__).resolve().parents[1]
        cls.workspace_src = cls.package.parent
        cls.system_path = cls.package / "launch" / "system.launch"
        cls.system = cls.system_path.read_text(encoding="utf-8")

    def test_launch_is_xml_and_exposes_real_robot_modes(self):
        root = ET.parse(str(self.system_path)).getroot()
        self.assertEqual(root.tag, "launch")
        self.assertIn('doc="shadow|autonomous"', self.system)
        self.assertIn('mode" default="shadow"', self.system)
        self.assertIn("arg('mode') == 'autonomous'", self.system)

    def test_production_entry_has_no_legacy_task_or_test_bypass(self):
        self.assertIn('allow_hardware_commands" default="false"', self.system)

    def test_only_autonomous_declares_hardware_command_outlets(self):
        marker = (
            "<group if=\"$(eval arg('mode') == 'autonomous' and "
            "arg('allow_hardware_commands') == 'true')\">")
        autonomous = self.system.split(marker, 1)[1]
        before = self.system.split(marker, 1)[0]
        internal = (self.package / "launch" /
                    "_hardware_runtime_internal.launch").read_text(
                        encoding="utf-8")
        self.assertIn("hardware_supervisor_node.py", autonomous)
        self.assertIn('<arg name="guard_token"/>', internal)
        for node in ("piper_trajectory_bridge_node.py",
                     "piper_gripper_action_node.py",
                     "piper_ctrl_single_node.py",
                     "gripper_joint_state_adapter_node.py"):
            self.assertIn(node, internal)
            self.assertNotIn(node, before)
            self.assertNotIn(node, autonomous)
        self.assertIn('command_rate_hz" value="20.0"', internal)

    def test_driver_construction_does_not_select_motion_mode(self):
        source = (self.workspace_src / "piper" / "scripts" /
                  "piper_ctrl_single_node.py").read_text(encoding="utf-8")
        construction = source.split("self.piper.ConnectPort()", 1)[1].split(
            "# 启动订阅线程", 1)[0]
        self.assertNotIn("self.piper.MotionCtrl", construction)
        self.assertNotIn("self.piper.JointCtrl", construction)
        self.assertNotIn("self.piper.GripperCtrl", construction)

    def _driver_module(self):
        path = self.workspace_src / "piper" / "scripts" / \
            "piper_ctrl_single_node.py"
        spec = importlib.util.spec_from_file_location(
            "rekpiper_test_piper_driver", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_mock_sdk_constructor_only_connects_telemetry(self):
        driver = self._driver_module()
        calls = []

        class FakePiper:
            def __init__(self, can_name):
                calls.append(("construct", can_name))

            def ConnectPort(self):
                calls.append(("ConnectPort",))

        class FakeThread:
            def __init__(self, target):
                self.target = target
                self.daemon = False

            def start(self):
                calls.append(("thread_start", self.target.__name__))

        parameters = {
            "~can_port": "mock_can",
            "~auto_enable": False,
            "~gripper_exist": True,
        }
        with mock.patch.object(driver, "check_ros_master"), \
                mock.patch.object(driver.rospy, "init_node"), \
                mock.patch.object(driver.rospy, "has_param",
                                  side_effect=lambda name: name in parameters), \
                mock.patch.object(driver.rospy, "get_param",
                                  side_effect=lambda name, default=None:
                                  parameters.get(name, default)), \
                mock.patch.object(driver.rospy, "resolve_name",
                                  side_effect=lambda name: name), \
                mock.patch.object(driver.rospy, "Publisher",
                                  return_value=SimpleNamespace()), \
                mock.patch.object(driver.rospy, "Service",
                                  return_value=SimpleNamespace()), \
                mock.patch.object(driver.rospy, "loginfo"), \
                mock.patch.object(driver.rospy, "logwarn"), \
                mock.patch.object(driver, "validate_release_bundle",
                                  return_value={"verified_artifacts": {}}), \
                mock.patch.object(driver, "C_PiperInterface", FakePiper), \
                mock.patch.object(driver.threading, "Thread", FakeThread):
            driver.C_PiperRosNode()
        self.assertIn(("ConnectPort",), calls)
        forbidden = {"MotionCtrl_1", "MotionCtrl_2", "JointCtrl",
                     "GripperCtrl", "EnableArm", "EnablePiper"}
        self.assertFalse(any(item[0] in forbidden for item in calls), calls)

    def test_rejected_release_never_constructs_sdk_or_connects_can(self):
        driver = self._driver_module()
        parameters = {"~can_port": "mock_can", "~auto_enable": False,
                      "~gripper_exist": True}
        with mock.patch.object(driver, "check_ros_master"), \
                mock.patch.object(driver.rospy, "init_node"), \
                mock.patch.object(driver.rospy, "has_param",
                                  side_effect=lambda name: name in parameters), \
                mock.patch.object(driver.rospy, "get_param",
                                  side_effect=lambda name, default=None:
                                  parameters.get(name, default)), \
                mock.patch.object(driver.rospy, "resolve_name",
                                  side_effect=lambda name: name), \
                mock.patch.object(driver.rospy, "Publisher",
                                  return_value=SimpleNamespace()), \
                mock.patch.object(driver.rospy, "Service",
                                  return_value=SimpleNamespace()), \
                mock.patch.object(driver.rospy, "loginfo"), \
                mock.patch.object(driver.rospy, "logwarn"), \
                mock.patch.object(driver, "validate_release_bundle",
                                  side_effect=driver.AcceptanceError("tampered")), \
                mock.patch.object(driver, "C_PiperInterface") as sdk:
            with self.assertRaises(driver.rospy.ROSInitException):
                driver.C_PiperRosNode()
        sdk.assert_not_called()

    def test_mock_sdk_motion_requires_confirmed_enable(self):
        driver = self._driver_module()
        calls = []

        class FakePiper:
            def __getattr__(self, name):
                return lambda *args: calls.append((name,) + args)

        node = driver.C_PiperRosNode.__new__(driver.C_PiperRosNode)
        node.piper = FakePiper()
        node.gripper_exist = True
        node.gripper_val_mutiple = 1
        node._joint_command_default_speed = 10
        node._motion_ctrl_refresh_s = 1.0
        node._last_motion_ctrl = None
        node._last_motion_ctrl_time = 0.0
        node._release = {}
        node._hold_only = False
        node._command_armed = True
        node._authorized_joint_caller = "/piper_trajectory_bridge"
        node._hold_reference = None
        node._hold_tolerance_rad = 0.01
        node._C_PiperRosNode__enable_flag = False
        message = SimpleNamespace(
            position=[0.1, 0.2, -0.3, 0.1, 0.0, 0.0, 0.02],
            velocity=[0.0] * 6 + [10.0], effort=[0.0] * 7,
            _connection_header={"callerid": "/piper_trajectory_bridge"})
        with mock.patch.object(driver.rospy, "logwarn_throttle"), \
                mock.patch.object(driver.rospy, "loginfo"), \
                mock.patch.object(driver.rospy, "logerr"):
            node.joint_callback(message)
            node.enable_callback(SimpleNamespace(data=True))
        self.assertEqual(calls, [])

        # This state is set only after /enable_srv receives stable six-motor
        # feedback.  The mock then observes the expected command outlet.
        node._C_PiperRosNode__enable_flag = True
        with mock.patch.object(driver.rospy, "loginfo"), \
                mock.patch.object(driver.rospy, "logdebug"), \
                mock.patch.object(driver, "assert_release_unchanged"):
            node.joint_callback(message)
        self.assertEqual(calls[0][0], "MotionCtrl_2")
        self.assertIn("JointCtrl", [item[0] for item in calls])

        calls.clear()
        message._connection_header = {"callerid": "/untrusted_publisher"}
        with mock.patch.object(driver.rospy, "logerr_throttle"):
            node.joint_callback(message)
        self.assertEqual(calls, [])

    def test_hardware_acceptance_driver_rejects_non_hold_target(self):
        driver = self._driver_module()
        calls = []
        node = driver.C_PiperRosNode.__new__(driver.C_PiperRosNode)
        node.piper = SimpleNamespace(
            MotionCtrl_2=lambda *args: calls.append(("MotionCtrl_2",) + args),
            JointCtrl=lambda *args: calls.append(("JointCtrl",) + args))
        node.gripper_exist = False
        node.gripper_val_mutiple = 1
        node._joint_command_default_speed = 10
        node._motion_ctrl_refresh_s = 1.0
        node._last_motion_ctrl = None
        node._last_motion_ctrl_time = 0.0
        node._C_PiperRosNode__enable_flag = True
        node._release = {}
        node._hold_only = True
        node._command_armed = True
        node._authorized_joint_caller = "/piper_trajectory_bridge"
        node._hold_reference = np.zeros(6)
        node._hold_tolerance_rad = 0.01
        message = SimpleNamespace(
            position=[0.02, 0, 0, 0, 0, 0],
            velocity=[0.0] * 7, effort=[0.0] * 7,
            _connection_header={"callerid": "/piper_trajectory_bridge"})
        with mock.patch.object(driver.rospy, "logerr_throttle"), \
                mock.patch.object(driver, "assert_release_unchanged"):
            node.joint_callback(message)
        self.assertEqual(calls, [])

    def test_hardware_acceptance_graph_has_no_gripper_or_cartesian_node(self):
        hold = (self.package / "launch" /
                "_hardware_hold_runtime_internal.launch").read_text(
                    encoding="utf-8")
        self.assertIn('hold_only" value="true"', hold)
        self.assertIn("hardware_hold_stream_node.py", hold)
        self.assertNotIn("piper_gripper_action_node.py", hold)
        self.assertNotIn("keypoint_anygrasp", hold)

    def test_piper_millidegree_round_trip_uses_pi(self):
        for raw in (-150000, -1, 0, 1, 123456, 180000):
            radians = raw * math.pi / 180000.0
            self.assertEqual(round(radians * 180000.0 / math.pi), raw)

    def test_closed_loop_public_api_and_short_horizon_are_fixed(self):
        source = (self.workspace_src / "rekpiper_execution" / "scripts" /
                  "closed_loop_node.py").read_text(encoding="utf-8")
        for service in ("arm", "pause", "resume", "abort"):
            self.assertIn(
                'rospy.Service("/rekpiper/execution/{}"'.format(service),
                source)
        self.assertIn("short_horizon_joint_prefix", source)
        self.assertIn("joint_path, current, self._joint_velocity, 0.10", source)
        self.assertIn("authorized_duration_s=0.1", source)
        self.assertIn("self._success_ticks_required", source)
        self.assertIn("_backtrack_if_needed", source)

    def test_production_topics_use_hierarchical_namespace(self):
        files = [
            self.system_path,
            self.workspace_src / "rekpiper_mapping" / "launch" /
            "safe_dual_mapping.launch",
            self.workspace_src / "rekpiper_mapping" / "launch" /
            "_safe_dual_mapping_runtime_internal.launch",
            self.workspace_src / "rekpiper_perception" / "launch" /
            "task_perception.launch",
        ]
        text = "\n".join(path.read_text(encoding="utf-8") for path in files)
        for legacy in ("/rekpiper_camera", "/rekpiper_mapping",
                       "/rekpiper_tracking", "/rekpiper_grasp",
                       "/rekpiper_planning", "/rekpiper_execution"):
            self.assertNotIn(legacy, text)
        self.assertNotIn('value="/nvblox_mapping/', text)
        self.assertNotIn('default="/nvblox_mapping/', text)
        self.assertNotIn('value="/task_perception/', text)
        self.assertNotIn('default="/task_perception/', text)
        self.assertIn("/rekpiper/mapping", text)
        self.assertIn('ns="/rekpiper" name="perception"', text)

    def test_acceptance_defaults_are_all_false_under_rekpiper(self):
        config = yaml.safe_load((self.package / "config" /
                                 "system.yaml").read_text(encoding="utf-8"))
        acceptance = config["rekpiper"]["acceptance"]
        self.assertEqual(
            set(acceptance), {"camera_extrinsics", "piper_workspace",
                              "gripper_baseline", "safe_map",
                              "emergency_stop"})
        self.assertFalse(any(acceptance.values()))

    def test_package_boundaries_match_rekep_runtime(self):
        packages = {
            path.parent.name for path in self.workspace_src.glob("*/package.xml")}
        self.assertNotIn("rekpiper_tracking", packages)
        self.assertIn("rekpiper_calibration", packages)

        mapping = self.workspace_src / "rekpiper_mapping"
        for moved in ("object_registry_node.py",
                      "multicamera_object_tracker_node.py",
                      "grasp_state_monitor_node.py"):
            self.assertFalse((mapping / "scripts" / moved).exists())

        perception = self.workspace_src / "rekpiper_perception"
        for online in ("dual_dino_tracker_node.py",
                       "multicamera_object_tracker_node.py",
                       "object_registry_node.py",
                       "grasp_state_monitor_node.py"):
            self.assertTrue((perception / "scripts" / online).is_file())

        camera = self.workspace_src / "rekpiper_camera"
        camera_files = "\n".join(
            str(path.relative_to(camera)) for path in camera.rglob("*"))
        for excluded in ("rviz", "rqt", "replay", "jog_gui"):
            self.assertNotIn(excluded, camera_files.lower())


if __name__ == "__main__":
    unittest.main()
