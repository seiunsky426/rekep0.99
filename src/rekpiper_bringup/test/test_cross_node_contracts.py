#!/usr/bin/env python3

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from rekpiper_msgs.srv import QuerySDFRequest, QuerySDFResponse


class CrossNodeContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = Path(__file__).resolve().parents[2]

    def _read(self, package, relative):
        return (self.src / package / relative).read_text(encoding="utf-8")

    def test_sdf_contract_carries_frame_and_generation(self):
        self.assertIn("header", QuerySDFRequest.__slots__)
        self.assertIn("map_generation_uuid", QuerySDFResponse.__slots__)
        snapshot = self._read(
            "rekpiper_mapping", "scripts/sdf_grid_snapshot_node.py")
        grasp = self._read(
            "rekpiper_grasp", "scripts/keypoint_anygrasp_node.py")
        self.assertIn("header=Header(frame_id=self._target_frame)", snapshot)
        self.assertIn("generation = str(response.map_generation_uuid)", snapshot)
        self.assertIn("header=Header(frame_id=\"base_link\")", grasp)

    def test_trajectory_is_bound_to_planning_map(self):
        coordinator = self._read(
            "rekpiper_execution", "scripts/closed_loop_node.py")
        bridge = self._read(
            "rekpiper_execution", "scripts/piper_trajectory_bridge_node.py")
        self.assertIn(
            "trajectory.header.frame_id = str(map_generation_uuid)",
            coordinator)
        self.assertIn("generation = str(goal.trajectory.header.frame_id)", bridge)
        self.assertIn("self._health(generation)", bridge)

    def test_gripper_feedback_uses_driver_total_opening(self):
        monitor = self._read(
            "rekpiper_perception", "scripts/grasp_state_monitor_node.py")
        self.assertIn(
            "self._gripper_opening = float(message.position[index])", monitor)
        self.assertNotIn(
            "self._gripper_opening = 2.0 * float(message.position[index])",
            monitor)

    def test_tracker_cannot_republish_the_same_camera_frame(self):
        tracker = self._read(
            "rekpiper_perception", "scripts/dual_dino_tracker_node.py")
        self.assertIn("stamps == self._last_processed_stamps", tracker)
        self.assertIn("camera frames are stale", tracker)

    def test_grasp_candidate_binds_attempt_and_semantic_target(self):
        grasp = self._read("rekpiper_msgs", "msg/GraspCandidate.msg")
        batch = self._read("rekpiper_msgs", "msg/GraspCandidateArray.msg")
        status = self._read("rekpiper_msgs", "msg/ClosedLoopStatus.msg")
        horizon = self._read("rekpiper_msgs", "msg/ReKepHorizon.msg")
        self.assertIn("uint32 grasp_attempt", grasp)
        self.assertIn("uint32 grasp_attempt", batch)
        self.assertIn("string map_generation_uuid", batch)
        self.assertIn("uint32 grasp_attempt", status)
        self.assertIn("geometry_msgs/Pose semantic_subgoal_pose", horizon)

    def test_anygrasp_target_is_requested_before_path_planning(self):
        generator = self._read(
            "rekpiper_grasp", "scripts/keypoint_anygrasp_node.py")
        planner = self._read(
            "rekpiper_planning",
            "src/rekpiper_planning/realtime_planner.py")
        self.assertIn("horizon_requests_grasp", generator)
        self.assertIn('status == "grasp_target_pending"', self._read(
            "rekpiper_grasp", "src/rekpiper_grasp/selection.py"))
        self.assertIn("request.grasp_target_pose", planner)

    def test_anygrasp_launch_uses_fused_cloud_and_rs1_reference(self):
        launch = ET.fromstring(self._read('rekpiper_bringup', 'launch/system.launch'))
        node = launch.find("node[@name='keypoint_anygrasp']")
        self.assertEqual(node.find("param[@name='points_topic']").get('value'),
                         '/rekpiper/camera/fused/points_base')
        self.assertEqual(node.find("param[@name='inference_frame']").get('value'),
                         'rs1_color_optical_frame')
        self.assertEqual(node.find("rosparam[@param='source_cameras']").text, '[rs1, rs3]')
        self.assertEqual(node.find("param[@name='restrict_horizontal']").get('value'), 'false')

    def test_planning_map_and_task_perception_use_only_rs1(self):
        launch = ET.fromstring(self._read('rekpiper_bringup', 'launch/system.launch'))
        includes = list(launch.iter('include'))
        mapping = next(node for node in includes if 'safe_dual_mapping.launch' in node.get('file'))
        self.assertEqual(mapping.find("arg[@name='use_rs3']").get('value'), 'false')
        self.assertTrue(mapping.find("arg[@name='mapping_cameras_config']").get('value').endswith('nvblox_cameras_rs1.yaml'))
        perception = next(node for node in includes if 'task_perception.launch' in node.get('file'))
        self.assertEqual(perception.find("arg[@name='points_topic']").get('value'),
                         '/rekpiper/camera/rs1/points_recognition')
        execution = self._read('rekpiper_execution', 'scripts/closed_loop_node.py')
        self.assertIn("rospy.Subscriber('/rekpiper/camera/rs1/points_recognition'", execution)
        self.assertNotIn("rospy.Subscriber('/rekpiper/camera/fused/points_base'", execution)

    def test_tracker_uses_one_reference_and_one_global_top100(self):
        tracker = self._read(
            "rekpiper_perception", "scripts/dual_dino_tracker_node.py")
        self.assertIn("tensor_reference_descriptor", tracker)
        self.assertIn("tensor_feature_observation", tracker)
        self.assertNotIn("ransac_rigid_transform", tracker)

    def test_rs3_is_excluded_from_tracking_launches(self):
        for filename in ("manual_tracking.launch", "online_tracking.launch"):
            launch = ET.fromstring(self._read("rekpiper_perception", "launch/" + filename))
            for name in ("object_registry", "multicamera_object_tracker", "dual_dino_tracker"):
                node = launch.find("node[@name='{}']".format(name))
                self.assertEqual(node.find("rosparam[@param='camera_names']").text, "[rs1]")
            tracker = launch.find("node[@name='dual_dino_tracker']")
            self.assertEqual(tracker.find("param[@name='use_object_masks']").get("value"), "true")
            if filename == "manual_tracking.launch":
                self.assertIsNone(launch.find("node[@name='grasp_state_monitor']"))


if __name__ == "__main__":
    unittest.main()
