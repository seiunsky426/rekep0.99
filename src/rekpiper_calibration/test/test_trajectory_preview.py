"""Preview must reject unsafe data and must not pollute live ROS streams."""

import copy
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from rekpiper_calibration.trajectory_preview import (
    JOINT_NAMES, playback_samples, validate_preview)
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver


class PreviewTest(unittest.TestCase):
    def setUp(self):
        self.package = Path(__file__).resolve().parents[1]
        xml = (self.package.parent / "piper_description/urdf/piper_description.urdf").read_text()
        self.solver = PiperURDFIKSolver.from_urdf_xml(xml, "base_link", "link6", JOINT_NAMES)
        self.plan = {"schema_version": 1, "status": "PREVIEW_ONLY",
                     "hardware_execution_allowed": False, "joint_names": JOINT_NAMES,
                     "base_frame": "base_link", "tip_frame": "link6", "poses": []}
        for i in range(20):
            q = [i*0.01, 0.8, -1.0, 0.3, 0.7, -0.4]
            self.plan["poses"].append({"positions_rad": q,
                "base_T_link6": self.solver.forward(q).tolist()})

    def test_rejects_nonfinite_limits_fk_mismatch_duplicates_and_hardware_flag(self):
        validate_preview(self.plan, self.solver)
        for value in (float("nan"), 100.0):
            plan = copy.deepcopy(self.plan)
            plan["poses"][0]["positions_rad"][0] = value
            with self.assertRaises(ValueError):
                validate_preview(plan, self.solver)
        plan = copy.deepcopy(self.plan)
        plan["poses"][0]["base_T_link6"][0][3] += 0.02
        with self.assertRaises(ValueError):
            validate_preview(plan, self.solver)
        plan = copy.deepcopy(self.plan)
        plan["poses"][1] = plan["poses"][0]
        with self.assertRaises(ValueError):
            validate_preview(plan, self.solver)
        self.plan["hardware_execution_allowed"] = True
        with self.assertRaises(ValueError):
            validate_preview(self.plan, self.solver)

    def test_playback_has_endpoints_and_stays_inside_joint_segment(self):
        q = np.asarray([p["positions_rad"] for p in self.plan["poses"]])
        samples = list(playback_samples(q, rate_hz=5))
        np.testing.assert_allclose(samples[0][2], q[0])
        np.testing.assert_allclose(samples[-1][2], q[-1])
        for _, index, point in samples:
            lo = np.minimum(q[max(0, index-1)], q[index])
            hi = np.maximum(q[max(0, index-1)], q[index])
            self.assertTrue(np.all(point >= lo-1e-12))
            self.assertTrue(np.all(point <= hi+1e-12))

    def test_launch_isolates_tf_and_has_no_driver(self):
        launch = ET.parse(str(self.package / "launch/calibration_preview.launch"))
        nodes = launch.findall(".//node")
        self.assertEqual({node.attrib["pkg"] for node in nodes},
                         {"rekpiper_calibration", "robot_state_publisher", "rviz"})
        for node in nodes:
            if node.attrib["pkg"] in ("robot_state_publisher", "rviz"):
                remaps = {r.attrib["from"]: r.attrib["to"] for r in node.findall("remap")}
                self.assertEqual(remaps["/tf"], "/calibration_preview/tf")
                self.assertEqual(remaps["/tf_static"], "/calibration_preview/tf_static")


if __name__ == "__main__":
    unittest.main()
