import importlib.util
from pathlib import Path
import unittest

import numpy as np
import rospy
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dual_extrinsics_publisher_node.py"
PACKAGE = Path(__file__).resolve().parents[1]
ACTIVE_CONFIG = PACKAGE / "config" / "active_fixed_camera_extrinsics"
SPEC = importlib.util.spec_from_file_location("dual_extrinsics_publisher_node", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def metadata():
    return {
        "status": "ACCEPTED", "publish_tf_allowed": True,
        "precision_operation_allowed": False,
        "logical_name": "rs1", "serial": "123", "parent_frame": "base_link",
        "child_frame": "rs1_link", "base_T_link": np.eye(4).tolist(),
        "translation_m": {"x": 0.0, "y": 0.0, "z": 0.0},
        "rotation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


class DualExtrinsicsPublisherTest(unittest.TestCase):
    def test_only_accepted_tf_release_is_allowed(self):
        MODULE.DualExtrinsicsPublisher._validate_metadata("rs1", "123", metadata())
        value = metadata()
        value["publish_tf_allowed"] = False
        with self.assertRaises(rospy.ROSInitException):
            MODULE.DualExtrinsicsPublisher._validate_metadata("rs1", "123", value)

    def test_provisional_release_requires_explicit_non_precision_mode(self):
        value = metadata()
        value["status"] = "PROVISIONAL_DIAGNOSTIC"
        with self.assertRaises(rospy.ROSInitException):
            MODULE.DualExtrinsicsPublisher._validate_metadata(
                "rs1", "123", value)
        MODULE.DualExtrinsicsPublisher._validate_metadata(
            "rs1", "123", value, allow_provisional=True)
        value["precision_operation_allowed"] = True
        with self.assertRaises(rospy.ROSInitException):
            MODULE.DualExtrinsicsPublisher._validate_metadata(
                "rs1", "123", value, allow_provisional=True)

    def test_identity_and_frame_mismatch_are_rejected(self):
        value = metadata()
        value["serial"] = "other"
        with self.assertRaises(rospy.ROSInitException):
            MODULE.DualExtrinsicsPublisher._validate_metadata("rs1", "123", value)
        value = metadata()
        value["child_frame"] = "camera_link"
        with self.assertRaises(rospy.ROSInitException):
            MODULE.DualExtrinsicsPublisher._validate_metadata("rs1", "123", value)

    def test_matrix_and_translation_must_agree(self):
        value = metadata()
        value["translation_m"]["x"] = 0.1
        with self.assertRaises(rospy.ROSInitException):
            MODULE.DualExtrinsicsPublisher._validate_metadata("rs1", "123", value)

    def test_rs3_rejects_previous_device_extrinsics(self):
        value = metadata()
        value.update(logical_name="rs3", child_frame="rs3_link",
                     serial="934222070377")
        MODULE.DualExtrinsicsPublisher._validate_metadata(
            "rs3", "934222070377", value)
        value["serial"] = "261922075819"
        with self.assertRaises(rospy.ROSInitException):
            MODULE.DualExtrinsicsPublisher._validate_metadata(
                "rs3", "934222070377", value)

    def test_active_pair_is_uncalibrated_and_rejected(self):
        for name, serial in (
                ("rs1", "346522071783"),
                ("rs3", "934222070377")):
            value = yaml.safe_load((
                ACTIVE_CONFIG / (name + "_extrinsics.yaml")).read_text(
                    encoding="utf-8"))
            self.assertEqual(value["status"], "UNCALIBRATED")
            self.assertFalse(value["publish_tf_allowed"])
            self.assertFalse(value["precision_operation_allowed"])
            with self.assertRaises(rospy.ROSInitException):
                MODULE.DualExtrinsicsPublisher._validate_metadata(
                    name, serial, value, allow_provisional=False)

    def test_runtime_launches_do_not_reference_archived_extrinsics(self):
        offenders = []
        for path in (PACKAGE.parent).glob("rekpiper_*/**/*.launch"):
            if "config/accepted_fixed_camera_extrinsics" in path.read_text(
                    encoding="utf-8"):
                offenders.append(str(path))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
