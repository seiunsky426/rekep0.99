import copy
import unittest

from rekpiper_calibration.teaching_checks import check_teaching_window


class TeachingChecksTest(unittest.TestCase):
    def setUp(self):
        self.joints = [{"stamp_s": 10+i*.01, "positions_rad": [0]*6} for i in range(101)]
        self.statuses = [{"received_ros_s": 10+i*.02, "valid": True} for i in range(51)]
        self.cameras = [{"received_ros_s": 10+i*.05, "status": {
            "state": "READY_TO_CAPTURE", "active_cameras": ["rs1", "rs3"],
            "quality": {"rs1": {}, "rs3": {}}}} for i in range(21)]

    def test_stationary_fresh_window(self):
        result = check_teaching_window(self.joints, self.statuses, self.cameras, 11.01)
        self.assertEqual(result["sample_count"], 101)

    def test_rejects_motion_and_nonfinite(self):
        for value in (.003, float("nan")):
            rows = copy.deepcopy(self.joints)
            rows[-1]["positions_rad"][2] = value
            with self.assertRaises(ValueError):
                check_teaching_window(rows, self.statuses, self.cameras, 11.01)

    def test_rejects_stale_missing_single_camera_and_initial_latch(self):
        for rows in ([], self.cameras[:1], self.cameras[:-8]):
            with self.assertRaises(ValueError):
                check_teaching_window(self.joints, self.statuses, rows, 11.01)
        cameras = copy.deepcopy(self.cameras)
        cameras[-1]["status"]["active_cameras"] = ["rs1"]
        with self.assertRaises(ValueError):
            check_teaching_window(self.joints, self.statuses, cameras, 11.01)
        with self.assertRaises(ValueError):
            check_teaching_window(self.joints, self.statuses, self.cameras, 12)

    def test_rejects_bad_mode_camera_failure_and_feedback_gaps(self):
        self.statuses[5]["valid"] = False
        with self.assertRaises(ValueError):
            check_teaching_window(self.joints, self.statuses, self.cameras, 11.01)
        self.statuses[5]["valid"] = True
        self.cameras[6]["status"]["state"] = "WAITING_FOR_VALID_INPUT"
        with self.assertRaises(ValueError):
            check_teaching_window(self.joints, self.statuses, self.cameras, 11.01)
        self.cameras[6]["status"]["state"] = "READY_TO_CAPTURE"
        for rows in (self.joints[:30]+self.joints[50:], self.joints[:-1]+self.joints[-2:-1]):
            with self.assertRaises(ValueError):
                check_teaching_window(rows, self.statuses, self.cameras, 11.01)


if __name__ == "__main__":
    unittest.main()
