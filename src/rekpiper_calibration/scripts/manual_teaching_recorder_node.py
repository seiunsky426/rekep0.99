#!/usr/bin/env python3
"""Read-only buffered teaching recorder. Trigger requests save one waypoint."""

from collections import deque
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
from piper_msgs.msg import PiperStatusMsg
import rospy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
import tf2_ros
import yaml

from rekpiper_calibration.teaching_checks import check_teaching_window
from rekpiper_calibration.trajectory_preview import JOINT_NAMES
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver


class TeachingRecorder:
    def __init__(self):
        self.path = Path(rospy.get_param("~waypoints")).resolve()
        self._read_session()
        self.lock = threading.Lock()
        self.record_lock = threading.Lock()
        self.joints, self.statuses, self.cameras = (deque(maxlen=500) for _ in range(3))
        self.images = {}
        self.tx_path = Path("/sys/class/net/can0/statistics/tx_packets")
        self.tx_baseline = int(self.tx_path.read_text())
        xml = rospy.get_param("/robot_description")
        self.urdf_hash = hashlib.sha256(xml.encode()).hexdigest()
        self.solver = PiperURDFIKSolver.from_urdf_xml(xml, "base_link", "link6", JOINT_NAMES)
        self.tf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf)
        self.bridge = CvBridge()
        self.subscribers = [
            rospy.Subscriber("/joint_states_single", JointState, self._joints, queue_size=200),
            rospy.Subscriber("/arm_status", PiperStatusMsg, self._status, queue_size=100),
            rospy.Subscriber("/dual_aruco18_capture/status", String, self._camera, queue_size=50),
        ]
        for name in ("rs1", "rs3"):
            self.subscribers.append(rospy.Subscriber(
                "/dual_aruco18_capture/{}/annotated_image".format(name), Image,
                self._image, callback_args=name, queue_size=1))
        self.check_service = rospy.Service("~check", Trigger, self._check)
        self.record_service = rospy.Service("~record", Trigger, self._record)
        rospy.loginfo("Teaching recorder ready: %s; waiting for explicit record requests", self.path)

    def _read_session(self):
        data = yaml.safe_load(self.path.read_text())
        if (data.get("hardware_execution_allowed") is not False
                or data.get("joint_names") != JOINT_NAMES
                or data.get("base_frame") != "base_link"
                or data.get("effector_frame") != "link6"):
            raise ValueError("invalid read-only teaching session")
        if [p["id"] for p in data["waypoints"]] != list(range(1, len(data["waypoints"])+1)):
            raise ValueError("waypoint IDs are not sequential")
        return data

    def _joints(self, message):
        mapping = dict(zip(message.name, message.position))
        if any(name not in mapping for name in JOINT_NAMES):
            return
        row = {"stamp_s": message.header.stamp.to_sec(),
               "positions_rad": [float(mapping[name]) for name in JOINT_NAMES]}
        with self.lock:
            self.joints.append(row)

    def _status(self, message):
        valid = message.ctrl_mode == 2 and message.arm_status == 0 and message.err_code == 0
        valid = valid and not any(
            getattr(message, "joint_{}_angle_limit".format(i)) or
            getattr(message, "communication_status_joint_{}".format(i)) for i in range(1, 7))
        with self.lock:
            self.statuses.append({"received_ros_s": rospy.Time.now().to_sec(),
                                  "valid": bool(valid), "ctrl_mode": message.ctrl_mode,
                                  "arm_status": message.arm_status, "err_code": message.err_code})

    def _camera(self, message):
        try:
            data = json.loads(message.data)
        except ValueError:
            return
        with self.lock:
            self.cameras.append({"received_ros_s": rospy.Time.now().to_sec(), "status": data})

    def _image(self, message, name):
        with self.lock:
            self.images[name] = message

    def _snapshot(self):
        now = rospy.Time.now().to_sec()
        if int(self.tx_path.read_text()) != self.tx_baseline:
            raise ValueError("CAN_TX_changed")
        with self.lock:
            joints = [r for r in self.joints if r["stamp_s"] >= now-1.10]
            statuses = [r for r in self.statuses if r["received_ros_s"] >= now-1.10]
            cameras = [r for r in self.cameras if r["received_ros_s"] >= now-1.10]
            images = dict(self.images)
        metrics = check_teaching_window(joints, statuses, cameras, now)
        for name in ("rs1", "rs3"):
            if name not in images or not -.05 <= now-images[name].header.stamp.to_sec() <= .3:
                raise ValueError(name+"_image_stale")
        selected = joints[metrics["representative_index"]]
        q = np.asarray(selected["positions_rad"])
        if np.any(q < self.solver._lower) or np.any(q > self.solver._upper):
            raise ValueError("joint_limit")
        t = self.tf.lookup_transform("base_link", "link6",
                                    rospy.Time.from_sec(selected["stamp_s"]), rospy.Duration(.05)).transform
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]).as_matrix()
        pose[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
        pe, re = self.solver._pose_errors(self.solver.forward(q), pose)
        if pe > .0005 or re > .001:
            raise ValueError("FK_TF_disagreement")
        return {"joints": joints, "statuses": statuses, "cameras": cameras,
                "images": images, "metrics": metrics, "selected": selected, "pose": pose}

    def _check(self, _request):
        try:
            snap = self._snapshot()
            return TriggerResponse(True, json.dumps({"state": "READY", "saved": False,
                "waypoint_count": len(self._read_session()["waypoints"]),
                "sample_count": snap["metrics"]["sample_count"],
                "cameras": snap["cameras"][-1]["status"]["quality"]}))
        except Exception as exc:
            return TriggerResponse(False, str(exc))

    def _record(self, _request):
        start = time.monotonic()
        if not self.record_lock.acquire(blocking=False):
            return TriggerResponse(False, "record_already_in_progress")
        try:
            snap = self._snapshot()
            # A second recorder or accidental retry must not duplicate a point.
            with self.path.with_suffix(".lock").open("a") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                session = self._read_session()
                for existing in session["waypoints"]:
                    dp, dr = self.solver._pose_errors(np.asarray(existing["base_T_link6"]), snap["pose"])
                    if dp < .010 and dr < np.deg2rad(5):
                        raise ValueError("near_existing_waypoint_{}".format(existing["id"]))
                result = self._save(session, snap)
            result["elapsed_s"] = round(time.monotonic()-start, 3)
            return TriggerResponse(True, json.dumps(result, ensure_ascii=False))
        except Exception as exc:
            rospy.logwarn("Teaching record rejected: %s", exc)
            return TriggerResponse(False, str(exc))
        finally:
            self.record_lock.release()

    def _save(self, session, snap):
        index = len(session["waypoints"])+1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        pose = snap["pose"]
        q = np.asarray(snap["selected"]["positions_rad"])
        record = {"id": index, "recorded_utc": stamp, "source": "operator_manual_teaching",
            "kind": "taught_waypoint", "hardware_execution_allowed": False, "joint_names": JOINT_NAMES,
            "positions_rad": q.tolist(), "positions_deg": np.rad2deg(q).tolist(),
            "base_T_link6": pose.tolist(), "translation_m": pose[:3, 3].tolist(),
            "quaternion_xyzw": Rotation.from_matrix(pose[:3, :3]).as_quat().tolist(),
            "rpy_xyz_deg": Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True).tolist(),
            "sample_stamp_s": snap["selected"]["stamp_s"], **snap["metrics"],
            "camera_status_snapshot": snap["cameras"][-1]["status"],
            "camera_readiness_window": {"fresh_status_count": len(snap["cameras"]), "all_fresh_statuses_ready": True},
            "calibration_rgbd_sample_captured": False, "urdf_text_sha256": self.urdf_hash,
            "can_tx_before": self.tx_baseline, "can_tx_after": int(self.tx_path.read_text())}
        if record["can_tx_after"] != self.tx_baseline:
            raise ValueError("CAN_TX_changed")
        prefix = "point_{:03d}_{}".format(index, stamp)
        record["annotated_images"] = {}
        for name, message in snap["images"].items():
            image_path = self.path.parent / (prefix+"_"+name+".png")
            if not cv2.imwrite(str(image_path), self.bridge.imgmsg_to_cv2(message, "bgr8")):
                raise OSError("image_save_failed")
            record["annotated_images"][name] = {"path": str(image_path),
                "stamp_s": message.header.stamp.to_sec(),
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()}
        raw_path = self.path.parent / (prefix+".yaml")
        with raw_path.open("x") as stream:
            yaml.safe_dump(dict(record, raw_feedback=snap["joints"], raw_status=snap["statuses"],
                                raw_camera_status=snap["cameras"]), stream, sort_keys=False, allow_unicode=True)
        record["raw_record_path"] = str(raw_path)
        record["raw_record_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        original = self.path.read_bytes()
        backup = self.path.parent / ("waypoints_before_"+prefix+".yaml")
        with backup.open("xb") as stream:
            stream.write(original)
        session["waypoints"].append(record)
        session["status"] = "COLLECTING_OPERATOR_WAYPOINTS"
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(yaml.safe_dump(session, sort_keys=False, allow_unicode=True))
        temporary.replace(self.path)
        return {"id": index, "positions_deg": record["positions_deg"],
                "translation_m": record["translation_m"], "rpy_xyz_deg": record["rpy_xyz_deg"],
                "cameras": record["camera_status_snapshot"]["quality"], "path": str(self.path)}


if __name__ == "__main__":
    rospy.init_node("manual_teaching_recorder")
    recorder = TeachingRecorder()
    rospy.spin()
