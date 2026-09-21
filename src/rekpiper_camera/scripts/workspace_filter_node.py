#!/usr/bin/env python3
"""Publish organized point clouds limited to the measured recognition box."""

import copy
import json
import threading

from cv_bridge import CvBridge
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String

from rekpiper_acceptance import (
    AcceptanceError, assert_release_unchanged, validate_release_bundle)
from rekpiper_camera.workspace import filter_xyz_bounds


class WorkspaceFilterNode:
    def __init__(self):
        self._signed_workspace = None
        self._release = None
        mode = str(rospy.get_param("~mode", "shadow")).strip().lower()
        if mode == "autonomous":
            try:
                self._release = validate_release_bundle(
                    str(rospy.get_param("~release_bundle", "")),
                    str(rospy.get_param("~acceptance_public_key", "")),
                    str(rospy.get_param("~minimum_release_counter", "")),
                    expected_robot_id=str(rospy.get_param(
                        "~robot_id", "piper-rekpiper")))
                self._signed_workspace = self._release[
                    "verified_artifacts"]["workspace"]["payload"]
                if (self._signed_workspace.get("status") != "ACCEPTED"
                        or self._signed_workspace.get(
                            "precision_operation_allowed") is not True):
                    raise ValueError("workspace payload is not accepted")
                if self._signed_workspace.get("frame_id") != "base_link":
                    raise ValueError("workspace frame_id must be base_link")
            except (AcceptanceError, KeyError, TypeError, ValueError) as exc:
                raise rospy.ROSInitException(
                    "signed workspace acceptance rejected: {}".format(exc))
        configured_cameras = rospy.get_param("~input_cameras", ["rs1", "rs3"])
        if not isinstance(configured_cameras, list):
            raise rospy.ROSInitException("input_cameras must be a list")
        self._camera_names = tuple(
            str(name).strip().lower() for name in configured_cameras
            if str(name).strip()
        )
        if (not self._camera_names
                or len(set(self._camera_names)) != len(self._camera_names)
                or set(self._camera_names) - {"rs1", "rs3"}):
            raise rospy.ROSInitException(
                "input_cameras must be a non-empty unique subset of rs1, rs3"
            )
        self._bounds_min = self._vector_param(
            "~workspace_bounds_min", "workspace_bounds_min")
        self._bounds_max = self._vector_param(
            "~workspace_bounds_max", "workspace_bounds_max")
        self._table_height_m = float(self._workspace_value(
            "~table_height_m", "table_height_m"))
        self._below_table_m = float(
            self._workspace_value(
                "~below_table_allowance_m", "below_table_allowance_m")
        )
        expected_lower_z = self._table_height_m - self._below_table_m
        if abs(self._bounds_min[2] - expected_lower_z) > 1e-6:
            raise rospy.ROSInitException(
                "workspace min Z must equal table_height - allowance"
            )
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._camera_status = {}
        self._publishers = {}
        self._mask_publishers = {}
        for name in self._camera_names:
            self._publishers[name] = rospy.Publisher(
                "/rekpiper/camera/{}/points_recognition".format(name),
                PointCloud2,
                queue_size=1,
            )
            self._mask_publishers[name] = rospy.Publisher(
                "/rekpiper/camera/{}/recognition_mask".format(name),
                Image,
                queue_size=1,
            )
            rospy.Subscriber(
                "/rekpiper/camera/{}/points_base".format(name),
                PointCloud2,
                self._callback,
                callback_args=name,
                queue_size=1,
                buff_size=8 * 1024 * 1024,
            )
        self._status_pub = rospy.Publisher(
            "/rekpiper/camera/recognition_workspace/status",
            String,
            queue_size=1,
            latch=True,
        )
        self._publish_status()
        rospy.loginfo(
            "Recognition workspace filter: min=%s max=%s",
            self._bounds_min.tolist(),
            self._bounds_max.tolist(),
        )

    def _workspace_value(self, parameter_name, payload_name):
        if self._signed_workspace is not None:
            if payload_name not in self._signed_workspace:
                raise rospy.ROSInitException(
                    "signed workspace field is missing: " + payload_name)
            return self._signed_workspace[payload_name]
        return rospy.get_param(parameter_name)

    def _vector_param(self, name, payload_name):
        value = np.asarray(
            self._workspace_value(name, payload_name), dtype=np.float32)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise rospy.ROSInitException(name + " must be a finite XYZ vector")
        return value

    @staticmethod
    def _organized_xyz(message):
        if message.header.frame_id != "base_link":
            raise ValueError("input cloud must use base_link")
        if (
            message.height <= 1
            or message.point_step != 12
            or message.row_step != message.width * message.point_step
            or len(message.fields) != 3
            or [field.name for field in message.fields] != ["x", "y", "z"]
            or [field.offset for field in message.fields] != [0, 4, 8]
            or message.is_bigendian
        ):
            raise ValueError("expected ReKep organized contiguous XYZ cloud")
        return np.frombuffer(message.data, dtype=np.float32).reshape(
            message.height, message.width, 3
        )

    def _callback(self, message, name):
        try:
            if self._release is not None:
                assert_release_unchanged(self._release)
            xyz = self._organized_xyz(message)
            filtered, valid = filter_xyz_bounds(
                xyz, self._bounds_min, self._bounds_max
            )
        except AcceptanceError as exc:
            rospy.logfatal("workspace release changed after validation: %s", exc)
            rospy.signal_shutdown("signed workspace release changed")
            return
        except ValueError as exc:
            rospy.logwarn_throttle(
                2.0, "%s workspace cloud rejected: %s", name, exc
            )
            return

        output = copy.copy(message)
        output.is_dense = False
        output.data = np.ascontiguousarray(
            filtered, dtype=np.float32
        ).tobytes()
        self._publishers[name].publish(output)
        mask_message = self._bridge.cv2_to_imgmsg(
            valid.astype(np.uint8) * 255, encoding="mono8"
        )
        mask_message.header = message.header
        self._mask_publishers[name].publish(mask_message)

        finite_input = int(np.count_nonzero(np.all(np.isfinite(xyz), axis=-1)))
        retained = int(np.count_nonzero(valid))
        with self._lock:
            self._camera_status[name] = {
                "stamp": message.header.stamp.to_sec(),
                "input_finite_points": finite_input,
                "retained_points": retained,
                "retained_ratio_of_finite": (
                    float(retained) / finite_input if finite_input else 0.0
                ),
            }
        self._publish_status()

    def _publish_status(self):
        with self._lock:
            cameras = dict(self._camera_status)
        ready = set(cameras) == set(self._camera_names)
        payload = {
            "schema_version": 1,
            "state": "READY" if ready else "WAITING",
            "frame_id": "base_link",
            "workspace_bounds_min": self._bounds_min.tolist(),
            "workspace_bounds_max": self._bounds_max.tolist(),
            "table_height_m": self._table_height_m,
            "below_table_allowance_m": self._below_table_m,
            "xy_margin_m": float(self._workspace_value(
                "~xy_margin_m", "xy_margin_m")),
            "input_cameras": list(self._camera_names),
            "recognition_cloud_topics": {
                name: "/rekpiper/camera/{}/points_recognition".format(name)
                for name in self._camera_names
            },
            "cameras": cameras,
            "planning_authorized": False,
            "motion_command_capable": False,
        }
        self._status_pub.publish(
            String(data=json.dumps(payload, sort_keys=True, separators=(",", ":")))
        )


if __name__ == "__main__":
    rospy.init_node("recognition_workspace_filter")
    try:
        WorkspaceFilterNode()
        rospy.spin()
    except rospy.ROSInitException as exc:
        rospy.logfatal(str(exc))
        raise
