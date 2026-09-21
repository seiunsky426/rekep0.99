#!/usr/bin/env python3
"""Publish the accepted, identity-bound fixed-camera TF pair."""

from pathlib import Path
import time

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TransformStamped
import numpy as np
import rospy
from std_msgs.msg import Header
import tf.transformations as transformations
import tf2_ros
import yaml

from rekpiper_acceptance import (
    AcceptanceError, assert_release_unchanged, validate_release_bundle)
from rekpiper_camera.extrinsics import validate_rigid_transform


class DualExtrinsicsPublisher:
    def __init__(self):
        self._diagnostics = rospy.Publisher("/diagnostics", DiagnosticArray, queue_size=1, latch=True)
        transforms = []
        precision_allowed = True
        statuses = []
        allow_provisional = bool(rospy.get_param("~allow_provisional", False))
        mode = str(rospy.get_param("~mode", "shadow"))
        if mode not in ("shadow", "autonomous"):
            raise rospy.ROSInitException("mode must be shadow or autonomous")
        signed_values = None
        self._release = None
        if mode == "autonomous":
            try:
                self._release = validate_release_bundle(
                    rospy.get_param("~release_bundle", ""),
                    rospy.get_param("~acceptance_public_key", ""),
                    rospy.get_param("~minimum_release_counter", ""),
                    rospy.get_param("~robot_id", ""))
                signed_values = {
                    name: self._release["verified_artifacts"][
                        "camera_extrinsics_" + name]["payload"]
                    for name in ("rs1", "rs3")}
            except (AcceptanceError, KeyError) as exc:
                raise rospy.ROSInitException(
                    "signed camera release rejected: " + str(exc))
        camera_names = [str(value) for value in rospy.get_param(
            "~camera_names", ["rs1", "rs3"])]
        if (not camera_names or len(set(camera_names)) != len(camera_names)
                or set(camera_names) - {"rs1", "rs3"}):
            raise rospy.ROSInitException(
                "camera_names must be a non-empty unique subset of rs1,rs3")
        for name in camera_names:
            path = Path(rospy.get_param("~{}_extrinsics".format(name))).expanduser()
            expected_serial = str(rospy.get_param("~serial_{}".format(name))).lstrip("_")
            metadata = (signed_values[name] if signed_values is not None else
                        (yaml.safe_load(path.read_text(encoding="utf-8")) or {}))
            if mode == "shadow" and metadata.get("status") == "UNCALIBRATED":
                statuses.append("UNCALIBRATED")
                precision_allowed = False
                continue
            self._validate_metadata(
                name, expected_serial, metadata,
                allow_provisional=allow_provisional)
            statuses.append(str(metadata["status"]))
            precision_allowed = precision_allowed and (
                metadata.get("precision_operation_allowed") is True)
            self._wait_for_driver_identity(name, expected_serial)
            transforms.append(self._message(metadata))
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        if transforms:
            broadcaster.sendTransform(transforms)
        self._broadcaster = broadcaster
        release_status = (
            "ACCEPTED" if set(statuses) == {"ACCEPTED"}
            else ("UNCALIBRATED" if "UNCALIBRATED" in statuses
                  else "PROVISIONAL_DIAGNOSTIC"))
        rospy.set_param(
            "/rekpiper/camera/extrinsics/precision_operation_allowed",
            bool(precision_allowed))
        rospy.set_param(
            "/rekpiper/camera/extrinsics/status", release_status)
        self._diagnostics.publish(DiagnosticArray(
            header=Header(stamp=rospy.Time.now()),
            status=[DiagnosticStatus(
                level=DiagnosticStatus.WARN if not precision_allowed else DiagnosticStatus.OK,
                name="rekpiper_camera/dual_extrinsics_publisher",
                message=(
                    "UNCALIBRATED_NO_TF_PUBLISHED"
                    if release_status == "UNCALIBRATED" else
                    "PROVISIONAL_EXTRINSICS_PUBLISHED_PRECISION_FORBIDDEN"
                    if release_status == "PROVISIONAL_DIAGNOSTIC"
                    else ("EXTRINSICS_PUBLISHED_PRECISION_FORBIDDEN"
                          if not precision_allowed
                          else "ACCEPTED_EXTRINSICS_PUBLISHED")),
                hardware_id="_".join(camera_names),
                values=[
                    KeyValue("frames", ",".join(
                        "base_link->{}_link".format(name)
                        for name in camera_names)),
                    KeyValue("status", release_status),
                    KeyValue("precision_operation_allowed",
                             "true" if precision_allowed else "false")])]))
        if self._release is not None:
            rospy.Timer(rospy.Duration(0.2), self._release_watchdog)

    def _release_watchdog(self, _event):
        try:
            assert_release_unchanged(self._release)
        except AcceptanceError as exc:
            rospy.set_param(
                "/rekpiper/camera/extrinsics/precision_operation_allowed",
                False)
            rospy.set_param(
                "/rekpiper/camera/extrinsics/status", "REVOKED")
            rospy.logfatal("camera release changed after validation: %s", exc)
            rospy.signal_shutdown("signed camera release changed")

    @staticmethod
    def _wait_for_driver_identity(name, expected_serial):
        deadline = time.monotonic() + 10.0
        parameters = ("/{}/realsense2_camera/serial_no".format(name),
                      "/{}/serial_no".format(name))
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            for parameter in parameters:
                if rospy.has_param(parameter):
                    actual = str(rospy.get_param(parameter)).lstrip("_")
                    if actual != expected_serial:
                        raise rospy.ROSInitException(name + " live driver serial mismatch")
                    return
            rospy.sleep(0.1)
        raise rospy.ROSInitException(name + " live driver serial parameter missing")

    @staticmethod
    def _validate_metadata(
            name, serial, metadata, allow_provisional=False):
        status = metadata.get("status")
        accepted = (
            status == "ACCEPTED"
            and metadata.get("publish_tf_allowed") is True)
        provisional = (
            allow_provisional
            and status == "PROVISIONAL_DIAGNOSTIC"
            and metadata.get("publish_tf_allowed") is True
            and metadata.get("precision_operation_allowed") is False)
        if not accepted and not provisional:
            raise rospy.ROSInitException(
                name + " calibration is neither ACCEPTED nor an explicitly "
                "enabled non-precision provisional diagnostic")
        if metadata.get("logical_name") != name or str(metadata.get("serial", "")).lstrip("_") != serial:
            raise rospy.ROSInitException(name + " calibration identity mismatch")
        if metadata.get("parent_frame") != "base_link" or metadata.get("child_frame") != name + "_link":
            raise rospy.ROSInitException(name + " calibration frame mismatch")
        matrix = validate_rigid_transform(np.asarray(metadata.get("base_T_link")), name + " base_T_link")
        saved_translation = metadata.get("translation_m", {})
        try:
            translation = np.asarray([saved_translation[axis] for axis in "xyz"], dtype=float)
        except (KeyError, TypeError, ValueError):
            raise rospy.ROSInitException(name + " translation fields invalid")
        if not np.allclose(matrix[:3, 3], translation, atol=1e-9):
            raise rospy.ROSInitException(name + " translation fields disagree")
        rotation_values = metadata.get("rotation_xyzw", {})
        try:
            quaternion = np.asarray([rotation_values[axis] for axis in "xyzw"], dtype=float)
        except (KeyError, TypeError, ValueError):
            raise rospy.ROSInitException(name + " quaternion fields invalid")
        if abs(np.linalg.norm(quaternion) - 1.0) > 1e-6:
            raise rospy.ROSInitException(name + " quaternion is not normalized")
        quaternion_matrix = transformations.quaternion_matrix(quaternion)
        if not np.allclose(matrix[:3, :3], quaternion_matrix[:3, :3], atol=1e-6):
            raise rospy.ROSInitException(name + " quaternion and matrix disagree")

    @staticmethod
    def _message(metadata):
        translation = metadata["translation_m"]
        rotation = metadata["rotation_xyzw"]
        quaternion = np.asarray([rotation[axis] for axis in "xyzw"], dtype=float)
        if not np.all(np.isfinite(quaternion)) or abs(np.linalg.norm(quaternion) - 1.0) > 1e-6:
            raise rospy.ROSInitException(metadata["logical_name"] + " quaternion invalid")
        matrix = transformations.quaternion_matrix(quaternion)
        if abs(np.linalg.det(matrix[:3, :3]) - 1.0) > 1e-6:
            raise rospy.ROSInitException(metadata["logical_name"] + " rotation invalid")
        message = TransformStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = metadata["parent_frame"]
        message.child_frame_id = metadata["child_frame"]
        message.transform.translation.x = float(translation["x"])
        message.transform.translation.y = float(translation["y"])
        message.transform.translation.z = float(translation["z"])
        message.transform.rotation.x = float(rotation["x"])
        message.transform.rotation.y = float(rotation["y"])
        message.transform.rotation.z = float(rotation["z"])
        message.transform.rotation.w = float(rotation["w"])
        return message

if __name__ == "__main__":
    rospy.init_node("dual_extrinsics_publisher")
    try:
        DualExtrinsicsPublisher()
        rospy.spin()
    except (OSError, ValueError, KeyError, yaml.YAMLError, rospy.ROSInitException) as exc:
        rospy.logfatal(str(exc))
        raise
