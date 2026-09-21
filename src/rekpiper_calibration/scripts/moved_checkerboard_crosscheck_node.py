#!/usr/bin/env python3
"""Live 88-corner cross-check for two already-calibrated fixed D435 cameras.

The checkerboard may be moved after the fixed-camera calibration.  Each camera
deprojects the aligned depth at every detected inner corner and transforms the
result with its saved base_T_link.  The two resulting base coordinates are
compared by matching corner index.  This node is deliberately read-only: it
never publishes base_link -> camera TF.

An explicitly enabled diagnostic fallback permits a checkerboard without the
configured ArUco orientation marker.  In that mode RS1's detector order is the
temporary numbering reference and RS3's only possible 180-degree reversal is
selected from the measured 3-D agreement.  The fallback is not an absolute
board-frame calibration and is never written to the camera extrinsics.
"""

import json
from collections import deque
from pathlib import Path
import threading

import cv2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
import tf2_ros
import yaml

from rekpiper_calibration.checkerboard_calibration import (
    CalibrationFailure, QualityThresholds, TargetDefinition,
    checkerboard_object_points, image_quality, solve_checkerboard_pose,
    validate_rigid_transform)
from rekpiper_camera.projection import transform_to_matrix


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class MovedCheckerboardCrosscheck:
    def __init__(self):
        config_path = Path(rospy.get_param("~checkerboard_config")).expanduser()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self._base_frame = str(config.get("base_frame", "base_link"))
        self._target = TargetDefinition.from_dict(config["target"])
        self._quality = QualityThresholds.from_dict(config.get("quality", {}))
        self._pair_max_skew_s = float(rospy.get_param("~pair_max_skew_s", 0.10))
        self._median_limit_m = float(rospy.get_param("~median_limit_m", 0.005))
        self._p95_limit_m = float(rospy.get_param("~p95_limit_m", 0.008))
        self._tf_timeout_s = float(rospy.get_param("~tf_timeout_s", 0.10))
        self._depth_scale_m = float(rospy.get_param(
            "~depth_scale_m", 0.001))
        self._depth_patch_radius = int(rospy.get_param(
            "~depth_patch_radius_px", 4))
        self._minimum_valid_corners = int(rospy.get_param(
            "~minimum_valid_depth_corners", 80))
        self._allow_missing_marker = bool(rospy.get_param(
            "~allow_missing_orientation_marker_for_depth_crosscheck", False))
        self._bridge = CvBridge()
        self._lock = threading.RLock()
        self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        self._cameras = {
            "rs1": self._load_camera("rs1", rospy.get_param("~rs1_extrinsics")),
            "rs3": self._load_camera("rs3", rospy.get_param("~rs3_extrinsics")),
        }
        # PnP takes a non-zero, camera-dependent amount of CPU time.  Keep a
        # short history and pair by acquisition timestamp, not callback
        # completion order.
        self._history = {"rs1": deque(maxlen=20), "rs3": deque(maxlen=20)}
        # Display frames are kept independently from valid geometric
        # observations, so the operator can still see both camera views when
        # a board, marker, or TF quality check fails.
        self._views = {}
        self._annotated = {
            name: rospy.Publisher("~{}/annotated_image".format(name), Image, queue_size=1)
            for name in self._cameras
        }
        self._combined = rospy.Publisher("~combined_image", Image, queue_size=1)
        self._status = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self._syncs, self._subscribers = [], []
        for name, camera in self._cameras.items():
            image = message_filters.Subscriber(camera["image_topic"], Image, queue_size=3)
            info = message_filters.Subscriber(camera["camera_info_topic"], CameraInfo, queue_size=3)
            depth = message_filters.Subscriber(
                camera["aligned_depth_topic"], Image, queue_size=3)
            sync = message_filters.ApproximateTimeSynchronizer(
                [image, info, depth], queue_size=10, slop=0.05,
                allow_headerless=False)
            sync.registerCallback(self._callback, name)
            self._subscribers.extend((image, info, depth))
            self._syncs.append(sync)
        self._publish("WAITING_FOR_RS1_RS3", "waiting for valid checkerboard in both views")

    def _load_camera(self, name, path_value):
        path = Path(str(path_value)).expanduser()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data.get("logical_name") != name:
            raise rospy.ROSInitException("{} metadata logical_name mismatch".format(name))
        if data.get("parent_frame") != self._base_frame or data.get("child_frame") != name + "_link":
            raise rospy.ROSInitException("{} metadata frame mismatch".format(name))
        matrix = validate_rigid_transform(np.asarray(data["base_T_link"], dtype=float),
                                          "base_T_" + name + "_link")
        return {
            "serial": str(data["serial"]).lstrip("_"), "base_T_link": matrix,
            "link_frame": name + "_link", "optical_frame": name + "_color_optical_frame",
            "image_topic": "/{}/color/image_raw".format(name),
            "camera_info_topic": "/{}/color/camera_info".format(name),
            "aligned_depth_topic":
                "/{}/aligned_depth_to_color/image_raw".format(name),
            "metadata": str(path),
        }

    def _driver_serial(self, name):
        for parameter in ("/{}/realsense2_camera/serial_no".format(name),
                          "/{}/serial_no".format(name)):
            if rospy.has_param(parameter):
                return str(rospy.get_param(parameter)).lstrip("_")
        return None

    def _publish(self, state, reason=None, comparison=None):
        payload = {"state": state, "reason": reason,
                   "checkerboard_inner_corner_count": self._target.pattern_cols * self._target.pattern_rows,
                   "limits_m": {"median": self._median_limit_m, "p95": self._p95_limit_m}}
        if comparison is not None:
            payload["comparison"] = comparison
        self._status.publish(String(data=json.dumps(_plain(payload), sort_keys=True)))

    @staticmethod
    def _base_points(base_T_board, object_points):
        points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
        return (base_T_board @ homogeneous.T).T[:, :3]

    def _depth_base_points(
            self, depth_message, camera_matrix, distortion, corners,
            base_T_optical):
        depth = self._bridge.imgmsg_to_cv2(
            depth_message, desired_encoding="passthrough")
        if depth.ndim != 2:
            raise CalibrationFailure("aligned_depth_not_single_channel")
        if depth_message.encoding in ("16UC1", "mono16"):
            depth_m = depth.astype(np.float64) * self._depth_scale_m
        elif depth_message.encoding == "32FC1":
            depth_m = depth.astype(np.float64)
        else:
            raise CalibrationFailure(
                "unsupported_aligned_depth_encoding_" +
                str(depth_message.encoding))
        height, width = depth_m.shape
        points = np.full((len(corners), 3), np.nan, dtype=np.float64)
        for index, pixel in enumerate(np.asarray(corners).reshape(-1, 2)):
            u, v = int(round(float(pixel[0]))), int(round(float(pixel[1])))
            radius = self._depth_patch_radius
            x0, x1 = max(0, u - radius), min(width, u + radius + 1)
            y0, y1 = max(0, v - radius), min(height, v + radius + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            values = depth_m[y0:y1, x0:x1]
            values = values[
                np.isfinite(values) & (values >= 0.10) & (values <= 3.0)]
            if values.size == 0:
                continue
            z = float(np.median(values))
            normalized = cv2.undistortPoints(
                np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2),
                camera_matrix, distortion).reshape(2)
            camera_point = np.asarray(
                [normalized[0] * z, normalized[1] * z, z, 1.0])
            points[index] = (base_T_optical @ camera_point)[:3]
        return points

    def _unoriented_checkerboard_pose(
            self, image, camera_matrix, distortion):
        """Detect all corners without assigning the board's absolute origin."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flags = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
                 | cv2.CALIB_CB_ACCURACY)
        found, corners = cv2.findChessboardCornersSB(
            gray, (self._target.pattern_cols, self._target.pattern_rows),
            flags=flags)
        expected = self._target.pattern_cols * self._target.pattern_rows
        if not found or corners is None or len(corners) != expected:
            raise CalibrationFailure("checkerboard_not_detected")

        metrics = image_quality(image, corners, self._target)
        checks = (
            (metrics["board_area_ratio"] <
             self._quality.minimum_board_area_ratio, "board_area_too_small"),
            (metrics["median_cell_px"] <
             self._quality.minimum_median_cell_px, "median_cell_too_small"),
            (metrics["minimum_cell_px"] <
             self._quality.minimum_cell_px, "cell_too_small"),
            (metrics["minimum_border_px"] <
             self._quality.image_margin_px, "checkerboard_near_image_edge"),
            (metrics["laplacian_variance"] <
             self._quality.minimum_laplacian_variance, "image_blurred"),
            (metrics["saturation_ratio"] >
             self._quality.maximum_saturation_ratio, "image_saturated"),
        )
        reasons = [reason for failed, reason in checks if failed]
        if reasons:
            raise CalibrationFailure(",".join(reasons))

        # This PnP value is diagnostic only; depth supplies every reported 3-D
        # coordinate.  It remains useful for spotting a poor RGB corner fit.
        object_points = checkerboard_object_points(self._target)
        ok, rvec, tvec = cv2.solvePnP(
            object_points, corners, camera_matrix, distortion,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise CalibrationFailure("diagnostic_pnp_failed")
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion)
        residuals = projected.reshape(-1, 2) - corners.reshape(-1, 2)
        metrics["reprojection_rmse_px"] = float(np.sqrt(
            np.mean(np.sum(residuals * residuals, axis=1))))

        annotated = image.copy()
        cv2.drawChessboardCorners(
            annotated,
            (self._target.pattern_cols, self._target.pattern_rows),
            corners, True)
        return {
            "corners": np.asarray(corners, dtype=np.float64),
            "annotated": annotated,
            "metrics": metrics,
            "orientation_source": "temporary_detector_order_no_marker",
        }

    def _observation(
            self, name, image_message, info_message, depth_message, image):
        camera = self._cameras[name]
        if image_message.header.frame_id != camera["optical_frame"] or \
                info_message.header.frame_id != camera["optical_frame"] or \
                depth_message.header.frame_id != camera["optical_frame"]:
            raise CalibrationFailure(name + "_optical_frame_mismatch")
        if (image_message.width, image_message.height) != \
                (depth_message.width, depth_message.height):
            raise CalibrationFailure(name + "_aligned_depth_size_mismatch")
        if self._driver_serial(name) != camera["serial"]:
            raise CalibrationFailure(name + "_serial_mismatch_or_missing")
        camera_matrix = np.asarray(
            info_message.K, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(info_message.D, dtype=np.float64)
        try:
            pose = solve_checkerboard_pose(
                image, camera_matrix, distortion, self._target, self._quality)
            pose["orientation_source"] = "aruco_marker"
        except CalibrationFailure as exc:
            if not self._allow_missing_marker or \
                    not str(exc).startswith("orientation_marker_count_"):
                raise
            pose = self._unoriented_checkerboard_pose(
                image, camera_matrix, distortion)
        link_T_camera = transform_to_matrix(self._tf_buffer.lookup_transform(
            camera["link_frame"], camera["optical_frame"], image_message.header.stamp,
            rospy.Duration(self._tf_timeout_s)).transform)
        base_T_optical = camera["base_T_link"] @ link_T_camera
        base_points = self._depth_base_points(
            depth_message, camera_matrix, distortion, pose["corners"],
            base_T_optical)
        valid_count = int(np.count_nonzero(
            np.all(np.isfinite(base_points), axis=1)))
        if valid_count < self._minimum_valid_corners:
            raise CalibrationFailure(
                "{}_valid_depth_corners_{}_below_{}".format(
                    name, valid_count, self._minimum_valid_corners))
        return {
            "stamp": image_message.header.stamp.to_sec(), "header": image_message.header,
            "annotated": pose["annotated"], "base_points": base_points,
            "valid_depth_corners": valid_count,
            "rmse_px": pose["metrics"]["reprojection_rmse_px"],
            "orientation_source": pose["orientation_source"],
        }

    @staticmethod
    def _label(image, text, ok):
        output = image.copy()
        color = (0, 190, 0) if ok else (0, 0, 220)
        cv2.rectangle(output, (0, 0), (output.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(output, text[:120], (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, color, 2, cv2.LINE_AA)
        return output

    def _publish_image(self, publisher, image, header):
        message = self._bridge.cv2_to_imgmsg(image, "bgr8")
        message.header = header
        publisher.publish(message)

    def _comparison(self):
        if not self._history["rs1"] or not self._history["rs3"]:
            return None, "waiting for valid checkerboard in both views"
        pairs = [(abs(first["stamp"] - second["stamp"]), first, second)
                 for first in self._history["rs1"] for second in self._history["rs3"]]
        skew, first, second = min(pairs, key=lambda value: (value[0],
                                                             -max(value[1]["stamp"],
                                                                  value[2]["stamp"])))
        if skew > self._pair_max_skew_s:
            return None, "rs1_rs3_image_skew_{:.1f}ms_exceeds_{:.1f}ms".format(
                skew * 1000.0, self._pair_max_skew_s * 1000.0)
        second_points_all = second["base_points"]
        rs3_reversed = False
        orientation_mode = "aruco_marker_absolute_numbering"
        if first["orientation_source"] != "aruco_marker" or \
                second["orientation_source"] != "aruco_marker":
            orientation_mode = "temporary_rs1_order_depth_matched"
            candidates = []
            for reversed_order, candidate in (
                    (False, second_points_all),
                    (True, second_points_all[::-1])):
                candidate_valid = \
                    np.all(np.isfinite(first["base_points"]), axis=1) & \
                    np.all(np.isfinite(candidate), axis=1)
                if np.count_nonzero(candidate_valid) < self._minimum_valid_corners:
                    continue
                candidate_errors = np.linalg.norm(
                    first["base_points"][candidate_valid] -
                    candidate[candidate_valid], axis=1)
                candidates.append((
                    float(np.median(candidate_errors)),
                    reversed_order, candidate))
            if not candidates:
                return None, "no_depth_valid_orientation_candidate"
            _, rs3_reversed, second_points_all = min(
                candidates, key=lambda item: item[0])

        valid = np.all(np.isfinite(first["base_points"]), axis=1) & \
            np.all(np.isfinite(second_points_all), axis=1)
        indices = np.flatnonzero(valid)
        if len(indices) < self._minimum_valid_corners:
            return None, "paired_valid_depth_corners_{}_below_{}".format(
                len(indices), self._minimum_valid_corners)
        first_points = first["base_points"][indices]
        second_points = second_points_all[indices]
        deltas = first_points - second_points
        errors = np.linalg.norm(deltas, axis=1)
        summary = {
            "image_skew_ms": float(skew * 1000.0),
            "valid_corner_count": int(len(indices)),
            "corner_indices": indices,
            "rs1_base_points_m": first_points,
            "rs3_base_points_m": second_points,
            "rs1_minus_rs3_m": deltas,
            "median_error_m": float(np.median(errors)),
            "p95_error_m": float(np.percentile(errors, 95)),
            "maximum_error_m": float(np.max(errors)),
            "corner_error_m": errors,
            "rs1_reprojection_rmse_px": first["rmse_px"],
            "rs3_reprojection_rmse_px": second["rmse_px"],
            "corner_numbering": orientation_mode,
            "rs3_order_reversed_for_correspondence": bool(rs3_reversed),
        }
        return summary, None

    def _combined_image(self, first, second, summary, reason):
        left, right = first["image"], second["image"]
        height = max(left.shape[0], right.shape[0])
        def resize(image):
            if image.shape[0] == height:
                return image
            return cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height))
        combined = np.hstack((resize(left), resize(right)))
        if summary is None:
            text, ok = "WAITING: " + reason, False
        else:
            ok = summary["median_error_m"] <= self._median_limit_m and \
                summary["p95_error_m"] <= self._p95_limit_m
            text = ("%d depth corners: median %.2f mm, P95 %.2f mm, max %.2f mm" %
                    (summary["valid_corner_count"],
                     summary["median_error_m"] * 1000.0, summary["p95_error_m"] * 1000.0,
                     summary["maximum_error_m"] * 1000.0))
        return self._label(combined, text, ok)

    def _callback(self, image_message, info_message, depth_message, name):
        image = None
        try:
            image = self._bridge.imgmsg_to_cv2(image_message, "bgr8")
            observation = self._observation(
                name, image_message, info_message, depth_message, image)
            observation["image"] = self._label(
                observation["annotated"],
                "{}: {}/88 depth corners, RGB RMS {:.3f}px".format(
                    name, observation["valid_depth_corners"],
                    observation["rmse_px"]), True)
        except (CalibrationFailure, CvBridgeError, ValueError, tf2_ros.TransformException) as exc:
            reason = str(exc)
            if image is not None:
                view = {"image": self._label(image, "{}: WAITING {}".format(name, reason),
                                               False), "header": image_message.header}
                with self._lock:
                    self._history[name].clear()
                    self._views[name] = view
                    if set(self._views) == {"rs1", "rs3"}:
                        combined = self._combined_image(self._views["rs1"], self._views["rs3"],
                                                        None, reason)
                        self._publish_image(self._combined, combined, image_message.header)
                    self._publish_image(self._annotated[name], view["image"], view["header"])
            self._publish("WAITING_FOR_VALID_OBSERVATION", reason)
            return
        with self._lock:
            self._history[name].append(observation)
            self._views[name] = observation
            summary, reason = self._comparison()
            self._publish_image(self._annotated[name], observation["image"], observation["header"])
            if set(self._views) == {"rs1", "rs3"}:
                combined = self._combined_image(self._views["rs1"], self._views["rs3"],
                                                summary, reason)
                self._publish_image(self._combined, combined, observation["header"])
            if summary is None:
                self._publish("WAITING_FOR_VALID_PAIR", reason)
            else:
                ok = summary["median_error_m"] <= self._median_limit_m and \
                    summary["p95_error_m"] <= self._p95_limit_m
                self._publish("CROSSCHECK_OK" if ok else "CROSSCHECK_OUT_OF_LIMIT",
                              None, summary)


if __name__ == "__main__":
    rospy.init_node("moved_checkerboard_crosscheck")
    MovedCheckerboardCrosscheck()
    rospy.spin()
