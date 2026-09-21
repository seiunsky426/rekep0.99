"""Geometry for a fixed-target, eye-in-hand board-to-base calibration.

All matrices use the project convention ``A_T_B``: a point in frame B is
expressed in frame A.  A fixed checkerboard observed by a camera mounted on
``link6`` satisfies::

    base_T_link6[i] * link6_T_camera * camera_T_board[i] = base_T_board

The two unknown constant transforms are solved together.  OpenCV supplies the
Robot-World/Hand-Eye initializer; this module normalizes its frame convention
and evaluates the resulting closure in the convention used by the ROS code.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from rekpiper_calibration.checkerboard_calibration import (
    rotation_distance_deg, validate_rigid_transform)


class RobotWorldHandeyeError(ValueError):
    """Raised when a board-to-base dataset is not safely solvable."""


def invert_transform(matrix, name="transform"):
    matrix = validate_rigid_transform(matrix, name)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inverse


def transform_deviation(reference, candidate):
    """Return translation and rotation distance from reference to candidate."""
    delta = invert_transform(reference, "reference") @ validate_rigid_transform(
        candidate, "candidate")
    return {
        "translation_m": float(np.linalg.norm(delta[:3, 3])),
        "rotation_deg": float(rotation_distance_deg(np.eye(4), delta)),
    }


def average_transforms(transforms):
    """Return a deterministic SE(3) average suitable for stopped-pose batches."""
    values = [validate_rigid_transform(value, "transform") for value in transforms]
    if not values:
        raise RobotWorldHandeyeError("cannot average an empty transform list")
    rotations = np.sum([value[:3, :3] for value in values], axis=0)
    left, _singular, right = np.linalg.svd(rotations)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.median(np.asarray([value[:3, 3] for value in values]), axis=0)
    return validate_rigid_transform(result, "average_transform")


def pose_span_metrics(base_T_effector):
    values = [validate_rigid_transform(value, "base_T_effector")
              for value in base_T_effector]
    if len(values) < 2:
        raise RobotWorldHandeyeError("at least two poses are required for span metrics")
    translations = np.asarray([value[:3, 3] for value in values])
    maximum_rotation = max(
        rotation_distance_deg(first, second)
        for index, first in enumerate(values) for second in values[index + 1:])
    return {
        "translation_axis_span_m": (np.max(translations, axis=0) -
                                    np.min(translations, axis=0)).tolist(),
        "maximum_pair_rotation_deg": float(maximum_rotation),
    }


def check_pose_diversity(base_T_effector, minimum_axis_span_m,
                         minimum_pair_rotation_deg, minimum_pose_count):
    if len(base_T_effector) < int(minimum_pose_count):
        raise RobotWorldHandeyeError(
            "requires at least {} poses".format(minimum_pose_count))
    metrics = pose_span_metrics(base_T_effector)
    spans = np.asarray(metrics["translation_axis_span_m"], dtype=float)
    minima = np.asarray(minimum_axis_span_m, dtype=float).reshape(3)
    if np.any(spans < minima):
        raise RobotWorldHandeyeError(
            "insufficient translation span: observed={}, required={}".format(
                spans.tolist(), minima.tolist()))
    if metrics["maximum_pair_rotation_deg"] < float(minimum_pair_rotation_deg):
        raise RobotWorldHandeyeError(
            "insufficient rotation span: observed={:.3f}deg, required={:.3f}deg".format(
                metrics["maximum_pair_rotation_deg"], minimum_pair_rotation_deg))
    return metrics


def _opencv_method(method):
    names = {
        "SHAH": "CALIB_ROBOT_WORLD_HAND_EYE_SHAH",
        "LI": "CALIB_ROBOT_WORLD_HAND_EYE_LI",
    }
    try:
        return getattr(cv2, names[method])
    except (AttributeError, KeyError):
        raise RobotWorldHandeyeError("unsupported Robot-World/Hand-Eye method " + str(method))


def _as_rt(matrices):
    rotations = [value[:3, :3].copy() for value in matrices]
    translations = [value[:3, 3].reshape(3, 1).copy() for value in matrices]
    return rotations, translations


def solve_robot_world_handeye(base_T_effector, camera_T_board, method="SHAH"):
    """Solve ``base_T_board`` and ``link6_T_camera`` from stopped poses.

    OpenCV takes ``camera_T_world`` and the *gripper-to-base* motion matrices,
    then returns ``world_T_base`` and ``camera_T_gripper``.  Its Python argument
    is named ``R_base2gripper``, but OpenCV's own sample convention and the
    synthetic recovery test require the inverse of ROS ``base_T_link6`` here.
    World is checkerboard and gripper is link6, so both outputs are inverted
    before returning them in the project's parent-to-child notation.
    """
    base_values = [validate_rigid_transform(value, "base_T_effector")
                   for value in base_T_effector]
    camera_values = [validate_rigid_transform(value, "camera_T_board")
                     for value in camera_T_board]
    if len(base_values) != len(camera_values) or len(base_values) < 3:
        raise RobotWorldHandeyeError("matching base and camera observations (at least 3) required")
    world_to_camera_rotation, world_to_camera_translation = _as_rt(camera_values)
    gripper_to_base = [invert_transform(value, "base_T_effector")
                       for value in base_values]
    base_to_gripper_rotation, base_to_gripper_translation = _as_rt(gripper_to_base)
    try:
        rotation_world_to_base, translation_world_to_base, \
            rotation_camera_to_gripper, translation_camera_to_gripper = \
            cv2.calibrateRobotWorldHandEye(
                world_to_camera_rotation, world_to_camera_translation,
                base_to_gripper_rotation, base_to_gripper_translation,
                method=_opencv_method(method))
    except cv2.error as exc:
        raise RobotWorldHandeyeError("OpenCV {} solve failed: {}".format(method, exc))
    world_T_base = np.eye(4, dtype=np.float64)
    world_T_base[:3, :3] = rotation_world_to_base
    world_T_base[:3, 3] = np.asarray(translation_world_to_base).reshape(3)
    camera_T_gripper = np.eye(4, dtype=np.float64)
    camera_T_gripper[:3, :3] = rotation_camera_to_gripper
    camera_T_gripper[:3, 3] = np.asarray(translation_camera_to_gripper).reshape(3)
    return {
        "method": method,
        "base_T_board": invert_transform(world_T_base, "world_T_base"),
        "link6_T_camera": invert_transform(camera_T_gripper, "camera_T_gripper"),
    }


def closure_metrics(base_T_effector, camera_T_board, base_T_board, link6_T_camera):
    """Measure fixed-board closure residuals in metres/degrees per pose."""
    residuals = []
    for base_T_link6, observed_camera_T_board in zip(base_T_effector, camera_T_board):
        predicted_camera_T_board = invert_transform(link6_T_camera) @ \
            invert_transform(base_T_link6) @ base_T_board
        residuals.append(transform_deviation(predicted_camera_T_board,
                                             observed_camera_T_board))
    translations = np.asarray([value["translation_m"] for value in residuals], dtype=float)
    rotations = np.asarray([value["rotation_deg"] for value in residuals], dtype=float)
    return {
        "per_pose": residuals,
        "translation_median_m": float(np.median(translations)),
        "translation_p95_m": float(np.percentile(translations, 95)),
        "translation_maximum_m": float(np.max(translations)),
        "rotation_median_deg": float(np.median(rotations)),
        "rotation_p95_deg": float(np.percentile(rotations, 95)),
        "rotation_maximum_deg": float(np.max(rotations)),
    }


def select_solution(base_T_effector, camera_T_board, methods=("SHAH", "LI")):
    """Select the legal OpenCV initializer with the smallest robust closure."""
    candidates = []
    failures = {}
    for method in methods:
        try:
            candidate = solve_robot_world_handeye(base_T_effector, camera_T_board, method)
            candidate["closure"] = closure_metrics(
                base_T_effector, camera_T_board, candidate["base_T_board"],
                candidate["link6_T_camera"])
            candidates.append(candidate)
        except RobotWorldHandeyeError as exc:
            failures[method] = str(exc)
    if not candidates:
        raise RobotWorldHandeyeError("all Robot-World/Hand-Eye methods failed: {}".format(failures))
    candidates.sort(key=lambda item: (
        item["closure"]["translation_median_m"],
        item["closure"]["rotation_median_deg"], item["method"]))
    selected = candidates[0]
    selected["all_method_metrics"] = [{
        "method": item["method"], "closure": item["closure"]} for item in candidates]
    selected["method_failures"] = failures
    return selected


def bootstrap_uncertainty(base_T_effector, camera_T_board, samples=200, seed=20260723):
    """Bootstrap the initializer; return spread around the full-data estimate."""
    if samples < 1:
        raise RobotWorldHandeyeError("bootstrap sample count must be positive")
    reference = select_solution(base_T_effector, camera_T_board)
    rng = np.random.default_rng(seed)
    board_errors, camera_errors = [], []
    count = len(base_T_effector)
    for _ in range(samples):
        indices = rng.integers(0, count, size=count)
        try:
            candidate = select_solution([base_T_effector[i] for i in indices],
                                        [camera_T_board[i] for i in indices])
        except RobotWorldHandeyeError:
            continue
        board_errors.append(transform_deviation(reference["base_T_board"],
                                                 candidate["base_T_board"]))
        camera_errors.append(transform_deviation(reference["link6_T_camera"],
                                                  candidate["link6_T_camera"]))
    if len(board_errors) < max(10, samples // 2):
        raise RobotWorldHandeyeError("insufficient successful bootstrap solves")
    def _spread(values):
        return {
            "translation_std_m": float(np.std([item["translation_m"] for item in values], ddof=1)),
            "rotation_std_deg": float(np.std([item["rotation_deg"] for item in values], ddof=1)),
        }
    return {
        "requested_samples": int(samples), "successful_samples": len(board_errors),
        "base_T_board": _spread(board_errors), "link6_T_camera": _spread(camera_errors),
    }
