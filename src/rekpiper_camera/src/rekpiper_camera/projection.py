"""Vectorized aligned-depth projection and rigid transforms.

Source classification: PORT + SAFETY.

Projection formula reference:
Everloom-129/ReKep@a8d94aa9332f2caf7a4df8082d428f4b5e533e9b
demonstrates real RGB-D back-projection. This implementation replaces its
hard-coded intrinsics and file input with validated CameraInfo values and
timestamped ROS TF supplied by the caller.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_camera_matrix(
        cls, matrix: Sequence[float], width: int, height: int
    ) -> "CameraIntrinsics":
        if len(matrix) != 9:
            raise ValueError("CameraInfo.K must contain exactly 9 values")
        intrinsics = cls(
            fx=float(matrix[0]),
            fy=float(matrix[4]),
            cx=float(matrix[2]),
            cy=float(matrix[5]),
            width=int(width),
            height=int(height),
        )
        intrinsics.validate()
        return intrinsics

    def validate(self) -> None:
        values = np.asarray([self.fx, self.fy, self.cx, self.cy], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        if not (0.0 <= self.cx < self.width and 0.0 <= self.cy < self.height):
            raise ValueError("camera principal point lies outside the image")


def depth_to_xyz(
    depth_raw: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_scale: float,
    min_depth_m: float = 0.10,
    max_depth_m: float = 2.00,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project aligned depth into the color optical frame.

    Invalid samples are represented as NaN in ``xyz`` and False in the mask.
    The input must be aligned to the color image because color intrinsics are
    used. No fallback or rescaling is performed for mismatched dimensions.
    """
    intrinsics.validate()
    depth = np.asarray(depth_raw)
    expected_shape = (intrinsics.height, intrinsics.width)
    if depth.ndim != 2 or depth.shape != expected_shape:
        raise ValueError(
            "aligned depth shape {} does not match CameraInfo {}".format(
                depth.shape, expected_shape
            )
        )
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("depth_scale must be finite and positive")
    if not (0.0 <= min_depth_m < max_depth_m):
        raise ValueError("expected 0 <= min_depth_m < max_depth_m")

    z = depth.astype(np.float32, copy=False) * np.float32(depth_scale)
    valid = (
        np.isfinite(z)
        & (z > 0.0)
        & (z >= np.float32(min_depth_m))
        & (z <= np.float32(max_depth_m))
    )

    v, u = np.indices(expected_shape, dtype=np.float32)
    x = (u - np.float32(intrinsics.cx)) * z / np.float32(intrinsics.fx)
    y = (v - np.float32(intrinsics.cy)) * z / np.float32(intrinsics.fy)
    xyz = np.stack((x, y, z), axis=-1).astype(np.float32, copy=False)
    xyz[~valid] = np.nan
    return xyz, valid


def transform_to_matrix(transform) -> np.ndarray:
    """Convert a geometry_msgs-like Transform into a 4x4 matrix."""
    q = np.asarray(
        [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("TF quaternion is invalid")
    x, y, z, w = q / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("TF transform contains non-finite values")
    return matrix


def transform_points(xyz: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Transform an organized or flat XYZ array while preserving NaNs."""
    points = np.asarray(xyz, dtype=np.float32)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("xyz must have final dimension 3")
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("matrix must be a finite 4x4 transform")

    result = points @ transform[:3, :3].T + transform[:3, 3]
    return result.astype(np.float32, copy=False)
