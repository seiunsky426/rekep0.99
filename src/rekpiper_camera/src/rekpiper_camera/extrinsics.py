"""Validation helpers for fixed-camera transforms used at runtime."""

import numpy as np


def validate_rigid_transform(matrix, name="transform"):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(name + " must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(name + " rotation is not orthonormal")
    if abs(np.linalg.det(rotation) - 1.0) > 1e-6:
        raise ValueError(name + " rotation determinant is not one")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(name + " homogeneous row is invalid")
    return matrix
