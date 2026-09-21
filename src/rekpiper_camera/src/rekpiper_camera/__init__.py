"""RealSense D435 geometry used by the ReKep real-robot adapters."""

from .projection import (
    CameraIntrinsics,
    depth_to_xyz,
    transform_points,
    transform_to_matrix,
)
from .workspace import (
    estimate_table_workspace,
    filter_xyz_bounds,
    normalized_polygon_mask,
)

__all__ = [
    "CameraIntrinsics",
    "depth_to_xyz",
    "transform_points",
    "transform_to_matrix",
    "estimate_table_workspace",
    "filter_xyz_bounds",
    "normalized_polygon_mask",
]
