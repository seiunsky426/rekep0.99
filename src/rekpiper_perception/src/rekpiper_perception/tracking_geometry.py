"""PointCloud2 decoding shared by the real-time DINOv2 tracker."""

import numpy as np
import cv2


def mask_patch_weights(mask, side):
    """Preserve fractional mask coverage when pooling image patch features."""
    weights = cv2.resize(np.asarray(mask, dtype=np.float32), (side, side),
                         interpolation=cv2.INTER_AREA).reshape(-1)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("object mask is empty")
    return weights / total


def organized_xyz(message):
    fields = [(field.name, field.offset) for field in message.fields]
    if fields != [("x", 0), ("y", 4), ("z", 8)] or message.point_step != 12:
        raise ValueError("PointCloud2 is not contiguous XYZ float32")
    if (message.height <= 1 or message.row_step != message.width * 12
            or len(message.data) != message.height * message.row_step):
        raise ValueError("PointCloud2 must be organized without row padding")
    return np.frombuffer(message.data, dtype=np.float32).reshape(
        message.height, message.width, 3)
