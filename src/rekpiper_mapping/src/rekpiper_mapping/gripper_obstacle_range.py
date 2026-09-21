"""Pure helpers for a 2-D view of 3-D TSDF surfaces around gripper_base."""

import cv2
import numpy as np


DISTANCE_LEVELS_M = (0.02, 0.05, 0.10, 0.15, 0.20)


def _validate_inputs(vertices, triangles, anchor_xyz, observed):
    xyz = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int64)
    anchor = np.asarray(anchor_xyz, dtype=np.float32)
    known = np.asarray(observed, dtype=bool)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
        raise ValueError("mesh vertices must be finite Nx3")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh triangles must be Mx3")
    if faces.size and (np.min(faces) < 0 or np.max(faces) >= len(xyz)):
        raise ValueError("mesh triangle index is outside vertices")
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        raise ValueError("gripper anchor must be finite XYZ")
    if known.ndim != 2 or known.shape[0] != known.shape[1] or known.size == 0:
        raise ValueError("observed background must be a non-empty square image")
    return xyz, faces, anchor, known


def distance_color(distance_m):
    """Return BGR safety bands for a true three-dimensional distance."""
    value = float(distance_m)
    if value <= 0.02:
        return 0, 0, 255
    if value <= 0.05:
        return 0, 128, 255
    if value <= 0.10:
        return 0, 255, 255
    if value <= 0.20:
        return 0, 190, 0
    return 255, 80, 0


def _world_to_pixel(points_xy, anchor, half_extent_m, pixels):
    result = np.empty_like(points_xy, dtype=np.float32)
    result[:, 0] = ((points_xy[:, 0] - (anchor[0] - half_extent_m))
                    / (2.0 * half_extent_m) * (pixels - 1))
    result[:, 1] = (((anchor[1] + half_extent_m) - points_xy[:, 1])
                    / (2.0 * half_extent_m) * (pixels - 1))
    return result


def project_tsdf_surface_range(vertices, triangles, anchor_xyz, observed_at_anchor_z,
                               half_extent_m=0.30,
                               contour_levels_m=DISTANCE_LEVELS_M):
    """Project TSDF triangles and retain the closest 3-D surface per XY cell.

    Dark gray means the ESDF voxel at ``gripper_base.z`` is observed but no
    reconstructed surface projects into that cell. Black remains unknown.
    """
    xyz, faces, anchor, known = _validate_inputs(
        vertices, triangles, anchor_xyz, observed_at_anchor_z)
    if not np.isfinite(half_extent_m) or half_extent_m <= 0.0:
        raise ValueError("half extent must be positive")
    pixels = known.shape[0]
    distance_grid = np.full((pixels, pixels), np.inf, dtype=np.float32)
    image = np.zeros((pixels, pixels, 3), dtype=np.uint8)
    image[known] = (42, 42, 42)

    if len(xyz):
        vertex_pixels = _world_to_pixel(xyz[:, :2], anchor, half_extent_m, pixels)
        vertex_distances = np.linalg.norm(xyz - anchor[None, :], axis=1)
        for pixel, distance in zip(vertex_pixels, vertex_distances):
            px, py = int(round(float(pixel[0]))), int(round(float(pixel[1])))
            if 0 <= px < pixels and 0 <= py < pixels:
                distance_grid[py, px] = min(distance_grid[py, px], float(distance))

    # Rasterize projected triangles with barycentric Z interpolation. Vertical
    # triangles have degenerate XY area; their vertices were retained above.
    for face in faces:
        tri = xyz[face]
        if (np.max(tri[:, 0]) < anchor[0] - half_extent_m
                or np.min(tri[:, 0]) > anchor[0] + half_extent_m
                or np.max(tri[:, 1]) < anchor[1] - half_extent_m
                or np.min(tri[:, 1]) > anchor[1] + half_extent_m):
            continue
        p = _world_to_pixel(tri[:, :2], anchor, half_extent_m, pixels)
        x0 = max(0, int(np.floor(np.min(p[:, 0]))))
        x1 = min(pixels - 1, int(np.ceil(np.max(p[:, 0]))))
        y0 = max(0, int(np.floor(np.min(p[:, 1]))))
        y1 = min(pixels - 1, int(np.ceil(np.max(p[:, 1]))))
        if x0 > x1 or y0 > y1:
            continue
        denominator = ((p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0])
                       + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1]))
        if abs(float(denominator)) < 1e-6:
            continue
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
        w0 = ((p[1, 1] - p[2, 1]) * (xx - p[2, 0])
              + (p[2, 0] - p[1, 0]) * (yy - p[2, 1])) / denominator
        w1 = ((p[2, 1] - p[0, 1]) * (xx - p[2, 0])
              + (p[0, 0] - p[2, 0]) * (yy - p[2, 1])) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not np.any(inside):
            continue
        world_x = anchor[0] - half_extent_m + xx / (pixels - 1) * 2.0 * half_extent_m
        world_y = anchor[1] + half_extent_m - yy / (pixels - 1) * 2.0 * half_extent_m
        world_z = w0 * tri[0, 2] + w1 * tri[1, 2] + w2 * tri[2, 2]
        distances = np.sqrt((world_x - anchor[0]) ** 2
                            + (world_y - anchor[1]) ** 2
                            + (world_z - anchor[2]) ** 2)
        target = distance_grid[y0:y1 + 1, x0:x1 + 1]
        update = inside & (distances < target)
        target[update] = distances[update]

    surface = np.isfinite(distance_grid)
    for y, x in np.argwhere(surface):
        image[y, x] = distance_color(distance_grid[y, x])

    # Contours come from the actual projected distance field. No geometric
    # rings are drawn when the TSDF mesh is empty.
    for level in contour_levels_m:
        region = (surface & (distance_grid <= float(level))).astype(np.uint8)
        if not np.any(region):
            continue
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    center = pixels // 2
    cv2.drawMarker(image, (center, center), (255, 255, 255), cv2.MARKER_CROSS, 13, 2)
    minimum = float(np.min(distance_grid[surface])) if np.any(surface) else None
    return image, distance_grid, minimum
