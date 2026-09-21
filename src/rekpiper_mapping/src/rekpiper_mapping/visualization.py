"""Human-readable robot/object mask and nvblox-input visualization."""

import cv2
import numpy as np

from rekpiper_mapping.frame_preparation import (
    FILTER_DEPTH_RANGE, FILTER_DYNAMIC, FILTER_RAW_INVALID,
    FILTER_RETAINED, FILTER_ROBOT, FILTER_WORKSPACE)


def _image(bgr):
    image = np.asarray(bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("RGB image must be a non-empty uint8 HxWx3 array")
    return image


def _mask(mask, shape, label):
    if mask is None:
        return None
    binary = np.asarray(mask)
    if binary.shape != shape:
        raise ValueError("{} mask dimensions differ from RGB".format(label))
    return binary != 0


def _label(panel, text, color=(255, 255, 255)):
    output = panel.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1] - 1, 38), (0, 0, 0), -1)
    cv2.putText(output, str(text), (9, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, tuple(int(value) for value in color), 2, cv2.LINE_AA)
    return output


def temporal_product_label(title, available, age_s, held_threshold_s=0.20):
    """Mark a retained map product without presenting it as a current frame."""
    if not available:
        return "{} | STALE".format(title), (80, 80, 255)
    if age_s is None:
        return title, (255, 255, 255)
    if not np.isfinite(age_s) or age_s < 0.0:
        raise ValueError("available temporal product must have a finite age")
    if not np.isfinite(held_threshold_s) or held_threshold_s <= 0.0:
        raise ValueError("held threshold must be positive")
    if age_s > held_threshold_s:
        return "{} | HELD {:.2f}s".format(title, age_s), (0, 200, 255)
    return title, (255, 255, 255)


def overlay_exclusion(bgr, mask, color_bgr, alpha=0.55):
    image = _image(bgr)
    binary = _mask(mask, image.shape[:2], "exclusion")
    if binary is None:
        return image.copy()
    output = image.copy()
    color = np.asarray(color_bgr, dtype=np.float32)
    output[binary] = np.clip(
        (1.0 - alpha) * image[binary].astype(np.float32) + alpha * color,
        0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, tuple(int(v) for v in color_bgr),
                     2, lineType=cv2.LINE_AA)
    return output


def colorize_static_depth(depth_m, shape, min_depth_m=0.10, max_depth_m=2.00):
    if depth_m is None:
        return np.zeros(shape + (3,), dtype=np.uint8)
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape != shape:
        raise ValueError("static depth dimensions differ from RGB")
    if not 0.0 <= min_depth_m < max_depth_m:
        raise ValueError("depth color range must be ordered")
    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    normalized = np.zeros(shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        255.0 * (depth[valid] - min_depth_m) / (max_depth_m - min_depth_m),
        0, 255).astype(np.uint8)
    # Invert so nearby geometry is warm and distant geometry is cool.
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def colorize_esdf_slice(distances_m, observed, truncation_m=0.15):
    """Color an actual nvblox ESDF slice using the ReKep sign convention."""
    distances = np.asarray(distances_m, dtype=np.float32)
    visible = np.asarray(observed, dtype=bool)
    if distances.ndim != 2 or distances.shape != visible.shape or distances.size == 0:
        raise ValueError("ESDF distances/observed must be matching non-empty HxW arrays")
    if not np.isfinite(truncation_m) or truncation_m <= 0.0:
        raise ValueError("ESDF visualization truncation must be positive")
    output = np.zeros(distances.shape + (3,), dtype=np.uint8)
    finite = visible & np.isfinite(distances)
    occupied = finite & (distances >= 0.0)
    free = finite & (distances < 0.0)

    occupied_strength = np.clip(distances / truncation_m, 0.0, 1.0)
    # Surface starts yellow and becomes red deeper inside collision geometry.
    output[..., 2][occupied] = 255
    output[..., 1][occupied] = (255.0 * (1.0 - occupied_strength[occupied])).astype(np.uint8)
    free_strength = np.clip(-distances / truncation_m, 0.0, 1.0)
    # Free space starts green near a surface and becomes blue with clearance.
    output[..., 0][free] = (255.0 * free_strength[free]).astype(np.uint8)
    output[..., 1][free] = (255.0 * (1.0 - free_strength[free])).astype(np.uint8)
    return output


def colorize_local_esdf_contours(distances_m, observed,
                                 contour_levels_m=(0.02, 0.05, 0.10, 0.15, 0.20)):
    """Color a point-clearance ESDF map and draw contours from queried values."""
    distances = np.asarray(distances_m, dtype=np.float32)
    visible = np.asarray(observed, dtype=bool)
    output = colorize_esdf_slice(distances, visible, truncation_m=0.20)
    free_clearance = -distances
    valid_free = visible & np.isfinite(distances) & (distances < 0.0)
    for level in contour_levels_m:
        region = (valid_free & (free_clearance <= float(level))).astype(np.uint8)
        if not np.any(region):
            continue
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    center = output.shape[0] // 2, output.shape[1] // 2
    cv2.drawMarker(output, (center[1], center[0]), (255, 255, 255),
                   cv2.MARKER_CROSS, 13, 2)
    return output


def colorize_filter_reasons(reasons, static_depth_m, min_depth_m=0.10, max_depth_m=2.00):
    """Color mutually exclusive causes for pixels rejected before nvblox."""
    values = np.asarray(reasons)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("filter reasons must be a non-empty HxW image")
    depth_color = colorize_static_depth(
        static_depth_m, values.shape, min_depth_m, max_depth_m)
    output = np.zeros(values.shape + (3,), dtype=np.uint8)
    output[values == FILTER_RAW_INVALID] = (0, 0, 0)
    output[values == FILTER_DEPTH_RANGE] = (255, 0, 0)
    output[values == FILTER_WORKSPACE] = (110, 110, 110)
    output[values == FILTER_ROBOT] = (0, 0, 255)
    output[values == FILTER_DYNAMIC] = (0, 220, 255)
    retained = values == FILTER_RETAINED
    output[retained] = depth_color[retained]
    return output


def _fit_to_panel(bgr, shape, crop_nonzero=False):
    if bgr is None:
        return np.zeros(shape + (3,), dtype=np.uint8)
    image = _image(bgr)
    if crop_nonzero:
        foreground = np.any(image != 0, axis=2)
        if np.any(foreground):
            ys, xs = np.nonzero(foreground)
            padding = max(2, int(round(0.08 * max(
                int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)))))
            y0 = max(0, int(ys.min()) - padding)
            y1 = min(image.shape[0], int(ys.max()) + padding + 1)
            x0 = max(0, int(xs.min()) - padding)
            x1 = min(image.shape[1], int(xs.max()) + padding + 1)
            image = image[y0:y1, x0:x1]
    target_h, target_w = shape
    scale = min(float(target_w) / image.shape[1], float(target_h) / image.shape[0])
    size = (max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    output = np.zeros(shape + (3,), dtype=np.uint8)
    y0 = (target_h - resized.shape[0]) // 2
    x0 = (target_w - resized.shape[1]) // 2
    output[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return output


def compose_robot_tsdf_esdf_quad(
        bgr, robot_mask=None, raw_tsdf_render_bgr=None,
        safe_tsdf_render_bgr=None, esdf_slice_bgr=None, status=None,
        camera_name="rs3"):
    """Compare the robot mask, raw TSDF, masked TSDF, and safe ESDF."""
    image = _image(bgr)
    shape = image.shape[:2]
    robot = _mask(robot_mask, shape, "rs3 robot")
    outlined = overlay_exclusion(
        image, robot, (0, 0, 255), alpha=0.0)
    mask_pixels = 0 if robot is None else int(np.count_nonzero(robot))
    raw_tsdf = _fit_to_panel(raw_tsdf_render_bgr, shape)
    safe_tsdf = _fit_to_panel(safe_tsdf_render_bgr, shape)
    esdf = _fit_to_panel(esdf_slice_bgr, shape, crop_nonzero=True)
    slice_z = dict(status or {}).get("esdf_slice_z_m", "?")
    top = np.hstack((
        _label(
            outlined,
            "{} RGB + PIPER MASK OUTLINE ({} px)".format(
                str(camera_name).upper(), mask_pixels),
            (80, 80, 255) if robot is None else (255, 255, 255)),
        _label(
            raw_tsdf,
            "RAW TSDF (ROBOT INCLUDED / DIAGNOSTIC ONLY)",
            (80, 80, 255)
            if raw_tsdf_render_bgr is None else (255, 255, 255)),
    ))
    bottom = np.hstack((
        _label(
            safe_tsdf, "SAFE TSDF (PIPER MASK REMOVED)",
            (80, 80, 255)
            if safe_tsdf_render_bgr is None else (255, 255, 255)),
        _label(
            esdf, "SAFE ESDF XY @ base_link.z={}m".format(slice_z),
            (80, 80, 255)
            if esdf_slice_bgr is None else (255, 255, 255)),
    ))
    return np.vstack((top, bottom))


def compose_mapping_panel(
    bgr, robot_mask=None, dynamic_mask=None, static_depth_m=None,
    tsdf_render_bgr=None, esdf_slice_bgr=None, status=None,
    min_depth_m=0.10, max_depth_m=2.00, robot_exclusion_enabled=True,
    dynamic_exclusion_enabled=True, validation_mode_label="",
    clearance=None,
):
    """Return RGB, exclusions, fusion input and actual ESDF plus safety status."""
    image = _image(bgr)
    robot = _mask(robot_mask, image.shape[:2], "robot")
    dynamic = _mask(dynamic_mask, image.shape[:2], "dynamic")
    robot_panel = overlay_exclusion(image, robot, (0, 0, 255))
    dynamic_panel = overlay_exclusion(image, dynamic, (0, 200, 255))
    depth_panel = colorize_static_depth(
        static_depth_m, image.shape[:2], min_depth_m, max_depth_m)
    tsdf_panel = _fit_to_panel(tsdf_render_bgr, image.shape[:2])
    esdf_panel = _fit_to_panel(esdf_slice_bgr, image.shape[:2], crop_nonzero=True)
    if esdf_slice_bgr is not None:
        cv2.rectangle(esdf_panel, (0, esdf_panel.shape[0] - 32),
                      (esdf_panel.shape[1] - 1, esdf_panel.shape[0] - 1),
                      (0, 0, 0), -1)
        cv2.putText(esdf_panel, "WHITE CROSS=gripper_base | BLACK=unknown",
                    (8, esdf_panel.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 1, cv2.LINE_AA)

    robot_title = ("Piper exclusion" if robot is not None
                   else "Piper exclusion: MISSING/STALE")
    if not robot_exclusion_enabled:
        robot_title = "Piper exclusion: DISABLED"
    dynamic_title = ("Attached-object exclusion" if dynamic is not None
                     else "Attached-object exclusion: MISSING/STALE")
    if not dynamic_exclusion_enabled:
        dynamic_title = "Dynamic exclusion: DISABLED (plan 1)"
    depth_title = ("Static depth -> nvblox" if static_depth_m is not None
                   else "Static depth: NOT INTEGRATED")
    tsdf_title = ("nvblox TSDF render (map output)" if tsdf_render_bgr is not None
                  else "nvblox TSDF render: UNAVAILABLE/STALE")
    values = dict(status or {})
    slice_z = values.get("esdf_slice_z_m", "?")
    following_anchor = bool(values.get(
        "esdf_slice_following_anchor",
        values.get("esdf_slice_following_gripper_base", False)))
    anchor_label = str(values.get(
        "esdf_slice_anchor_label", "gripper_base")).replace("_", " ").upper()
    esdf_title = ("nvblox ESDF XY @ {} z={}m".format(anchor_label, slice_z)
                  if following_anchor else
                  "nvblox ESDF XY @ z={}m (anchor stale)".format(slice_z))
    clearance_values = dict(clearance or {})
    clearance_known = bool(clearance_values.get("clearance_known", False))
    clearance_m = clearance_values.get("minimum_clearance_m")
    if clearance_known and isinstance(clearance_m, (int, float)):
        clearance_text = "{} CLEARANCE: {:+.3f} m".format(
            anchor_label, float(clearance_m))
        clearance_color = ((70, 220, 70) if float(clearance_m) > 0.0
                           else (60, 170, 255))
    else:
        clearance_text = "{} CLEARANCE: UNKNOWN".format(anchor_label)
        clearance_color = (80, 80, 255)
    if esdf_slice_bgr is not None:
        cv2.rectangle(esdf_panel, (0, 39), (esdf_panel.shape[1] - 1, 70), (0, 0, 0), -1)
        cv2.putText(esdf_panel, clearance_text, (8, 62), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, clearance_color, 1, cv2.LINE_AA)
    top_row = [
        _label(image, "D435 RGB (raw)"),
        _label(robot_panel, robot_title,
               (80, 80, 255) if robot_exclusion_enabled and robot is None else (255, 255, 255)),
        _label(dynamic_panel, dynamic_title,
               (80, 200, 255) if dynamic_exclusion_enabled and dynamic is None else (255, 255, 255)),
    ]
    bottom_row = [
        _label(depth_panel, depth_title, (80, 80, 255) if static_depth_m is None else (255, 255, 255)),
        _label(tsdf_panel, tsdf_title,
               (80, 80, 255) if tsdf_render_bgr is None else (255, 255, 255)),
        _label(esdf_panel, esdf_title,
               (80, 80, 255) if esdf_slice_bgr is None else (255, 255, 255)),
    ]
    body = np.vstack((np.hstack(top_row), np.hstack(bottom_row)))

    state = str(values.get("state", "status_unavailable"))
    integrated = int(values.get("integrated_frames", 0))
    esdf = int(values.get("esdf_updates", 0))
    allowed = bool(values.get("map_query_allowed", False))
    observed = float(values.get("esdf_slice_observed_fraction", 0.0))
    mode = ("MODE: {} | ".format(validation_mode_label) if validation_mode_label else "")
    text = (mode + "STATE: {} | TSDF frames: {} | ESDF updates: {} | "
            "SLICE observed: {:.1f}% | {} | MAP QUERY: {}").format(
                state, integrated, esdf, 100.0 * observed,
                clearance_text,
                "ALLOWED" if allowed else "BLOCKED")
    color = (70, 220, 70) if allowed else (60, 170, 255)
    footer = np.zeros((46, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(footer, text, (10, 31), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, color, 2, cv2.LINE_AA)
    return np.vstack((body, footer))


def _text_panel(shape, lines, title="STATUS", title_color=(255, 255, 255)):
    panel = np.zeros(shape + (3,), dtype=np.uint8)
    panel = _label(panel, title, title_color)
    y = 67
    for line, color in lines:
        cv2.putText(panel, str(line), (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.56, tuple(int(value) for value in color), 1, cv2.LINE_AA)
        y += 27
        if y >= shape[0] - 8:
            break
    return panel


def camera_diagnostic_tiles(camera_name, bgr, robot_mask=None, filter_reasons=None,
                            static_depth_m=None, tsdf_render_bgr=None, status=None,
                            min_depth_m=0.10, max_depth_m=2.00,
                            dynamic_exclusion_enabled=False,
                            product_ages_s=None, held_threshold_s=0.20):
    """Return the five requested camera tiles plus a per-source status tile."""
    image = _image(bgr)
    shape = image.shape[:2]
    robot = _mask(robot_mask, shape, camera_name + " robot")
    robot_panel = overlay_exclusion(image, robot, (0, 0, 255))
    if filter_reasons is None:
        reason_panel = np.zeros(shape + (3,), dtype=np.uint8)
    else:
        reasons = np.asarray(filter_reasons)
        if reasons.shape != shape:
            raise ValueError("{} filter reason dimensions differ from RGB".format(camera_name))
        reason_panel = colorize_filter_reasons(
            reasons, static_depth_m, min_depth_m, max_depth_m)
    depth_panel = colorize_static_depth(static_depth_m, shape, min_depth_m, max_depth_m)
    tsdf_panel = _fit_to_panel(tsdf_render_bgr, shape)
    product_ages = dict(product_ages_s or {})
    filter_title, filter_color = temporal_product_label(
        camera_name.upper() + " filter reasons", filter_reasons is not None,
        product_ages.get("filter"), held_threshold_s)
    static_title, static_color = temporal_product_label(
        camera_name.upper() + " static depth -> nvblox",
        static_depth_m is not None, product_ages.get("static"),
        held_threshold_s)
    values = dict(status or {})
    frames = dict(values.get("integrated_frames_by_camera", {})).get(camera_name, 0)
    counts = dict(values.get("filter_counts_by_camera", {})).get(camera_name, {})
    age = dict(values.get("camera_age_s", {})).get(camera_name)
    stamp = dict(values.get("last_integrated_stamp_by_camera", {})).get(camera_name)
    lines = [
        ("integrated frames: {}".format(frames), (255, 255, 255)),
        ("retained: {}".format(counts.get("retained", "?")), (120, 230, 120)),
        ("robot: {}  dynamic: {}".format(
            counts.get("robot", "?"), counts.get("dynamic", "?")), (80, 180, 255)),
        ("workspace: {}  range: {}".format(
            counts.get("workspace", "?"), counts.get("depth_range", "?")), (200, 200, 200)),
        ("raw invalid: {}".format(counts.get("raw_invalid", "?")), (160, 160, 160)),
        ("source age: {}".format("?" if age is None else "{:.3f}s".format(float(age))),
         (255, 255, 255)),
        ("last stamp: {}".format(
            "?" if stamp is None else "{:.6f}".format(float(stamp))), (255, 255, 255)),
        ("dynamic exclusion: {}".format("ON" if dynamic_exclusion_enabled else "DISABLED"),
         (0, 220, 255) if dynamic_exclusion_enabled else (255, 255, 255)),
    ]
    status_panel = _text_panel(shape, lines, camera_name.upper() + " DATA HEALTH")
    return [
        _label(image, camera_name.upper() + " RGB (raw)"),
        _label(robot_panel, camera_name.upper() + " Piper exclusion",
               (80, 80, 255) if robot is None else (255, 255, 255)),
        _label(reason_panel, filter_title, filter_color),
        _label(depth_panel, static_title, static_color),
        _label(tsdf_panel, camera_name.upper() + " fused TSDF render",
               (80, 80, 255) if tsdf_render_bgr is None else (255, 255, 255)),
        status_panel,
    ]


def compose_camera_diagnostic_page(camera_name, bgr, robot_mask=None, filter_reasons=None,
                                   static_depth_m=None, tsdf_render_bgr=None, status=None,
                                   min_depth_m=0.10, max_depth_m=2.00,
                                   dynamic_exclusion_enabled=False,
                                   product_ages_s=None, held_threshold_s=0.20):
    tiles = camera_diagnostic_tiles(
        camera_name, bgr, robot_mask, filter_reasons, static_depth_m,
        tsdf_render_bgr, status, min_depth_m, max_depth_m,
        dynamic_exclusion_enabled, product_ages_s, held_threshold_s)
    return np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:])))


def fused_diagnostic_tiles(global_esdf_bgr, local_esdf_bgr, obstacle_range_bgr,
                           status=None, clearance=None, shape=(320, 320)):
    values = dict(status or {})
    clearance_values = dict(clearance or {})
    known = bool(clearance_values.get("clearance_known", False))
    clearance_m = clearance_values.get("minimum_clearance_m")
    clearance_text = ("{:+.3f} m".format(float(clearance_m))
                      if known and isinstance(clearance_m, (int, float)) else "UNKNOWN")
    frames = dict(values.get("integrated_frames_by_camera", {}))
    lines = [
        ("state: {}".format(values.get("state", "unavailable")), (255, 255, 255)),
        ("rs1 frames: {}".format(frames.get("rs1", 0)), (255, 255, 255)),
        ("rs3 frames: {}".format(frames.get("rs3", 0)), (255, 255, 255)),
        ("ESDF updates: {}".format(values.get("esdf_updates", 0)), (255, 255, 255)),
        ("ESDF rate: {:.2f} Hz".format(float(values.get("esdf_rate_hz", 0.0))),
         (255, 255, 255)),
        ("MIN 3D CLEARANCE: {}".format(clearance_text),
         (70, 220, 70) if known and float(clearance_m) > 0.0 else (60, 170, 255)),
        ("MAP QUERY: {}".format(
            "ALLOWED" if values.get("map_query_allowed", False) else "BLOCKED"),
         (70, 220, 70) if values.get("map_query_allowed", False) else (60, 170, 255)),
    ]
    global_panel = _label(
        _fit_to_panel(global_esdf_bgr, shape), "Global ESDF @ gripper_base.z",
        (80, 80, 255) if global_esdf_bgr is None else (255, 255, 255))
    local_panel = _label(
        _fit_to_panel(local_esdf_bgr, shape), "Local ESDF contours (+/-0.30m)",
        (80, 80, 255) if local_esdf_bgr is None else (255, 255, 255))
    obstacle_panel = _label(
        _fit_to_panel(obstacle_range_bgr, shape),
        "Gripper obstacle range | MIN {}".format(
            "UNKNOWN" if values.get("obstacle_range_minimum_m") is None
            else "{:.3f}m".format(float(values["obstacle_range_minimum_m"]))),
        (80, 80, 255) if obstacle_range_bgr is None else (255, 255, 255))
    status_panel = _text_panel(shape, lines, "DUAL FUSION STATUS")
    return [global_panel, local_panel, obstacle_panel, status_panel]


def compose_fused_diagnostic_page(global_esdf_bgr, local_esdf_bgr,
                                  obstacle_range_bgr, status=None, clearance=None):
    tiles = fused_diagnostic_tiles(
        global_esdf_bgr, local_esdf_bgr, obstacle_range_bgr, status, clearance)
    return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))


def compose_dual_overview(rs1_tiles, rs3_tiles, fused_tiles,
                          validation_mode_label=""):
    """Compose three diagnostic rows without shrinking away per-camera evidence."""
    if len(rs1_tiles) < 5 or len(rs3_tiles) < 5 or len(fused_tiles) < 4:
        raise ValueError("overview needs five camera tiles and four fused tiles")
    shape = rs1_tiles[0].shape[:2]
    row1 = np.hstack([_fit_to_panel(tile, shape) for tile in rs1_tiles[:5]])
    row2 = np.hstack([_fit_to_panel(tile, shape) for tile in rs3_tiles[:5]])
    fused_width = row1.shape[1] // 5
    fused_shape = (shape[0], fused_width)
    fused_row = [_fit_to_panel(tile, fused_shape) for tile in fused_tiles]
    legend = _text_panel(fused_shape, [
        ("BLACK raw/unknown", (180, 180, 180)),
        ("RED robot/collision", (0, 0, 255)),
        ("GRAY workspace/known free", (150, 150, 150)),
        ("BLUE range/far", (255, 80, 0)),
        ("YELLOW dynamic/near", (0, 220, 255)),
    ], "COLOR LEGEND")
    row3 = np.hstack(fused_row + [legend])
    body = np.vstack((row1, row2, row3))
    footer = np.zeros((48, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(footer, "MODE: {} | DUAL FUSION | PLANNING REMAINS BLOCKED".format(
        validation_mode_label or "MAPPING"), (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
        0.72, (60, 170, 255), 2, cv2.LINE_AA)
    return np.vstack((body, footer))
