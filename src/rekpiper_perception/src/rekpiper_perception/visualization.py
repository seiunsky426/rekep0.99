"""Phase D visual acceptance panels."""

import cv2
import numpy as np


_VLM_GROUP_COLORS = (
    (190, 225, 255),  # light orange
    (220, 255, 205),  # light green
    (255, 220, 205),  # light blue
    (235, 210, 255),  # light magenta
    (205, 245, 255),  # light yellow
    (255, 235, 200),  # light cyan
)


def _vlm_group_color(group_id):
    return _VLM_GROUP_COLORS[(int(group_id) - 1) % len(_VLM_GROUP_COLORS)]


def draw_state_banner(bgr, text, color_bgr=(0, 170, 255)):
    image = np.asarray(bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("banner image must be uint8 HxWx3")
    output = image.copy()
    label = str(text)
    (width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x0 = max(0, output.shape[1] - width - 24)
    cv2.rectangle(output, (x0, 0), (output.shape[1] - 1, 36), (0, 0, 0), -1)
    cv2.putText(output, label, (x0 + 8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, tuple(int(value) for value in color_bgr), 2, cv2.LINE_AA)
    return output


def overlay_mask(bgr, mask, color_bgr, alpha=0.45):
    image = np.asarray(bgr, dtype=np.uint8)
    binary = np.asarray(mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or binary.shape != image.shape[:2]:
        raise ValueError("image and mask dimensions differ")
    output = image.copy()
    color = np.asarray(color_bgr, dtype=np.float32)
    output[binary] = np.clip(
        (1.0 - alpha) * output[binary].astype(np.float32) + alpha * color,
        0,
        255,
    ).astype(np.uint8)
    return output


def colorize_instance_mask(label_map):
    labels = np.asarray(label_map, dtype=np.uint16)
    output = np.zeros(labels.shape + (3,), dtype=np.uint8)
    for value in np.unique(labels):
        if value:
            output[labels == value] = ((37 * value) % 255, (97 * value) % 255, (173 * value) % 255)
    return output


def overlay_instance_masks(bgr, label_map, alpha=0.22, outline_px=2):
    """Draw subtle instance fills and crisp boundaries over the source image."""
    image = np.asarray(bgr, dtype=np.uint8)
    labels = np.asarray(label_map, dtype=np.uint16)
    if image.ndim != 3 or image.shape[2] != 3 or labels.shape != image.shape[:2]:
        raise ValueError("image and instance mask dimensions differ")
    colored = colorize_instance_mask(labels)
    foreground = labels > 0
    output = image.copy()
    output[foreground] = np.clip(
        (1.0 - alpha) * image[foreground].astype(np.float32)
        + alpha * colored[foreground].astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    # A saturated contour makes mask quality readable without hiding texture.
    for value in np.unique(labels):
        if not value:
            continue
        binary = (labels == value).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = tuple(int(channel) for channel in colorize_instance_mask(
            np.array([[value]], dtype=np.uint16)
        )[0, 0])
        cv2.drawContours(output, contours, -1, color, int(outline_px), lineType=cv2.LINE_AA)
    return output


def annotate_vlm_candidates(bgr, masks, pixels_rc, rigid_group_ids, sources=None):
    """Create a VLM-only overlay without changing SAM/DINO model inputs.

    Eligible instances receive a light contour band and bounding frame.  Every
    candidate keeps a small dot at its exact source pixel; its number is drawn
    in an offset box connected by a leader line so the surface stays visible.
    """
    image = np.asarray(bgr, dtype=np.uint8)
    instances = np.asarray(masks, dtype=bool)
    pixels = np.asarray(pixels_rc, dtype=np.int32).reshape(-1, 2)
    groups = np.asarray(rigid_group_ids, dtype=np.int32).reshape(-1)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("candidate image must be uint8 HxWx3")
    if instances.ndim != 3 or instances.shape[1:] != image.shape[:2]:
        raise ValueError("instance masks must be NxHxW and match the image")
    if len(pixels) != len(groups):
        raise ValueError("candidate pixels and rigid groups must have equal length")
    if sources is not None and len(sources) != len(pixels):
        raise ValueError("candidate sources must match candidate count")
    if np.any(groups < 1) or np.any(groups > len(instances)):
        raise ValueError("candidate rigid group is outside the instance set")

    output = image.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for offset, mask in enumerate(instances):
        group_id = offset + 1
        binary = mask.astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        color = _vlm_group_color(group_id)
        band = cv2.dilate(binary, kernel) > 0
        band &= cv2.erode(binary, kernel) == 0
        output[band] = np.clip(
            0.72 * output[band].astype(np.float32)
            + 0.28 * np.asarray(color, dtype=np.float32), 0, 255).astype(np.uint8)
        cv2.drawContours(output, contours, -1, color, 2, lineType=cv2.LINE_AA)
        x, y, width, height = cv2.boundingRect(binary)
        cv2.rectangle(output, (x, y), (x + width - 1, y + height - 1), color, 1,
                      lineType=cv2.LINE_AA)
        group_label = "G{}".format(group_id)
        cv2.putText(output, group_label, (x + 3, max(14, y + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (25, 25, 25), 2, cv2.LINE_AA)
        cv2.putText(output, group_label, (x + 3, max(14, y + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    height, width = image.shape[:2]
    offsets = ((10, -12), (10, 25), (-42, -12), (-42, 25))
    for candidate_id, ((row, col), group_id) in enumerate(zip(pixels, groups)):
        if row < 0 or row >= height or col < 0 or col >= width:
            raise ValueError("candidate pixel is outside the image")
        color = _vlm_group_color(group_id)
        cv2.circle(output, (int(col), int(row)), 5, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(output, (int(col), int(row)), 3, color, -1, cv2.LINE_AA)
        if sources is not None and sources[candidate_id] == "mask_3d_medoid":
            cv2.circle(output, (int(col), int(row)), 7, color, 1, cv2.LINE_AA)
        dx, dy = offsets[candidate_id % len(offsets)]
        x0 = int(np.clip(int(col) + dx, 1, max(1, width - 35)))
        y0 = int(np.clip(int(row) + dy, 18, max(18, height - 3)))
        label = "K{}".format(candidate_id)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        box_left, box_top = x0, y0 - text_height - 5
        box_right, box_bottom = x0 + text_width + 8, y0 + 4
        cv2.line(output, (int(col), int(row)), (box_left, box_bottom), color, 1,
                 cv2.LINE_AA)
        cv2.rectangle(output, (box_left, box_top), (box_right, box_bottom),
                      (245, 245, 245), -1)
        cv2.rectangle(output, (box_left, box_top), (box_right, box_bottom), color, 2)
        cv2.putText(output, label, (x0 + 4, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (20, 20, 20), 2, cv2.LINE_AA)
    return output


def compose_scene_panel(bgr, scene_mask, candidate_bgr):
    image = np.asarray(bgr, dtype=np.uint8)
    candidate = np.asarray(candidate_bgr, dtype=np.uint8)
    if candidate.shape != image.shape:
        raise ValueError("candidate image dimensions differ from RGB")
    mask_panel = overlay_instance_masks(image, scene_mask)
    panels = [image.copy(), mask_panel, candidate.copy()]
    labels = ["D435 RGB (raw)", "SAM masks overlay", "Official DINOv2 candidates"]
    for panel, label in zip(panels, labels):
        cv2.rectangle(panel, (0, 0), (290, 36), (0, 0, 0), -1)
        cv2.putText(panel, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.hstack(panels)
