"""Perception primitives for workspace candidates and downstream task geometry."""

from .task_geometry import (
    InstanceGeometryEstimate,
    erode_foreground_mask,
    clean_organized_mask,
    estimate_instance_geometry,
    filter_point_samples,
    organized_workspace_mask,
    prepare_sam_visual_mask,
    workspace_mask_area_ratio,
)
from .visualization import (
    annotate_vlm_candidates,
    colorize_instance_mask,
    compose_scene_panel,
    draw_state_banner,
    overlay_instance_masks,
    overlay_mask,
)

__all__ = [
    "InstanceGeometryEstimate",
    "erode_foreground_mask",
    "clean_organized_mask",
    "estimate_instance_geometry",
    "filter_point_samples",
    "organized_workspace_mask",
    "prepare_sam_visual_mask",
    "workspace_mask_area_ratio",
    "colorize_instance_mask",
    "annotate_vlm_candidates",
    "compose_scene_panel",
    "draw_state_banner",
    "overlay_instance_masks",
    "overlay_mask",
]
