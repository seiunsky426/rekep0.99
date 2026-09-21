"""Segment Anything automatic masks used by the ReKep keypoint proposer."""

from pathlib import Path
import os

import numpy as np


class SAMAutomaticSegmenter:
    """Thin, fail-closed adapter around Meta's original SAM generator."""

    def __init__(self, checkpoint, device, model_type="vit_h",
                 points_per_side=32, pred_iou_thresh=0.88,
                 stability_score_thresh=0.95, min_area_px=150,
                 max_area_ratio=0.50, max_masks=64, use_float16=True):
        import torch
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("SAM checkpoint is missing: {}".format(path))
        try:
            import segment_anything as sam_package
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc:
            raise RuntimeError("segment-anything is not installed") from exc
        workspace = Path(__file__).resolve().parents[4]
        vendor_root = Path(os.environ.get(
            "REKPIPER_VENDOR_ROOT", str(workspace / "runtime" / "vendor")))
        expected_root = vendor_root / "segment-anything"
        actual_source = Path(sam_package.__file__).resolve()
        try:
            actual_source.relative_to(expected_root.resolve())
        except ValueError as exc:
            raise RuntimeError(
                "Segment Anything resolved outside the pinned source") from exc
        head = expected_root / ".git" / "HEAD"
        if (not head.is_file()
                or head.read_text(encoding="utf-8").strip()
                != "6fdee8f2727f4506cfbbe553e23b895e27956588"):
            raise RuntimeError("Segment Anything source commit mismatch")
        if model_type not in sam_model_registry:
            raise ValueError("unsupported SAM model type: {}".format(model_type))
        self._minimum_area = int(min_area_px)
        self._maximum_ratio = float(max_area_ratio)
        self._maximum_masks = int(max_masks)
        # ViT-H's full-resolution attention exceeds this site's 8 GB GPU in
        # float32. Convert parameters after loading the unchanged checkpoint.
        self._use_float16 = bool(use_float16) and str(device).startswith("cuda")
        model = sam_model_registry[model_type](checkpoint=str(path)).to(
            device=device, dtype=torch.float16 if self._use_float16 else torch.float32)
        model.eval()
        self._generator = SamAutomaticMaskGenerator(
            model=model, points_per_side=int(points_per_side),
            pred_iou_thresh=float(pred_iou_thresh),
            stability_score_thresh=float(stability_score_thresh),
            min_mask_region_area=self._minimum_area,
        )

    def segment(self, bgr):
        import cv2
        import torch
        image = np.asarray(bgr, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("SAM input must be a BGR HxWx3 image")
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=self._use_float16):
            records = self._generator.generate(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        maximum = image.shape[0] * image.shape[1] * self._maximum_ratio
        records = [record for record in records
                   if self._minimum_area <= int(record["area"]) <= maximum]
        records.sort(key=lambda record: (
            -float(record.get("predicted_iou", 0.0)),
            -float(record.get("stability_score", 0.0)),
            -int(record["area"])))
        return [np.asarray(record["segmentation"], dtype=bool)
                for record in records[:self._maximum_masks]]


def masks_to_label_map(masks):
    values = np.asarray(masks, dtype=bool)
    if values.ndim != 3:
        raise ValueError("masks must have shape NxHxW")
    labels = np.zeros(values.shape[1:], dtype=np.uint16)
    for index, mask in enumerate(values, start=1):
        labels[mask & (labels == 0)] = index
    return labels


def arbitrate_quality_ordered_masks(masks):
    """Give every pixel to the first (highest-quality) SAM instance only."""
    values = [np.asarray(mask, dtype=bool) for mask in masks]
    if not values:
        return []
    shape = values[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in values):
        raise ValueError("quality-ordered masks must share one HxW shape")
    claimed = np.zeros(shape, dtype=bool)
    result = []
    for value in values:
        unique = value & ~claimed
        claimed |= value
        if np.any(unique):
            result.append(unique)
    return result
