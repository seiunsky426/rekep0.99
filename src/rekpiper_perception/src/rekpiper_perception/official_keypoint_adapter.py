"""Use the pristine official ReKep KeypointProposer with local DINOv2 assets.

The upstream constructor downloads DINOv2 with ``torch.hub.load`` and only
returns 3-D candidates plus an annotated image.  This adapter injects the
pinned local model while constructing the class and captures the candidate
pixels at the official projection boundary.  No upstream source is patched.
"""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import yaml


OFFICIAL_KEYPOINT_SHA256 = (
    "de120fe4365a735a614de50d3fae948926ae8b058882447b8372c5ab1ba5a04e")
DINOV2_COMMIT = "e1277af2ba9496fbadf7aec6eba56e8d882d1e35"
DINOV2_ARCHITECTURE = "dinov2_vits14_reg"
DINOV2_NUM_REGISTER_TOKENS = 4
DINOV2_SHA256 = (
    "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb")


def _git_head(root: Path) -> str:
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return ""
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        target = root / ".git" / value[5:]
        return target.read_text(encoding="utf-8").strip() if target.is_file() else ""
    return value


def _verify_dinov2_source(root: Path) -> None:
    head = _git_head(root)
    if head:
        if head != DINOV2_COMMIT:
            raise RuntimeError("DINOv2 source commit mismatch")
        return
    lock_path = root.parent / "UPSTREAM.lock.yaml"
    if not lock_path.is_file():
        raise RuntimeError("DINOv2 source has neither Git identity nor source lock")
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))["dinov2"]
    if str(lock.get("commit", "")) != DINOV2_COMMIT:
        raise RuntimeError("DINOv2 source lock commit mismatch")
    for name, expected in lock.get("files", {}).items():
        path = root / name
        if (not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise RuntimeError("DINOv2 source hash mismatch: {}".format(name))


@dataclass(frozen=True)
class CandidateSet:
    points: np.ndarray
    pixels_rc: np.ndarray
    rigid_group_ids: np.ndarray
    annotated_rgb: np.ndarray


def _load_official_class(official_root: Path):
    module_path = official_root / "keypoint_proposal.py"
    if not module_path.is_file():
        raise FileNotFoundError("official keypoint_proposal.py not found: {}".format(module_path))
    if hashlib.sha256(module_path.read_bytes()).hexdigest() != OFFICIAL_KEYPOINT_SHA256:
        raise RuntimeError("official keypoint_proposal.py hash mismatch")
    root_text = str(official_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec = importlib.util.spec_from_file_location("rekep_official_keypoint_proposal", str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.KeypointProposer


def load_local_dinov2(repo_path: str, weights_path: str, device: str):
    repo = Path(repo_path).expanduser().resolve()
    weights = Path(weights_path).expanduser().resolve()
    backbone_source = repo / "dinov2" / "hub" / "backbones.py"
    if not backbone_source.is_file():
        raise FileNotFoundError(
            "local DINOv2 repository is missing the backbone source: {}".format(repo))
    _verify_dinov2_source(repo)
    if not weights.is_file():
        raise FileNotFoundError(
            "local DINOv2 ViT-S/14 reg4 weights missing: {}".format(weights))
    if hashlib.sha256(weights.read_bytes()).hexdigest() != DINOV2_SHA256:
        raise RuntimeError("DINOv2 ViT-S/14 reg4 weight SHA256 mismatch")
    repo_text = str(repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    backbones = importlib.import_module("dinov2.hub.backbones")
    loaded_source = Path(backbones.__file__).resolve()
    if repo not in loaded_source.parents:
        raise RuntimeError("DINOv2 was imported from an unpinned source")
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    validate_reg4_checkpoint(state)
    model = backbones.dinov2_vits14_reg(pretrained=False)
    if int(getattr(model, "num_register_tokens", -1)) \
            != DINOV2_NUM_REGISTER_TOKENS:
        raise RuntimeError("DINOv2 ViT-S/14 reg4 token count mismatch")
    model.load_state_dict(state, strict=True)
    return model.eval().to(torch.device(device))


def validate_reg4_checkpoint(state) -> None:
    """Reject plain, malformed, and non-ViT-S/14 reg4 checkpoints."""
    if (not isinstance(state, dict)
            or "register_tokens" not in state
            or tuple(state["register_tokens"].shape) != (1, 4, 384)):
        raise RuntimeError(
            "DINOv2 ViT-S/14 checkpoint is not the pinned reg4 architecture")


class OfficialKeypointProposerAdapter:
    def __init__(
        self,
        official_root: str,
        dinov2_repo: str,
        dinov2_weights: str,
        config: Dict,
    ):
        requested = config.get("device", "auto")
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for DINOv2 keypoint inference")
        config = dict(config)
        config["device"] = requested
        model = load_local_dinov2(dinov2_repo, dinov2_weights, requested)
        official_class = _load_official_class(Path(official_root).expanduser().resolve())
        original_hub_load = torch.hub.load

        def _local_hub_load(repo_or_dir, model_name, *args, **kwargs):
            # The byte-exact public demo requests ``dinov2_vits14``.  The
            # real-world pipeline described in paper v2 uses ViT-S/14 with
            # registers, so the Piper environment adapter injects the pinned
            # reg4 instance without patching the official source.
            if str(model_name) != "dinov2_vits14":
                raise RuntimeError("official proposer requested an unapproved model: {}".format(model_name))
            return model

        torch.hub.load = _local_hub_load
        try:
            self._proposer = official_class(config)
        finally:
            torch.hub.load = original_hub_load

        self._last_projection: Tuple[np.ndarray, np.ndarray] = (
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )
        self._last_candidate_keep = np.empty((0,), dtype=bool)
        self._active_unique_labels = np.empty((0,), dtype=np.int32)
        original_project = self._proposer._project_keypoints_to_img

        def _capture_projection(rgb, candidate_pixels, candidate_groups, masks, features):
            pixels = np.asarray(candidate_pixels, dtype=np.int32).reshape(-1, 2)
            raw_groups = np.asarray(candidate_groups, dtype=np.int32).reshape(-1)
            if len(pixels) != len(raw_groups):
                raise RuntimeError("official proposer pixel/group output is inconsistent")
            keep = ((raw_groups >= 0)
                    & (raw_groups < len(self._active_unique_labels)))
            labels = np.zeros(raw_groups.shape, dtype=np.int32)
            labels[keep] = self._active_unique_labels[raw_groups[keep]]
            # Official ReKep enumerates every unique label, including label 0.
            # SAM reserves 0 for unassigned/background pixels, so it must never
            # become a movable rigid group in the Piper environment adapter.
            keep &= labels > 0
            mapped_groups = labels[keep] - 1
            self._last_candidate_keep = keep.copy()
            self._last_projection = (
                pixels[keep].copy(),
                mapped_groups.copy(),
            )
            return original_project(
                rgb, pixels[keep], mapped_groups, masks, features)

        self._proposer._project_keypoints_to_img = _capture_projection
        self.device = requested

    def propose(self, rgb: np.ndarray, points: np.ndarray, masks: np.ndarray) -> CandidateSet:
        rgb = np.asarray(rgb, dtype=np.uint8)
        points = np.asarray(points, dtype=np.float32)
        masks = np.asarray(masks)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape HxWx3")
        if points.shape != rgb.shape[:2] + (3,):
            raise ValueError("organized points must have shape HxWx3")
        if masks.ndim == 3:
            if masks.shape[1:] != rgb.shape[:2]:
                raise ValueError("binary instance masks must have shape NxHxW")
            label_map = np.zeros(rgb.shape[:2], dtype=np.int32)
            for index, binary in enumerate(masks, start=1):
                binary = np.asarray(binary, dtype=bool)
                if np.any(binary & (label_map != 0)):
                    raise ValueError("binary instance masks must not overlap")
                label_map[binary] = index
            masks = label_map
        elif masks.shape != rgb.shape[:2]:
            raise ValueError("instance label map must have shape HxW")
        else:
            if np.any(masks < 0):
                raise ValueError("instance label map cannot contain negative labels")
            # Public rigid groups are always contiguous even if an upstream
            # label image used sparse positive identifiers.
            normalized = np.zeros(rgb.shape[:2], dtype=np.int32)
            for index, label in enumerate(np.unique(masks[masks > 0]), start=1):
                normalized[masks == label] = index
            masks = normalized

        self._active_unique_labels = np.unique(masks).astype(np.int32)
        self._last_candidate_keep = np.empty((0,), dtype=bool)
        # The official k-means dependency flushes both output streams.
        # Detached roslaunch sessions can inherit closed terminal pipes.
        with open(os.devnull, 'w') as progress, redirect_stderr(progress), redirect_stdout(progress):
            candidates, annotated = self._proposer.get_keypoints(rgb, points, masks)
        candidates = np.asarray(candidates, dtype=np.float32).reshape(-1, 3)
        if len(self._last_candidate_keep) != len(candidates):
            raise RuntimeError("official proposer candidate capture is inconsistent")
        candidates = candidates[self._last_candidate_keep]
        pixels, groups = self._last_projection
        if len(pixels) != len(candidates) or len(groups) != len(candidates):
            raise RuntimeError("official proposer output capture is inconsistent")
        return CandidateSet(
            points=candidates,
            pixels_rc=pixels.reshape(-1, 2),
            rigid_group_ids=groups.reshape(-1),
            annotated_rgb=np.asarray(annotated, dtype=np.uint8),
        )
