"""Pinned, local-only Cutie v1.0 inference adapter.

The adapter deliberately fails closed: source provenance, weight digest and CUDA
availability are checked before importing the upstream network.  It contains no
download path and no fallback tracker.
"""

import hashlib
import importlib.machinery
from pathlib import Path
import sys
import types

import numpy as np
import torch
import yaml


CUTIE_V1_COMMIT = "7db8f9c9133f4884a9a13d865f9e3df20868d569"
CUTIE_BASE_MEGA_SHA256 = (
    "9c05402ee36d3a356fb72715d263ba7e1ea06ad3bada48c1306491792da43023"
)


class AttrDict(dict):
    """Small read-only-compatible subset of OmegaConf used by Cutie inference."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _convert(value):
    if isinstance(value, dict):
        return AttrDict({key: _convert(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_convert(item) for item in value]
    return value


def sha256_file(path, chunk_size=4 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(source_root):
    head = source_root / ".git" / "HEAD"
    if not head.is_file():
        return ""
    value = head.read_text().strip()
    if value.startswith("ref: "):
        target = source_root / ".git" / value[5:]
        return target.read_text().strip() if target.is_file() else ""
    return value


def build_runtime_config(source_root, max_internal_size=480):
    with open(source_root / "cutie" / "config" / "model" / "base.yaml", "r") as stream:
        model = yaml.safe_load(stream)
    model["object_transformer"]["embed_dim"] = model["embed_dim"]
    model["object_summarizer"]["embed_dim"] = model["embed_dim"]
    model["object_summarizer"]["num_summaries"] = model["object_transformer"]["num_queries"]
    return _convert({
        "model": model,
        "amp": True,
        "mem_every": 5,
        "top_k": 30,
        "stagger_updates": 5,
        "chunk_size": -1,
        "save_aux": False,
        "max_internal_size": int(max_internal_size),
        "flip_aug": False,
        "use_long_term": True,
        "max_mem_frames": 5,
        "long_term": {
            "count_usage": True,
            "max_mem_frames": 10,
            "min_mem_frames": 5,
            "num_prototypes": 128,
            "max_num_tokens": 10000,
            "buffer_tokens": 2000,
        },
    })


def ensure_omegaconf_shim():
    """Install the minimal Cutie inference shim with valid import metadata.

    Torch/TorchVision may call ``importlib.util.find_spec("omegaconf")`` while
    importing unrelated optional features.  A synthetic module with a missing
    ``__spec__`` makes that standard-library call raise ``ValueError``.
    """
    shim = sys.modules.get("omegaconf")
    if shim is None:
        shim = types.ModuleType("omegaconf")
        sys.modules["omegaconf"] = shim
    if getattr(shim, "__spec__", None) is None:
        shim.__spec__ = importlib.machinery.ModuleSpec(
            "omegaconf", loader=None
        )
    if not hasattr(shim, "DictConfig"):
        shim.DictConfig = AttrDict
    return shim


class CutieBackend:
    def __init__(self, source_root, weights_path, max_internal_size=480,
                 require_cuda=True):
        self.source_root = Path(source_root).expanduser().resolve()
        self.weights_path = Path(weights_path).expanduser().resolve()
        if _source_commit(self.source_root) != CUTIE_V1_COMMIT:
            raise RuntimeError("Cutie source commit does not match pinned v1.0")
        if not self.weights_path.is_file():
            raise FileNotFoundError("Cutie weights are missing: {}".format(self.weights_path))
        if sha256_file(self.weights_path) != CUTIE_BASE_MEGA_SHA256:
            raise RuntimeError("Cutie weight SHA256 mismatch")
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for safe online Cutie tracking")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Upstream inference imports DictConfig only for attribute-style access.
        ensure_omegaconf_shim()
        if str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))
        from cutie.inference.inference_core import InferenceCore
        from cutie.model.cutie import CUTIE
        from cutie.model.utils import resnet as cutie_resnet

        # Official Cutie constructors request ImageNet backbones even though the
        # complete checkpoint replaces those parameters immediately afterwards.
        # Force local initialization so an offline runtime can never contact
        # download.pytorch.org.  The pinned full checkpoint remains authoritative.
        original_resnet18 = cutie_resnet.resnet18
        original_resnet50 = cutie_resnet.resnet50
        cutie_resnet.resnet18 = lambda pretrained=True, extra_dim=0: original_resnet18(
            pretrained=False, extra_dim=extra_dim)
        cutie_resnet.resnet50 = lambda pretrained=True, extra_dim=0: original_resnet50(
            pretrained=False, extra_dim=extra_dim)

        self._inference_class = InferenceCore
        self.config = build_runtime_config(self.source_root, max_internal_size)
        try:
            self.network = CUTIE(self.config).eval().to(self.device)
            try:
                state = torch.load(str(self.weights_path), map_location="cpu",
                                   weights_only=True)
            except TypeError:  # PyTorch < 2.0
                state = torch.load(str(self.weights_path), map_location="cpu")
            self.network.load_weights(state)
        finally:
            cutie_resnet.resnet18 = original_resnet18
            cutie_resnet.resnet50 = original_resnet50

    def new_processor(self):
        return self._inference_class(self.network, self.config)

    @torch.inference_mode()
    def step(self, processor, bgr, label_map=None, object_labels=None):
        image = np.asarray(bgr, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Cutie image must be uint8 HxWx3 BGR")
        rgb = image[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(self.device)
        mask = None
        objects = None
        if label_map is not None:
            mask_np = np.asarray(label_map)
            if mask_np.shape != image.shape[:2]:
                raise ValueError("Cutie seed label map size mismatch")
            mask = torch.from_numpy(mask_np.astype(np.int64)).to(self.device)
            objects = [int(value) for value in object_labels]
        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            probabilities = processor.step(tensor, mask=mask, objects=objects)
        temporary = probabilities.argmax(dim=0)
        labels = processor.object_manager.tmp_to_obj_cls(temporary)
        confidence = probabilities.max(dim=0).values
        return (labels.detach().cpu().numpy().astype(np.uint16),
                confidence.detach().cpu().numpy().astype(np.float32))
