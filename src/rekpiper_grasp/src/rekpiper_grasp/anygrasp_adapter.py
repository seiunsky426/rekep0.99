"""Fail-closed adapter for the licensed official AnyGrasp SDK."""

from argparse import Namespace
from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import sys
from typing import List, Optional, Sequence

import numpy as np


class AnyGraspUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SDKPreflight:
    ready: bool
    reasons: tuple


@dataclass
class CameraGrasp:
    translation: np.ndarray
    rotation: np.ndarray
    width_m: float
    depth_m: float
    score: float
    source_camera: str
    candidate_origin: str = "licensed_anygrasp"


def check_anygrasp_runtime(sdk_root, checkpoint_path, license_dir,
                           torch_module=None):
    reasons = []
    root = Path(sdk_root).expanduser()
    checkpoint = Path(checkpoint_path).expanduser()
    license_path = Path(license_dir).expanduser()
    if not root.is_dir():
        reasons.append("sdk_root_missing")
    elif not any(root.glob("gsnet*.so")):
        reasons.append("gsnet_binary_missing")
    if not checkpoint.is_file():
        reasons.append("checkpoint_missing")
    license_ready = (
        license_path.is_dir()
        and (license_path / "licenseCfg.json").is_file()
        and any(license_path.glob("*.lic"))
        and any(license_path.glob("*.signature"))
        and any(license_path.glob("*.public_key")))
    if not license_ready:
        reasons.append("license_missing")
    elif root.is_dir() and license_path.resolve() != (root / "license").resolve():
        reasons.append("license_not_inside_sdk_root")
    try:
        torch = torch_module or importlib.import_module("torch")
        if not bool(torch.cuda.is_available()):
            reasons.append("cuda_unavailable")
    except (ImportError, AttributeError):
        reasons.append("torch_unavailable")
    for module_name, reason in (
            ("MinkowskiEngine", "minkowski_engine_missing"),
            ("graspnetAPI", "graspnetapi_missing"),
            ("grasp_nms", "grasp_nms_missing"),
            ("pointnet2", "pointnet2_missing"),
            ("open3d", "open3d_missing")):
        try:
            if importlib.util.find_spec(module_name) is None:
                raise ImportError(module_name)
            importlib.import_module(module_name)
        except (ImportError, OSError):
            reasons.append(reason)
    return SDKPreflight(not reasons, tuple(reasons))


class AnyGraspAdapter:
    def __init__(self, sdk_root, checkpoint_path, license_dir,
                 max_gripper_width_m=0.070, gripper_height_m=0.03):
        self.sdk_root = str(Path(sdk_root).expanduser().resolve())
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.license_dir = str(Path(license_dir).expanduser().resolve())
        self.max_gripper_width_m = float(max_gripper_width_m)
        self.gripper_height_m = float(gripper_height_m)
        if not 0.0 < self.max_gripper_width_m <= 0.070:
            raise ValueError("max_gripper_width_m must lie in (0, 0.070]")
        preflight = check_anygrasp_runtime(
            self.sdk_root, self.checkpoint_path, self.license_dir)
        if not preflight.ready:
            raise AnyGraspUnavailable(",".join(preflight.reasons))
        if self.sdk_root not in sys.path:
            sys.path.insert(0, self.sdk_root)
        try:
            create_detector = importlib.import_module("gsnet").create_detector
            self._detector = create_detector(Namespace(
                checkpoint_path=self.checkpoint_path,
                max_gripper_width=self.max_gripper_width_m,
                gripper_height=self.gripper_height_m))
        except Exception as exc:
            raise AnyGraspUnavailable(
                "sdk_initialization_failed:{}".format(exc)) from exc
        if self._detector is None:
            raise AnyGraspUnavailable("license_validation_failed")

    def infer(self, points_camera, region_mask, source_camera,
              approach_steering: Optional[Sequence[float]] = None,
              approach_thresh_rad=np.pi, max_candidates=50,
              dense_grasp=False, collision_detection=True) -> List[CameraGrasp]:
        """Return ranked native grasps; max_candidates=None retains all after NMS."""
        points = np.asarray(points_camera)
        mask = np.asarray(region_mask)
        if points.ndim != 2 or points.shape[1] != 3 or points.dtype != np.float32:
            raise ValueError("points_camera must be float32 Nx3")
        if mask.shape != (points.shape[0],) or mask.dtype != np.bool_:
            raise ValueError("region_mask must be bool N and align with points")
        if not np.all(np.isfinite(points)) or not np.any(mask):
            raise ValueError("AnyGrasp input points or region are invalid")
        steering = None
        if approach_steering is not None:
            steering = np.asarray(approach_steering, dtype=float)
            if steering.shape != (3,) or not np.all(np.isfinite(steering)):
                raise ValueError("approach_steering must be a finite 3-vector")
            steering = steering.tolist()
        grasps = self._detector.get_grasp(points, {
            "dense_grasp": bool(dense_grasp), "collision_detection": bool(collision_detection),
            "region_steering": mask, "approach_steering": steering,
            "approach_thresh": float(approach_thresh_rad)})
        if grasps is None:
            return []
        ranked = grasps.nms().sort_by_score()
        if max_candidates is not None:
            ranked = ranked[:int(max_candidates)]
        output = []
        for grasp in ranked:
            rotation = np.asarray(grasp.rotation_matrix, dtype=float)
            translation = np.asarray(grasp.translation, dtype=float)
            scalars = [grasp.width, grasp.depth, grasp.score]
            if (rotation.shape != (3, 3) or translation.shape != (3,)
                    or not np.all(np.isfinite(rotation))
                    or not np.all(np.isfinite(translation))
                    or not np.all(np.isfinite(scalars))):
                continue
            output.append(CameraGrasp(
                translation, rotation, float(grasp.width), float(grasp.depth),
                float(grasp.score), str(source_camera)))
        return output

    def infer_directional(self, points_camera, region_mask, source_camera,
                          base_from_camera, table_normal, table_offset, gripper,
                          cancelled=lambda: False):
        """Top/side model proposals with Piper/table audit; contact is separate.

        Returned records retain every merged-NMS proposal, including rejections.
        No IK, control client, or execution authority is created here.
        """
        import random
        import torch
        from graspnetAPI import GraspGroup
        from .blue_cloud_comparison import native_pose_record, transform_points
        from .directional_grasp import direction_queries, DirectionalAudit

        transform = np.asarray(base_from_camera)
        target = transform_points(points_camera[region_mask], transform)
        audit = DirectionalAudit(target, table_normal, table_offset, gripper)
        records = []
        for mode in ('top', 'side'):
            directions, cone = direction_queries(table_normal, mode)
            raw = []
            for direction in directions:
                if cancelled():
                    raise InterruptedError('planning_cancelled')
                random.seed(0); np.random.seed(0); torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                proposals = self.infer(
                    points_camera, region_mask, source_camera,
                    approach_steering=transform[:3, :3].T @ direction,
                    approach_thresh_rad=np.deg2rad(cone), max_candidates=None,
                    dense_grasp=False, collision_detection=False)
                for grasp in proposals:
                    raw.append(np.r_[grasp.score, grasp.width_m, gripper.finger_height_m,
                                     grasp.depth_m, grasp.rotation.ravel(), grasp.translation, -1.])
            merged = GraspGroup(np.asarray(raw).reshape(-1, 17)).nms().sort_by_score() if raw else []
            for index, value in enumerate(merged):
                if cancelled():
                    raise InterruptedError('planning_cancelled')
                grasp = CameraGrasp(value.translation, value.rotation_matrix, value.width,
                                    value.depth, value.score, source_camera)
                identifier = '{}-{:03d}'.format(mode, index+1)
                row = native_pose_record(grasp, transform, identifier)
                row.update(audit.audit(grasp, transform, mode, identifier))
                row['mode'] = mode
                records.append(row)
        return records
