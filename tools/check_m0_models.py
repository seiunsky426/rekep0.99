#!/usr/bin/env python3
"""Run one offline CUDA smoke check in the current ReKpiper environment.

These checks verify loading and computation, not perception quality, camera
calibration, or permission to send robot commands. Run models in separate
processes so their CUDA allocations do not overlap.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
VENDOR = Path(os.environ.get("REKPIPER_VENDOR_ROOT", ROOT / "runtime/vendor"))
MODELS = Path(os.environ.get("REKPIPER_MODEL_ROOT", ROOT / "runtime/models"))


def dinov2():
    from rekpiper_perception.official_keypoint_adapter import load_local_dinov2
    model = load_local_dinov2(str(VENDOR / "dinov2"),
                             str(MODELS / "dinov2_vits14_reg4_pretrain.pth"), "cuda")
    with torch.inference_mode():
        output = model.forward_features(torch.zeros(1, 3, 56, 56, device="cuda"))
    features = output["x_norm_patchtokens"]
    assert tuple(features.shape) == (1, 16, 384)
    assert torch.isfinite(features).all()
    return {"stage": "reg4_patch_features", "shape": list(features.shape)}


def sam():
    from rekpiper_perception.sam_automatic import SAMAutomaticSegmenter
    backend = SAMAutomaticSegmenter(
        str(MODELS / "sam_vit_h_4b8939.pth"), "cuda", points_per_side=2,
        pred_iou_thresh=0.0, stability_score_thresh=0.0,
        min_area_px=16, max_area_ratio=1.0)
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[32:96, 32:96] = [220, 80, 30]
    masks = backend.segment(frame)
    assert masks and all(m.shape == (128, 128) and m.dtype == bool for m in masks)
    return {"stage": "vit_h_automatic_masks", "mask_count": len(masks),
            "thresholds": "low thresholds for compute check only"}


def cutie():
    from rekpiper_perception.cutie_backend import CutieBackend
    backend = CutieBackend(str(VENDOR / "Cutie"), str(MODELS / "cutie-base-mega.pth"),
                           max_internal_size=128, require_cuda=True)
    processor = backend.new_processor()
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[40:88, 40:88, 1] = 200
    seed = np.zeros((128, 128), dtype=np.uint16)
    seed[40:88, 40:88] = 1
    backend.step(processor, frame, seed, [1])
    labels, confidence = backend.step(processor, frame)
    assert labels.shape == (128, 128) and 1 in np.unique(labels)
    assert np.isfinite(confidence).all()
    return {"stage": "seed_and_next_frame", "shape": list(labels.shape),
            "labels": np.unique(labels).tolist()}


def curobo():
    import yaml
    from curobo.types.base import TensorDeviceType
    from curobo.types.camera import CameraObservation
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.model.robot_segmenter import RobotSegmenter
    config = yaml.safe_load((ROOT / "src/rekpiper_mapping/config/piper_curobo.yml").read_text())
    description = ROOT / "src/piper_description"
    config["robot_cfg"]["kinematics"].update(
        urdf_path=str(description / "urdf/piper_description.urdf"),
        asset_root_path=str(description))
    tensor_args = TensorDeviceType.from_basic("cuda", 0)
    segmenter = RobotSegmenter.from_robot_file(
        config, collision_sphere_buffer=0.005, distance_threshold=0.015,
        use_cuda_graph=False, tensor_args=tensor_args)
    names = config["robot_cfg"]["kinematics"]["cspace"]["joint_names"]
    state = JointState.from_numpy(names, np.zeros(8, dtype=np.float32),
                                 tensor_args=tensor_args)
    depth = torch.full((1, 32, 32), 1000.0, device="cuda")
    # The central sample lies inside the configured base-link sphere;
    # the surrounding one-metre plane is outside the zero-pose robot.
    depth[0, 16, 16] = 43.0
    observation = CameraObservation(
        depth_image=depth,
        intrinsics=torch.tensor([[30., 0., 15.5], [0., 30., 15.5], [0., 0., 1.]],
                                device="cuda"),
        pose=Pose.from_matrix(torch.eye(4, device="cuda")))
    mask, filtered = segmenter.get_robot_mask(observation, state)
    assert tuple(mask.shape) == (1, 32, 32) and torch.isfinite(filtered).all()
    assert mask[0, 16, 16] and (~mask).any() and filtered[0, 16, 16] == 0
    return {"stage": "piper_fk_and_robot_depth_mask", "shape": list(mask.shape),
            "robot_pixels": int(mask.sum().item())}


def nvblox():
    from nvblox_torch.mapper import Mapper
    mapper = Mapper(voxel_sizes=[0.02], integrator_types=["tsdf"],
                    free_on_destruction=True)
    depth = torch.ones((32, 32), device="cuda")
    pose = torch.eye(4, device="cuda")
    intrinsics = torch.tensor([[30., 0., 15.5], [0., 30., 15.5], [0., 0., 1.]])
    for _ in range(3):
        mapper.add_depth_frame(depth, pose, intrinsics, 0)
    mapper.update_esdf(0)
    mapper.update_hashmaps()
    query = torch.tensor([[0., 0., 0.8, 0.], [0., 0., 0.98, 0.]], device="cuda")
    distances = mapper.query_sdf(query, mapper_id=0)
    assert distances.numel() == 2 and torch.isfinite(distances).all()
    assert (distances.abs() < 1.0).all(), "query returned unknown-space sentinel"
    return {"stage": "depth_tsdf_esdf_query", "distances": distances.cpu().tolist()}


def anygrasp():
    import MinkowskiEngine as ME
    from pointnet2 import pointnet2_utils
    from rekpiper_grasp.anygrasp_adapter import AnyGraspAdapter, check_anygrasp_runtime
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32)
    sparse = ME.SparseTensor(torch.ones(2, 1), coordinates=coords, device="cuda")
    layer = ME.MinkowskiConvolution(1, 2, kernel_size=1, dimension=3).cuda()
    assert torch.isfinite(layer(sparse).F).all()
    xyz = torch.rand(1, 64, 3, device="cuda")
    samples = pointnet2_utils.furthest_point_sample(xyz, 8)
    assert tuple(samples.shape) == (1, 8)
    sdk = VENDOR / "anygrasp_sdk/grasp_detection"
    result = check_anygrasp_runtime(sdk, sdk / "checkpoint_detection.tar", sdk / "license")
    assert result.ready, result.reasons
    detector = AnyGraspAdapter(sdk, sdk / "checkpoint_detection.tar", sdk / "license")
    rng = np.random.RandomState(0)
    points = rng.uniform([-0.10, -0.10, 0.6], [0.10, 0.10, 0.6], (4096, 3))
    points[:1024] = rng.uniform([-0.025, -0.025, 0.55], [0.025, 0.025, 0.59], (1024, 3))
    points = points.astype(np.float32)
    grasps = detector.infer(points, np.ones(len(points), dtype=bool), "synthetic")
    return {"stage": "sparse_conv_pointnet2_licensed_detector_inference",
            "candidate_count": len(grasps), "preflight_ready": result.ready}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["sam", "dinov2", "cutie", "curobo", "nvblox", "anygrasp"])
    parser.add_argument("--output")
    args = parser.parse_args()
    report = {"model": args.model, "python": sys.executable, "torch": torch.__version__,
              "compiled_cuda": torch.version.cuda, "hardware_execution_authorized": False}
    try:
        assert torch.cuda.is_available(), "CUDA is unavailable"
        torch.set_num_threads(2)
        torch.manual_seed(0)
        report.update(globals()[args.model]())
        torch.cuda.synchronize()
        report["imports"] = {
            name: str(Path(module.__file__).resolve())
            for name in ("segment_anything", "dinov2.hub.backbones",
                         "cutie.model.cutie", "curobo", "nvblox_torch.mapper",
                         "MinkowskiEngine", "MinkowskiEngineBackend._C",
                         "pointnet2._ext", "grasp_nms", "gsnet")
            for module in [sys.modules.get(name)]
            if module is not None and getattr(module, "__file__", None)}
        report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        report["passed"] = True
    except Exception as exc:
        report.update(passed=False, error=type(exc).__name__, message=str(exc))
        traceback.print_exc()
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
