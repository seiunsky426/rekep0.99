#!/usr/bin/env python3
"""Offline paired-view SAM geometry check. Never publishes or commands a robot."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch

from rekpiper_perception.sam_automatic import SAMAutomaticSegmenter


def cloud(data, mask):
    z = data['depth'].astype(np.float32)
    encoding = str(data['depth_encoding'].item())
    if encoding in ('16UC1', 'mono16'):
        z *= .001
    elif encoding != '32FC1':
        raise ValueError('unknown_depth_units')
    k = data['K']; v, u = np.indices(z.shape)
    camera = np.stack([(u-k[2])*z/k[0], (v-k[5])*z/k[4], z], axis=-1)
    valid = mask & np.isfinite(camera).all(axis=2) & (z > .15) & (z < 1.5)
    t = data['base_from_camera']
    return camera[valid] @ t[:3, :3].T + t[:3, 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-directory', required=True)
    parser.add_argument('--reference-group', required=True, type=int)
    parser.add_argument('--rs3-pixel', nargs=2, required=True, type=float)
    args = parser.parse_args()
    root = Path(args.evidence_directory)
    out = root / 'rs3_target_check'
    out.mkdir(exist_ok=False)
    start = time.monotonic(); done = threading.Event()
    def heartbeat():
        while not done.wait(5):
            print('RS3_SAM_CHECK_RUNNING', round(time.monotonic()-start, 1), flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    report = {'motion_allowed': False, 'hardware_commands_sent': False,
              'full_task_accepted': False, 'cross_view_identity_accepted': False,
              'scope': 'paired_snapshot_geometry_diagnostic',
              'reference_group': args.reference_group,
              'rs3_positive_pixel': args.rs3_pixel}
    try:
        capture = json.loads((root/'capture_report.json').read_text())
        report['snapshot_id'] = capture['snapshot_id']
        a = np.load(root/'rs1_rgbd.npz', allow_pickle=False)
        b = np.load(root/'rs3_rgbd.npz', allow_pickle=False)
        stamps = [float(d[key]) for d in (a, b) for key in ('rgb_stamp_s', 'depth_stamp_s')]
        report['maximum_pair_skew_s'] = max(stamps)-min(stamps)
        if not capture['capture_success'] or max(stamps)-min(stamps) > .025:
            raise ValueError('paired_capture_invalid_or_skew_exceeds_25ms')
        rs1_mask = cv2.imread(str(root/'scene_mask.png'), -1) == args.reference_group
        first = cloud(a, rs1_mask)
        if len(first) < 120:
            raise ValueError('insufficient_rs1_target_points')
        positive = np.asarray([args.rs3_pixel], dtype=np.float32)
        height, width = b['rgb'].shape[:2]
        if not (0 <= positive[0, 0] < width and 0 <= positive[0, 1] < height):
            raise ValueError('prompt_pixel_outside_image')
        sampler = SAMAutomaticSegmenter(
            Path(os.environ['REKPIPER_MODEL_ROOT'])/'sam_vit_h_4b8939.pth', 'cuda')
        predictor = sampler._generator.predictor
        began = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            predictor.set_image(b['rgb'])
            masks, scores, _ = predictor.predict(
                point_coords=positive, point_labels=np.ones(1, dtype=np.int32),
                multimask_output=True)
        index = int(np.argmax(scores)); mask = masks[index]
        report['sam_inference_s'] = time.monotonic()-began
        report['sam_scores'] = scores.tolist()
        report['selected_mask'] = index
        report['mask_pixels'] = int(mask.sum())
        report['mask_sha256'] = hashlib.sha256(mask.tobytes()).hexdigest()
        report['mask_touches_image_edge'] = bool(
            mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        cv2.imwrite(str(out/'rs3_source.png'), cv2.cvtColor(b['rgb'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out/'rs3_target_mask.png'), mask.astype(np.uint8)*255)
        second = cloud(b, mask)
        if len(second) < 120:
            raise ValueError('insufficient_rs3_target_points')
        del sampler, predictor; gc.collect(); torch.cuda.empty_cache()
        forward = cKDTree(second).query(first)[0]
        backward = cKDTree(first).query(second)[0]
        report['target_point_counts'] = [len(first), len(second)]
        report['median_centers_base'] = [np.median(p, axis=0).tolist() for p in (first, second)]
        report['median_center_delta_m'] = float(np.linalg.norm(
            np.median(first, axis=0)-np.median(second, axis=0)))
        report['rs1_to_rs3_nearest_p50_p95_m'] = np.quantile(forward, [.5, .95]).tolist()
        report['rs3_to_rs1_nearest_p50_p95_m'] = np.quantile(backward, [.5, .95]).tolist()
        report['centroid_gate_20mm_pass'] = report['median_center_delta_m'] <= .020
        report['reason'] = ('cross_view_centroid_exceeds_existing_20mm_gate'
            if not report['centroid_gate_20mm_pass'] else 'geometry_only_not_tracked_uuid_acceptance')
        np.savez_compressed(out/'paired_target_clouds.npz',
                            rs1_points_base=first, rs3_points_base=second, motion_allowed=False)
    except Exception as exc:
        report['error'] = type(exc).__name__+': '+str(exc)
    finally:
        report['elapsed_s'] = time.monotonic()-start
        (out/'report.json').write_text(json.dumps(report, indent=2))
        done.set(); print(json.dumps(report, indent=2), flush=True)
    return 1 if 'error' in report else 0


if __name__ == '__main__':
    raise SystemExit(main())
