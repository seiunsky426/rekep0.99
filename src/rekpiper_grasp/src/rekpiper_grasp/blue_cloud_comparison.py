"""Pure geometry for frozen, color-selected AnyGrasp input comparisons."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from rekpiper_planning.offline_observation import rgbd_points

CASES = ('rs1', 'rs3', 'fused')
POLICY = dict(hsv_lower=[78, 91, 46], hsv_upper=[115, 255, 255],
              erosion_px=3, minimum_target_depth_points=120,
              outlier_neighbors=20, outlier_std_ratio=2.5,
              target_voxel_m=.002, background_voxel_m=.003,
              maximum_background_points=30000, random_seed=0)
MODEL_POLICY = dict(max_gripper_width_m=.070, gripper_height_m=.03,
                    dense_grasp=False, collision_detection=False,
                    approach_steering=None, max_candidates=None)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def transform_points(points, matrix):
    matrix = np.asarray(matrix)
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def select_blue(frame, lower, upper, workspace_from_arm):
    """Frame XYZ is arm_base; configured workspace bounds are in base_link."""
    xyz, rgb, pixels = rgbd_points(frame)
    workspace = transform_points(xyz, workspace_from_arm)
    inside = np.all(workspace >= lower, axis=1) & np.all(workspace <= upper, axis=1)
    valid = np.zeros(frame['depth'].shape, bool)
    valid[pixels] = inside
    hsv = cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2HSV)
    blue = cv2.inRange(hsv, np.array(POLICY['hsv_lower']), np.array(POLICY['hsv_upper'])) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats((blue & valid).astype(np.uint8), 8)
    component = np.zeros_like(valid)
    if count > 1:
        component = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = cv2.erode(component.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & valid
    target_pixels = mask[pixels]
    raw, raw_rgb = xyz[target_pixels], rgb[target_pixels]
    keep = np.ones(len(raw), bool)
    if len(raw) >= 21:
        distances = cKDTree(raw).query(raw, k=21)[0][:, 1:].mean(axis=1)
        keep = distances <= distances.mean() + 2.5 * distances.std()
    filtered_mask = mask.copy()
    positions = np.column_stack(pixels)[target_pixels]
    if len(positions):
        filtered_mask[tuple(positions[~keep].T)] = False
    # Discard eroded boundary and rejected blue points, rather than relabeling
    # them as scene background and weakening region_steering.
    background = inside & ~component[pixels]
    status = ('ready' if len(raw) >= 120 and int(keep.sum()) >= 120
              else 'insufficient_blue_depth_points')
    return dict(target=raw[keep], target_rgb=raw_rgb[keep], raw=raw, raw_rgb=raw_rgb,
                background=xyz[background], background_rgb=rgb[background],
                mask=filtered_mask, component=component,
                report=dict(status=status, raw_target_points=len(raw),
                            filtered_target_points=int(keep.sum()),
                            workspace_points=int(inside.sum()),
                            background_points=int(background.sum())))


def voxel_average(points, rgb, sources, size):
    points = np.asarray(points).reshape(-1, 3)
    if not len(points):
        return points.astype(np.float32), np.empty((0, 3), np.uint8), np.empty(0, np.uint8)
    _, inverse = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    sums, colors = np.zeros((len(counts), 3)), np.zeros((len(counts), 3))
    np.add.at(sums, inverse, points)
    np.add.at(colors, inverse, rgb)
    bits = np.zeros(len(counts), np.uint8)
    np.bitwise_or.at(bits, inverse, sources)
    return (sums/counts[:, None]).astype(np.float32), np.rint(colors/counts[:, None]).astype(np.uint8), bits


def prepare_case(selected, names, arm_from_reference):
    def combine(kind, size):
        xyz = np.concatenate([selected[n][kind] for n in names])
        rgb = np.concatenate([selected[n][kind+'_rgb'] for n in names])
        source = np.concatenate([np.full(len(selected[n][kind]), 1 if n == 'rs1' else 2,
                                         np.uint8) for n in names])
        return voxel_average(xyz, rgb, source, size)
    target, tc, ts = combine('target', .002)
    background, bc, bs = combine('background', .003)
    # Remove background falling in a target voxel at either resolution.
    keep = np.ones(len(background), bool)
    for size in (.002, .003):
        occupied = set(map(tuple, np.floor(target / size).astype(np.int64)))
        keep &= np.array([tuple(cell) not in occupied
                          for cell in np.floor(background / size).astype(np.int64)], dtype=bool)
    background, bc, bs = background[keep], bc[keep], bs[keep]
    background_voxels = len(background)
    if len(background) > 30000:
        indices = np.sort(np.random.default_rng(0).choice(len(background), 30000, replace=False))
        background, bc, bs = background[indices], bc[indices], bs[indices]
    points = np.concatenate([target, background])
    mask = np.arange(len(points)) < len(target)
    return dict(points_arm=points, points_camera=transform_points(
                    points, np.linalg.inv(arm_from_reference)).astype(np.float32),
                region_mask=mask, rgb=np.concatenate([tc, bc]), source_bits=np.r_[ts, bs],
                arm_from_reference=np.asarray(arm_from_reference),
                background_voxels_before_sampling=np.int64(background_voxels))


def native_pose_record(grasp, arm_from_reference, identifier):
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = grasp.rotation, grasp.translation
    return dict(id=identifier, pose_reference=pose.tolist(),
                pose_arm_base=(arm_from_reference @ pose).tolist(),
                width_m=float(grasp.width_m), depth_m=float(grasp.depth_m), score=float(grasp.score))


def gripper_segments(row):
    """Native GraspNet x approach, y closing; finger tips at x=depth."""
    width, depth = row['width_m'], row['depth_m']
    rear = -.022
    width += .004  # Center lines of official 4 mm thick fingers.
    local = np.array([[rear, -width/2, 0], [depth, -width/2, 0],
                      [rear, width/2, 0], [depth, width/2, 0],
                      [rear, -width/2, 0], [rear, width/2, 0],
                      [rear-.035, 0, 0], [rear, 0, 0]])
    return transform_points(local, np.asarray(row['pose_arm_base']))


def view_label(case, report, top_k):
    rows = report.get('candidates', [])[:top_k]
    if rows:
        return '{} | top {} / {} | native AnyGrasp | no collision/IK'.format(
            case.upper(), len(rows), len(report['candidates']))
    return '{} | NO CANDIDATES | {}'.format(case.upper(), report.get('error', report['status']))


def write_ply(path, points, rgb):
    data = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for i, key in enumerate(('x', 'y', 'z')):
        data[key] = points[:, i]
    for i, key in enumerate(('red', 'green', 'blue')):
        data[key] = rgb[:, i]
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex {}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    Path(path).write_bytes(header.format(len(points)).encode()+data.tobytes())
