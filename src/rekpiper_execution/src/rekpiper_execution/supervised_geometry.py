"""Measured calibration and conservative per-segment RGB-D collision scenes."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import yaml

from rekpiper_camera.extrinsics import validate_rigid_transform
from rekpiper_planning.offline_observation import build_depth_collision_grid, rgbd_points, depth_metres
from rekpiper_planning.table_surface import fit_table_plane, table_footprint
from .supervised_session import digest


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_experiment(config):
    hand_path, stereo_path = config['rs1_extrinsics'], config['stereo_extrinsics']
    hand = yaml.safe_load(Path(hand_path).read_text())
    stereo = yaml.safe_load(Path(stereo_path).read_text())
    entry = hand['calibration']['base_to_camera_optical']
    if (entry['parent_frame'] != 'arm_base'
            or entry['child_frame'] != 'rs1_color_optical_frame'):
        raise ValueError('unsupported_handeye_frame_direction')
    # This is an explicit installation transform, never a frame-name substitution.
    base_from_arm = validate_rigid_transform(config['base_link_T_arm_base'])
    first = base_from_arm @ validate_rigid_transform(entry['matrix_4x4'])
    relative = validate_rigid_transform(stereo['rs1_optical_T_rs3_optical'])
    for name in ('rs1', 'rs3'):
        if str(stereo['cameras'][name]['serial']) != str(config['serials'][name]):
            raise ValueError('stereo_camera_identity_mismatch:'+name)
    if str(hand['camera']['serial']) != str(config['serials']['rs1']):
        raise ValueError('handeye_camera_identity_mismatch')
    binding = dict(rs1=file_hash(hand_path), stereo=file_hash(stereo_path),
                   base_link_T_arm_base=base_from_arm.tolist(), serials=config['serials'])
    return {'rs1': first, 'rs3': first @ relative}, binding


def load_segmentation_snapshot(path, stamp_ns):
    with np.load(path, allow_pickle=False) as data:
        saved_stamp = int(data['stamp_ns'])
        mask = data['mask'].copy()
        xyz = data['xyz'].copy()
        rs3 = None
        if 'rs3_mask' in data:
            rs3 = dict(stamp_ns=int(data['rs3_stamp_ns']),
                       mask=data['rs3_mask'].copy(), xyz=data['rs3_xyz'].copy())
    if (saved_stamp != stamp_ns or mask.ndim != 2 or xyz.shape != mask.shape + (3,)
            or not np.issubdtype(mask.dtype, np.integer) or not np.any(mask > 0)
            or not np.isfinite(xyz[mask > 0]).all()):
        raise ValueError('segmentation_snapshot_mismatch')
    if rs3 is not None and (
            abs(rs3['stamp_ns'] - saved_stamp) > 50_000_000
            or rs3['mask'].ndim != 2
            or rs3['xyz'].shape != rs3['mask'].shape + (3,)
            or not np.issubdtype(rs3['mask'].dtype, np.integer)
            or not np.any(rs3['mask'] > 0)
            or not np.isfinite(rs3['xyz'][rs3['mask'] > 0]).all()):
        raise ValueError('rs3_segmentation_snapshot_mismatch')
    return dict(stamp_ns=saved_stamp, mask=mask, xyz=xyz, rs3=rs3)


def point_at(frame, pixel):
    u, v = [int(x) for x in pixel]
    depth = depth_metres(frame)
    if not (2 <= u < depth.shape[1]-2 and 2 <= v < depth.shape[0]-2):
        raise ValueError('calibration_pixel_outside_image')
    patch = depth[v-2:v+3, u-2:u+3]
    good = patch[np.isfinite(patch) & (patch > .15) & (patch < 1.5)]
    if len(good) < 15 or np.ptp(good) > .01:
        raise ValueError('calibration_depth_missing_or_discontinuous')
    z = float(np.median(good)); k = np.asarray(frame['K']).reshape(3, 3)
    point = np.array([(u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z])
    t = frame['base_from_camera']
    return t[:3, :3] @ point+t[:3, 3]


def validate_known_points(rows, tolerance=.005):
    if len(rows) < 4:
        raise ValueError('at_least_four_independent_known_points_required')
    reference = np.asarray([r['base'] for r in rows], dtype=float)
    if (reference.shape != (len(rows), 3) or not np.isfinite(reference).all()
            or np.linalg.matrix_rank(reference-reference.mean(axis=0), tol=.005) < 2
            or np.ptp(reference[:, 2]) < .03
            or np.max(np.ptp(reference[:, :2], axis=0)) < .10):
        raise ValueError('known_points_need_spread_and_two_heights')
    errors = {}
    for camera in ('rs1', 'rs3'):
        values = np.asarray([r[camera] for r in rows], dtype=float)
        if values.shape != reference.shape or not np.isfinite(values).all():
            raise ValueError('invalid_known_point_measurements')
        errors[camera] = np.linalg.norm(values-reference, axis=1).tolist()
    errors['cross_view'] = np.linalg.norm(np.asarray([r['rs1'] for r in rows])-
                                         np.asarray([r['rs3'] for r in rows]), axis=1).tolist()
    if max(max(x) for x in errors.values()) > tolerance:
        raise ValueError('known_point_error_exceeds_5mm:'+str(errors))
    return errors


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T+matrix[:3, 3]


def mask_near_geometry(frame, points, radii):
    """Remove only depth samples explained by the measured robot geometry."""
    xyz, _, pixels = rgbd_points(frame)
    distance, index = cKDTree(points).query(xyz)
    selected = distance <= np.asarray(radii)[index]+.003
    mask = np.zeros(frame['depth'].shape, dtype=bool)
    mask[pixels[0][selected], pixels[1][selected]] = True
    return mask


class SegmentScene:
    def __init__(self, frames, config, sampler, joints, opening, held_world=None):
        self.frames = deepcopy(frames)
        self.config = config
        robot, radii, _ = sampler.contact_samples(joints, opening)
        exclusions, environment = {}, {}
        for name, frame in frames.items():
            exclusions[name] = mask_near_geometry(frame, robot, radii)
            if held_world is not None:
                exclusions[name] |= mask_near_geometry(
                    frame, held_world, np.full(len(held_world), .005))
            xyz, _, pixels = rgbd_points(frame)
            use = ~exclusions[name][pixels]
            lo, hi = np.array(config['bounds_min']), np.array(config['bounds_max'])
            use &= np.all((xyz >= lo) & (xyz <= hi), axis=1)
            environment[name] = xyz[use]
        self.points = np.vstack(list(environment.values()))
        if len(self.points) < 300:
            raise ValueError('insufficient_environment_cloud')
        # Experimental preview uses RS1 for the table while retaining both views
        # as collision obstacles. Normal operation still fits both views.
        table_cloud = (environment['rs1'] if config.get('extrinsics_status') == 'EXPERIMENTAL_PREVIEW_ONLY'
                       else self.points)
        self.table = fit_table_plane(table_cloud)
        normal = np.asarray(self.table['table_plane_normal'])
        offset = self.table['table_plane_offset_m']
        table_points = table_cloud[np.abs(table_cloud @ normal+offset) < .006]
        self.table['table_footprint_xy_m'] = table_footprint(table_points)
        self.grid = build_depth_collision_grid(frames, config['bounds_min'], config['bounds_max'],
                                               config['voxel_m'], vacated_masks=exclusions,
                                               table_model=self.table)
        self.id = digest(dict(stamps={k: float(v['rgb_stamp_s']) for k, v in frames.items()},
                              points=hashlib.sha256(self.points.tobytes()).hexdigest()))
        # Do not republish this as an accepted production SDFGrid.
        self.grid['motion_allowed'] = False
        self.lower, self.upper = self.grid['bounds_min'], self.grid['bounds_max']
        self.shape = np.array(self.grid['observed'].shape)
        axes = [np.linspace(a, b, int(n)) for a, b, n in zip(self.lower, self.upper, self.shape)]
        coordinates = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
        # The physical robot already occupies this precisely modeled volume.
        # Its interior is known, but no unobserved space around it is cleared.
        distance, index = cKDTree(robot).query(coordinates)
        inside_robot = (distance <= radii[index]).reshape(tuple(self.shape))
        self.grid['observed'][inside_robot] = True
        free = (self.grid['distances_m'] < 0) | inside_robot
        occupied = self.grid['occupied'] & ~inside_robot
        distances = -distance_transform_edt(free & ~occupied, sampling=config['voxel_m'])
        distances[occupied] = distance_transform_edt(occupied, sampling=config['voxel_m'])[occupied]
        distances[~self.grid['observed']] = 1.
        self.grid['distances_m'] = distances.astype(np.float32)
        self.tree = cKDTree(self.points)

    def sample(self, points):
        points = np.asarray(points)
        index = np.rint((points-self.lower)*(self.shape-1)/(self.upper-self.lower)).astype(int)
        inside = np.all((index >= 0) & (index < self.shape), axis=1)
        known = np.zeros(len(points), dtype=bool)
        distance = np.ones(len(points))
        ix = index[inside]
        known[inside] = self.grid['observed'][ix[:, 0], ix[:, 1], ix[:, 2]]
        distance[inside] = self.grid['distances_m'][ix[:, 0], ix[:, 1], ix[:, 2]]
        return distance, known & inside

    def target_points(self, frame, mask, group, source_points=None):
        if source_points is None:
            xyz, _, pixels = rgbd_points(frame)
            target = xyz[np.asarray(mask)[pixels] == group]
        else:
            xyz = np.asarray(source_points)
            labels = np.asarray(mask)
            if xyz.shape != labels.shape + (3,):
                raise ValueError('segmentation_snapshot_shape_mismatch')
            target = xyz[(labels == group) & np.isfinite(xyz).all(axis=2)]
        if len(target) < 120:
            raise ValueError('target_mask_has_insufficient_depth')
        # A bounding box alone can include the table or a neighbouring object.
        # Keep second-view points only where the selected first-view surface agrees.
        other, _, _ = rgbd_points(self.frames['rs3'])
        matching = cKDTree(target).query(other)[0] <= .008
        return np.vstack([target, other[matching]])

    def unchanged(self, other, expected_removed=None):
        points = other.points
        if expected_removed is not None:
            points = points[cKDTree(expected_removed).query(points)[0] > .01]
        distances = self.tree.query(points)[0]
        if len(distances) < 100 or np.count_nonzero(distances > .015) > max(20, .01*len(distances)):
            raise ValueError('environment_changed_repreview_required')
