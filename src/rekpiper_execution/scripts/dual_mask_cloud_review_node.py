#!/usr/bin/env python3
"""Display-only review of two selected, frozen SAM depth masks in RViz."""
import hashlib
import json
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import numpy as np
import rospy
import yaml
from geometry_msgs.msg import Point, Quaternion
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, String
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl

from rekpiper_execution.supervised_session import select_instance_group
from rekpiper_planning.vlm_program_generator import create_backend


NS = '/rekpiper/supervised/blue_review'
FIELDS = [PointField(name=name, offset=4 * index, datatype=PointField.FLOAT32, count=1)
          for index, name in enumerate(('x', 'y', 'z'))]
FIELDS.append(PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1))
COLORS = {'rs1': 0x00D9FF, 'rs3': 0xFF39C0}


def selected_clouds(snapshot_path, selection_path):
    snapshot = Path(snapshot_path).read_bytes()
    selection = json.loads(Path(selection_path).read_text())
    if selection.get('snapshot_sha256') != hashlib.sha256(snapshot).hexdigest():
        raise ValueError('selection_does_not_match_frozen_snapshot')
    with np.load(snapshot_path, allow_pickle=False) as data:
        if 'rs3_mask' not in data:
            raise ValueError('rs3_segmentation_not_in_snapshot')
        if abs(float(selection['stamp']) - int(data['stamp_ns']) * 1e-9) > .001:
            raise ValueError('selection_stamp_mismatch')
        groups = {'rs1': int(selection['selection'][0]['rigid_group_id']),
                  'rs3': int(selection['rs3_group_id'])}
        result = {}
        for name, prefix in (('rs1', ''), ('rs3', 'rs3_')):
            mask = data[prefix + 'mask']
            xyz = data[prefix + 'xyz']
            if xyz.shape != mask.shape + (3,) or groups[name] <= 0:
                raise ValueError(name + '_invalid_mask_or_xyz_shape')
            points = xyz[(mask == groups[name]) & np.isfinite(xyz).all(axis=2)].copy()
            if len(points) < 120:
                raise ValueError(name + '_selected_mask_has_insufficient_depth')
            result[name] = points
    return result


class DualMaskCloudReview:
    def __init__(self):
        self.snapshot_path = Path(rospy.get_param('~snapshot_output'))
        self.selection_path = Path(rospy.get_param('~selection_output', '') or
                                   self.snapshot_path.with_name('vlm_selection.json'))
        self.system = yaml.safe_load(Path(rospy.get_param('~system_config')).read_text())
        self.bridge = CvBridge()
        self.rs3_image = None
        self.rs3_image_stamp_ns = None
        rospy.Subscriber('/rekpiper/perception/rs3_candidate_image', Image,
                         self.rs3_image_cb, queue_size=1)
        self.clouds = None
        self.centers = {}
        self.offsets = {'rs1': 0., 'rs3': 0.}
        self.loaded_hash = None
        self.loaded_snapshot_sha = None
        self.publishers = {name: rospy.Publisher(NS + '/' + name, PointCloud2, queue_size=1, latch=True)
                           for name in ('rs1', 'rs3', 'fused')}
        self.status = rospy.Publisher(NS + '/status', String, queue_size=1, latch=True)
        self.server = InteractiveMarkerServer(NS + '/drag')
        self.status.publish(String('waiting_for_two_selected_static_masks'))
        self.timer = rospy.Timer(rospy.Duration(1.), self.poll)

    def rs3_image_cb(self, message):
        self.rs3_image = self.bridge.imgmsg_to_cv2(message, 'bgr8').copy()
        self.rs3_image_stamp_ns = message.header.stamp.to_nsec()

    def enrich_selection(self, snapshot_sha):
        """Support an already-running workbench that only chose RS1."""
        selection = json.loads(self.selection_path.read_text())
        if 'rs3_group_id' in selection:
            return
        with np.load(self.snapshot_path, allow_pickle=False) as data:
            stamp_ns = int(data['stamp_ns'])
            rs3_stamp_ns = int(data['rs3_stamp_ns'])
            labels = data['rs3_mask'].copy()
        if (abs(float(selection['stamp']) - stamp_ns * 1e-9) > .001
                or self.rs3_image_stamp_ns != rs3_stamp_ns):
            raise ValueError('waiting_for_matching_rs3_vlm_image')
        path = self.snapshot_path.with_name('rs3_review_vlm.png')
        if not cv2.imwrite(str(path), self.rs3_image):
            raise ValueError('rs3_candidate_image_save_failed')
        prompt = ('Select the small cyan-blue cube from the locked RS3 image. SAM masks are labeled G1, G2, etc. '
                  'Ignore all K keypoint labels; select only a G instance, not the robot arm or yellow disk. Return only JSON {"group_id": integer}; do not provide coordinates. '
                  'Available IDs: ' + str(sorted(int(v) for v in np.unique(labels) if v > 0)))
        self.status.publish(String('rs3_vlm_confirming_blue_mask'))
        answer = create_backend(self.system['vlm']).generate_text(prompt, str(path))
        group = select_instance_group(answer, labels)
        selection['rs3_group_id'] = group
        selection['rs3_raw'] = answer
        selection['snapshot_sha256'] = snapshot_sha
        temporary = self.selection_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(selection, ensure_ascii=False, indent=2))
        temporary.replace(self.selection_path)

    def poll(self, _event):
        if not self.snapshot_path.is_file():
            return
        try:
            snapshot_sha = hashlib.sha256(self.snapshot_path.read_bytes()).hexdigest()
            if self.loaded_snapshot_sha is not None and snapshot_sha != self.loaded_snapshot_sha:
                self.clear()
            if not self.selection_path.is_file():
                return
            self.enrich_selection(snapshot_sha)
            marker = hashlib.sha256(self.selection_path.read_bytes()).hexdigest()
            if marker == self.loaded_hash and snapshot_sha == self.loaded_snapshot_sha:
                return
            clouds = selected_clouds(self.snapshot_path, self.selection_path)
        except (OSError, KeyError, ValueError, IndexError, TypeError) as exc:
            self.status.publish(String(str(exc)))
            return
        self.clouds = clouds
        self.loaded_hash = marker
        self.loaded_snapshot_sha = snapshot_sha
        self.offsets = {'rs1': 0., 'rs3': 0.}
        self.centers = {name: np.median(points, axis=0) for name, points in clouds.items()}
        self.server.clear()
        for name in ('rs1', 'rs3'):
            self.add_handle(name)
        self.server.applyChanges()
        self.publish()

    def clear(self):
        self.clouds = None
        self.loaded_hash = None
        self.loaded_snapshot_sha = None
        self.server.clear()
        self.server.applyChanges()
        empty = point_cloud2.create_cloud(
            Header(stamp=rospy.Time.now(), frame_id='base_link'), FIELDS, [])
        for publisher in self.publishers.values():
            publisher.publish(empty)
        self.status.publish(String('snapshot_changed_waiting_for_matching_selection'))

    def add_handle(self, name):
        center = self.centers[name]
        handle = InteractiveMarker(name=name, description=name.upper() + ' blue mask: drag Z only',
                                   scale=.08)
        handle.header.frame_id = 'base_link'
        handle.pose.position = Point(*[float(value) for value in center])
        handle.pose.orientation.w = 1.
        half = float(np.sqrt(.5))
        handle.controls.append(InteractiveMarkerControl(
            name='vertical', orientation=Quaternion(0., half, 0., half),
            interaction_mode=InteractiveMarkerControl.MOVE_AXIS))
        self.server.insert(handle, lambda feedback, camera=name: self.drag(camera, feedback))

    def drag(self, name, feedback):
        if self.clouds is None:
            return
        self.offsets[name] = float(feedback.pose.position.z - self.centers[name][2])
        self.publish()

    def publish(self):
        if self.clouds is None:
            return
        header = Header(stamp=rospy.Time.now(), frame_id='base_link')
        fused = []
        for name in ('rs1', 'rs3'):
            moved = self.clouds[name].copy()
            moved[:, 2] += self.offsets[name]
            rows = [(float(x), float(y), float(z), COLORS[name]) for x, y, z in moved]
            self.publishers[name].publish(point_cloud2.create_cloud(header, FIELDS, rows))
            fused.extend(rows)
        self.publishers['fused'].publish(point_cloud2.create_cloud(header, FIELDS, fused))
        self.status.publish(String('static_blue_masks rs1={} rs3={} z_offsets_m={:.4f},{:.4f}'
            .format(len(self.clouds['rs1']), len(self.clouds['rs3']),
                    self.offsets['rs1'], self.offsets['rs3'])))


if __name__ == '__main__':
    rospy.init_node('dual_mask_cloud_review')
    DualMaskCloudReview()
    rospy.spin()
