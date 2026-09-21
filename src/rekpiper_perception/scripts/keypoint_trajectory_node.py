#!/usr/bin/env python3
"""Display and record measured M5 keypoint trajectories without commanding motion."""

from collections import defaultdict, deque
import csv
import json
from pathlib import Path
import threading

import cv2
from cv_bridge import CvBridge
import numpy as np
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

from rekpiper_camera.projection import transform_to_matrix
from rekpiper_msgs.msg import Keypoint3DArray, TrackedObjectArray


class TrajectoryView:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.image = None
        self.info = None
        self.tracks = None
        self.last = {}
        self.segments = defaultdict(lambda: deque(maxlen=400))
        self.colors = {}
        self.directory = Path(rospy.get_param('~output_directory')).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.csv_file = (self.directory / 'keypoint_trajectories.csv').open('w')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['capture_stamp_s', 'received_stamp_s', 'id', 'name',
                              'rigid_group_id', 'x_m', 'y_m', 'z_m', 'valid', 'source'])
        self.events = (self.directory / 'object_states.jsonl').open('w')
        self.marker_pub = rospy.Publisher('/rekpiper/tracking/trajectory_markers', MarkerArray, queue_size=1)
        self.image_pub = rospy.Publisher('/rekpiper/tracking/trajectory_image', Image, queue_size=1)
        self.tf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf)
        rospy.Subscriber('/rs1/color/image_raw', Image, self.picture, queue_size=1)
        rospy.Subscriber('/rs1/color/camera_info', CameraInfo, self.camera_info, queue_size=1)
        rospy.Subscriber('/rekpiper/tracking/keypoints', Keypoint3DArray, self.update, queue_size=1)
        rospy.Subscriber('/rekpiper/objects/registry', TrackedObjectArray, self.objects, queue_size=1)

    def picture(self, message):
        with self.lock:
            self.image = self.bridge.imgmsg_to_cv2(message, 'bgr8').copy()

    def camera_info(self, message):
        self.info = message

    def objects(self, message):
        record = dict(stamp=message.header.stamp.to_sec(), objects=[dict(
            uuid=o.object_uuid, group=o.rigid_group_id, state=o.state,
            camera_status=list(o.camera_status), confidence=list(o.camera_confidence),
            status=o.status) for o in message.objects])
        with self.lock:
            self.events.write(json.dumps(record) + '\n')
            self.events.flush()

    def update(self, message):
        with self.lock:
            self.tracks = message
            markers = []
            for key in message.keypoints:
                point = np.array([key.position.x, key.position.y, key.position.z])
                previous = self.last.get(key.id)
                if key.valid and np.isfinite(point).all():
                    if previous is not None and np.linalg.norm(previous - point) <= .25:
                        self.segments[key.id].append((previous.copy(), point.copy()))
                    self.last[key.id] = point.copy()
                else:
                    self.last[key.id] = None
                color = ((53 * key.rigid_group_id + 70) % 220 + 35,
                         (97 * key.rigid_group_id + 30) % 220 + 35,
                         (137 * key.rigid_group_id + 20) % 220 + 35)
                self.colors[key.id] = color
                self.writer.writerow([message.header.stamp.to_sec(), rospy.Time.now().to_sec(),
                    key.id, key.name, key.rigid_group_id, *point, key.valid, key.source])
                for marker_type, offset in ((Marker.LINE_LIST, 0), (Marker.SPHERE, 1000),
                                            (Marker.TEXT_VIEW_FACING, 2000)):
                    marker = Marker(header=message.header, ns='M5_keypoints',
                                    id=key.id + offset, type=marker_type, action=Marker.ADD)
                    marker.pose.orientation.w = 1.0
                    marker.lifetime = rospy.Duration(1.0)
                    marker.color.r, marker.color.g, marker.color.b = [value / 255.0 for value in color[::-1]]
                    marker.color.a = 1.0 if key.valid else .35
                    marker.scale.x = .002 if offset == 0 else .008
                    marker.scale.y = marker.scale.z = .008
                    if offset == 0:
                        marker.points = [Point(*p) for segment in self.segments[key.id] for p in segment]
                    else:
                        marker.pose.position = key.position
                    if offset == 2000:
                        marker.scale.z = .02
                        marker.pose.position = Point(point[0], point[1], point[2] + .02)
                        marker.text = key.name + ('' if key.valid else ' LOST')
                    markers.append(marker)
            self.csv_file.flush()
            self.marker_pub.publish(MarkerArray(markers=markers))

    def render(self):
        with self.lock:
            if self.image is None or self.info is None:
                return
            image = self.image.copy()
            if self.tracks is None:
                cv2.putText(image, 'WAITING FOR TRACKING', (12, 25), 0, .6, (0, 200, 255), 2)
            else:
                try:
                    transform = self.tf.lookup_transform(self.info.header.frame_id, 'base_link', rospy.Time(0))
                    matrix = transform_to_matrix(transform.transform)
                except tf2_ros.TransformException:
                    return
                def pixel(p):
                    q = matrix[:3, :3] @ p + matrix[:3, 3]
                    if q[2] <= 0:
                        return None
                    return tuple(np.rint([self.info.K[0] * q[0] / q[2] + self.info.K[2],
                                          self.info.K[4] * q[1] / q[2] + self.info.K[5]]).astype(int))
                for key in self.tracks.keypoints:
                    color = self.colors[key.id]
                    for a, b in self.segments[key.id]:
                        pa, pb = pixel(a), pixel(b)
                        if pa is not None and pb is not None:
                            cv2.line(image, pa, pb, color, 1, cv2.LINE_AA)
                    point = pixel(np.array([key.position.x, key.position.y, key.position.z]))
                    if point is not None:
                        cv2.circle(image, point, 4, color if key.valid else (0, 0, 255), -1)
                        location = (point[0] + 6, point[1] - 8)
                        cv2.putText(image, key.name, location, 0, .55, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(image, key.name, location, 0, .55, color, 1, cv2.LINE_AA)
                age = (rospy.Time.now() - self.tracks.header.stamp).to_sec()
                valid = sum(k.valid for k in self.tracks.keypoints)
                label = '{}  {}/{} valid  age {:.2f}s'.format(
                    'LIVE' if age < 1.0 else 'STALE', valid, len(self.tracks.keypoints), age)
                cv2.putText(image, label, (12, 25), 0, .55, (0, 220, 0) if age < 1 else (0, 0, 255), 2)
            message = self.bridge.cv2_to_imgmsg(image, 'bgr8')
            message.header.stamp = rospy.Time.now()
            self.image_pub.publish(message)
            cv2.imwrite(str(self.directory / 'latest_tracking.png'), image)


if __name__ == '__main__':
    rospy.init_node('keypoint_trajectory')
    view = TrajectoryView()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        view.render()
        rate.sleep()
