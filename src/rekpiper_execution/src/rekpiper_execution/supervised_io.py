"""ROS observation cache and display-only workbench publishers."""
from collections import deque
from copy import deepcopy
import threading
import time

import cv2
from cv_bridge import CvBridge
import message_filters
import numpy as np
import rospy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from urdf_parser_py.urdf import URDF
from rekpiper_planning.piper_collision_sampling import _origin_matrix
from rekpiper_planning.offline_observation import rgbd_points
from .supervised_geometry import depth_metres


NS = '/rekpiper/supervised'


class Observations:
    def __init__(self):
        self.lock = threading.RLock()
        self.frames = {name: deque(maxlen=15) for name in ('rs1', 'rs3')}
        self.bridge = CvBridge()
        self.syncs = []
        self.transforms = {}
        self.clouds = {n: rospy.Publisher(NS+'/'+n+'/points', PointCloud2, queue_size=1)
                       for n in self.frames}
        self.fused = rospy.Publisher(NS+'/fused_points', PointCloud2, queue_size=1)
        for camera in self.frames:
            subs = [message_filters.Subscriber('/'+camera+topic, typ, queue_size=2)
                    for topic, typ in (('/color/image_raw', Image),
                                      ('/aligned_depth_to_color/image_raw', Image),
                                      ('/color/camera_info', CameraInfo))]
            sync = message_filters.ApproximateTimeSynchronizer(subs, 5, .025)
            sync.registerCallback(lambda rgb, depth, info, name=camera: self.update(name, rgb, depth, info))
            self.syncs.append((subs, sync))

    def update(self, name, rgb, depth, info):
        try:
            frame = dict(rgb=self.bridge.imgmsg_to_cv2(rgb, 'rgb8').copy(),
                         depth=self.bridge.imgmsg_to_cv2(depth, 'passthrough').copy(),
                         K=np.asarray(info.K).reshape(3, 3), depth_encoding=depth.encoding,
                         rgb_stamp_s=rgb.header.stamp.to_sec(), depth_stamp_s=depth.header.stamp.to_sec(),
                         info_stamp_s=info.header.stamp.to_sec(), arrival=time.monotonic(),
                         optical_frame=info.header.frame_id)
            if frame['optical_frame'] != name+'_color_optical_frame':
                return
            with self.lock:
                self.frames[name].append(frame)
        except (ValueError, TypeError) as exc:
            rospy.logwarn_throttle(2., 'supervised RGB-D: %s', exc)

    def capture(self, maximum_age=.5):
        now = rospy.Time.now().to_sec()
        with self.lock:
            if not all(self.frames.values()) or set(self.transforms) != {'rs1', 'rs3'}:
                raise ValueError('dual_rgbd_or_experiment_transforms_missing')
            pairs = [(abs(a['rgb_stamp_s']-b['rgb_stamp_s']), a, b)
                     for a in self.frames['rs1'] for b in self.frames['rs3']
                     if 0 <= now-a['rgb_stamp_s'] <= maximum_age
                     and 0 <= now-b['rgb_stamp_s'] <= maximum_age]
            if not pairs:
                raise ValueError('dual_camera_frames_stale')
            _, a, b = min(pairs, key=lambda x: x[0])
            result = dict(rs1=deepcopy(a), rs3=deepcopy(b))
            stamps = [f[k] for f in result.values() for k in ('rgb_stamp_s', 'depth_stamp_s', 'info_stamp_s')]
            if max(stamps)-min(stamps) > .025 or min(stamps) <= 0:
                raise ValueError('dual_rgbd_time_skew_exceeds_25ms')
            for name, frame in result.items():
                frame['base_from_camera'] = self.transforms[name].copy()
            return result

    def publish(self, frames):
        rows = []
        xyz_fields = [PointField(name=n, offset=i*4, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate(('x', 'y', 'z'))]
        rgb_fields = xyz_fields + [PointField(name='rgb', offset=12,
                                             datatype=PointField.UINT32, count=1)]
        for name, f in frames.items():
            xyz, rgb, pixels = rgbd_points(f)
            packed = (rgb[:, 0].astype(np.uint32)<<16) | (rgb[:, 1].astype(np.uint32)<<8) | rgb[:, 2]
            # SAM and the DINO tracker require an organized, contiguous XYZ cloud.
            array = np.full(f['depth'].shape + (3,), np.nan, dtype=np.float32)
            array[pixels] = xyz
            header = Header(stamp=rospy.Time.from_sec(f['rgb_stamp_s']), frame_id='base_link')
            cloud = PointCloud2(header=header, height=array.shape[0], width=array.shape[1],
                                fields=xyz_fields, is_bigendian=False, point_step=12,
                                row_step=array.shape[1]*12, data=array.tobytes(), is_dense=False)
            self.clouds[name].publish(cloud)
            step = max(1, len(xyz)//40000)
            rows.extend((float(p[0]), float(p[1]), float(p[2]), int(c)) for p,c in zip(xyz[::step],packed[::step]))
        self.fused.publish(point_cloud2.create_cloud(
            Header(stamp=rospy.Time.from_sec(min(f['rgb_stamp_s'] for f in frames.values())), frame_id='base_link'),
            rgb_fields, rows))


class Display:
    def __init__(self, xml, ik):
        self.ik = ik
        self.robot = URDF.from_xml_string(xml)
        self.preview = rospy.Publisher(NS+'/preview_markers', MarkerArray, queue_size=1, latch=True)
        self.keys = rospy.Publisher(NS+'/keypoint_markers', MarkerArray, queue_size=1, latch=True)
        self.actual = rospy.Publisher(NS+'/actual_trace', MarkerArray, queue_size=1, latch=True)
        self.trace = deque(maxlen=20000)

    @staticmethod
    def marker(namespace, index, kind, color=(.2,.7,1.,1.)):
        return Marker(header=Header(frame_id='base_link'), ns=namespace, id=index, type=kind,
                      action=Marker.ADD, color=dict_to_color(color))

    def clear(self):
        clear = Marker(action=Marker.DELETEALL)
        self.preview.publish(MarkerArray(markers=[clear]))
        self.actual.publish(MarkerArray(markers=[clear]))
        self.trace.clear()

    def keypoints(self, keys):
        values = [Marker(action=Marker.DELETEALL)]
        for k in keys:
            text = self.marker('keypoints', k['id'], Marker.TEXT_VIEW_FACING, (1.,1.,.1,1.))
            text.pose.position = Point(*k['position']); text.pose.position.z += .015
            text.pose.orientation.w = 1.; text.scale.z = .015; text.text = 'K'+str(k['id'])
            values.append(text)
            point = self.marker('keypoint_dots', k['id'], Marker.SPHERE, (1.,.2,.2,1.))
            point.pose.position = Point(*k['position']); point.pose.orientation.w = 1.
            point.scale.x = point.scale.y = point.scale.z = .006; values.append(point)
        self.keys.publish(MarkerArray(markers=values))

    def ghosts(self, joints, opening, stage, detail):
        values = [Marker(action=Marker.DELETEALL)]
        line = self.marker('planned_tcp', 0, Marker.LINE_STRIP); line.pose.orientation.w = 1.
        line.scale.x = .003
        line.points = [Point(*self.ik.forward(q)[:3, 3]) for q in joints]
        values.append(line)
        for ghost, index in enumerate(np.unique(np.linspace(0, len(joints)-1, 12).astype(int))):
            transforms = self.ik.link_transforms(joints[index])
            for joint in self.robot.joints:
                if joint.type == 'prismatic' and joint.parent in transforms:
                    move = np.eye(4); amount = opening/2 if joint.limit.upper > 0 else -opening/2
                    move[:3, 3] = np.asarray(joint.axis)*amount
                    transforms[joint.child] = transforms[joint.parent] @ _origin_matrix(joint.origin) @ move
            for link_index, link in enumerate(self.robot.links):
                if link.name not in transforms: continue
                for mesh_index, collision in enumerate(link.collisions):
                    if not hasattr(collision.geometry, 'filename'): continue
                    m = self.marker('ghost_'+str(ghost), link_index*10+mesh_index, Marker.MESH_RESOURCE,
                                    (.2,.65,1.,.18 if ghost < 11 else .65))
                    matrix = transforms[link.name] @ _origin_matrix(collision.origin)
                    m.pose.position = Point(*matrix[:3, 3]); q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
                    m.pose.orientation.x,m.pose.orientation.y,m.pose.orientation.z,m.pose.orientation.w = q
                    m.mesh_resource = collision.geometry.filename
                    scale = collision.geometry.scale or [1.,1.,1.]
                    m.scale.x,m.scale.y,m.scale.z = scale
                    values.append(m)
        for n, key in enumerate(('pregrasp_pose','grasp_pose','goal_matrix')):
            pose = np.asarray(detail[key])
            for axis in range(3):
                color = [0.,0.,0.,1.]; color[axis] = 1.
                m = self.marker(key, axis, Marker.ARROW, color)
                m.points = [Point(*pose[:3,3]),Point(*(pose[:3,3]+.06*pose[:3,axis]))]
                m.scale.x=.004; m.scale.y=.008; m.scale.z=.01; values.append(m)
        self.preview.publish(MarkerArray(markers=values))

    def feedback(self, joints):
        p = self.ik.forward(joints)[:3, 3]
        if not self.trace or np.linalg.norm(p-self.trace[-1]) > .001:
            self.trace.append(p)
        line = self.marker('actual_tcp', 0, Marker.LINE_STRIP, (1.,.45,.1,1.))
        line.pose.orientation.w = 1.; line.scale.x = .004
        line.points = [Point(*p) for p in self.trace]
        self.actual.publish(MarkerArray(markers=[line]))


def dict_to_color(values):
    from std_msgs.msg import ColorRGBA
    return ColorRGBA(*values)
