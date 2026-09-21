#!/usr/bin/env python3
"""Capture a new local perception snapshot; never publish execution messages."""
import argparse
from collections import deque
import io
import json
from pathlib import Path
import threading
import time
import zlib

import cv2
from cv_bridge import CvBridge
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
import tf2_ros
from tf.transformations import quaternion_matrix

from rekpiper_msgs.msg import SafeMappingStatus, SceneSnapshot
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args()
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=False)
    rospy.init_node('m8_shadow_scene_capture', anonymous=True, disable_signals=True)
    bridge = CvBridge()
    lock = threading.Lock()
    images, frames, counts, latest = {}, {}, {'rs1': 0, 'rs3': 0}, {}
    depths = {'rs1': deque(maxlen=4000), 'rs3': deque(maxlen=4000)}
    snapshots = deque(maxlen=4)
    feedback = deque(maxlen=2000)
    subs = []
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30))
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    def frame(name, rgb, depth, info):
        # SAM may finish long after capture. Keep compressed depth and the
        # corresponding feedback, not a latest-frame substitute for that stamp.
        packed = zlib.compress(depth.data, 1)
        packed_rgb = zlib.compress(rgb.data, 1) if name == 'rs3' else None
        with lock:
            frames[name] = (rgb, depth, info)
            depths[name].append((depth.header.stamp.to_sec(), packed, info,
                                 depth.height, depth.width, depth.encoding,
                                 latest.get('joints'), packed_rgb,
                                 rgb.header, rgb.encoding, rgb.step))
            counts[name] += 1

    def receive(name, msg):
        with lock:
            latest[name] = msg
            if name == 'joints':
                feedback.append(msg)

    for name in counts:
        topics = [message_filters.Subscriber('/' + name + suffix, kind)
                  for suffix, kind in [('/color/image_raw', Image),
                     ('/aligned_depth_to_color/image_raw', Image),
                     ('/color/camera_info', CameraInfo)]]
        sync = message_filters.ApproximateTimeSynchronizer(topics, 10, .025)
        sync.registerCallback(lambda *msgs, n=name: frame(n, *msgs))
        subs.append((topics, sync))
    for name, topic, kind in [
            ('joints', '/joint_states_single', JointState),
            ('safe_map', '/rekpiper/mapping/safe_status', SafeMappingStatus),
            ('status', '/rekpiper/perception/task_status', String)]:
        subs.append(rospy.Subscriber(topic, kind, lambda m, n=name: receive(n, m)))
    subs.append(rospy.Subscriber('/rekpiper/perception/scene_snapshot',
                                SceneSnapshot, snapshots.append))
    for name in ('source_image', 'candidate_image', 'scene_mask'):
        subs.append(rospy.Subscriber('/rekpiper/perception/' + name, Image,
                    lambda m, n=name: images.__setitem__(n, m)))
    publisher = rospy.Publisher('/rekpiper/perception/task_request', String, queue_size=1)
    report = {'motion_allowed': False, 'hardware_commands_sent': False,
              'full_task_accepted': False, 'capture_success': False}
    began = time.monotonic()
    try:
        deadline = began + 15
        while (publisher.get_num_connections() == 0 or len(frames) != 2
               or not feedback) and time.monotonic() < deadline:
            time.sleep(.05)
        if publisher.get_num_connections() != 1:
            raise RuntimeError('expected_one_perception_trigger_subscriber')
        if len(frames) != 2 or not feedback:
            raise RuntimeError('dual_camera_or_joint_feedback_missing')
        trigger = rospy.Time.now()
        report['trigger_stamp_s'] = trigger.to_sec()
        publisher.publish(String(data='将蓝色方块放在黄色圆盘上；重新识别，仅供无运动 Shadow 审核。'))
        print('FRESH_PERCEPTION_TRIGGERED', trigger.to_sec(), flush=True)
        selected = None
        deadline, next_print = time.monotonic() + 120, 0
        while time.monotonic() < deadline and not rospy.is_shutdown():
            for snapshot in reversed(list(snapshots)):
                if (snapshot.valid and snapshot.immutable_layout and snapshot.time_consistent
                        and snapshot.header.stamp >= trigger
                        and all(n in images and images[n].header.stamp == snapshot.header.stamp
                                for n in ('source_image', 'candidate_image', 'scene_mask'))):
                    selected = snapshot
                    break
            if selected is not None:
                break
            if time.monotonic() >= next_print:
                print('CAPTURE_STATUS', counts, latest.get('status'), flush=True)
                next_print = time.monotonic() + 5
            time.sleep(.05)
        if selected is None:
            raise RuntimeError('fresh_matching_snapshot_timeout_120s')
        stamp = selected.header.stamp.to_sec()
        buffer = io.BytesIO()
        selected.serialize(buffer)
        (output / 'scene_snapshot.rosmsg').write_bytes(buffer.getvalue())
        (output / 'scene_snapshot.yaml').write_text(str(selected))
        for name in ('source_image', 'candidate_image', 'scene_mask'):
            encoding = 'passthrough' if name == 'scene_mask' else 'bgr8'
            cv2.imwrite(str(output / (name + '.png')),
                        bridge.imgmsg_to_cv2(images[name], encoding))
        with lock:
            camera_frames = dict(frames)
            depth_cache = {n: list(values) for n, values in depths.items()}
            safe_map = latest.get('safe_map')
        packed_depth = min(depth_cache['rs1'], key=lambda item: abs(item[0] - stamp))
        depth_stamp, packed, info, height, width, encoding, near_joint = packed_depth[:7]
        if near_joint is None:
            raise RuntimeError('snapshot_joint_feedback_missing')
        joint_offset = abs(near_joint.header.stamp.to_sec() - stamp)
        if joint_offset > .15:
            raise RuntimeError('snapshot_joint_timestamp_offset_exceeds_0.15s')
        report['snapshot_id'] = selected.snapshot_id
        report['snapshot_stamp_s'] = stamp
        report['joint_stamp_offset_s'] = joint_offset
        report['joints'] = dict(zip(near_joint.name, near_joint.position))
        report['camera_frame_counts'] = counts.copy()
        depth = Image()
        depth.header.stamp = rospy.Time.from_sec(depth_stamp)
        depth.height, depth.width, depth.encoding = height, width, encoding
        depth.data = zlib.decompress(packed)
        depth.step = len(depth.data) // height
        offset = abs(depth_stamp-stamp)
        if offset > .05:
            raise RuntimeError('source_image_depth_offset_exceeds_0.05s')
        camera_frames['rs1'] = (images['source_image'], depth, info)
        report['rs1_source_depth_offset_s'] = offset
        def pair_span(item):
            stamps = (stamp, depth_stamp, item[0], item[8].stamp.to_sec())
            return max(stamps) - min(stamps)
        pair = min(depth_cache['rs3'], key=pair_span)
        pair_stamp, packed, pair_info, height, width, encoding = pair[:6]
        report['dual_camera_stamp_delta_s'] = abs(pair_stamp-stamp)
        report['maximum_rgb_depth_pair_skew_s'] = pair_span(pair)
        if pair_span(pair) > .025:
            raise RuntimeError('dual_camera_rgb_depth_timestamp_span_exceeds_0.025s')
        pair_depth = Image()
        pair_depth.header.stamp = rospy.Time.from_sec(pair_stamp)
        pair_depth.height, pair_depth.width, pair_depth.encoding = height, width, encoding
        pair_depth.data = zlib.decompress(packed)
        pair_depth.step = len(pair_depth.data)//height
        pair_rgb = Image()
        pair_rgb.header, pair_rgb.encoding, pair_rgb.step = pair[8:11]
        pair_rgb.height, pair_rgb.width = height, width
        pair_rgb.data = zlib.decompress(pair[7])
        camera_frames['rs3'] = (pair_rgb, pair_depth, pair_info)
        for name, (rgb, depth, info) in camera_frames.items():
            tf = tf_buffer.lookup_transform('base_link', info.header.frame_id,
                                            rgb.header.stamp, rospy.Duration(2)).transform
            q, t = tf.rotation, tf.translation
            matrix = quaternion_matrix([q.x, q.y, q.z, q.w])
            matrix[:3, 3] = [t.x, t.y, t.z]
            np.savez_compressed(output / (name + '_rgbd.npz'),
                rgb=bridge.imgmsg_to_cv2(rgb, 'rgb8'),
                depth=bridge.imgmsg_to_cv2(depth, 'passthrough'), K=info.K,
                depth_encoding=depth.encoding, base_from_camera=matrix,
                rgb_stamp_s=rgb.header.stamp.to_sec(), depth_stamp_s=depth.header.stamp.to_sec())
        xml = rospy.get_param('/robot_description')
        (output / 'robot_description.urdf').write_text(xml)
        names = ['joint{}'.format(i) for i in range(1, 7)]
        ik = PiperURDFIKSolver.from_urdf_xml(xml, 'base_link', 'rekep_tcp', names)
        q = np.array([report['joints'][n] for n in names])
        lower, upper = ik._lower, ik._upper
        report['joint_bounds_violations'] = [
            {'joint': n, 'position_rad': float(v), 'lower_rad': float(lo), 'upper_rad': float(hi)}
            for n, v, lo, hi in zip(names, q, lower, upper) if not lo <= v <= hi]
        report['fk_base_tcp'] = ik.forward(q).tolist()
        report['safe_map'] = None if safe_map is None else {
            'state': safe_map.state_name, 'reason': safe_map.reason,
            'planning_safe': safe_map.planning_safe,
            'map_query_allowed': safe_map.map_query_allowed,
            'generation_uuid': safe_map.map_generation_uuid}
        report['capture_success'] = True
    except Exception as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        report['elapsed_s'] = time.monotonic() - began
        (output / 'capture_report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report['capture_success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
