#!/usr/bin/env python3
"""Display/replay native AnyGrasp poses and record actual RViz window images."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import rosbag
import rospy
from geometry_msgs.msg import Point, TransformStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
from tf2_msgs.msg import TFMessage
import yaml

from rekpiper_grasp.blue_cloud_comparison import (
    CASES, sha256, transform_points, gripper_segments, view_label, write_json)

TOPIC = '/rekpiper/blue_cloud_anygrasp_compare'
COLORS = [(1., .75, .05, 1.), (1., .28, .22, 1.), (.4, 1., .3, 1.),
          (.85, .35, 1., 1.), (.15, 1., 1., 1.)]


def point(xyz):
    return Point(*map(float, xyz))


def marker(identifier, kind, color, ns='native_grasps'):
    value = Marker(header=Header(frame_id='arm_base', stamp=rospy.Time.now()),
                   ns=ns, id=identifier, type=kind, action=Marker.ADD)
    value.pose.orientation.w = 1.
    value.color = ColorRGBA(*color)
    return value


def make_markers(case, report, top_k, center):
    values = [Marker(header=Header(frame_id='arm_base'), action=Marker.DELETEALL)]
    title = marker(0, Marker.TEXT_VIEW_FACING, (1., 1., 1., 1.), 'title')
    title.pose.position = point(center + np.array([0, 0, .13]))
    title.scale.z = .012
    title.text = view_label(case, report, top_k)
    if report.get('audit_kind') == 'piper_table_direction':
        title.scale.z = .010
        title.text = '{} | geometry {}/{} | contact-supported {} | no IK'.format(
            case.upper(), len(report['candidates']), report['raw_count'], report['contact_supported_count'])
        if not report['candidates']:
            title.text += '\nNO GEOMETRY PASS'
        if report.get('table'):
            table = report['table']
            n, offset = np.asarray(table['normal']), table['offset_m']
            corners = np.array([[-.14,-.14], [.14,-.14], [.14,.14], [-.14,.14], [-.14,-.14]])+center[:2]
            z = (-offset-corners @ n[:2])/n[2]
            plane = marker(0, Marker.LINE_STRIP, (.6,.8,.6,.6), 'measured_table')
            plane.scale.x = .0015
            plane.points = [point(p) for p in np.c_[corners,z]]
            values.append(plane)
    values.append(title)
    for index, row in enumerate(report.get('candidates', [])[:top_k]):
        color = COLORS[index % len(COLORS)]
        pose = np.asarray(row['pose_arm_base'])
        lines = marker(index*3, Marker.LINE_LIST, color)
        lines.scale.x = .003
        lines.points = [point(p) for p in gripper_segments(row)]
        if 'piper_pose_arm_base' in row:
            for part_index, (name, mesh) in enumerate(report['_gripper'].meshes(
                    np.asarray(row['piper_pose_arm_base']), row['preopen_width_m'])):
                body = marker(index*3+part_index, Marker.TRIANGLE_LIST,
                              (*color[:3], .45), 'piper_collision_mesh')
                body.scale.x = body.scale.y = body.scale.z = 1.
                body.points = [point(p) for p in mesh.triangles.reshape(-1,3)]
                values.append(body)
            # Keep native axes out of the grasp body; the full URDF mesh is shown.
            lines.points = []
        arrow = marker(index*3+1, Marker.ARROW, color)
        arrow.scale.x, arrow.scale.y, arrow.scale.z = .003, .007, .010
        ends = transform_points(np.array([[-.055, 0, 0], [-.025, 0, 0]]), pose)
        arrow.points = [point(p) for p in ends]
        label = marker(index*3+2, Marker.TEXT_VIEW_FACING, color)
        label.scale.z = .009
        # Legend placement shared across all groups, away from the grippers.
        label.pose.position = point(center + np.array([0, -.09, .105-index*.016]))
        label.text = '#{}  score {:.3f}  width {:.1f}mm'.format(index+1, row['score'], row['width_m']*1000)
        if 'piper_pose_arm_base' in row:
            label.scale.z = .008
            label.text = '#{}  {:.1f}mm clear  {}'.format(index+1, row['minimum_table_clearance_m']*1000,
                'CONTACT SUPPORTED' if row['contact_supported'] else 'CONTACT UNVERIFIED')
        values.extend([arrow, label] if 'piper_pose_arm_base' in row else [lines, arrow, label])
    return MarkerArray(markers=values)


def load_case(root, case):
    report = json.loads((root/case/'grasps.json').read_text())
    capture = json.loads((root/'capture.json').read_text())
    if report['snapshot_sha256'] != capture['snapshot_sha256'] or report['input_sha256'] != sha256(root/case/'input.npz'):
        raise ValueError('display_input_hash_mismatch')
    with np.load(root/case/'input.npz', allow_pickle=False) as data:
        points, mask = data['points_arm'], data['region_mask']
    return report, points, mask


def configuration(center):
    displays = [dict(Class='rviz/Grid', Name='arm_base XY grid', Enabled=True,
                     **{'Cell Size': .05, 'Plane Cell Count': 30, 'Alpha': .12})]
    for name, topic, color, alpha, size in (
            ('Scene background', 'background', '150; 155; 165', .20, .002),
            ('Color-selected blue cube', 'target', '35; 190; 255', 1., .002)):
        displays.append(dict(Class='rviz/PointCloud2', Name=name, Enabled=True, Topic=TOPIC+'/'+topic,
                             **{'Queue Size': 1, 'Color Transformer': 'FlatColor', 'Color': color,
                                'Position Transformer': 'XYZ', 'Style': 'Flat Squares', 'Size (m)': size,
                                'Alpha': alpha, 'Decay Time': 0, 'Use Fixed Frame': True}))
    displays.append(dict(Class='rviz/MarkerArray', Name='Native grasp candidates', Enabled=True,
                         **{'Marker Topic': TOPIC+'/markers', 'Queue Size': 1}))
    return {'Panels': [{'Class': 'rviz/Displays', 'Name': 'Displays'}],
            'Visualization Manager': {'Class': '', 'Global Options': {
                'Fixed Frame': 'arm_base', 'Background Color': '30; 34; 40', 'Frame Rate': 30},
                'Displays': displays, 'Tools': [{'Class': 'rviz/MoveCamera'}, {'Class': 'rviz/Select'}],
                'Views': {'Current': {'Class': 'rviz/Orbit', 'Distance': .48, 'Pitch': .8, 'Yaw': 3.2,
                                     'Focal Point': dict(zip(('X', 'Y', 'Z'), map(float, center+[0, 0, .035]))),
                                     'Target Frame': '<Fixed Frame>'}}},
            'Window Geometry': {'Width': 1450, 'Height': 1050, 'X': 60, 'Y': 40}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--case', default='fused')
    parser.add_argument('--cases', nargs='+', default=list(CASES))
    parser.add_argument('--top-k', type=int, choices=[1, 5], default=5)
    parser.add_argument('--record', action='store_true', help='Open RViz and save six screenshots/bags; stay live afterwards')
    args = parser.parse_args()
    if args.case not in args.cases:
        parser.error('--case must be included in --cases')
    root = args.output.resolve()
    rospy.init_node('blue_cloud_anygrasp_comparison_display')
    clouds = {n: rospy.Publisher(TOPIC+'/'+n, PointCloud2, queue_size=1, latch=True)
              for n in ('background', 'target')}
    pub = rospy.Publisher(TOPIC+'/markers', MarkerArray, queue_size=1, latch=True)
    # Private display-only TF topic: never publish an experiment transform to /tf.
    tf_pub = rospy.Publisher(TOPIC+'/tf_static', TFMessage, queue_size=1, latch=True)
    gripper = None
    if (root/'piper.urdf').exists():
        from rekpiper_grasp.directional_grasp import load_gripper
        gripper = load_gripper((root/'piper.urdf').read_text())
    _, fused, mask = load_case(root, 'fused')
    center = np.median(fused[mask], axis=0) if mask.any() else np.array([.4, 0, .1])
    config_path = root/'comparison.rviz'
    config_path.write_text(yaml.safe_dump(configuration(center), sort_keys=False))

    def publish(case, top_k, record=False):
        report, points, mask = load_case(root, case)
        if report.get('audit_kind') == 'piper_table_direction':
            report['_gripper'] = gripper
            report['table'] = json.loads((root/'table.json').read_text())
        header = Header(frame_id='arm_base', stamp=rospy.Time.now())
        origin = TransformStamped(header=Header(frame_id='comparison_reference', stamp=header.stamp),
                                  child_frame_id='arm_base')
        origin.transform.rotation.w = 1.
        messages = {TOPIC+'/tf_static': TFMessage(transforms=[origin]),
                    TOPIC+'/background': point_cloud2.create_cloud_xyz32(header, points[~mask]),
                    TOPIC+'/target': point_cloud2.create_cloud_xyz32(header, points[mask]),
                    TOPIC+'/markers': make_markers(case, report, top_k, center)}
        tf_pub.publish(messages[TOPIC+'/tf_static'])
        clouds['background'].publish(messages[TOPIC+'/background'])
        clouds['target'].publish(messages[TOPIC+'/target'])
        pub.publish(messages[TOPIC+'/markers'])
        if record:
            with rosbag.Bag(str(root/case/('top{}.bag'.format(top_k))), 'w') as bag:
                for topic, message in messages.items():
                    bag.write(topic, message, t=header.stamp,
                              connection_header={'latching': '1', 'topic': topic,
                                                 'type': message._type, 'md5sum': message._md5sum,
                                                 'message_definition': message._full_text,
                                                 'callerid': rospy.get_name()})
        return report

    qt = None
    if args.record:
        from PyQt5.QtWidgets import QApplication
        # Initialize only the C++ ROS client, with private display TF remaps.
        # This does not construct a MoveGroup or any hardware control client.
        from moveit_ros_planning_interface import _moveit_roscpp_initializer
        _moveit_roscpp_initializer.roscpp_init('blue_cloud_comparison_rviz',
            ['/tf:='+TOPIC+'/tf', '/tf_static:='+TOPIC+'/tf_static'])
        from rviz import bindings as rviz
        os.environ['DISABLE_ROS1_EOL_WARNINGS'] = '1'
        qt = QApplication(['blue_cloud_comparison_rviz',
                           '/tf:='+TOPIC+'/tf', '/tf_static:='+TOPIC+'/tf_static'])
        window = rviz.VisualizationFrame()
        window.setSplashPath('')
        window.initialize()
        config = rviz.Config()
        rviz.YamlConfigReader().readFile(config, str(config_path))
        window.load(config)
        window.setWindowTitle('Blue-cloud AnyGrasp comparison - RViz')
        window.resize(1450, 1050)
        window.show()
        view = window.getManager().getViewManager().getCurrent()
        expected = configuration(center)['Visualization Manager']['Views']['Current']

        def fixed_view():
            for name in ('Distance', 'Pitch', 'Yaw'):
                view.subProp(name).setValue(expected[name])
            for name in ('X', 'Y', 'Z'):
                view.subProp('Focal Point').subProp(name).setValue(expected['Focal Point'][name])

        def view_values():
            result = {name: float(view.subProp(name).getValue()) for name in ('Distance', 'Pitch', 'Yaw')}
            result['Focal Point'] = {name: float(view.subProp('Focal Point').subProp(name).getValue())
                                     for name in ('X', 'Y', 'Z')}
            return result

        publish(args.cases[0], 1)
        deadline = time.monotonic()+30
        while time.monotonic() < deadline and not rospy.is_shutdown():
            qt.processEvents()
            if pub.get_num_connections() and all(p.get_num_connections() for p in clouds.values()):
                break
            time.sleep(.05)
        else:
            raise RuntimeError('rviz_subscription_timeout')
        records = []
        for case in args.cases:
            for top_k in (1, 5):
                report = publish(case, top_k, record=True)
                deadline = time.monotonic()+2.
                while time.monotonic() < deadline:
                    qt.processEvents()
                    fixed_view()
                    time.sleep(.02)
                qt.processEvents()
                actual_view = view_values()
                np.testing.assert_allclose([actual_view[n] for n in ('Distance', 'Pitch', 'Yaw')],
                                           [expected[n] for n in ('Distance', 'Pitch', 'Yaw')], atol=1e-6)
                np.testing.assert_allclose(list(actual_view['Focal Point'].values()),
                                           list(expected['Focal Point'].values()), atol=1e-6)
                target = root/case/('top{}.png'.format(top_k))
                pixmap = qt.primaryScreen().grabWindow(int(window.winId()))
                if pixmap.isNull() or pixmap.width() < 800 or not pixmap.save(str(target)):
                    raise RuntimeError('rviz_screenshot_failed')
                records.append(dict(case=case, top_k=top_k, screenshot=str(target.relative_to(root)),
                                    screenshot_sha256=sha256(target), candidates=len(report['candidates']),
                                    actual_camera=actual_view,
                                    bag=str((root/case/('top{}.bag'.format(top_k))).relative_to(root))))
                print('RVIZ_RECORDED', case, top_k, str(target), flush=True)
        write_json(root/'rviz_records.json', dict(frame='arm_base', camera=expected,
                                                 records=records, window_id=int(window.winId()),
                                                 display_pid=os.getpid()))
    rospy.set_param('~case', args.case)
    rospy.set_param('~top_k', args.top_k)
    publish(args.case, args.top_k)
    previous = (args.case, args.top_k)
    while not rospy.is_shutdown():
        if qt is not None:
            qt.processEvents()
        selected = (rospy.get_param('~case'), int(rospy.get_param('~top_k')))
        if selected != previous and selected[0] in args.cases and selected[1] in (1, 5):
            publish(*selected)
            previous = selected
        rospy.sleep(.2)


if __name__ == '__main__':
    main()
