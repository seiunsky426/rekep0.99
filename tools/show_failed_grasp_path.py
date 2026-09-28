#!/usr/bin/env python3
"""Replay a saved failed path in RViz; publish visualization messages only."""
import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
from scipy.spatial.transform import Rotation

from rekpiper_grasp.directional_grasp import load_gripper
from rekpiper_planning.continuous_ik import validate_continuous_ik
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver
from rekpiper_execution.trajectory import JOINT_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source = json.loads(args.trace.read_text())
    candidate = json.loads(args.candidate.read_text())
    match = re.search(r'IK sequence failed at (\d+):', source.get('error', ''))
    if not match or source['base_frame'] != 'base_link':
        raise ValueError('expected_saved_base_link_ik_failure')
    failed = int(match.group(1))
    poses = np.asarray(source['dense_poses'], dtype=float)
    matrices = np.tile(np.eye(4), (len(poses), 1, 1))
    matrices[:, :3, 3] = poses[:, :3]
    matrices[:, :3, :3] = Rotation.from_quat(poses[:, 3:]).as_matrix()
    repo = Path(__file__).resolve().parents[1]
    xml = (repo/'src/piper_description/urdf/piper_description.urdf').read_text()
    ik = PiperURDFIKSolver.from_urdf_xml(xml, 'base_link', 'rekep_tcp', JOINT_NAMES,
                                       position_tolerance=.003, orientation_tolerance=.03)
    attempts = []
    report = validate_continuous_ik(ik, matrices, source['initial_joint_positions'], trace=attempts)
    if report['valid'] or report['failed_pose_index'] != failed:
        raise ValueError('saved_failure_did_not_reproduce')
    args.output.mkdir(parents=True, exist_ok=True)
    report['poses'] = report['poses'].tolist()
    report.update(source_trace=str(args.trace.resolve()), candidate=candidate['id'],
                  recorded_failure_index=failed, attempts=attempts,
                  full_scene_collision_checked=False, executable=False,
                  hardware_commands_sent=False)
    (args.output/'ik_replay.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.output/'path_review.npz', dense_poses=poses,
                        ik_prefix_poses=np.asarray(report['poses']), failed_pose_index=failed,
                        grasp_pose=candidate['grasp_pose'], pregrasp_pose=candidate['pregrasp_pose'])

    import rospy
    import rosbag
    from geometry_msgs.msg import Point
    from std_msgs.msg import Header, ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray
    rospy.init_node('failed_grasp_path_review', anonymous=True)
    topic = '/rekpiper/supervised/preview_markers'
    publisher = rospy.Publisher(topic, MarkerArray, queue_size=1, latch=True)
    markers = [Marker(action=Marker.DELETEALL)]
    green, red, grey = (0.1, 1., .25, 1.), (1., .1, .1, 1.), (.65, .65, .65, .85)
    gold, cyan = (1., .72, .05, .55), (.1, .8, 1., .25)

    def marker(name, kind, color, scale):
        value = Marker(header=Header(frame_id='base_link', stamp=rospy.Time.now()),
                       ns='failed_path_'+name, id=len(markers), type=kind, action=Marker.ADD)
        value.pose.orientation.w = 1.
        value.color = ColorRGBA(*color)
        value.scale.x, value.scale.y, value.scale.z = scale
        markers.append(value)
        return value

    def line(name, points, color, width=.004, dashed=False):
        points = np.asarray(points)
        if dashed:
            segments = []
            for a, b in zip(points[:-1], points[1:]):
                count = max(2, int(np.ceil(np.linalg.norm(b-a)/.006)))
                for i in range(0, count, 2):
                    segments.extend([a+(b-a)*i/count, a+(b-a)*min(i+1,count)/count])
            points = np.asarray(segments)
        value = marker(name, Marker.LINE_LIST if dashed else Marker.LINE_STRIP, color, (width, 0., 0.))
        value.points = [Point(*p) for p in points]

    def label(name, text, position, color, size=.013):
        value = marker(name, Marker.TEXT_VIEW_FACING, color, (0., 0., size))
        value.pose.position = Point(*position)
        value.text = text

    accepted = np.asarray(report['poses'])
    line('ik_prefix', accepted[:, :3, 3], green)
    line('failed_segment', [accepted[-1, :3, 3], poses[failed, :3]], red, .006)
    line('unchecked_remainder', poses[failed:, :3], grey, .003, True)
    dots = marker('dense_samples', Marker.SPHERE_LIST, grey, (.007,)*3)
    dots.points = [Point(*p) for p in poses[:, :3]]
    dots.colors = [ColorRGBA(*(green if i < failed else red if i == failed else grey)) for i in range(len(poses))]
    fail = marker('failure', Marker.SPHERE, red, (.018,)*3)
    fail.pose.position = Point(*poses[failed, :3])
    label('failed_label', 'IK FAIL #{} (sample {})'.format(failed,failed+1), poses[failed,:3]+[.02,0,.045],red)
    label('legend', 'GREEN: IK prefix only | RED: failed | GREY: unchecked\nGOLD: grasp | CYAN: pregrasp | NOT EXECUTABLE',
          poses[0,:3]+[.06,0,.12], (1.,1.,1.,1.), .012)
    label('last_ok', 'last IK pass #{}'.format(failed-1), accepted[-1,:3,3]+[-.05,0,.025],green,.011)

    gripper = load_gripper(xml)
    for name, pose, color in (
            ('grasp', np.asarray(candidate['grasp_pose']), gold),
            ('pregrasp', np.asarray(candidate['pregrasp_pose']), cyan),
            ('failed_orientation', matrices[failed], (1.,.1,.1,.2))):
        for part, mesh in gripper.meshes(pose, candidate['opening_m']):
            value = marker(name+'_'+part, Marker.TRIANGLE_LIST, color, (1.,1.,1.))
            value.points = [Point(*p) for p in mesh.triangles.reshape(-1,3)]
        arrow = marker(name+'_approach', Marker.ARROW, (*color[:3],1.), (.003,.007,.01))
        arrow.points = [Point(*(pose[:3,3]-.06*pose[:3,2])),Point(*pose[:3,3])]
    label('grasp_label', candidate['id']+' GRASP: contact unverified',
          np.asarray(candidate['grasp_pose'])[:3,3]+[.02,0,-.025], (*gold[:3],1.),.011)
    label('pregrasp_label', 'PREGRASP: stage 1 goal',
          np.asarray(candidate['pregrasp_pose'])[:3,3]+[.035,0,.015], (*cyan[:3],1.),.011)
    values = MarkerArray(markers=markers)
    with rosbag.Bag(str(args.output/'review_markers.bag'),'w') as bag:
        bag.write(topic,values,connection_header={
            'latching':'1', 'topic':topic, 'type':values._type,
            'md5sum':values._md5sum, 'message_definition':values._full_text,
            'callerid':rospy.get_name()})
    deadline = time.monotonic()+10
    while publisher.get_num_connections()==0 and time.monotonic()<deadline:
        rospy.sleep(.1)
    if not publisher.get_num_connections():
        raise RuntimeError('rviz_preview_markers_has_no_subscriber')
    publisher.publish(values)
    rospy.sleep(2.)
    print(json.dumps(dict(failed_index=failed, failed_position_m=poses[failed,:3].tolist(),
                         last_pass_position_m=accepted[-1,:3,3].tolist(),
                         prefix_samples=len(accepted), total_samples=len(poses),
                         markers=len(markers), subscribers=publisher.get_num_connections(),
                         output=str(args.output.resolve()))),flush=True)


if __name__ == '__main__':
    main()
