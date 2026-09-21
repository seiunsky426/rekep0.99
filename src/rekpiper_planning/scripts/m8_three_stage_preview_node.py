#!/usr/bin/env python3
"""Display archived RGB clouds and three-stage diagnostics on private topics."""
import json
from pathlib import Path

import numpy as np
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState, PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


def marker(kind, namespace, index, color):
    value = Marker()
    value.header = Header(frame_id='base_link', stamp=rospy.Time.now())
    value.ns, value.id, value.type = namespace, index, kind
    value.action = Marker.ADD
    value.pose.orientation.w = 1.
    value.color.r, value.color.g, value.color.b = color
    value.color.a = 1.
    return value


def main():
    rospy.init_node('three_stage_preview')
    root = Path(rospy.get_param('~evidence_directory')).resolve()
    planning = root/rospy.get_param('~planning_directory', 'planning')
    binding = json.loads((root/'scene_binding.json').read_text())
    report = json.loads((planning/'report.json').read_text())
    if binding['execution_authorized'] or report['motion_allowed']:
        raise ValueError('display_requires_offline_artifacts')
    scene = np.load(root/'scene_rgb.npz',allow_pickle=False)
    display = np.load(root/'static_rgb.npz',allow_pickle=False)
    fields = [PointField(n,i*4,PointField.FLOAT32,1) for i,n in enumerate(('x','y','z'))]
    fields.append(PointField('rgb',12,PointField.UINT32,1))
    rgb = display['rgb'].astype(np.uint32)
    packed = (rgb[:,0]<<16) | (rgb[:,1]<<8) | rgb[:,2]
    step = max(1,int(np.ceil(len(rgb)/100000)))
    rows = [(float(p[0]),float(p[1]),float(p[2]),int(c))
            for p,c in zip(display['points_base'][::step],packed[::step])]
    cloud = point_cloud2.create_cloud(Header(frame_id='base_link',stamp=rospy.Time.now()),fields,rows)
    cloud_pub = rospy.Publisher('~rgb_points',PointCloud2,queue_size=1,latch=True)
    marker_pub = rospy.Publisher('~markers',MarkerArray,queue_size=1,latch=True)
    joint_pub = rospy.Publisher('~joint_states',JointState,queue_size=1)
    map_pub = rospy.Publisher('~occupied_voxels',PointCloud2,queue_size=1,latch=True)
    grid = np.load(root/'map_before_grasp.npz',allow_pickle=False)
    indices = np.argwhere(grid['occupied'])
    spacing = (grid['bounds_max']-grid['bounds_min'])/(np.array(grid['occupied'].shape)-1)
    occupied_points = grid['bounds_min']+indices*spacing
    map_cloud = point_cloud2.create_cloud_xyz32(Header(frame_id='base_link'),occupied_points)
    colors = [(.15,.65,1.),(1.,.65,.1),(.8,.25,1.)]
    names = ['1 GRASP','2 TRANSPORT (ASSUMED HELD)','3 PLACE (ASSUMED HELD)']
    values = []
    statuses = []
    for row in report['stages']:
        stage = int(row['stage'])
        path = planning/('stage{}_cartesian_candidate.npz'.format(stage))
        if not path.is_file():
            statuses.append('{}: no path output'.format(stage)); continue
        data = np.load(path,allow_pickle=False)
        if bool(data['motion_allowed']): raise ValueError('motion_enabled_path')
        matrices = data['matrices']
        if matrices.ndim != 3 or matrices.shape[1:] != (4,4) or not np.isfinite(matrices).all():
            raise ValueError('invalid_display_path')
        color = colors[stage-1]
        line = marker(Marker.LINE_STRIP,'stage_paths',stage,color)
        line.scale.x = .004
        line.points = [Point(*p) for p in matrices[:,:3,3]]
        values.append(line)
        status = ('CHECKED PREVIEW' if row['qualified'] else
                  'REJECTED CANDIDATE' if bool(data['solver_returned']) else 'SEED ONLY; SOLVER FAILED')
        if (not row['qualified'] and report.get('numerical_complete', False)
                and row['path'].get('debug', {}).get('optimizer_success', False)):
            status = 'NUMERICALLY CONVERGED; CHECKS FAILED'
        if not row['qualified'] and row['path'].get('unchanged_initialization', False):
            status = 'UNCHANGED INITIAL SEED; REJECTED'
        statuses.append('{}: {}'.format(stage,status))
        text = marker(Marker.TEXT_VIEW_FACING,'stage_labels',stage,color)
        text.pose.position = Point(*(matrices[-1,:3,3]+[0,0,.045+.035*(stage-1)]))
        text.scale.z = .021; text.text = names[stage-1]+'\n'+status
        values.append(text)
        axes = marker(Marker.ARROW,'stage_endpoints',stage,color)
        axes.scale.x=.006; axes.scale.y=.013; axes.scale.z=.020
        endpoint=matrices[-1]
        axes.points=[Point(*endpoint[:3,3]),Point(*(endpoint[:3,3]+.05*endpoint[:3,2]))]
        values.append(axes)
    for index,label in [(9,'K9 BLUE CUBE'),(10,'K10 YELLOW DISK')]:
        text=marker(Marker.TEXT_VIEW_FACING,'objects',index,(1.,1.,1.))
        text.pose.position=Point(*(scene['keypoints_base'][index]+[0,0,.025]))
        text.scale.z=.02; text.text=label;values.append(text)
    legend=marker(Marker.TEXT_VIEW_FACING,'assumptions',0,(1.,.85,.2))
    legend.pose.position=Point(.10,-.34,.58);legend.scale.z=.024
    legend.text=('OFFLINE SNAPSHOT | ZERO START ASSUMED\n'
                 'RIGID GRASP ASSUMED AFTER STAGE 1 | NO MOTION\n'+'\n'.join(statuses))
    values.append(legend)
    # Keep the robot at the declared zero start. A Cartesian candidate that
    # fails IK is never used to animate invented robot joint positions.
    rate=rospy.Rate(5)
    while not rospy.is_shutdown():
        stamp=rospy.Time.now()
        cloud.header.stamp=stamp
        map_cloud.header.stamp=stamp
        for value in values:value.header.stamp=stamp
        cloud_pub.publish(cloud);map_pub.publish(map_cloud);marker_pub.publish(MarkerArray(markers=values))
        joints=JointState(header=Header(stamp=stamp),name=['joint'+str(i) for i in range(1,9)],
                          position=[0.]*6+[.035,-.035])
        joint_pub.publish(joints);rate.sleep()


if __name__=='__main__':
    main()
