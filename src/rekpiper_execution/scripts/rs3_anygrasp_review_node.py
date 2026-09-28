#!/usr/bin/env python3
"""RViz-only display of one frozen RS3 AnyGrasp diagnostic pose."""
import json
from pathlib import Path

import numpy as np
import rospy
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def color(red, green, blue, alpha=1.):
    return ColorRGBA(red, green, blue, alpha)


def marker(namespace, index, kind, rgba):
    item = Marker()
    item.header.frame_id = 'base_link'
    item.header.stamp = rospy.Time.now()
    item.ns, item.id, item.type, item.action = namespace, index, kind, Marker.ADD
    item.pose.orientation.w = 1.
    item.color = rgba
    return item


def point(values):
    return Point(*[float(value) for value in values])


def build_markers(report):
    markers = [Marker(action=Marker.DELETEALL)]
    row = next((item for item in report['candidates'] if item['id'] == 'rs3-006'), None)
    if row is None or 'grasp_pose' not in row:
        return MarkerArray(markers=markers)
    grasp = np.asarray(row['grasp_pose'], dtype=float)
    pre = np.asarray(row['pregrasp_pose'], dtype=float)
    tint = color(1., .65, .05)
    approach = marker('rs3_anygrasp_approach', 0, Marker.ARROW, tint)
    approach.points = [point(pre[:3, 3]), point(grasp[:3, 3])]
    approach.scale = Vector3(.003, .006, .009)
    markers.append(approach)
    dot = marker('rs3_anygrasp_tcp', 0, Marker.SPHERE, tint)
    dot.pose.position = point(grasp[:3, 3])
    dot.scale = Vector3(.012, .012, .012)
    markers.append(dot)
    label = marker('rs3_anygrasp_label', 0, Marker.TEXT_VIEW_FACING, tint)
    label.pose.position = point(grasp[:3, 3] + [0., 0., .045])
    label.scale.z = .018
    label.text = 'RS3-006: IK reachable, contact failed'
    markers.append(label)
    for axis, shade in enumerate((color(1., .1, .1), color(.1, 1., .1), color(.1, .5, 1.))):
        item = marker('rs3_anygrasp_axes', axis, Marker.ARROW, shade)
        item.points = [point(grasp[:3, 3]), point(grasp[:3, 3] + .025 * grasp[:3, axis])]
        item.scale = Vector3(.002, .004, .006)
        markers.append(item)
    return MarkerArray(markers=markers)


if __name__ == '__main__':
    rospy.init_node('rs3_anygrasp_review')
    report = json.loads(Path(rospy.get_param('~report')).read_text())
    publisher = rospy.Publisher('/rekpiper/supervised/blue_review/anygrasp_markers',
                                MarkerArray, queue_size=1, latch=True)
    publisher.publish(build_markers(report))
    rospy.loginfo('Published frozen RS3 AnyGrasp diagnostic markers; no grasp passed')
    rospy.spin()
