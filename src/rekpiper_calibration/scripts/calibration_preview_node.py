#!/usr/bin/env python3
"""Publish an isolated RViz robot and candidate path; never connect to Piper."""

import hashlib
from pathlib import Path

import rospy
import numpy as np
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from rekpiper_calibration.trajectory_preview import (
    JOINT_NAMES, playback_samples, validate_preview)
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver
from rekpiper_calibration.workspace_bounds import workspace_limits
from rekpiper_calibration.checkerboard_calibration import matrix_to_quaternion_xyzw


def main():
    rospy.init_node("calibration_preview")
    plan_path = Path(rospy.get_param("~plan"))
    plan = yaml.safe_load(plan_path.read_text())
    xml = rospy.get_param("robot_description")
    if hashlib.sha256(xml.encode()).hexdigest() != plan["urdf_sha256"]:
        raise ValueError("URDF differs from the planned model")
    solver = PiperURDFIKSolver.from_urdf_xml(xml, "base_link", "link6", JOINT_NAMES)
    joints = validate_preview(plan, solver)
    samples = list(playback_samples(joints))
    # Private outputs are never named /joint_states_single or /joint_ctrl_single.
    joint_pub = rospy.Publisher("~joint_states", JointState, queue_size=1)
    markers_pub = rospy.Publisher("~markers", MarkerArray, queue_size=1, latch=True)
    status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
    markers = MarkerArray()
    workspace_path = rospy.get_param("~workspace", "")
    if workspace_path:
        workspace = yaml.safe_load(Path(workspace_path).read_text())
        lower, upper = workspace_limits(workspace)
        volume = Marker()
        volume.header.frame_id = "base_link"
        volume.ns, volume.id = "user_workspace_unverified", 0
        volume.type, volume.action = Marker.CUBE, Marker.ADD
        volume.pose.orientation.w = 1.0
        volume.pose.position = Point(*((lower+upper)/2))
        volume.scale.x, volume.scale.y, volume.scale.z = upper-lower
        volume.color.g, volume.color.b, volume.color.a = 0.8, 1.0, 0.08
        markers.markers.append(volume)
    geometry_report = rospy.get_param("~geometry_report", "")
    if geometry_report:
        if not workspace_path:
            raise ValueError("workspace required for board geometry display")
        report = yaml.safe_load(Path(geometry_report).read_text())
        if (report["plan_sha256"] != hashlib.sha256(plan_path.read_bytes()).hexdigest()
                or report["workspace_sha256"] != hashlib.sha256(Path(workspace_path).read_bytes()).hexdigest()):
            raise ValueError("geometry report does not match plan/workspace")
        transform = np.asarray(report["link6_T_marker_candidate_mean"], dtype=float)
        center = np.asarray(workspace["board_and_mount"]["center_in_marker_m"])
        plate = Marker()
        plate.header.frame_id = "link6"
        plate.ns, plate.id = "board_candidate_from_old_calibration", 0
        plate.type, plate.action = Marker.CUBE, Marker.ADD
        plate.pose.position = Point(*(transform[:3, :3] @ center + transform[:3, 3]))
        quaternion = matrix_to_quaternion_xyzw(transform)
        (plate.pose.orientation.x, plate.pose.orientation.y,
         plate.pose.orientation.z, plate.pose.orientation.w) = quaternion
        plate.scale.x, plate.scale.y, plate.scale.z = workspace["board_and_mount"]["full_dimensions_m"]
        plate.color.r = plate.color.g = plate.color.b = 0.9
        plate.color.a = 0.8
        markers.markers.append(plate)
        for axis in range(3):
            arrow = Marker()
            arrow.header.frame_id = "link6"
            arrow.ns, arrow.id = "link6_axes", axis
            arrow.type, arrow.action = Marker.ARROW, Marker.ADD
            arrow.pose.orientation.w = 1.0
            arrow.points = [Point(0, 0, 0), Point(*(np.eye(3)[axis]*0.07))]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.003, 0.007, 0.010
            arrow.color.r, arrow.color.g, arrow.color.b = np.eye(3)[axis]
            arrow.color.a = 1.0
            markers.markers.append(arrow)
    path = Marker()
    path.header.frame_id = "base_link"
    path.ns, path.id = "candidate_path", 0
    path.type, path.action = Marker.LINE_STRIP, Marker.ADD
    path.pose.orientation.w = 1.0
    path.scale.x = 0.003
    path.color.r, path.color.g, path.color.a = 1.0, 0.6, 1.0
    for _, _, q in samples:
        xyz = solver.forward(q)[:3, 3]
        path.points.append(Point(*xyz))
    markers.markers.append(path)
    for index, q in enumerate(joints):
        label = Marker()
        label.header.frame_id = "base_link"
        label.ns, label.id = "pose_numbers", index
        label.type, label.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        xyz = solver.forward(q)[:3, 3]
        label.pose.position = Point(*xyz)
        label.pose.position.z += 0.015
        label.pose.orientation.w = 1.0
        label.scale.z = 0.018
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = plan["poses"][index].get("label", str(index+1))
        markers.markers.append(label)
    markers_pub.publish(markers)
    rate = rospy.Rate(20)
    for _, index, q in samples:
        if rospy.is_shutdown():
            return
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = JOINT_NAMES + ["joint7", "joint8"]
        message.position = list(q) + [0.0, 0.0]
        joint_pub.publish(message)
        status_pub.publish(String(data="PREVIEW_ONLY pose {}/20; no hardware commands".format(index+1)))
        rate.sleep()
    status_pub.publish(String(data="PREVIEW_COMPLETE; collision and visibility NOT verified"))
    while not rospy.is_shutdown():
        message.header.stamp = rospy.Time.now()
        joint_pub.publish(message)
        rate.sleep()


if __name__ == "__main__":
    main()
