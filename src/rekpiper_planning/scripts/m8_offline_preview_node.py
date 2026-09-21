#!/usr/bin/env python3
"""Display-only M8 endpoint and rejected-path playback; never commands Piper."""

from pathlib import Path
import hashlib
import json

import numpy as np
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from rekpiper_planning.ik_path_diagnostics import COLORS


JOINT_NAMES = ["joint{}".format(index) for index in range(1, 7)]
ENDPOINT_COLORS = [
    (0.7, 0.7, 0.7),
    (1.0, 0.5, 0.0),
    (0.1, 0.4, 1.0),
    (1.0, 0.85, 0.0),
    (0.1, 0.9, 0.2),
]


def _marker(marker_type, namespace, marker_id, color, scale):
    value = Marker()
    value.header.frame_id = "base_link"
    value.header.stamp = rospy.Time.now()
    value.ns = namespace
    value.id = marker_id
    value.type = marker_type
    value.action = Marker.ADD
    value.pose.orientation.w = 1.0
    value.color.r, value.color.g, value.color.b = color
    value.color.a = 1.0
    value.scale.x, value.scale.y, value.scale.z = scale
    return value


def _line(namespace, marker_id, points, color, width):
    value = _marker(Marker.LINE_STRIP, namespace, marker_id, color,
                    (width, 0.0, 0.0))
    value.points = [Point(*point[:3]) for point in points]
    return value


def _gripper_outline(pose, width_m):
    """Schematic U: Piper +Z approach, Y closing; 4 cm illustrative fingers.

    Tip separation is the predicted opening, not an audited collision mesh.
    """
    pose = np.asarray(pose, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.isfinite(width_m) or width_m <= 0):
        raise ValueError('invalid grasp glyph pose or width')
    half = width_m / 2
    local = np.array([[0., -half, 0.], [0., -half, -.04],
                      [0., half, -.04], [0., half, 0.]])
    return local @ pose[:3, :3].T + pose[:3, 3]


def _pose_markers(poses, labels):
    markers = []
    for index, (pose, label) in enumerate(zip(poses, labels)):
        color = ENDPOINT_COLORS[index]
        arrow = _marker(Marker.ARROW, "endpoint_axes", index, color,
                        (0.08, 0.012, 0.018))
        arrow.pose.position = Point(*pose[:3])
        (arrow.pose.orientation.x, arrow.pose.orientation.y,
         arrow.pose.orientation.z, arrow.pose.orientation.w) = pose[3:]
        markers.append(arrow)

        text = _marker(Marker.TEXT_VIEW_FACING, "endpoint_labels", index,
                       color, (0.0, 0.0, 0.025))
        text.pose.position = Point(*pose[:3])
        text.pose.position.z += 0.04
        text.text = str(label)
        markers.append(text)
    return markers


def _keypoint_markers(keypoints):
    markers = []
    values = [
        ("K8 yellow disk anchor", (1.0, 0.9, 0.0)),
        ("K10 blue cube", (0.1, 0.35, 1.0)),
        ("K6 excluded", (0.1, 0.9, 0.2)),
    ]
    for index, (point, (label, color)) in enumerate(zip(keypoints, values)):
        sphere = _marker(Marker.SPHERE, "semantic_keypoints", index, color,
                         (0.025, 0.025, 0.025))
        sphere.pose.position = Point(*point)
        markers.append(sphere)
        text = _marker(Marker.TEXT_VIEW_FACING, "semantic_labels", index,
                       color, (0.0, 0.0, 0.021))
        text.pose.position = Point(*point)
        text.pose.position.z += 0.025
        text.text = label
        markers.append(text)
    return markers


def _validate(data):
    endpoints = np.asarray(data["endpoint_poses7"], dtype=float)
    joints = np.asarray(data["endpoint_joints_rad"], dtype=float)
    candidate = np.asarray(data["candidate_path_poses7"], dtype=float)
    keypoints = np.asarray(data["semantic_keypoints"], dtype=float)
    if (bool(np.asarray(data["motion_allowed"]).item())
            or endpoints.shape != (5, 7) or joints.shape != (5, 6)
            or candidate.ndim != 2 or candidate.shape[1] != 7 or len(candidate) < 2
            or keypoints.shape != (3, 3)
            or not all(np.all(np.isfinite(value)) for value in (
                endpoints, joints, candidate, keypoints))):
        raise ValueError("invalid PREVIEW_ONLY M8 data")
    return endpoints, joints, candidate, keypoints


def _candidate_only(data):
    """Display new replay artifacts without implying environment acceptance."""
    joints=np.asarray(data['joint_path'],dtype=float)
    matrices=np.asarray(data['cartesian_matrices'],dtype=float)
    if (bool(np.asarray(data['motion_allowed']).item()) or joints.ndim != 2
            or joints.shape[1] != 6 or len(joints)<2
            or matrices.shape != (len(joints),4,4)
            or not np.all(np.isfinite(joints)) or not np.all(np.isfinite(matrices))):
        raise ValueError('invalid candidate-only preview')
    marker_pub=rospy.Publisher('~markers',MarkerArray,queue_size=1,latch=True)
    joint_pub=rospy.Publisher('~joint_states',JointState,queue_size=1)
    status_pub=rospy.Publisher('~status',String,queue_size=1,latch=True)
    legend=_marker(Marker.TEXT_VIEW_FACING,'legend',0,(1.,.6,.1),(0.,0.,.025))
    legend.pose.position=Point(.3,-.4,.5)
    legend.text='DIAGNOSTIC ONLY | self-collision checked | environment NOT accepted'
    marker_pub.publish(MarkerArray(markers=[
        _line('diagnostic_joint_candidate',0,matrices[:,:3,3],(1.,.6,.1),.005),legend]))
    status_pub.publish(String(data=legend.text))
    rate=rospy.Rate(20); index=0
    while not rospy.is_shutdown():
        msg=JointState(); msg.header.stamp=rospy.Time.now()
        msg.name=JOINT_NAMES+['joint7','joint8']
        msg.position=joints[index].tolist()+[.035,-.035]
        joint_pub.publish(msg); index=(index+1)%len(joints); rate.sleep()


def _target_snapshot(data):
    """Live feedback + frozen grasp poses on the live fusion cloud, never playback."""
    target = np.asarray(data['target_points_base'], dtype=float)
    grasps = np.asarray(data['grasp_matrices'], dtype=float)
    pregrasps = np.asarray(data['pregrasp_matrices'], dtype=float)
    widths = np.asarray(data['widths_m'], dtype=float)
    if (bool(np.asarray(data['motion_allowed']).item())
            or target.ndim != 2 or target.shape[1] != 3
            or grasps.ndim != 3 or grasps.shape[1:] != (4, 4)
            or pregrasps.shape != grasps.shape
            or widths.shape != (len(grasps),) or np.any(widths <= 0)
            or not all(np.all(np.isfinite(v)) for v in (target, grasps, pregrasps, widths))):
        raise ValueError('invalid snapshot-only grasp preview')
    marker_pub = rospy.Publisher('~markers', MarkerArray, queue_size=1, latch=True)
    joint_pub = rospy.Publisher('~joint_states', JointState, queue_size=1)
    status_pub = rospy.Publisher('~status', String, queue_size=1, latch=True)
    last_feedback = [rospy.Time(0)]
    def feedback(value):
        positions = dict(zip(value.name, value.position))
        if not all(n in positions for n in JOINT_NAMES) or 'gripper' not in positions:
            return
        q = [positions[n] for n in JOINT_NAMES]
        if not np.isfinite(q + [positions['gripper']]).all():
            return
        # Display measured feedback, including model-boundary disagreement.
        # This topic is private and connects only to the isolated TF publisher.
        opening = float(np.clip(positions['gripper'], 0., .070))
        message = JointState()
        message.header = value.header
        message.name = JOINT_NAMES + ['joint7', 'joint8']
        message.position = q + [opening/2, -opening/2]
        last_feedback[0] = value.header.stamp
        joint_pub.publish(message)
    subscriber = rospy.Subscriber('/joint_states_single', JointState, feedback, queue_size=1)
    markers = []
    cloud = _marker(Marker.POINTS, 'frozen_target_mask', 0, (.1, .8, 1.), (.004, .004, .004))
    cloud.points = [Point(*p) for p in target[::max(1, len(target)//1500)]]
    markers.append(cloud)
    target_label = _marker(Marker.TEXT_VIEW_FACING, 'target_label', 0,
                           (.2, 1., 1.), (0., 0., .025))
    target_label.pose.position = Point(*(np.median(target, axis=0) + [0., 0., .10]))
    target_label.text = 'BLUE CUBE | frozen SAM instance'
    markers.append(target_label)
    for i, (grasp, pregrasp, width) in enumerate(zip(grasps, pregrasps, widths)):
        # Same namespaces/IDs replace the old arrows; no connecting trajectory.
        for kind, pose, color in [('grasp', grasp, (1., .5, .05)),
                                  ('pregrasp', pregrasp, (.7, .7, .7))]:
            markers.append(_line(kind, i, _gripper_outline(pose, width), color, .002))
    legend = _marker(Marker.TEXT_VIEW_FACING, 'legend', 0, (1., .4, .1), (0., 0., .022))
    legend.pose.position = Point(.35, -.35, .35)
    prefix = ('NO MOTION | LIVE CLOUD + MEASURED ROBOT\n'
              '{} FROZEN GRASPS (orange) / PREGRASPS (gray)\n'
              'U: predicted opening / schematic fingers | NO AUDITED TRAJECTORY\n').format(len(grasps))
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        age = (rospy.Time.now()-last_feedback[0]).to_sec()
        legend.text = prefix + ('Feedback LIVE' if 0 <= age <= .15 else 'Feedback STALE')
        for marker in markers + [legend]:
            marker.header.stamp = rospy.Time.now()
        marker_pub.publish(MarkerArray(markers=markers + [legend]))
        status_pub.publish(String(data=legend.text))
        rate.sleep()


def _continuous_ik_markers(candidate, report):
    """Only committed prefix points are green; disconnected points stay gray."""
    markers = [_line('rejected_pathsolver_path', 0, candidate, (1., .05, .05), .003)]
    for row in report['waypoints']:
        index = row['pose_index']
        marker = _marker(Marker.SPHERE, 'continuous_ik_waypoints', index,
                         COLORS[row['display_status']], (.008, .008, .008))
        marker.pose.position = Point(*candidate[index, :3])
        markers.append(marker)
        if row['risk_flag']:
            halo = _marker(Marker.SPHERE, 'limit_singularity_warning', index,
                           COLORS['near_limit_or_singularity'], (.014, .014, .014))
            halo.color.a = .3
            halo.pose.position = marker.pose.position
            markers.append(halo)
    failed = report['summary']['first_failed_pose_index']
    if failed is not None:
        row = report['waypoints'][failed]
        attempt = report['attempts'][row['primary_attempt_id']]
        label = _marker(Marker.TEXT_VIEW_FACING, 'failure_detail', failed,
                        COLORS[row['display_status']], (0., 0., .013))
        label.pose.position = Point(*(candidate[failed, :3]+[0., 0., .07]))
        label.text = ('#{} first failed original waypoint\n{} | max dq={:.4f} rad\n'
                      'pos={:.3f} mm rot={:.4f} rad\nq={}\ndq={}').format(
                          failed, attempt['status'], attempt['maximum_joint_step_rad'],
                          attempt['position_error']*1000, attempt['rotation_error'],
                          np.round(attempt['joint_positions'], 3).tolist(),
                          np.round(attempt['joint_delta_rad'], 3).tolist())
        markers.append(label)
        axis = _marker(Marker.ARROW, 'failed_target_orientation', failed, (1., 1., 1.), (.06, .006, .01))
        axis.pose.position = Point(*candidate[failed, :3])
        (axis.pose.orientation.x, axis.pose.orientation.y,
         axis.pose.orientation.z, axis.pose.orientation.w) = candidate[failed, 3:]
        markers.append(axis)
    legend = _marker(Marker.TEXT_VIEW_FACING, 'legend', 0, (1., 1., 1.), (0., 0., .019))
    legend.pose.position = Point(.30, -.40, .55)
    legend.text = ('SAME RED PATH | OFFLINE IK DIAGNOSTIC | robot fixed at archived q0\n'
                   'green=continuous prefix | yellow=joint jump | purple=IK residual\n'
                   'orange halo=limit/singularity warning | gray=NOT reached\n'
                   'red line=rejected candidate | collision/task acceptance NOT checked')
    markers.append(legend)
    return markers


def _continuous_ik_snapshot(data):
    report = json.loads(str(data['continuous_ik_report_json'].item()))
    candidate = np.asarray(data['candidate_path_poses7'], dtype=float)
    q0 = np.asarray(data['initial_joints_rad'], dtype=float)
    if (bool(data['motion_allowed']) or report['motion_allowed']
            or candidate.shape != (len(report['waypoints']), 7) or q0.shape != (6,)
            or not np.isfinite(candidate).all() or not np.isfinite(q0).all()
            or hashlib.sha256(candidate.tobytes()).hexdigest() != report['provenance']['path_array_sha256']
            or not np.array_equal(q0, report['provenance']['initial_joints_rad'])):
        raise ValueError('invalid frozen-path IK diagnostic')
    marker_pub = rospy.Publisher('~markers', MarkerArray, queue_size=1, latch=True)
    joint_pub = rospy.Publisher('~joint_states', JointState, queue_size=1)
    status_pub = rospy.Publisher('~status', String, queue_size=1, latch=True)
    markers = _continuous_ik_markers(candidate, report)
    status_pub.publish(String(data=json.dumps(dict(
        report['summary'], motion_allowed=False, hardware_commands_sent=False,
        robot_display='fixed archived q0; not measured feedback',
        initial_joint_source=report['provenance']['initial_joint_source']))))
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        for marker in markers:
            marker.header.stamp = rospy.Time.now()
        marker_pub.publish(MarkerArray(markers=markers))
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = JOINT_NAMES + ['joint7', 'joint8']
        message.position = q0.tolist() + [0., 0.]
        joint_pub.publish(message)
        rate.sleep()


def main():
    rospy.init_node("m8_offline_preview")
    path = Path(rospy.get_param("~preview_data")).expanduser().resolve()
    with np.load(str(path), allow_pickle=False) as data:
        if 'continuous_ik_report_json' in data:
            return _continuous_ik_snapshot(data)
        if 'target_points_base' in data and 'grasp_matrices' in data:
            return _target_snapshot(data)
        if 'cartesian_matrices' in data and 'joint_path' in data:
            return _candidate_only(data)
        endpoints, joints, candidate, keypoints = _validate(data)
        labels = [str(value) for value in data["endpoint_labels"]]
        valid_segments = [np.asarray(data[name], dtype=float) for name in (
            "valid_grasp_approach_poses7",
            "valid_transport_poses7",
            "valid_place_poses7",
        )]
        failure_index = int(np.asarray(data["candidate_failure_index"]).item())

    joint_pub = rospy.Publisher("~joint_states", JointState, queue_size=1)
    marker_pub = rospy.Publisher("~markers", MarkerArray, queue_size=1,
                                 latch=True)
    status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)

    markers = [_line("rejected_pathsolver_path", 0, candidate,
                     (1.0, 0.05, 0.05), 0.005)]
    for index, segment in enumerate(valid_segments):
        markers.append(_line("isolated_ik_valid_segments", index, segment,
                             (0.1, 0.9, 0.2), 0.004))
    failed = _marker(Marker.SPHERE, "first_ik_failure", 0,
                     (1.0, 0.0, 0.8), (0.035, 0.035, 0.035))
    failed.pose.position = Point(*candidate[failure_index, :3])
    markers.append(failed)
    markers.extend(_pose_markers(endpoints, labels))
    markers.extend(_keypoint_markers(keypoints))
    legend = _marker(Marker.TEXT_VIEW_FACING, "legend", 0,
                     (1.0, 1.0, 1.0), (0.0, 0.0, 0.025))
    legend.pose.position = Point(0.30, -0.40, 0.50)
    legend.text = "M8 PREVIEW ONLY | red=rejected PathSolver | green=isolated IK-valid"
    markers.append(legend)
    marker_pub.publish(MarkerArray(markers=markers))

    rate = rospy.Rate(20)
    current = 0
    step = 0
    transition_samples = 40
    dwell_samples = 30
    while not rospy.is_shutdown():
        following = (current + 1) % len(joints)
        if step < transition_samples:
            u = (step + 1) / float(transition_samples)
            blend = 10*u**3 - 15*u**4 + 6*u**5
            position = joints[current] + blend * (joints[following] - joints[current])
            label = labels[following]
        else:
            position = joints[following]
            label = labels[following]
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = JOINT_NAMES + ["joint7", "joint8"]
        message.position = position.tolist() + [0.0, 0.0]
        joint_pub.publish(message)
        status_pub.publish(String(data=(
            "PREVIEW_ONLY {} | no controller/CAN | rejected path stays red"
            .format(label))))
        step += 1
        if step >= transition_samples + dwell_samples:
            current, step = following, 0
        rate.sleep()


if __name__ == "__main__":
    main()
