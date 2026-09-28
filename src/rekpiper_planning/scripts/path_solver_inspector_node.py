#!/usr/bin/env python3
"""RViz-only PathSolver inspector. No driver, command topic or execution client."""

import json
from pathlib import Path
import queue
import threading
import time
import uuid

import numpy as np
import rospy
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion, Vector3
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, String
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import (
    InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback,
    Marker, MarkerArray)

from rekpiper_planning.path_preview import (
    display_indices, live_joint_seed, matrix_pose, pose_matrices, preview_ik)
from rekpiper_planning.continuous_ik import densify_joint_path
from rekpiper_planning.path_solver_trace import default_trace_directory, write_json
from rekpiper_planning.path_trace_report import cost_plot, joint_plot
from rekpiper_planning.piper_urdf_ik import PiperURDFIKSolver


def ros_pose(value):
    return Pose(Point(*value[:3]), Quaternion(*value[3:]))


class PathSolverInspector:
    def __init__(self):
        self.frame = rospy.get_param('~base_frame', 'base_link')
        self.tip = rospy.get_param('~tip_frame', 'rekep_tcp')
        self.root = Path(rospy.get_param('~trace_directory', '') or default_trace_directory()).expanduser()
        self.pinned = rospy.get_param('~run_directory', '')
        self.limit = int(rospy.get_param('~max_interpolation_points', 35))
        display_indices(2, self.limit)
        self.ik = PiperURDFIKSolver(
            self.frame, self.tip, ['joint{}'.format(i) for i in range(1, 7)],
            robot_description_param='/path_solver_preview/robot_description')
        self.markers = rospy.Publisher('~markers', MarkerArray, queue_size=1, latch=True)
        self.poses = rospy.Publisher('~all_poses', PoseArray, queue_size=1, latch=True)
        self.joints = rospy.Publisher('~joint_states', JointState, queue_size=1)
        self.status = rospy.Publisher('~status', String, queue_size=1, latch=True)
        self.server = InteractiveMarkerServer('~interactive')
        self.feedback = None
        self.directory = None
        self.summary = {}
        self.costs = []
        self.clear_markers = True
        self.target = None
        self.playback = None
        self.play_index = 0
        self.seed = None
        self.busy = False
        self.notice = 'Waiting for PathSolver logs and fresh joint feedback'
        self.commands = queue.Queue(maxsize=1)
        self.results = queue.Queue()
        self.last_poll = 0.
        self._buttons()
        self.server.applyChanges()
        rospy.Subscriber(rospy.get_param('~joint_state_topic', '/joint_states_single'),
                         JointState, self._feedback, queue_size=1)
        self.services = [rospy.Service('~' + command, Trigger,
                         lambda request, action=command: self._request(action))
                         for command in ('reset_target', 'solve_target', 'solve_path', 'replay')]
        self.timer = rospy.Timer(rospy.Duration(.1), self._tick)

    def _feedback(self, message):
        self.feedback = (message, time.monotonic())

    def _live_seed(self, with_stamp=False):
        if self.feedback is None:
            raise ValueError('No live joint feedback; start your existing read-only feedback source')
        message, received = self.feedback
        seed = live_joint_seed(message.name, message.position, message.header.stamp.to_sec(),
                               rospy.Time.now().to_sec(), time.monotonic() - received, self.ik)
        return (seed, message.header.stamp.to_sec()) if with_stamp else seed

    def _request(self, action):
        if self.busy:
            return TriggerResponse(False, 'IK preview is busy')
        try:
            self.commands.put_nowait(action)
        except queue.Full:
            return TriggerResponse(False, 'A preview request is already queued')
        return TriggerResponse(True, 'Queued ' + action + '; inspect status and RViz')

    def _buttons(self):
        for index, (name, label) in enumerate((
                ('reset_target', '1  Live EE\nto target'),
                ('solve_target', '2  IK target'),
                ('solve_path', '3  Path IK\nfrom live joints'),
                ('replay', '4  Replay IK'))):
            button = InteractiveMarker(name=name, scale=.10)
            button.header.frame_id = self.frame
            button.pose.position = Point(.12 + .19 * index, -.40, .12)
            button.pose.orientation.w = 1.
            control = InteractiveMarkerControl(always_visible=True,
                                               interaction_mode=InteractiveMarkerControl.BUTTON)
            block = Marker(type=Marker.CUBE, scale=Vector3(.16, .07, .045),
                           color=ColorRGBA(.12, .55, .85, 1.))
            block.pose.orientation.w = 1.
            control.markers.append(block)
            caption = Marker(type=Marker.TEXT_VIEW_FACING, text=label,
                             scale=Vector3(0., 0., .025), color=ColorRGBA(1., 1., 1., 1.))
            caption.pose.orientation.w = 1.
            caption.pose.position.z = .075
            control.markers.append(caption)
            button.controls.append(control)
            self.server.insert(button, self._button_feedback)

    def _button_feedback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            response = self._request(feedback.marker_name)
            if not response.success:
                rospy.logwarn(response.message)

    def _target_feedback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            p, q = feedback.pose.position, feedback.pose.orientation
            self.target = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])

    def _set_target(self, pose):
        self.target = pose.copy()
        marker = InteractiveMarker(name='target', description='Drag target; then click IK target', scale=.13)
        marker.header.frame_id = self.frame
        marker.pose = ros_pose(pose)
        half = np.sqrt(.5)
        for name, orientation in (('x', Quaternion(half, 0, 0, half)),
                                  ('y', Quaternion(0, 0, half, half)),
                                  ('z', Quaternion(0, half, 0, half))):
            for mode, label in ((InteractiveMarkerControl.MOVE_AXIS, 'move'),
                                (InteractiveMarkerControl.ROTATE_AXIS, 'rotate')):
                marker.controls.append(InteractiveMarkerControl(
                    name=label + '_' + name, orientation=orientation, interaction_mode=mode))
        self.server.insert(marker, self._target_feedback)
        self.server.applyChanges()

    def _load(self):
        if self.busy:
            return
        if self.pinned:
            directory = Path(self.pinned).expanduser()
        else:
            pointer = self.root / 'latest.json'
            if not pointer.exists():
                return
            directory = Path(json.loads(pointer.read_text())['directory'])
        if directory == self.directory:
            return
        summary = json.loads((directory / 'summary.json').read_text())
        if summary.get('base_frame') != self.frame or summary.get('tip_frame') != self.tip:
            raise ValueError('Trace base/tip frame does not match preview URDF settings')
        for key in ('control_poses', 'spline_poses', 'dense_poses'):
            if len(summary.get(key, [])):
                pose_matrices(summary[key])
        costs = cost_plot(directory, summary.get('reset_reg_in_total', False))
        self.directory, self.summary, self.costs = directory, summary, costs
        self.clear_markers = True
        self.playback = None
        self.notice = '{} | {}'.format(directory.name, summary['status'])
        if summary.get('validation_only'):
            self.notice = 'SYNTHETIC TEST | ' + self.notice
        poses = PoseArray()
        poses.header.frame_id = self.frame
        poses.header.stamp = rospy.Time.now()
        poses.poses = [ros_pose(p) for p in summary.get('dense_poses', [])]
        self.poses.publish(poses)

    def _work(self, action):
        seed, seed_stamp = self._live_seed(with_stamp=True)
        if action == 'reset_target':
            self.playback = None
            self.seed = seed
            self._set_target(matrix_pose(self.ik.forward(seed)))
            self.notice = 'Target copied from fresh live joints / URDF FK'
            return
        if action == 'solve_path':
            poses = np.asarray(self.summary.get('dense_poses', []), dtype=float)
            if not len(poses):
                raise ValueError('This trace has no completed PathSolver dense path')
            # The first edge starts at measured FK and is checked by continuous IK.
            sequence = True
        else:
            if self.target is None:
                self._set_target(matrix_pose(self.ik.forward(seed)))
            poses = self.target[None].copy()
            sequence = False
        self.seed = seed
        self.playback = None
        self.busy = True
        self.notice = 'Solving {} from fresh live joints...'.format(action)
        source = self.directory
        destination = self.root / 'ik_previews' / (time.strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8])

        def solve():
            try:
                report = preview_ik(self.ik, poses, seed, sequence=sequence)
                report.update(source_trace=str(source) if source else None,
                              base_frame=self.frame, tip_frame=self.tip,
                              target_poses=poses, seed_stamp_s=seed_stamp)
                destination.mkdir(parents=True)
                write_json(destination / 'ik.json', report)
                joint_plot(destination, report)
                self.results.put((report, str(destination), ''))
            except Exception as exc:
                self.results.put((None, '', str(exc)))
        threading.Thread(target=solve, daemon=True).start()

    def _marker(self, array, namespace, kind, color, scale):
        marker = Marker(ns=namespace, id=len(array.markers), type=kind, action=Marker.ADD)
        marker.header.frame_id = self.frame
        marker.pose.orientation.w = 1.
        marker.color = ColorRGBA(*color)
        marker.scale = Vector3(*scale)
        array.markers.append(marker)
        return marker

    def _draw(self, q):
        array = MarkerArray()
        styles = [('dense_poses', (.2, .75, 1., .65), .004),
                  ('spline_poses', (1., .8, .05, 1.), .010),
                  ('control_poses', (1., .15, .7, 1.), .023)]
        for key, color, size in styles:
            poses = np.asarray(self.summary.get(key, []))
            if not len(poses):
                continue
            indices = display_indices(len(poses), self.limit) if key == 'dense_poses' else range(len(poses))
            points = self._marker(array, key, Marker.SPHERE_LIST, color, (size,) * 3)
            points.points = [Point(*poses[i, :3]) for i in indices]
            matrices = pose_matrices(poses)
            for axis, axis_color in enumerate(((1., .2, .2, 1.), (.2, 1., .2, 1.), (.2, .4, 1., 1.))):
                axes = self._marker(array, key + '_axis_' + str(axis), Marker.LINE_LIST,
                                    axis_color, (.0015, 0., 0.))
                length = .018 if key == 'dense_poses' else .04
                for i in indices:
                    axes.points.extend([Point(*poses[i, :3]),
                                        Point(*(poses[i, :3] + length * matrices[i, :3, axis]))])
            for i in indices:
                if key == 'control_poses':
                    label = self._marker(array, 'control_labels', Marker.TEXT_VIEW_FACING,
                                         (1., 1., 1., 1.), (0., 0., .020))
                    label.pose.position = Point(*(poses[i, :3] + [0, 0, .028]))
                    label.text = 'START' if i == 0 else 'GOAL' if i == len(poses)-1 else 'C{}'.format(i)
            if key == 'dense_poses':
                line = self._marker(array, 'full_path', Marker.LINE_STRIP, color, (.002, 0., 0.))
                line.points = [Point(*p[:3]) for p in poses]
        finite_costs = [row for row in self.costs if np.isfinite(row['total_cost'])]
        if finite_costs:
            low = min(row['best_cost'] for row in finite_costs)
            high = max(row['total_cost'] for row in finite_costs)
            for key, color in (('total_cost', (.1, .7, 1., 1.)), ('best_cost', (.2, 1., .3, 1.))):
                line = self._marker(array, key, Marker.LINE_STRIP, color, (.002, 0., 0.))
                line.points = [Point(.05 + .55 * row['evaluation'] / max(1, self.costs[-1]['evaluation']),
                                     .42, .12 + .25 * (row[key] - low) / max(1.e-12, high-low))
                               for row in finite_costs]
        label = self._marker(array, 'status', Marker.TEXT_VIEW_FACING,
                             (1., 1., 1., 1.), (0., 0., .030))
        label.pose.position = Point(.35, 0., .72)
        label.text = ('PREVIEW ONLY | magenta=controls; yellow=spline; cyan=dense\n'
                      'Cost graph: cyan=total; green=best; x=evaluations\n' + self.notice)
        if self.summary.get('validation_only'):
            label.text = 'SYNTHETIC TEST DATA\n' + label.text
        if q is not None:
            delta = q - (self.seed if self.seed is not None else q)
            label.text += '\nq deg: ' + ' / '.join('{:.1f}'.format(v) for v in np.rad2deg(q))
            label.text += '\ndelta deg: ' + ' / '.join('{:+.1f}'.format(v) for v in np.rad2deg(delta))
        if self.clear_markers:
            array.markers.insert(0, Marker(action=Marker.DELETEALL))
            self.clear_markers = False
        self.markers.publish(array)
        self.status.publish(self.notice)

    def _tick(self, _event):
        try:
            if time.monotonic() - self.last_poll > 1.:
                self.last_poll = time.monotonic()
                self._load()
            try:
                report, destination, error = self.results.get_nowait()
                self.busy = False
                if error:
                    raise ValueError(error)
                self.notice = '{} | {} | {}'.format(
                    'IK PASS (collision unchecked)' if report['valid'] else 'IK FAILED',
                    report['preview_kind'], report['reason'])
                rospy.loginfo('IK preview evidence: %s', destination)
                if report['valid']:
                    self.playback = np.asarray(report['joint_positions'])
                    if report['preview_kind'] == 'endpoint_only':
                        self.playback = densify_joint_path(self.playback, .04)
                        self.notice += ' | animation edges unchecked'
                    self.play_index = 0
                else:
                    self.playback = None
                    self.notice += ' | failed pose {}'.format(report.get('failed_pose_index'))
            except queue.Empty:
                pass
            if not self.busy:
                try:
                    action = self.commands.get_nowait()
                    if action == 'replay':
                        if self.playback is None:
                            raise ValueError('No successful IK preview to replay')
                        self.play_index = 0
                    else:
                        self._work(action)
                except queue.Empty:
                    pass
            if self.playback is not None:
                q = self.playback[min(self.play_index, len(self.playback)-1)]
                self.play_index += 1
            else:
                try:
                    q = self._live_seed()
                    if self.target is None:
                        self._set_target(matrix_pose(self.ik.forward(q)))
                except ValueError:
                    q = None
            if q is not None:
                # Fingers are drawn closed; this tool previews only the six arm joints.
                message = JointState(name=self.ik.joint_names + ['joint7', 'joint8'],
                                     position=q.tolist() + [0., 0.])
                message.header.stamp = rospy.Time.now()
                self.joints.publish(message)
            self._draw(q)
        except Exception as exc:
            self.notice = str(exc)
            self.status.publish(self.notice)
            rospy.logwarn_throttle(5., 'PathSolver inspector: %s', exc)
            self._draw(None)


if __name__ == '__main__':
    rospy.init_node('inspector')
    PathSolverInspector()
    rospy.spin()
