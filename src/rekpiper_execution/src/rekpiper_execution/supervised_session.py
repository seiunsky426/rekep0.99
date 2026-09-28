"""Hardware-free, single-use preview authority for supervised experiments."""

from dataclasses import dataclass
import hashlib
import json
import threading
import uuid

import numpy as np

from .trajectory import JOINT_NAMES, validate_trajectory


STAGES = ("预抓取", "抓取", "抬升并运输", "下放释放")
FIXED_TASK = "把蓝色方块放到黄色圆盘上"


def fixed_constraints(selection):
    """Bind measured keypoints to the unchanging four-stage task."""
    if len(selection) != 2:
        raise ValueError("blue_cube_and_yellow_disk_keypoints_required")
    return dict(task=FIXED_TASK, stages=list(STAGES),
                blue_cube=dict(keypoint_id=int(selection[0]["id"]),
                               rigid_group_id=int(selection[0]["rigid_group_id"]),
                               position_m=list(selection[0]["position"])),
                yellow_disk=dict(keypoint_id=int(selection[1]["id"]),
                                 rigid_group_id=int(selection[1]["rigid_group_id"]),
                                 position_m=list(selection[1]["position"])))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def trajectory_digest(names, positions, times):
    return digest(dict(names=list(names), positions=np.asarray(positions).tolist(),
                       times=np.asarray(times).tolist()))


@dataclass(frozen=True)
class Speed:
    velocity: float = .05
    acceleration: float = .10
    jerk: float = .50
    driver_percent: int = 5

    def __post_init__(self):
        if (not all(np.isfinite(x) and x > 0 for x in
                    (self.velocity, self.acceleration, self.jerk))
                or self.velocity > .20 or self.acceleration > .50
                or self.jerk > .50 or not 1 <= self.driver_percent <= 20
                or int(self.driver_percent) != self.driver_percent):
            raise ValueError("invalid_supervised_speed")


class Session:
    def __init__(self):
        self.lock = threading.RLock()
        self.id = uuid.uuid4().hex
        self.revision = 0
        self.stage = 1
        self.state = "IDLE"
        self.speed = Speed()
        self.preview = None
        self.held = False
        self.reason = "等待任务"

    def invalidate(self, reason):
        with self.lock:
            self.revision += 1
            self.preview = None
            self.state = "IDLE"
            self.reason = str(reason)

    def new_task(self):
        with self.lock:
            if self.state == "EXECUTING" or self.held:
                raise ValueError("cannot_replace_task_during_motion_or_holding")
            self.id = uuid.uuid4().hex
            self.stage = 1
            self.invalidate("任务已更新")

    def begin_plan(self):
        with self.lock:
            if self.state in ("PLANNING", "EXECUTING", "WAITING_GRASP", "COMPLETE"):
                raise ValueError("stage_not_ready_for_planning")
            if self.stage >= 3 and not self.held:
                raise ValueError("operator_grasp_confirmation_required")
            self.preview = None
            self.state = "PLANNING"
            return self.id, self.stage, self.revision

    def install(self, ticket, positions, times, binding, detail):
        with self.lock:
            if ticket != (self.id, self.stage, self.revision) or self.state != "PLANNING":
                raise ValueError("discarded_obsolete_planning_result")
            q, t = np.asarray(positions, dtype=float), np.asarray(times, dtype=float)
            validate_trajectory(JOINT_NAMES, q, t, q[0], self.speed.velocity,
                                self.speed.acceleration, .01, self.speed.jerk)
            self.preview = dict(id=uuid.uuid4().hex, session_id=self.id, stage=self.stage,
                                positions=q.tolist(), times=t.tolist(), binding=binding,
                                trajectory_sha256=trajectory_digest(JOINT_NAMES, q, t),
                                detail=detail, revision=self.revision)
            self.state = "PREVIEW_READY"
            self.reason = "检查通过，等待执行本段"
            return self.preview

    def claim(self, session_id, stage, preview_id, joints, binding):
        with self.lock:
            p = self.preview
            if (self.state != "PREVIEW_READY" or p is None
                    or (session_id, stage, preview_id) != (self.id, self.stage, p['id'])
                    or p['revision'] != self.revision or p['binding'] != binding):
                raise ValueError("preview_missing_changed_or_already_consumed")
            validate_trajectory(JOINT_NAMES, p['positions'], p['times'], joints,
                                self.speed.velocity, self.speed.acceleration, .01,
                                self.speed.jerk)
            if trajectory_digest(JOINT_NAMES, p['positions'], p['times']) != p['trajectory_sha256']:
                raise ValueError("preview_trajectory_changed")
            self.state = "EXECUTING"
            return p

    def finish(self):
        with self.lock:
            if self.state != "EXECUTING":
                raise ValueError("execution_was_revoked")
            self.preview = None
            self.revision += 1
            if self.stage == 2:
                self.state = "WAITING_GRASP"
                self.reason = "请人工确认是否夹稳"
            elif self.stage == 4:
                self.held = False
                self.state = "COMPLETE"
                self.reason = "下放释放完成，无撤离动作"
            else:
                self.stage += 1
                self.state = "IDLE"
                self.reason = "本段完成，请生成下一段预览"

    def confirm_grasp(self, confirmed):
        with self.lock:
            if self.state != "WAITING_GRASP":
                raise ValueError("not_waiting_for_grasp_confirmation")
            self.held = bool(confirmed)
            self.stage = 3 if confirmed else 2
            self.invalidate("人工确认已夹稳" if confirmed else "未夹稳，请重新检查抓取")

    def status(self):
        with self.lock:
            return dict(session_id=self.id, stage=self.stage, state=self.state,
                        reason=self.reason, held=self.held, speed=vars(self.speed),
                        preview=self.preview, revision=self.revision,
                        stage_names=list(STAGES))


def select_keypoints(text, keypoints):
    """VLM selects IDs only; metric coordinates never come from the model."""
    value = str(text).strip()
    if value.startswith('```json') and value.endswith('```'):
        value = value[7:-3].strip()
    data = json.loads(value)
    if set(data) != {'grasp_id', 'target_id'}:
        raise ValueError('vlm_must_return_grasp_id_and_target_id_only')
    ids = [data['grasp_id'], data['target_id']]
    if any(type(i) is not int for i in ids) or ids[0] == ids[1]:
        raise ValueError('invalid_or_duplicate_keypoint_ids')
    lookup = {int(k['id']): k for k in keypoints}
    result = []
    for i in ids:
        k = lookup.get(i)
        if (k is None or not k['valid'] or k.get('valid_depth_pixels', 0) <= 0
                or not np.isfinite(k['position']).all() or k['rigid_group_id'] <= 0):
            raise ValueError('selected_keypoint_has_no_valid_measured_geometry')
        result.append(dict(k))
    if result[0]['rigid_group_id'] == result[1]['rigid_group_id']:
        raise ValueError('grasp_and_support_must_be_different_objects')
    return result


def select_instance_group(text, labels):
    """Accept a VLM object label only if a saved 3-D instance exists."""
    value = str(text).strip()
    if value.startswith('```json') and value.endswith('```'):
        value = value[7:-3].strip()
    data = json.loads(value)
    if set(data) != {'group_id'} or type(data['group_id']) is not int:
        raise ValueError('vlm_must_return_group_id_only')
    group = data['group_id']
    if group <= 0 or np.count_nonzero(np.asarray(labels) == group) < 120:
        raise ValueError('rs3_selected_group_has_insufficient_depth')
    return group
