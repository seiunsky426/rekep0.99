"""Pure safety supervisor state used by the ROS mapping runtime."""

from dataclasses import dataclass, field
from enum import IntEnum
import uuid


class SafeMapState(IntEnum):
    DIRTY = 0
    WAITING_FOR_INPUTS = 1
    CLEARING = 2
    WARMUP = 3
    BUILDING = 4
    VALIDATING = 5
    READY = 6
    PAUSED_UNSAFE = 7


@dataclass
class SafeMapSupervisor:
    camera_names: tuple = ("rs1", "rs3")
    warmup_frames_per_camera: int = 30
    build_frames_per_camera: int = 150
    minimum_esdf_updates: int = 10
    state: SafeMapState = SafeMapState.DIRTY
    reason: str = "not_built"
    generation_uuid: str = ""
    configuration_hash: str = ""
    valid_frames: dict = field(default_factory=dict)
    esdf_updates: int = 0

    def __post_init__(self):
        self.camera_names = tuple(self.camera_names)
        self.valid_frames = {name: 0 for name in self.camera_names}

    @property
    def planning_safe(self):
        return self.state == SafeMapState.READY

    def mark_dirty(self, reason):
        self.state = SafeMapState.DIRTY
        self.reason = str(reason)
        self.generation_uuid = ""
        return self.state

    def begin_rebuild(self, configuration_hash, generation_uuid=None):
        self.state = SafeMapState.CLEARING
        self.reason = "clearing"
        self.configuration_hash = str(configuration_hash)
        self.generation_uuid = str(generation_uuid or uuid.uuid4())
        self.valid_frames = {name: 0 for name in self.camera_names}
        self.esdf_updates = 0
        return self.generation_uuid

    def cleared(self):
        if self.state != SafeMapState.CLEARING:
            raise ValueError("clear completion is out of sequence")
        self.state = SafeMapState.WARMUP
        self.reason = "collecting_warmup"

    def record_valid_frame(self, camera_name):
        if camera_name not in self.valid_frames:
            raise ValueError("unknown camera")
        if self.state not in (SafeMapState.WARMUP, SafeMapState.BUILDING,
                              SafeMapState.VALIDATING):
            return self.state
        self.valid_frames[camera_name] += 1
        if (self.state == SafeMapState.WARMUP
                and all(value >= self.warmup_frames_per_camera
                        for value in self.valid_frames.values())):
            self.state = SafeMapState.BUILDING
            self.reason = "building"
            self.valid_frames = {name: 0 for name in self.camera_names}
        if (self.state == SafeMapState.BUILDING
                and all(value >= self.build_frames_per_camera
                        for value in self.valid_frames.values())):
            self.state = SafeMapState.VALIDATING
            self.reason = "validating"
        self._maybe_ready()
        return self.state

    def record_esdf_update(self):
        if self.state in (SafeMapState.BUILDING, SafeMapState.VALIDATING):
            self.esdf_updates += 1
            self._maybe_ready()
        return self.state

    def pause(self, reason):
        self.state = SafeMapState.PAUSED_UNSAFE
        self.reason = str(reason)
        return self.state

    def _maybe_ready(self):
        if (self.state == SafeMapState.VALIDATING
                and self.esdf_updates >= self.minimum_esdf_updates):
            self.state = SafeMapState.READY
            self.reason = "safe_tsdf_ready"
