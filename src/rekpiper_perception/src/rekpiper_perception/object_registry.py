"""Session-scoped object identities and fail-closed camera tracking state."""

from dataclasses import dataclass, field
import time
import uuid


CAMERA_UNKNOWN = 0
CAMERA_TRACKED = 1
CAMERA_TRUSTED_NOT_VISIBLE = 2
CAMERA_LOST = 3
SAFE_CAMERA_STATES = frozenset((CAMERA_TRACKED, CAMERA_TRUSTED_NOT_VISIBLE))


def visually_tracked(camera_states, ages_s, maximum_age_s):
    """An ungrasped object is visible only with a fresh tracked observation."""
    return any(state == CAMERA_TRACKED and 0.0 <= age <= maximum_age_s
               for state, age in zip(camera_states, ages_s))


@dataclass
class CameraTrack:
    status: int = CAMERA_UNKNOWN
    confidence: float = 0.0
    stamp_s: float = 0.0


@dataclass
class ObjectRecord:
    object_uuid: str
    display_name: str
    rigid_group_id: int
    excluded_from_static_map: bool
    state: int = 1
    camera_tracks: dict = field(default_factory=dict)
    active: bool = True
    created_s: float = field(default_factory=time.time)
    retired_s: float = 0.0


class ObjectRegistry:
    def __init__(self, camera_names=("rs1", "rs3"), maximum_active=4,
                 uuid_factory=None):
        self.camera_names = tuple(str(name) for name in camera_names)
        if len(set(self.camera_names)) != len(self.camera_names) or not self.camera_names:
            raise ValueError("camera names must be unique and non-empty")
        self.maximum_active = int(maximum_active)
        if self.maximum_active <= 0:
            raise ValueError("maximum_active must be positive")
        self._uuid_factory = uuid_factory or (lambda: str(uuid.uuid4()))
        self._records = {}
        self.revision = 0
        self.exclusion_revision = 0

    @property
    def records(self):
        return tuple(self._records.values())

    @property
    def active_records(self):
        return tuple(record for record in self._records.values() if record.active)

    def register(self, source_camera, display_name, rigid_group_id,
                 excluded_from_static_map=True):
        if source_camera not in self.camera_names:
            raise ValueError("unknown source camera")
        rigid_group_id = int(rigid_group_id)
        if rigid_group_id <= 0:
            raise ValueError("rigid_group_id must be positive")
        if any(record.active and record.rigid_group_id == rigid_group_id
               for record in self._records.values()):
            raise ValueError("rigid_group_id already has an active object")
        if len(self.active_records) >= self.maximum_active:
            raise ValueError("maximum active object count reached")
        identity = str(self._uuid_factory())
        if not identity or identity in self._records:
            raise ValueError("UUID factory returned an invalid or reused identity")
        record = ObjectRecord(
            object_uuid=identity,
            display_name=str(display_name).strip() or "object",
            rigid_group_id=rigid_group_id,
            excluded_from_static_map=bool(excluded_from_static_map),
            camera_tracks={name: CameraTrack() for name in self.camera_names},
        )
        self._records[identity] = record
        self.revision += 1
        if record.excluded_from_static_map:
            self.exclusion_revision += 1
        return record

    def require(self, object_uuid, active=True):
        record = self._records.get(str(object_uuid))
        if record is None or (active and not record.active):
            raise KeyError("unknown active object_uuid")
        return record

    def retire(self, object_uuid, now_s=None):
        record = self.require(object_uuid)
        record.active = False
        record.retired_s = time.time() if now_s is None else float(now_s)
        self.revision += 1
        if record.excluded_from_static_map:
            self.exclusion_revision += 1
        return record

    def set_excluded_from_static_map(self, object_uuid, excluded):
        """Atomically change only the map-exclusion lifecycle flag."""
        record = self.require(object_uuid)
        value = bool(excluded)
        if record.excluded_from_static_map == value:
            return False
        record.excluded_from_static_map = value
        self.revision += 1
        self.exclusion_revision += 1
        return True

    def update_camera(self, object_uuid, camera_name, status, confidence, stamp_s):
        record = self.require(object_uuid)
        if camera_name not in self.camera_names:
            raise ValueError("unknown camera")
        if int(status) not in (CAMERA_UNKNOWN, CAMERA_TRACKED,
                               CAMERA_TRUSTED_NOT_VISIBLE, CAMERA_LOST):
            raise ValueError("invalid camera tracking status")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("camera confidence must be in [0, 1]")
        track = record.camera_tracks[camera_name]
        if float(stamp_s) < track.stamp_s:
            raise ValueError("camera track timestamps must be monotonic")
        record.camera_tracks[camera_name] = CameraTrack(
            status=int(status), confidence=confidence, stamp_s=float(stamp_s))

    def all_excluded_objects_safe(self, now_s, maximum_age_s):
        now_s = float(now_s)
        maximum_age_s = float(maximum_age_s)
        for record in self.active_records:
            if not record.excluded_from_static_map:
                continue
            for track in record.camera_tracks.values():
                if track.status not in SAFE_CAMERA_STATES:
                    return False
                if now_s - track.stamp_s > maximum_age_s:
                    return False
        return True

    def exclusion_signature(self):
        return tuple(sorted(record.object_uuid for record in self.active_records
                            if record.excluded_from_static_map))
