#!/usr/bin/env python3
"""Materialize an immutable full 3-D SDF generation for each planning tick."""

from collections import deque
import io
import json
import time

from geometry_msgs.msg import Point
import numpy as np
import rospy
from std_msgs.msg import Header, String

from rekpiper_msgs.msg import SDFGrid
from rekpiper_msgs.srv import QuerySDF, QuerySDFRequest
from rekpiper_mapping.sdf_conventions import checked_grid_response, inclusive_grid_axes


def _distribution(values):
    data = np.asarray(list(values), dtype=float)
    if not len(data):
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": int(len(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


class SDFGridSnapshotNode:
    def __init__(self):
        self._bounds_min = np.asarray(rospy.get_param("~bounds_min"), dtype=float)
        self._bounds_max = np.asarray(rospy.get_param("~bounds_max"), dtype=float)
        self._resolution = float(rospy.get_param("~resolution_m", 0.015))
        self._rate = float(rospy.get_param("~rate_hz", 10.0))
        self._target_frame = str(rospy.get_param("~target_frame", "base_link"))
        self._diagnostic_only = bool(rospy.get_param("~diagnostic_only", False))
        if (self._bounds_min.shape != (3,) or self._bounds_max.shape != (3,)
                or np.any(self._bounds_min >= self._bounds_max)
                or self._resolution <= 0.0 or self._rate <= 0.0):
            raise rospy.ROSInitException("invalid SDF grid bounds, resolution, or rate")
        axes = inclusive_grid_axes(
            self._bounds_min, self._bounds_max, self._resolution)
        mesh = np.meshgrid(*axes, indexing="ij")
        self._shape = tuple(len(axis) for axis in axes)
        self._grid_bounds_max = self._bounds_max.astype(np.float32)
        self._points = np.stack(mesh, axis=-1).reshape(-1, 3)
        self._request = QuerySDFRequest(
            header=Header(frame_id=self._target_frame),
            points=[Point(*map(float, value)) for value in self._points],
            radii_m=[])
        request_buffer = io.BytesIO()
        self._request.serialize(request_buffer)
        self._request_bytes = request_buffer.tell()
        self._query = rospy.ServiceProxy(
            rospy.get_param("~query_service", "/rekpiper/mapping/query_sdf"),
            QuerySDF, persistent=False)
        self._publisher = rospy.Publisher(
            "/rekpiper/mapping/sdf_grid", SDFGrid, queue_size=1)
        self._diagnostics = rospy.Publisher(
            "/rekpiper/mapping/sdf_grid_snapshot_diagnostics",
            String, queue_size=1, latch=True)
        self._latencies = deque(maxlen=10000)
        self._publish_times = deque(maxlen=10000)
        self._deadline_misses = 0
        self._sequence = 0
        self._unavailable_since = None
        self._unavailable_duration_s = 0.0
        rospy.Timer(rospy.Duration(1.0 / self._rate), self._tick)

    def _invalid(self, reason):
        self._publisher.publish(SDFGrid(
            header=Header(stamp=rospy.Time.now(), frame_id="base_link"),
            valid=False, status=str(reason)))

    def _tick(self, _event):
        started = time.monotonic()
        self._sequence += 1
        sequence = self._sequence
        try:
            response = self._query(self._request)
            count = int(np.prod(self._shape))
            distances, observed, valid = checked_grid_response(
                response, count, self._target_frame, self._diagnostic_only)
            generation = str(response.map_generation_uuid)
            if not generation:
                raise RuntimeError("SDF query is not bound to a map generation")
            message = SDFGrid(
                header=response.header, map_generation_uuid=generation,
                bounds_min=Point(*self._bounds_min.tolist()),
                bounds_max=Point(*self._grid_bounds_max.tolist()),
                resolution_m=self._resolution,
                size_x=self._shape[0], size_y=self._shape[1], size_z=self._shape[2],
                distances_m=distances.tolist(), observed=observed.tolist(),
                valid=valid, status=("diagnostic_sdf_generation_not_for_planning"
                                    if self._diagnostic_only else "immutable_sdf_generation"))
            message_buffer = io.BytesIO()
            message.serialize(message_buffer)
            response_buffer = io.BytesIO()
            response.serialize(response_buffer)
            self._publisher.publish(message)
            finished = time.monotonic()
            if self._unavailable_since is not None:
                self._unavailable_duration_s += (
                    finished - self._unavailable_since)
                self._unavailable_since = None
            latency = finished - started
            self._latencies.append(latency)
            self._publish_times.append(finished)
            if latency > 1.0 / self._rate:
                self._deadline_misses += 1
            gaps = np.diff(np.asarray(self._publish_times, dtype=float))
            duration = (self._publish_times[-1] - self._publish_times[0]
                        if len(self._publish_times) > 1 else 0.0)
            payload = {
                "state": "diagnostic_published" if self._diagnostic_only else "published",
                "sequence": sequence,
                "event_monotonic_ns": time.monotonic_ns(),
                "target_rate_hz": self._rate,
                "rate_valid": len(self._publish_times) > 1,
                "rate_sample_count": len(self._publish_times),
                "actual_rate_hz": ((len(self._publish_times) - 1) / duration
                                   if duration > 0.0 else 0.0),
                "query_and_publish_latency_s": _distribution(self._latencies),
                "publish_period_s": _distribution(gaps),
                "deadline_miss_count": self._deadline_misses,
                "maximum_gap_s": float(np.max(gaps)) if len(gaps) else 0.0,
                "unavailable_duration_s": self._unavailable_duration_s,
                "grid_point_count": count,
                "array_payload_bytes": int(count * 5),
                "query_request_serialized_bytes": self._request_bytes,
                "query_response_serialized_bytes": response_buffer.tell(),
                "sdf_message_serialized_bytes": message_buffer.tell(),
                "map_generation_uuid": generation,
            }
            self._diagnostics.publish(String(
                data=json.dumps(payload, sort_keys=True)))
        except Exception as exc:
            failed_at = time.monotonic()
            if self._unavailable_since is None:
                self._unavailable_since = failed_at
            rospy.logwarn_throttle(2.0, "SDF grid snapshot unavailable: %s", exc)
            self._publisher.publish(SDFGrid(
                header=Header(stamp=rospy.Time.now(), frame_id="base_link"),
                map_generation_uuid="", valid=False,
                status="{}:{}".format(type(exc).__name__, exc)))
            self._diagnostics.publish(String(data=json.dumps({
                "state": "unavailable",
                "sequence": sequence,
                "event_monotonic_ns": time.monotonic_ns(),
                "target_rate_hz": self._rate,
                "error": "{}:{}".format(type(exc).__name__, exc),
                "query_and_publish_latency_s": _distribution(self._latencies),
                "deadline_miss_count": self._deadline_misses,
                "maximum_gap_s": (
                    float(np.max(np.diff(np.asarray(
                        self._publish_times, dtype=float))))
                    if len(self._publish_times) > 1 else 0.0),
                "unavailable_duration_s": (
                    self._unavailable_duration_s
                    + failed_at - self._unavailable_since),
                "grid_point_count": int(np.prod(self._shape)),
            }, sort_keys=True)))


if __name__ == "__main__":
    rospy.init_node("sdf_grid_snapshot")
    SDFGridSnapshotNode()
    rospy.spin()
