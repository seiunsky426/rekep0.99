"""Continuous, sequence-aware aggregation of runtime benchmark evidence."""

from __future__ import annotations

import json
import math

import numpy as np


class RuntimeEvidenceError(ValueError):
    pass


class RuntimeEvidenceAccumulator:
    def __init__(self, started_monotonic, profile="software"):
        if profile not in ("software", "hardware_hold"):
            raise RuntimeEvidenceError("runtime profile is invalid")
        self.started = float(started_monotonic)
        self.profile = profile
        self.sdf_rates, self.sdf_p99, self.sdf_sizes = [], [], []
        self.event_times = {"sdf": [], "trajectory": [], "planning": []}
        self.trajectory_rates, self.trajectory_p99 = [], []
        self.trajectory_samples, self.trajectory_max_gaps = [], []
        self.warm_planning = []
        self.last_sequence = {}
        self.failures = []
        self.raw_events = []

    @staticmethod
    def _payload(value):
        result = json.loads(value) if isinstance(value, str) else value
        if not isinstance(result, dict):
            raise RuntimeEvidenceError("diagnostic payload must be a mapping")
        return result

    def _event(self, stream, value, allow_synthetic_sequence=False):
        sequence = value.get("sequence")
        event_ns = value.get("event_monotonic_ns")
        if sequence is None and allow_synthetic_sequence:
            sequence = self.last_sequence.get(stream, 0) + 1
        if sequence is not None:
            sequence = int(sequence)
            previous = self.last_sequence.get(stream)
            if previous is not None and sequence != previous + 1:
                self.failures.append("{}_sequence_discontinuity".format(stream))
            self.last_sequence[stream] = sequence
        if event_ns is not None:
            event_s = int(event_ns) / 1e9
            times = self.event_times[stream]
            if times and event_s <= times[-1]:
                self.failures.append("{}_clock_not_monotonic".format(stream))
            times.append(event_s)
        self.raw_events.append({"stream": stream, "payload": value})

    def add_sdf(self, payload):
        value = self._payload(payload)
        self._event("sdf", value, allow_synthetic_sequence=True)
        if value.get("state") != "published":
            self.failures.append("sdf_" + str(value.get("state", "invalid")))
            return
        if int(value.get("grid_point_count", 0)) != 324972:
            raise RuntimeEvidenceError("SDF benchmark grid must contain 324972 points")
        if bool(value.get("rate_valid", True)):
            rate = float(value["actual_rate_hz"])
            if not math.isfinite(rate) or rate <= 0.0:
                raise RuntimeEvidenceError("valid SDF rate must be positive and finite")
            self.sdf_rates.append(rate)
        self.sdf_p99.append(float(value["query_and_publish_latency_s"]["p99"]))
        self.sdf_sizes.append(int(value["sdf_message_serialized_bytes"]))

    def add_trajectory(self, payload):
        value = self._payload(payload)
        self._event("trajectory", value, allow_synthetic_sequence=True)
        state = str(value.get("state", "invalid"))
        if state in ("aborted", "invalid", "unavailable"):
            self.failures.append("trajectory_" + state)
        if state not in ("completed", "streaming"):
            return
        timing = value["command_timing"]
        self.trajectory_rates.append(float(timing["actual_rate_hz"]))
        self.trajectory_p99.append(float(timing["p99"]))
        self.trajectory_samples.append(int(timing.get(
            "command_samples", int(timing.get("count", 0)) + 1)))
        self.trajectory_max_gaps.append(float(timing.get(
            "maximum_gap_s", timing.get("p99", float("inf")))))

    def add_horizon(self, latency_s, from_scratch, valid,
                    sequence=None, event_monotonic_ns=None):
        value = {"sequence": sequence, "event_monotonic_ns": event_monotonic_ns,
                 "valid": bool(valid), "from_scratch": bool(from_scratch),
                 "planning_latency_s": float(latency_s)}
        if sequence is not None or event_monotonic_ns is not None:
            self._event("planning", value)
        else:
            self.raw_events.append({"stream": "planning", "payload": value})
        if not bool(valid):
            self.failures.append("planning_invalid")
        elif not bool(from_scratch):
            latency = float(latency_s)
            if np.isfinite(latency) and latency >= 0.0:
                self.warm_planning.append(latency)
            else:
                self.failures.append("planning_latency_invalid")

    @staticmethod
    def _maximum_gap(times):
        return float(np.max(np.diff(times))) if len(times) > 1 else 0.0

    def report(self, finished_monotonic, raw_event_log_sha256=""):
        duration = float(finished_monotonic) - self.started
        if self.failures:
            raise RuntimeEvidenceError("runtime stream failed: " + self.failures[-1])
        if (duration <= 0.0 or not self.sdf_rates or not self.sdf_p99
                or not self.trajectory_rates or not self.warm_planning):
            raise RuntimeEvidenceError("runtime benchmark evidence is incomplete")
        return {
            "schema_version": 2,
            "profile": self.profile,
            "duration_s": duration,
            "continuous": True,
            "raw_event_log_sha256": str(raw_event_log_sha256),
            "raw_event_count": len(self.raw_events),
            "sdf": {
                "actual_rate_hz": float(np.min(self.sdf_rates)),
                "latency_p99_s": float(np.max(self.sdf_p99)),
                "serialized_message_bytes_maximum": int(max(self.sdf_sizes)),
                "diagnostic_samples": len(self.sdf_p99),
                "valid_rate_samples": len(self.sdf_rates),
                "maximum_gap_s": self._maximum_gap(self.event_times["sdf"]),
            },
            "trajectory": {
                "actual_rate_hz": float(np.min(self.trajectory_rates)),
                "period_p99_s": float(np.max(self.trajectory_p99)),
                # Bridge diagnostics expose one cumulative monotonic counter;
                # summing snapshots would manufacture coverage.
                "command_samples": int(max(self.trajectory_samples)),
                "completed_prefixes": len(self.trajectory_rates),
                "maximum_gap_s": float(max(self.trajectory_max_gaps)),
            },
            "planning": {
                "warm_latency_p99_s": float(np.percentile(self.warm_planning, 99)),
                "warm_samples": len(self.warm_planning),
                "maximum_gap_s": self._maximum_gap(
                    self.event_times["planning"]),
            },
        }
