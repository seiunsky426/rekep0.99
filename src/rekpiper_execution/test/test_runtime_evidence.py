#!/usr/bin/env python3

import json
import unittest

from rekpiper_execution.runtime_evidence import (
    RuntimeEvidenceAccumulator, RuntimeEvidenceError)


class RuntimeEvidenceTest(unittest.TestCase):
    def test_conservative_complete_report(self):
        accumulator = RuntimeEvidenceAccumulator(10.0)
        for rate, latency in ((10.0, 0.08), (9.7, 0.09)):
            accumulator.add_sdf(json.dumps({
                "state": "published", "grid_point_count": 324972,
                "actual_rate_hz": rate,
                "query_and_publish_latency_s": {"p99": latency},
                "sdf_message_serialized_bytes": 1700000,
            }))
        accumulator.add_trajectory({
            "state": "completed",
            "command_timing": {"actual_rate_hz": 19.5, "p99": 0.049},
        })
        accumulator.add_horizon(0.05, False, True)
        report = accumulator.report(610.0)
        self.assertEqual(report["duration_s"], 600.0)
        self.assertEqual(report["sdf"]["actual_rate_hz"], 9.7)
        self.assertEqual(report["sdf"]["latency_p99_s"], 0.09)

    def test_rejects_wrong_grid_and_incomplete_evidence(self):
        accumulator = RuntimeEvidenceAccumulator(0.0)
        with self.assertRaises(RuntimeEvidenceError):
            accumulator.add_sdf({
                "state": "published", "grid_point_count": 1,
            })
        with self.assertRaises(RuntimeEvidenceError):
            accumulator.report(600.0)

    def test_first_zero_rate_is_excluded_but_discontinuity_fails(self):
        accumulator = RuntimeEvidenceAccumulator(0.0)
        accumulator.add_sdf({
            "state": "published", "sequence": 1,
            "event_monotonic_ns": 1000000000, "rate_valid": False,
            "grid_point_count": 324972, "actual_rate_hz": 0.0,
            "query_and_publish_latency_s": {"p99": 0.01},
            "sdf_message_serialized_bytes": 1,
        })
        self.assertEqual(accumulator.sdf_rates, [])
        accumulator.add_sdf({
            "state": "published", "sequence": 3,
            "event_monotonic_ns": 1100000000, "rate_valid": True,
            "grid_point_count": 324972, "actual_rate_hz": 10.0,
            "query_and_publish_latency_s": {"p99": 0.01},
            "sdf_message_serialized_bytes": 1,
        })
        accumulator.add_trajectory({
            "state": "completed", "sequence": 1,
            "event_monotonic_ns": 1000000000,
            "command_timing": {"actual_rate_hz": 20.0, "p99": 0.049,
                               "command_samples": 12000,
                               "maximum_gap_s": 0.05},
        })
        accumulator.add_horizon(
            0.05, False, True, sequence=1,
            event_monotonic_ns=1000000000)
        with self.assertRaisesRegex(RuntimeEvidenceError, "discontinuity"):
            accumulator.report(600.0, "a" * 64)


if __name__ == "__main__":
    unittest.main()
