#!/usr/bin/env python3

import unittest

from rekpiper_execution.runtime_acceptance import (
    RuntimeAcceptanceError, approve_evidence, validate_evidence)


class RuntimeAcceptanceTest(unittest.TestCase):
    def evidence(self):
        return {
            "duration_s": 600.0,
            "sdf": {"actual_rate_hz": 9.8, "latency_p99_s": 0.09},
            "trajectory": {"actual_rate_hz": 19.5, "period_p99_s": 0.05},
            "planning": {"warm_latency_p99_s": 0.09},
        }

    def test_approved_only_when_all_nominal_periods_pass(self):
        self.assertTrue(approve_evidence(
            self.evidence(), "operator")["approved"])
        failed = self.evidence()
        failed["sdf"]["latency_p99_s"] = 0.101
        with self.assertRaises(RuntimeAcceptanceError):
            approve_evidence(failed, "operator")

    def test_schema2_requires_raw_coverage_and_profile_contract(self):
        evidence = {
            "schema_version": 2, "profile": "software",
            "duration_s": 600.0, "continuous": True,
            "raw_event_log_sha256": "a" * 64,
            "raw_event_count": 24000,
            "software_contract": {"piper_driver_absent": True,
                                  "can_disabled_verified": True,
                                  "replay_dataset_sha256": "b" * 64,
                                  "program_sha256": "c" * 64},
            "sdf": {"actual_rate_hz": 9.8, "latency_p99_s": 0.09,
                    "diagnostic_samples": 6000, "maximum_gap_s": 0.11},
            "trajectory": {"actual_rate_hz": 19.5, "period_p99_s": 0.049,
                           "command_samples": 12000, "maximum_gap_s": 0.06},
            "planning": {"warm_latency_p99_s": 0.09,
                         "warm_samples": 6000, "maximum_gap_s": 0.11},
        }
        self.assertTrue(validate_evidence(evidence, "software"))
        evidence["trajectory"]["command_samples"] = 10
        with self.assertRaises(RuntimeAcceptanceError):
            validate_evidence(evidence, "software")


if __name__ == "__main__":
    unittest.main()
