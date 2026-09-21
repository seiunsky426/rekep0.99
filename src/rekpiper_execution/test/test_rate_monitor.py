#!/usr/bin/env python3

import unittest

from rekpiper_execution.rate_monitor import EventRate, distribution


class RateMonitorTest(unittest.TestCase):
    def test_measures_arrival_rate_and_zeroes_when_stale(self):
        rate = EventRate(window_s=2.0)
        for index in range(21):
            rate.mark(index * 0.05)
        self.assertAlmostEqual(rate.rate(1.0, 0.2), 20.0)
        self.assertEqual(rate.rate(1.3, 0.2), 0.0)

    def test_reports_latency_percentiles(self):
        values = distribution([0.01, 0.02, 0.03, 0.04])
        self.assertEqual(values["count"], 4)
        self.assertGreater(values["p99"], values["p95"])


if __name__ == "__main__":
    unittest.main()
