#!/usr/bin/env python3
"""Select a paper-real weight candidate from deterministic replay evidence."""

import argparse
import json
from pathlib import Path
import sys

import yaml

from rekpiper_planning.solver_acceptance import (
    SolverAcceptanceError, calibrate_replay_report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_report")
    parser.add_argument("output")
    args = parser.parse_args()
    try:
        report = yaml.safe_load(Path(args.replay_report).read_text(encoding="utf-8"))
        candidate = calibrate_replay_report(report)
        Path(args.output).write_text(
            yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
        print(json.dumps({"success": True, "output": args.output,
                          "weights": candidate["weights"]}, sort_keys=True))
        return 0
    except (OSError, yaml.YAMLError, SolverAcceptanceError, ValueError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
