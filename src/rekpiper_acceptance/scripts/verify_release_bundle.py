#!/usr/bin/env python3

import argparse
import json
import os
import sys

from rekpiper_acceptance import AcceptanceError, validate_release_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("--public-key", required=True)
    parser.add_argument("--minimum-counter", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--bundle-type", choices=(
        "release_bundle", "hardware_acceptance_bundle"),
        default="release_bundle")
    args = parser.parse_args()
    os.environ["REKPIPER_ACCEPTANCE_BUNDLE_TYPE"] = args.bundle_type
    try:
        result = validate_release_bundle(
            args.bundle, args.public_key, args.minimum_counter, args.robot_id)
        print(json.dumps({"success": True,
                          "release_counter": result["payload"]["release_counter"],
                          "artifacts": sorted(result["verified_artifacts"])},
                         sort_keys=True))
        return 0
    except AcceptanceError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
