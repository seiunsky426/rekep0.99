#!/usr/bin/env python3
"""Sequential entrypoint: validate autonomous release before roslaunch."""

import argparse
import os
import subprocess
import sys

from rekpiper_acceptance import AcceptanceError, validate_release_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("shadow", "autonomous"), default="shadow")
    parser.add_argument("--release-bundle", default="")
    parser.add_argument("--acceptance-bundle-type", choices=(
        "release_bundle", "hardware_acceptance_bundle"),
        default="release_bundle")
    parser.add_argument("--public-key", default="")
    parser.add_argument("--minimum-counter", default="")
    parser.add_argument("--robot-id", default="piper-rekpiper")
    parser.add_argument("--source-root", default=os.environ.get(
        "REKPIPER_ROOT", ""))
    parser.add_argument("--vendor-root", default=os.environ.get(
        "REKPIPER_VENDOR_ROOT", ""))
    parser.add_argument("--model-manifest", default=os.path.join(
        os.environ.get("REKPIPER_MODEL_ROOT", ""), "MANIFEST.sha256"))
    parser.add_argument("launch_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    child_environment = dict(os.environ)
    child_environment["REKPIPER_ACCEPTANCE_BUNDLE_TYPE"] = \
        args.acceptance_bundle_type
    os.environ["REKPIPER_ACCEPTANCE_BUNDLE_TYPE"] = args.acceptance_bundle_type
    offline = [
        "rosrun", "rekpiper_bringup", "preflight_cli.py",
        "--mode", args.mode, "--source-root", args.source_root,
        "--vendor-root", args.vendor_root,
        "--model-manifest", args.model_manifest,
    ]
    if args.mode == "autonomous":
        try:
            validate_release_bundle(
                args.release_bundle, args.public_key, args.minimum_counter,
                args.robot_id)
        except AcceptanceError as exc:
            print("autonomous preflight failed before roslaunch: {}".format(exc),
                  file=sys.stderr)
            return 2
        offline.extend([
            "--release-bundle", args.release_bundle,
            "--public-key", args.public_key,
            "--minimum-counter", args.minimum_counter,
            "--robot-id", args.robot_id,
        ])
    if subprocess.call(offline, env=child_environment) != 0:
        return 2
    command = ["roslaunch", "rekpiper_bringup", "system.launch",
               "mode:=" + args.mode,
               "acceptance_bundle_type:=" + args.acceptance_bundle_type,
               "release_bundle:=" + args.release_bundle,
               "acceptance_public_key:=" + args.public_key,
               "minimum_release_counter:=" + args.minimum_counter,
               "robot_id:=" + args.robot_id] + list(args.launch_args)
    return subprocess.call(command, env=child_environment)


if __name__ == "__main__":
    sys.exit(main())
