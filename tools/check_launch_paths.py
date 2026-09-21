#!/usr/bin/env python3
"""Validate launch XML and package-local paths in the active ROS layout."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import rospkg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--dump-shadow", action="store_true")
    args = parser.parse_args()
    root = Path(args.package_root).expanduser().resolve()
    errors = []
    ros_pack = rospkg.RosPack()
    pattern = re.compile(r"\$\(find\s+([^)]+)\)([^\"'<\s]*)")
    for launch in root.rglob("*.launch"):
        try:
            ET.parse(str(launch))
            text = launch.read_text(encoding="utf-8")
        except (OSError, ET.ParseError) as exc:
            errors.append("launch_xml_invalid:{}:{}".format(launch, exc))
            continue
        for package, suffix in pattern.findall(text):
            if not suffix or "$" in suffix:
                continue
            try:
                package_path = Path(ros_pack.get_path(package)).resolve()
            except rospkg.ResourceNotFound:
                errors.append("launch_package_missing:{}:{}".format(launch, package))
                continue
            target = (package_path / suffix.lstrip("/")).resolve()
            if not target.exists():
                errors.append("launch_file_missing:{}:{}".format(launch, target))
    if args.dump_shadow:
        result = subprocess.run(
            ["roslaunch", "--dump-params", "rekpiper_bringup", "system.launch",
             "mode:=shadow"], capture_output=True, text=True, check=False)
        if result.returncode:
            errors.append("shadow_dump_failed:" + result.stderr.strip())
    print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
