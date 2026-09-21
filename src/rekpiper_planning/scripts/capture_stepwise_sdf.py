#!/usr/bin/env python3
"""Capture and save a complete SDF for one ReKep planning batch."""
import argparse
import io
import json
from pathlib import Path
import time

import numpy as np
import rospy

from rekpiper_msgs.srv import CaptureSDFSnapshot
from rekpiper_planning.planning_snapshot import PlanningSDFSnapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(rospy.myargv()[1:])
    output = Path(args.output_dir).expanduser()
    if output.exists():
        raise SystemExit('output directory already exists; use a new capture directory')
    rospy.init_node('capture_stepwise_sdf', anonymous=True)
    service = '/rekpiper/mapping/capture_sdf_snapshot'
    rospy.wait_for_service(service, timeout=10)
    started = time.monotonic()
    response = rospy.ServiceProxy(service, CaptureSDFSnapshot, persistent=False)()
    elapsed = time.monotonic() - started
    if not response.success:
        raise SystemExit(response.message)
    data = PlanningSDFSnapshot(response.grid)
    output.mkdir(parents=True)
    buffer = io.BytesIO()
    response.grid.serialize(buffer)
    (output / 'sdf_grid.rosmsg').write_bytes(buffer.getvalue())
    np.savez_compressed(output / 'sdf_grid.npz', sdf_voxels=data.sdf_voxels,
        observed=data.observed, bounds_min=data.bounds_min, bounds_max=data.bounds_max,
        resolution_m=data.resolution_m, generation_uuid=data.generation_uuid,
        stamp_s=data.stamp_s, motion_allowed=False)
    details = json.loads(response.message)
    details.update(roundtrip_s=elapsed, shape=list(data.sdf_voxels.shape),
                   observed_fraction=float(data.observed.mean()), motion_allowed=False)
    (output / 'capture.json').write_text(json.dumps(details, indent=2))
    print(json.dumps(details, indent=2))


if __name__ == '__main__':
    main()
