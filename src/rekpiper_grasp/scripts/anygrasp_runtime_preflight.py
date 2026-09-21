#!/usr/bin/env python3
"""Print a secret-free readiness report for licensed AnyGrasp detection."""

import argparse
import json
from pathlib import Path
import platform

from rekpiper_grasp.anygrasp_adapter import check_anygrasp_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--license-dir", required=True)
    arguments = parser.parse_args()
    try:
        import torch
        torch_report = {
            "version": str(torch.__version__),
            "compiled_cuda": str(torch.version.cuda),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": (int(torch.cuda.device_count())
                             if torch.cuda.is_available() else 0),
        }
    except Exception as exc:
        torch_report = {"error": type(exc).__name__, "cuda_available": False}
        torch = None
    result = check_anygrasp_runtime(
        arguments.sdk_root, arguments.checkpoint, arguments.license_dir,
        torch_module=torch)
    report = {
        "python": platform.python_version(),
        "python_executable": str(Path(__import__("sys").executable).absolute()),
        "torch": torch_report,
        "ready": result.ready,
        "reasons": list(result.reasons),
        "checkpoint_configured": bool(arguments.checkpoint),
        "planning_authorized": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
