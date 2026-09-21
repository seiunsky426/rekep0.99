#!/usr/bin/env python3
"""Atomically advance the root-managed minimum accepted release counter."""

import argparse
import os
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_file")
    parser.add_argument("counter", type=int)
    args = parser.parse_args()
    target = Path(args.state_file).expanduser().resolve()
    if args.counter < 0:
        raise SystemExit("counter must be non-negative")
    current = int(target.read_text().strip()) if target.exists() else -1
    if args.counter <= current:
        raise SystemExit("release counter must advance monotonically")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=str(target.parent))
    try:
        data = (str(args.counter) + "\n").encode("ascii")
        written = 0
        while written < len(data):
            written += os.write(descriptor, data[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
        directory = os.open(str(target.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
