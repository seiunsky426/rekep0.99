#!/usr/bin/env python3
"""Create an immutable hash manifest for a Git checkout or source archive."""

import argparse
import hashlib
import json
from pathlib import Path


EXCLUDED_PARTS = {".git", "build", "devel", "install", "runtime", "__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.source_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit("refusing to overwrite source manifest")
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (not path.is_file() or any(part in EXCLUDED_PARTS for part in relative.parts)
                or path.suffix in EXCLUDED_SUFFIXES or path.resolve() == output):
            continue
        files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"schema_version": 1, "files": files}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["tree_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
