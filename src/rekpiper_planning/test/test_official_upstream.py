#!/usr/bin/env python3

import hashlib
from pathlib import Path
import unittest

import yaml

from rekpiper_planning import EXPECTED_OFFICIAL_COMMIT
from rekpiper_planning.upstream import resolve_official_root


class OfficialUpstreamTest(unittest.TestCase):
    def test_pinned_official_core_loads_and_hashes_match(self):
        workspace = Path(__file__).resolve().parents[3]
        manifest = yaml.safe_load(
            (workspace / "third_party" / "UPSTREAM.lock.yaml").read_text())
        self.assertEqual(manifest["rekep"]["commit"], EXPECTED_OFFICIAL_COMMIT)
        root = workspace / "third_party" / "ReKep"
        for relative, expected in manifest["rekep"]["files"].items():
            digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
            self.assertEqual(digest, expected)
        self.assertEqual(resolve_official_root(), root)


if __name__ == "__main__":
    unittest.main()
