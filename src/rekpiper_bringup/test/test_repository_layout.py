#!/usr/bin/env python3

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RepositoryLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[3]

    def test_repository_audit_passes(self):
        command = [
            sys.executable, str(self.root / "tools/check_repository.py"),
            "--source-root", str(self.root)]
        for _index in range(2):
            result = subprocess.run(
                command, check=False, capture_output=True, text=True)
            report = json.loads(result.stdout)
            self.assertEqual(result.returncode, 0, report.get("errors"))
            self.assertTrue(report["clean"])

    def test_setup_has_one_external_runtime_contract(self):
        setup = (self.root / "setup.bash").read_text(encoding="utf-8")
        self.assertIn(
            "REKPIPER_RUNTIME_ROOT:-${REKPIPER_ROOT}/runtime", setup)
        self.assertIn("REKPIPER_SITE_CONFIG_ROOT", setup)
        self.assertIn("REKPIPER_TRUST_ROOT", setup)
        for obsolete in ("/.venv", "/.runtime", "third_party/local",
                         "_REKPIPER_EXTERNAL"):
            self.assertNotIn(obsolete, setup)

    def test_repository_audit_rejects_nested_or_missing_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "src" / "src" / "nested_package"
            nested.mkdir(parents=True)
            (nested / "package.xml").write_text(
                "<package format=\"2\"></package>\n", encoding="utf-8")
            result = subprocess.run([
                sys.executable, str(self.root / "tools/check_repository.py"),
                "--source-root", str(root),
            ], check=False, capture_output=True, text=True)
            report = json.loads(result.stdout)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("nested_source_space_not_allowed", report["errors"])
            self.assertIn("ros_packages_missing", report["errors"])

    def test_empty_git_directory_is_not_a_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            result = subprocess.run([
                sys.executable, str(self.root / "tools/check_repository.py"),
                "--source-root", str(root),
            ], check=False, capture_output=True, text=True)
            report = json.loads(result.stdout)
            self.assertTrue(report["source_archive"])

    def test_checkout_or_source_archive_has_canonical_contract(self):
        for name in (
                ".catkin_workspace", ".gitignore", "README.md", "docs",
                "requirements.in", "requirements.lock", "setup.bash", "src", "third_party",
                "tools"):
            self.assertTrue((self.root / name).exists(), name)
        self.assertFalse((self.root / "src" / "src").exists())
        self.assertTrue(any((self.root / "src").glob("*/package.xml")))


if __name__ == "__main__":
    unittest.main()
