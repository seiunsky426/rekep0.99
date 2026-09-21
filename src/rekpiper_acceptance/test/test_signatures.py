#!/usr/bin/env python3

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import yaml

from rekpiper_acceptance import (
    AcceptanceError, assert_release_unchanged, create_signed_artifact,
    load_signed_artifact, sha256_file,
    validate_program_approval, validate_release_bundle,
    verify_signed_artifact)
from rekpiper_acceptance.signed_artifact import EXPECTED_OFFICIAL_REKEP_COMMIT


class SignatureTest(unittest.TestCase):
    def keys(self, root):
        private = Ed25519PrivateKey.generate()
        private_path, public_path = root / "private.pem", root / "public.pem"
        private_path.write_bytes(private.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()))
        public_path.write_bytes(private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo))
        return private_path, public_path

    def test_payload_and_signature_tampering_fail(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            private, public = self.keys(root)
            document = create_signed_artifact(
                "workspace", "w1", {"table_height_m": 0.0}, [],
                "operator", str(private))
            verify_signed_artifact(document, str(public), expected_type="workspace")
            changed = deepcopy(document)
            changed["payload"]["table_height_m"] = 1.0
            with self.assertRaises(AcceptanceError):
                verify_signed_artifact(changed, str(public))
            changed = deepcopy(document)
            changed["approval"]["operator"] = "somebody_else"
            with self.assertRaises(AcceptanceError):
                verify_signed_artifact(changed, str(public))

    def test_evidence_and_post_validation_changes_fail(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            private, public = self.keys(root)
            evidence = root / "dataset.json"
            evidence.write_text("{}\n", encoding="utf-8")
            manifest = root / "artifact.yaml"
            manifest.write_text(yaml.safe_dump(create_signed_artifact(
                "test", "test-1", {"accepted": True}, [{
                    "role": "dataset", "path": evidence.name,
                    "sha256": sha256_file(str(evidence))}],
                "operator", str(private)), sort_keys=False), encoding="utf-8")
            loaded = load_signed_artifact(str(manifest), str(public), "test")
            loaded["_validated_files"] = {
                str(evidence): sha256_file(str(evidence))}
            stat = evidence.stat()
            loaded["_validated_stats"] = {str(evidence): (
                stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
                stat.st_ctime_ns)}
            assert_release_unchanged(loaded)
            evidence.write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaises(AcceptanceError):
                assert_release_unchanged(loaded)
            with self.assertRaises(AcceptanceError):
                load_signed_artifact(str(manifest), str(public), "test")

    def test_bundle_checks_counter_hash_and_robot(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            private, public = self.keys(root)
            required = {"test_alpha", "test_beta"}
            entries = {}
            for name in required:
                path = root / (name + ".yaml")
                path.write_text(yaml.safe_dump(create_signed_artifact(
                    name, name + "-1", {"approved": True}, [], "operator",
                    str(private)), sort_keys=False), encoding="utf-8")
                entries[name] = {"path": path.name, "sha256": sha256_file(str(path))}
            binding_evidence = []
            bindings = {}
            for index, role in enumerate((
                    "source_manifest", "urdf", "system_config",
                    "runtime_fingerprint"), 1):
                path = root / (role + ".txt")
                path.write_text(str(index), encoding="utf-8")
                digest = sha256_file(str(path))
                binding_evidence.append({"role": role, "path": path.name,
                                         "sha256": digest})
                bindings[role + "_sha256"] = digest
            bundle_path = root / "release.yaml"
            bundle_path.write_text(yaml.safe_dump(create_signed_artifact(
                "release_bundle", "release-7", {
                    "release_counter": 7, "robot_id": "piper-a",
                    "official_rekep_commit": EXPECTED_OFFICIAL_REKEP_COMMIT,
                    "bindings": bindings,
                    "artifacts": entries}, binding_evidence, "operator", str(private)),
                sort_keys=False), encoding="utf-8")
            counter = root / "minimum"; counter.write_text("7\n")
            validate_release_bundle(
                str(bundle_path), str(public), str(counter), "piper-a", required,
                source_root="")
            counter.write_text("8\n")
            with self.assertRaisesRegex(AcceptanceError, "counter"):
                validate_release_bundle(
                    str(bundle_path), str(public), str(counter), "piper-a", required,
                    source_root="")
            counter.write_text("6\n")
            with self.assertRaisesRegex(AcceptanceError, "counter"):
                validate_release_bundle(
                    str(bundle_path), str(public), str(counter), "piper-a", required,
                    source_root="")

    def test_program_manifest_binds_serialized_snapshot(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            private, public = self.keys(root)
            snapshot = root / "scene_snapshot.rosmsg"
            snapshot.write_bytes(b"immutable serialized snapshot")
            snapshot_hash = sha256_file(str(snapshot))
            manifest = root / "program_approval.yaml"
            payload = {
                "session_id": "task-1", "snapshot_id": "snap-1",
                "snapshot_sha256": snapshot_hash,
                "program_sha256": "a" * 64,
                "official_prompt_sha256": "b" * 64,
                "local_safety_contract_version": "ast-v1",
                "annotated_image_sha256": "c" * 64,
                "raw_response_sha256": "d" * 64,
                "effective_prompt_sha256": "e" * 64,
            }
            manifest.write_text(yaml.safe_dump(create_signed_artifact(
                "rekep_program", "task-1", payload, [{
                    "role": "scene_snapshot_rosmsg", "path": snapshot.name,
                    "sha256": snapshot_hash}], "operator", str(private)),
                sort_keys=False), encoding="utf-8")
            validate_program_approval(
                str(manifest), str(public), "task-1", "snap-1", "a" * 64,
                "b" * 64, "ast-v1", snapshot_hash)
            snapshot.write_bytes(b"changed")
            with self.assertRaises(AcceptanceError):
                validate_program_approval(
                    str(manifest), str(public), "task-1", "snap-1", "a" * 64,
                    "b" * 64, "ast-v1", snapshot_hash)


if __name__ == "__main__":
    unittest.main()
