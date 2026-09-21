#!/usr/bin/env python3
"""Generate an offline Ed25519 approval key pair without overwriting files."""

import argparse
import json
from pathlib import Path
import sys

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("private_key")
    parser.add_argument("public_key")
    args = parser.parse_args()
    private_path = Path(args.private_key).expanduser()
    public_path = Path(args.public_key).expanduser()
    if private_path.exists() or public_path.exists():
        print(json.dumps({"success": False, "error": "key output already exists"}))
        return 2
    key = Ed25519PrivateKey.generate()
    private_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    private_path.chmod(0o600)
    public_path.write_bytes(key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo))
    print(json.dumps({"success": True, "private_key": str(private_path),
                      "public_key": str(public_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
