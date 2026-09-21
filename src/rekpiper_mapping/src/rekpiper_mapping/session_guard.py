"""Linux process lock used before any dual-hand-eye hardware node starts."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import tempfile
from datetime import datetime, timezone


class SessionAlreadyRunning(RuntimeError):
    """Raised when another process still owns the calibration session lock."""


class CANInterfaceNotReady(RuntimeError):
    """Raised before Piper startup when the required SocketCAN link is unsafe."""


def parse_can_link_details(output):
    """Parse the small, stable subset of ``ip -details link`` needed by Piper."""
    text = str(output)
    flags_match = re.search(r"<([^>]+)>", text)
    state_match = re.search(r"\bcan state\s+([A-Z-]+)", text)
    bitrate_match = re.search(r"\bbitrate\s+(\d+)", text)
    if not flags_match or not state_match or not bitrate_match:
        raise CANInterfaceNotReady("could not parse SocketCAN state from ip output")
    return {
        "flags": set(flags_match.group(1).split(",")),
        "can_state": state_match.group(1),
        "bitrate": int(bitrate_match.group(1)),
    }


def require_can_interface_ready(interface, expected_bitrate, run=subprocess.run):
    """Fail closed before any Piper hardware process is created."""
    name = str(interface)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise CANInterfaceNotReady("invalid CAN interface name")
    try:
        completed = run(
            ["ip", "-details", "link", "show", "dev", name],
            capture_output=True, text=True, check=False, timeout=2.0)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CANInterfaceNotReady("could not inspect {}: {}".format(name, exc))
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "interface missing").strip()
        raise CANInterfaceNotReady("{} is unavailable: {}".format(name, detail))
    details = parse_can_link_details(completed.stdout)
    if "UP" not in details["flags"]:
        raise CANInterfaceNotReady("{} is not UP".format(name))
    if details["can_state"] != "ERROR-ACTIVE":
        raise CANInterfaceNotReady(
            "{} CAN state is {}, expected ERROR-ACTIVE".format(
                name, details["can_state"]))
    if details["bitrate"] != int(expected_bitrate):
        raise CANInterfaceNotReady(
            "{} bitrate is {}, expected {}".format(
                name, details["bitrate"], int(expected_bitrate)))
    return details


def default_lock_path() -> Path:
    return Path(tempfile.gettempdir()) / (
        "rekep_dual_aruco_handeye_uid_{}.lock".format(os.getuid()))


def process_snapshot(proc_root=Path("/proc")):
    """Return {pid: {ppid, command}} while tolerating process churn."""
    result = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace").strip()
            result[int(entry.name)] = {
                "ppid": int(stat_fields[1]), "command": command,
            }
        except (IndexError, OSError, ValueError):
            continue
    return result


def process_tree_contains_rosmaster(root_pid, processes):
    """Whether a roslaunch process owns a rosmaster descendant."""
    descendants = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, details in processes.items():
            if pid not in descendants and details.get("ppid") in descendants:
                descendants.add(pid)
                changed = True
    for pid in descendants:
        command = str(processes.get(pid, {}).get("command", ""))
        tokens = command.split()
        if any(Path(token).name == "rosmaster" for token in tokens):
            return True
        if "-m rosmaster" in command:
            return True
    return False


class ExclusiveSessionLock:
    """Advisory lock whose file descriptor can be inherited by roslaunch."""

    def __init__(self, path=None):
        self.path = Path(path) if path else default_lock_path()
        self._fd = None

    @property
    def fileno(self):
        if self._fd is None:
            raise RuntimeError("session lock is not held")
        return self._fd

    def acquire(self, metadata=None):
        if self._fd is not None:
            raise RuntimeError("session lock is already held by this process")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(self.path), flags, 0o600)
        details = os.fstat(fd)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid():
            os.close(fd)
            raise RuntimeError("unsafe session lock file ownership or type")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.lseek(fd, 0, os.SEEK_SET)
            owner = os.read(fd, 8192).decode("utf-8", errors="replace").strip()
            os.close(fd)
            raise SessionAlreadyRunning(
                "dual hand-eye session already running: " + (owner or "owner unknown")) from exc
        payload = {
            "pid": os.getpid(),
            "uid": os.getuid(),
            "hostname": socket.gethostname(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        payload.update(metadata or {})
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, encoded)
        os.fsync(fd)
        self._fd = fd
        return payload

    def release(self):
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.release()
