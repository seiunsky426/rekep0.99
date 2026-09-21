#!/usr/bin/env python3
"""Disable all six Piper drives and verify feedback; never send a pose command."""

import argparse
import re
import subprocess
import time

from piper_sdk import C_PiperInterface_V2


def require_can_ready(interface):
    result = subprocess.run(
        ["ip", "-details", "link", "show", "dev", interface],
        capture_output=True, text=True, check=False, timeout=2)
    flags = re.search(r"<([^>]+)>", result.stdout)
    if (result.returncode != 0 or flags is None
            or "UP" not in flags.group(1).split(",")
            or "can state ERROR-ACTIVE" not in result.stdout
            or not re.search(r"\bbitrate 1000000\b", result.stdout)):
        raise SystemExit("can0 must be UP, ERROR-ACTIVE, 1000000 bit/s")


def motor_states(driver):
    feedback = driver.GetArmLowSpdInfoMsgs()
    return [bool(getattr(feedback, "motor_{}".format(index))
                 .foc_status.driver_enable_status)
            for index in range(1, 7)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can-port", default="can0")
    args = parser.parse_args()
    require_can_ready(args.can_port)
    driver = C_PiperInterface_V2(can_name=args.can_port)
    try:
        driver.ConnectPort(piper_init=False)
        deadline = time.monotonic()+5.0
        while time.monotonic() < deadline:
            driver.DisableArm(7)
            time.sleep(0.10)
            states = motor_states(driver)
            print("motor_enable=" + "".join("1" if state else "0" for state in states))
            if not any(states):
                print("PASS: all six Piper motor drives are disabled")
                return 0
        raise SystemExit("FAIL: disable sent but six-motor feedback did not become 000000")
    finally:
        driver.DisconnectPort()


if __name__ == "__main__":
    raise SystemExit(main())
