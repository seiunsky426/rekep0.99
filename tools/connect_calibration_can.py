#!/usr/bin/env python3
"""Bring can0 up at 1 Mbit/s if DOWN; never restart an active bus or enable motors."""

import re
import subprocess


def main():
    def inspect():
        return subprocess.check_output(["ip", "-details", "link", "show", "dev", "can0"], text=True)

    text = inspect()
    flags = re.search(r"<([^>]+)>", text)
    if flags is None:
        raise SystemExit("无法识别can0状态")
    if "UP" not in flags.group(1).split(","):
        subprocess.run(["sudo", "ip", "link", "set", "can0", "up", "type", "can", "bitrate", "1000000"], check=True)
        text = inspect()
    print(text)
    if "can state ERROR-ACTIVE" not in text or not re.search(r"\bbitrate 1000000\b", text):
        raise SystemExit("CAN状态或速率不符合要求；未自动重启总线，请检查设备。")
    print("CAN ready: can0, 1000000 bit/s. No motor commands sent.")


if __name__ == "__main__":
    main()
