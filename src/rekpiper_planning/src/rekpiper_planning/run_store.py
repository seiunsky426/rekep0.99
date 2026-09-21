"""统一管理 ReKep 四层的运行数据目录。"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


class RunStore:
    """在一个固定根目录下创建不会互相覆盖的运行记录。"""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run(self, layer: str, label: str) -> Path:
        """创建 ``data/<layer>/<UTC时间>_<label>/`` 并返回它。"""
        clean_layer = self._safe_name(layer)
        clean_label = self._safe_name(label)[:48] or "run"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = self.root / clean_layer / (timestamp + "_" + clean_label)
        target.mkdir(parents=True, exist_ok=False)
        return target

    @staticmethod
    def write_json(path, payload: Any) -> None:
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n", encoding="utf-8")

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
