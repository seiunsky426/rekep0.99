"""Load the pinned official ReKep numerical core.

Source classification: NEW adapter.
Upstream algorithm source:
huangwl18/ReKep@63c43fdba60354980258beaeb8a7d48e088e1e3e.

The official repository is a flat Python directory rather than an installable
package. This module narrowly adapts that layout while verifying that imported
modules really come from the fixed snapshot.
"""

from dataclasses import dataclass
import importlib
import hashlib
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Optional


EXPECTED_OFFICIAL_COMMIT = "63c43fdba60354980258beaeb8a7d48e088e1e3e"
_OFFICIAL_ENV = "REKEP_OFFICIAL_ROOT"
EXPECTED_SOURCE_HASHES = {
    "constraint_generation.py": "4dc2f0506a92a5d5e6798a3a411da073d61b07c3cf46f4fd430c27e549b991ef",
    "keypoint_proposal.py": "de120fe4365a735a614de50d3fae948926ae8b058882447b8372c5ab1ba5a04e",
    "path_solver.py": "a6f858c57a921f6d47594b234413ee1afe11cf86cdcdff80d1577f3fbece0cd2",
    "subgoal_solver.py": "dbb0b69b44c4a7a167f1c0686214eef0d47eeffbdebad6b5ce5bfcd56a852218",
    "transform_utils.py": "ca0a0ecafbd193bf05a471e407e4d9cfbfab6ed1dc9b9262fdadc689a6dc5d12",
    "utils.py": "5e2ab110e941f044b5afefa40c53fa89caa2a21e96add669af7b815b535ceb66",
    "vlm_query/prompt_template.txt": "0cda5043e2353e20a3e11c8a830f76bcf1441b7878af288452949898b8c078a0",
}
_MINIMUM_NUMBA_RECURSION_LIMIT = 10000


@dataclass(frozen=True)
class OfficialCoreModules:
    """References to unmodified algorithm modules from the official snapshot."""

    transform_utils: ModuleType
    utils: ModuleType
    subgoal_solver: ModuleType
    path_solver: ModuleType


def resolve_official_root(explicit_root: Optional[str] = None) -> Path:
    """Resolve and validate the flat official ReKep source directory."""
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
    elif os.environ.get(_OFFICIAL_ENV):
        root = Path(os.environ[_OFFICIAL_ENV]).expanduser().resolve()
    else:
        source = Path(__file__).resolve()
        workspace = next((parent for parent in source.parents
                          if (parent / ".catkin_workspace").is_file()), None)
        if workspace is None:
            raise FileNotFoundError(
                "cannot discover ReKpiper workspace root from " + str(source))
        root = workspace / "third_party" / "ReKep"

    missing = [name for name in EXPECTED_SOURCE_HASHES
               if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "official ReKep snapshot is incomplete at {}: missing {}".format(
                root, ", ".join(missing)
            )
        )
    mismatched = []
    for name, expected in EXPECTED_SOURCE_HASHES.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            mismatched.append(name)
    if mismatched:
        raise RuntimeError(
            "official ReKep source hash mismatch: {}".format(
                ", ".join(mismatched)))
    return root


def _assert_origin(module: ModuleType, root: Path) -> None:
    source = Path(module.__file__).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ImportError(
            "module {!r} resolved outside the fixed official snapshot: {}".format(
                module.__name__, source
            )
        ) from exc


def load_official_core(
    explicit_root: Optional[str] = None,
    require_exact_commit: bool = True,
) -> OfficialCoreModules:
    """Load official transform, utility, subgoal, and path modules.

    The snapshot path is prepended because the official files use sibling
    absolute imports. Origin checks prevent an unrelated top-level ``utils``
    module on ROS/Python paths from being silently accepted.
    """
    root = resolve_official_root(explicit_root)
    # Exported production source deliberately has no nested repository.  The
    # byte hashes above are the exact snapshot identity; retain the argument
    # for API compatibility with earlier adapters.
    _ = require_exact_commit

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    # Numba 0.57 serializes the official dynamic-assert lowering graph through
    # cloudpickle.  Python's default recursion limit is too small for that
    # compiler graph, although the byte-exact function itself is valid.  This
    # process-level adapter setting leaves upstream source and numerics intact.
    if sys.getrecursionlimit() < _MINIMUM_NUMBA_RECURSION_LIMIT:
        sys.setrecursionlimit(_MINIMUM_NUMBA_RECURSION_LIMIT)
    importlib.import_module("numba")
    importlib.import_module("open3d")

    loaded = {}
    for name in ("transform_utils", "utils", "subgoal_solver", "path_solver"):
        module = importlib.import_module(name)
        _assert_origin(module, root)
        loaded[name] = module

    return OfficialCoreModules(**loaded)
