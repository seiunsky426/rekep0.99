"""官方 ReKep 风格 Python 约束的受限生成、存储和加载。

官方项目把 VLM 的 Python 分别保存为 ``stageN_*_constraints.txt``，并在
运行时 ``exec`` 为 callable。本模块保持该磁盘格式，但在 exec 之前用 AST
白名单验证：模型代码不能导入模块、访问文件/网络，也不能调用任意内建函数。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .upstream import EXPECTED_SOURCE_HASHES, resolve_official_root


class OfficialProgramError(ValueError):
    """VLM Python 不是可接受的官方 ReKep 约束程序。"""


def compute_program_sha256(directory: str) -> str:
    """Hash every file that can affect execution of an approved program."""
    root = Path(directory)
    names = ["program.py", "metadata.json"]
    names.extend(sorted(
        path.name for path in root.glob("stage*_constraints.txt")
        if path.is_file()))
    if len(names) < 2:
        raise OfficialProgramError("program files are unavailable")
    digest = hashlib.sha256()
    for name in names:
        path = root / name
        if not path.is_file():
            raise OfficialProgramError("program file is unavailable: {}".format(name))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


_MAXIMUM_STAGES = 32
_FUNCTION_RE = re.compile(
    r"^stage([1-9][0-9]*)_(subgoal|path)_constraint([1-9][0-9]*)$")
_ALLOWED_NP_CALLS = {
    "abs", "arccos", "array", "clip", "cos", "cross", "dot", "maximum",
    "mean", "minimum", "norm", "sin", "sqrt", "sum", "stack",
    "concatenate", "vstack", "hstack", "min", "max",
}
_ALLOWED_NP_VALUES = {"pi"}
LOCAL_SAFETY_CONTRACT_VERSION = "rekpiper_vector_ast_v1"
OFFICIAL_PROMPT_SHA256 = EXPECTED_SOURCE_HASHES["vlm_query/prompt_template.txt"]
_ALLOWED_NP_KEYWORDS = {
    "mean": {"axis", "keepdims"},
    "sum": {"axis", "keepdims"},
    "min": {"axis", "keepdims"},
    "max": {"axis", "keepdims"},
    "norm": {"ord", "axis", "keepdims"},
    "stack": {"axis"},
    "concatenate": {"axis"},
    "vstack": set(),
    "hstack": set(),
    "clip": set(),
    "array": set(),
    "abs": set(),
    "arccos": set(),
    "cos": set(),
    "cross": set(),
    "dot": set(),
    "maximum": set(),
    "minimum": set(),
    "sin": set(),
    "sqrt": set(),
}


def _strip_python_fence(text: str) -> str:
    """仅接受裸 Python 或唯一的 python 代码块。"""
    value = str(text).strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if not lines or lines[0].strip().lower() not in ("```python", "```py", "```"):
        raise OfficialProgramError("VLM code fence must be marked python")
    if len(lines) < 2 or lines[-1].strip() != "```":
        raise OfficialProgramError("VLM Python code block is not closed")
    return "\n".join(lines[1:-1]).strip()


def _np_name(node: ast.AST) -> Optional[str]:
    """返回 ``np.linalg.norm`` 的末级名字；拒绝非 np 属性链。"""
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        if current.attr.startswith("_"):
            return None
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name) or current.id != "np":
        return None
    return parts[0] if parts else None


class _FunctionValidator(ast.NodeVisitor):
    """一个很小的数值表达式子集，足以覆盖官方 prompt 的约束写法。"""

    def __init__(self, function: ast.FunctionDef):
        self.function = function
        self.locals = {"end_effector", "keypoints"}

    def fail(self, message: str) -> None:
        raise OfficialProgramError("{}: {}".format(self.function.name, message))

    def validate(self) -> None:
        args = self.function.args
        if (args.posonlyargs or args.vararg or args.kwarg or args.kwonlyargs
                or len(args.args) != 2
                or [item.arg for item in args.args] != ["end_effector", "keypoints"]):
            self.fail("signature must be (end_effector, keypoints)")
        if not self.function.body:
            self.fail("function body is empty")
        returns = 0
        for statement in self.function.body:
            if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)):
                continue
            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                    self.fail("only a single local variable assignment is allowed")
                name = statement.targets[0].id
                if name.startswith("_") or name in {"end_effector", "keypoints", "np"}:
                    self.fail("invalid local variable name")
                self.visit(statement.value)
                self.locals.add(name)
                continue
            if isinstance(statement, ast.Return):
                returns += 1
                self.visit(statement.value)
                continue
            control_flow = (ast.For, ast.AsyncFor, ast.While, ast.If,
                            ast.With, ast.Try)
            match_statement = getattr(ast, "Match", None)
            if (isinstance(statement, control_flow)
                    or (match_statement is not None
                        and isinstance(statement, match_statement))):
                self.fail(
                    "control flow is forbidden (use local numerical "
                    "assignments followed by one vectorized return expression)")
            self.fail("only local assignments and one return expression are allowed")
        if returns != 1 or not isinstance(self.function.body[-1], ast.Return):
            self.fail("function must finish with exactly one return")

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            return
        if node.id not in self.locals | {"np", "get_grasping_cost_by_keypoint_idx"}:
            self.fail("unknown name {!r}".format(node.id))
        if node.id.startswith("__"):
            self.fail("dunder names are forbidden")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        leaf = _np_name(node)
        if leaf is None or (leaf not in _ALLOWED_NP_CALLS and leaf not in _ALLOWED_NP_VALUES):
            self.fail("only whitelisted numpy attributes are allowed")

    def visit_Call(self, node: ast.Call) -> None:
        leaf = None
        if isinstance(node.func, ast.Name):
            if node.func.id != "get_grasping_cost_by_keypoint_idx":
                self.fail("only get_grasping_cost_by_keypoint_idx may be called directly")
            if node.keywords:
                self.fail("grasping-cost call does not accept keyword arguments")
        elif isinstance(node.func, ast.Attribute):
            leaf = _np_name(node.func)
            if leaf not in _ALLOWED_NP_CALLS:
                self.fail("numpy call is not whitelisted")
            allowed = _ALLOWED_NP_KEYWORDS.get(leaf, set())
            for keyword in node.keywords:
                if keyword.arg is None or keyword.arg not in allowed:
                    self.fail(
                        "numpy {} keyword {!r} is not allowed".format(
                            leaf, keyword.arg))
                if keyword.arg in {"axis", "ord"}:
                    if not (isinstance(keyword.value, ast.Constant)
                            and (keyword.value.value is None
                                 or (isinstance(keyword.value.value, (int, float))
                                     and not isinstance(keyword.value.value, bool)))):
                        self.fail("{} must be a numeric literal or None".format(
                            keyword.arg))
                elif keyword.arg == "keepdims":
                    if not (isinstance(keyword.value, ast.Constant)
                            and isinstance(keyword.value.value, bool)):
                        self.fail("keepdims must be a boolean literal")
        else:
            self.fail("indirect function calls are forbidden")
        for argument in node.args:
            self.visit(argument)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if not isinstance(node.value, ast.Name) or node.value.id not in self.locals:
            self.fail("subscript base must be a function argument or local array")
        self.visit(node.slice)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            self.fail("unsupported arithmetic operator")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            self.fail("unsupported unary operator")
        self.visit(node.operand)

    def visit_List(self, node: ast.List) -> None:
        for element in node.elts:
            self.visit(element)

    visit_Tuple = visit_List

    def visit_Slice(self, node: ast.Slice) -> None:
        for item in (node.lower, node.upper, node.step):
            if item is not None:
                self.visit(item)

    # Python 3.8 wraps ``keypoints[0]`` in ast.Index; 3.9 removed it.
    def visit_Index(self, node) -> None:  # pragma: no cover - version dependent
        self.visit(node.value)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, (bool, str, bytes, type(None), complex)):
            self.fail("only finite numeric constants are allowed in expressions")
        if not isinstance(node.value, (int, float)) or not math.isfinite(float(node.value)):
            self.fail("numeric constant must be finite")

    def generic_visit(self, node: ast.AST) -> None:
        self.fail("unsupported syntax {}".format(type(node).__name__))


def _literal_int_list(node: ast.AST, name: str, num_stages: int,
                      num_keypoints: int) -> List[int]:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError) as exc:
        raise OfficialProgramError("{} must be a literal integer list".format(name)) from exc
    if (not isinstance(value, list) or len(value) != num_stages
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)):
        raise OfficialProgramError("{} must have {} integer entries".format(name, num_stages))
    if any(item < -1 or item >= num_keypoints for item in value):
        raise OfficialProgramError("{} contains a keypoint index outside [-1, {})".format(
            name, num_keypoints))
    return value


@dataclass
class OfficialConstraintProgram:
    instruction: str
    source: str
    num_stages: int
    grasp_keypoints: List[int]
    release_keypoints: List[int]
    functions: Dict[int, Dict[str, List[Tuple[str, str]]]]

    def metadata(self, keypoints: Sequence[Sequence[float]]) -> Dict[str, Any]:
        return {
            "schema_version": "rekep_official_python_v1",
            "instruction": self.instruction,
            "num_stages": self.num_stages,
            "num_keypoints": len(keypoints),
            "init_keypoint_positions": np.asarray(keypoints, dtype=float).tolist(),
            "grasp_keypoints": self.grasp_keypoints,
            "release_keypoints": self.release_keypoints,
            "execution_authorized": False,
            "semantic_review_required": True,
            "official_prompt_sha256": OFFICIAL_PROMPT_SHA256,
            "local_safety_contract_version": LOCAL_SAFETY_CONTRACT_VERSION,
        }


def parse_official_program(text: str, instruction: str,
                           num_keypoints: int) -> OfficialConstraintProgram:
    """解析并验证 VLM 输出；尚未执行任何 Python。"""
    if not 1 <= int(num_keypoints):
        raise OfficialProgramError("at least one numbered keypoint is required")
    source = _strip_python_fence(text)
    if not source:
        raise OfficialProgramError("VLM returned empty Python")
    try:
        module = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise OfficialProgramError("VLM Python syntax error: {}".format(exc.msg)) from exc
    assignments: Dict[str, ast.AST] = {}
    raw_functions: List[ast.FunctionDef] = []
    allowed_assignments = {"num_stages", "grasp_keypoints", "release_keypoints"}
    for statement in module.body:
        if isinstance(statement, ast.FunctionDef):
            if statement.decorator_list or statement.returns or statement.type_comment:
                raise OfficialProgramError("constraint functions may not use decorators or annotations")
            raw_functions.append(statement)
        elif isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(
                statement.targets[0], ast.Name):
            name = statement.targets[0].id
            if name not in allowed_assignments or name in assignments:
                raise OfficialProgramError("unsupported or duplicated top-level assignment {!r}".format(name))
            assignments[name] = statement.value
        elif (isinstance(statement, ast.Expr)
              and isinstance(statement.value, ast.Constant)
              and isinstance(statement.value.value, str)):
            # The official prompt permits explanatory triple-quoted text.
            continue
        else:
            raise OfficialProgramError("only metadata assignments and constraint functions are allowed")
    if set(assignments) != allowed_assignments:
        raise OfficialProgramError("num_stages, grasp_keypoints and release_keypoints are all required")
    try:
        num_stages = ast.literal_eval(assignments["num_stages"])
    except (ValueError, TypeError) as exc:
        raise OfficialProgramError("num_stages must be an integer literal") from exc
    if (isinstance(num_stages, bool) or not isinstance(num_stages, int)
            or not 1 <= num_stages <= _MAXIMUM_STAGES):
        raise OfficialProgramError(
            "num_stages must be an integer in [1, {}]".format(
                _MAXIMUM_STAGES))
    grasp = _literal_int_list(assignments["grasp_keypoints"], "grasp_keypoints",
                              num_stages, num_keypoints)
    release = _literal_int_list(assignments["release_keypoints"], "release_keypoints",
                                num_stages, num_keypoints)
    held = None
    for stage, (grasp_index, release_index) in enumerate(zip(grasp, release), start=1):
        if grasp_index != -1 and release_index != -1:
            raise OfficialProgramError(
                "stage {} cannot both grasp and release".format(stage))
        if grasp_index != -1:
            if held is not None:
                raise OfficialProgramError("stage {} grasps while another object is held".format(stage))
            held = grasp_index
        if release_index != -1:
            if held != release_index:
                raise OfficialProgramError("stage {} releases a keypoint that is not held".format(stage))
            held = None
    functions: Dict[int, Dict[str, List[Tuple[str, str]]]] = {
        stage: {"subgoal": [], "path": []} for stage in range(1, num_stages + 1)}
    seen = set()
    for function in raw_functions:
        match = _FUNCTION_RE.fullmatch(function.name)
        if not match:
            raise OfficialProgramError("invalid constraint function name {!r}".format(function.name))
        stage, category, index = int(match.group(1)), match.group(2), int(match.group(3))
        if stage > num_stages:
            raise OfficialProgramError("{} refers to a nonexistent stage".format(function.name))
        if function.name in seen:
            raise OfficialProgramError("duplicated function {!r}".format(function.name))
        seen.add(function.name)
        _FunctionValidator(function).validate()
        snippet = ast.get_source_segment(source, function)
        if not snippet:
            raise OfficialProgramError("cannot recover source for {}".format(function.name))
        functions[stage][category].append((function.name, snippet.strip()))
    for stage in functions:
        for category in functions[stage]:
            functions[stage][category].sort(key=lambda item: int(_FUNCTION_RE.fullmatch(item[0]).group(3)))
        if grasp[stage - 1] != -1:
            if len(functions[stage]["subgoal"]) != 1:
                raise OfficialProgramError(
                    "grasp stage {} must contain exactly one subgoal constraint".format(
                        stage))
            if functions[stage]["path"]:
                raise OfficialProgramError(
                    "grasp stage {} may not contain path constraints".format(stage))
    return OfficialConstraintProgram(str(instruction).strip(), source, num_stages,
                                     grasp, release, functions)


def build_official_python_prompt(instruction: str,
                                 keypoints: Sequence[Sequence[float]],
                                 robot_capabilities: Dict[str, Any],
                                 scene_metadata: Optional[Dict[str, Any]] = None) -> str:
    del keypoints, robot_capabilities, scene_metadata
    template = (resolve_official_root() / "vlm_query" /
                "prompt_template.txt").read_text(encoding="utf-8")
    official = template.format(instruction=str(instruction).strip())
    local_contract = """

REKPIPER LOCAL EXECUTION CONTRACT ({version})
The generated program is executed by a restricted AST evaluator. Every
constraint must have exactly the signature (end_effector, keypoints), contain
only local single-name assignments, and finish with exactly one return.
Do not use imports, loops, comprehensions, conditionals, exceptions, attribute
access outside NumPy, native Python calls, or indirect calls. Use vectorized
arithmetic and only these NumPy operations: {calls}. The only direct callable
is get_grasping_cost_by_keypoint_idx(index). Supported NumPy keywords are
axis, keepdims, and ord only where applicable. Numeric literals must be finite.
""".format(
        version=LOCAL_SAFETY_CONTRACT_VERSION,
        calls=", ".join(sorted(_ALLOWED_NP_CALLS)))
    return official.rstrip() + local_contract


def generate_official_program(instruction: str, annotated_image_path: Optional[str],
                              scene_keypoints: Sequence[Sequence[float]],
                              robot_capabilities: Dict[str, Any], backend,
                              scene_metadata: Optional[Dict[str, Any]] = None) -> OfficialConstraintProgram:
    instruction = str(instruction).strip()
    if not instruction or len(instruction) > 1000:
        raise OfficialProgramError("instruction must contain 1..1000 characters")
    prompt = build_official_python_prompt(instruction, scene_keypoints,
                                          robot_capabilities, scene_metadata)
    if not hasattr(backend, "generate_text"):
        raise OfficialProgramError("VLM backend does not support Python constraint generation")
    output = backend.generate_text(prompt, annotated_image_path)
    return parse_official_program(output, instruction, len(scene_keypoints))


def save_official_program(program: OfficialConstraintProgram, directory,
                          keypoints: Sequence[Sequence[float]]) -> Path:
    """写成官方兼容的 metadata + 每阶段 txt 文件，供后续受限加载。"""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    stale_pattern = re.compile(
        r"^stage([1-9][0-9]*)_(subgoal|path)_constraints\.txt$")
    for stale in target.iterdir():
        if stale.is_file() and stale_pattern.fullmatch(stale.name):
            stale.unlink()
    (target / "program.py").write_text(program.source.rstrip() + "\n", encoding="utf-8")
    (target / "metadata.json").write_text(
        json.dumps(program.metadata(keypoints), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    for stage in range(1, program.num_stages + 1):
        for category in ("subgoal", "path"):
            snippets = [snippet for _name, snippet in program.functions[stage][category]]
            if snippets:
                (target / "stage{}_{}_constraints.txt".format(stage, category)).write_text(
                    "\n\n".join(snippets) + "\n", encoding="utf-8")
    return target


def export_stage_constraint_texts(program_directory, output_directory) -> List[Path]:
    """Mirror the validated stage txt files into one operator-facing folder."""
    source = Path(program_directory)
    target = Path(output_directory).expanduser().resolve()
    if not source.is_dir():
        raise OfficialProgramError("official program directory is unavailable")
    target.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(
        r"^stage([1-9][0-9]*)_(subgoal|path)_constraints\.txt$")
    stage_files = []
    for path in source.iterdir():
        match = pattern.fullmatch(path.name) if path.is_file() else None
        if match and int(match.group(1)) <= _MAXIMUM_STAGES:
            stage_files.append(path)
    stage_files.sort(key=lambda path: (
        int(pattern.fullmatch(path.name).group(1)),
        pattern.fullmatch(path.name).group(2)))
    if not stage_files:
        raise OfficialProgramError("official program has no stage constraint txt")
    for stale in target.iterdir():
        if stale.is_file() and pattern.fullmatch(stale.name):
            stale.unlink()
    exported = []
    for stage_file in stage_files:
        destination = target / stage_file.name
        temporary = target / (stage_file.name + ".tmp")
        temporary.write_text(
            stage_file.read_text(encoding="utf-8"), encoding="utf-8")
        temporary.replace(destination)
        exported.append(destination)
    return exported


def _load_stage_file(path: Path, stage: int, category: str, num_keypoints: int,
                     grasping_cost_fn: Callable[[int], float]) -> List[Callable]:
    if not path.exists():
        return []
    source = path.read_text(encoding="utf-8")
    try:
        module = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise OfficialProgramError("{} syntax error: {}".format(path.name, exc.msg)) from exc
    if not module.body or any(not isinstance(item, ast.FunctionDef) for item in module.body):
        raise OfficialProgramError("{} may contain only constraint functions".format(path.name))
    functions = []
    for item in module.body:
        match = _FUNCTION_RE.fullmatch(item.name)
        if not match or int(match.group(1)) != stage or match.group(2) != category:
            raise OfficialProgramError("{} has a function in the wrong stage/category".format(path.name))
        _FunctionValidator(item).validate()
        functions.append(item)
    names = [item.name for item in functions]
    if len(names) != len(set(names)):
        raise OfficialProgramError("{} contains duplicate function names".format(path.name))
    namespace: Dict[str, Any] = {}
    safe_globals = {
        "__builtins__": {}, "np": np,
        "get_grasping_cost_by_keypoint_idx": grasping_cost_fn,
    }
    # exec 的输入已经过完整 AST 白名单校验；globals 没有内建函数。
    exec(compile(module, str(path), "exec"), safe_globals, namespace)
    ordered = sorted(names, key=lambda name: int(_FUNCTION_RE.fullmatch(name).group(3)))
    return [namespace[name] for name in ordered]


def load_official_stage_constraints(directory, stage: int, num_keypoints: int,
                                    grasping_cost_fn: Callable[[int], float]) -> Tuple[List[Callable], List[Callable]]:
    """从官方 ``stage*_constraints.txt`` 受限 exec 成求解器的 callable。"""
    if not 1 <= int(stage) <= _MAXIMUM_STAGES:
        raise OfficialProgramError(
            "stage must be in [1, {}]".format(_MAXIMUM_STAGES))
    root = Path(directory)
    return (
        _load_stage_file(root / "stage{}_subgoal_constraints.txt".format(stage),
                         stage, "subgoal", num_keypoints, grasping_cost_fn),
        _load_stage_file(root / "stage{}_path_constraints.txt".format(stage),
                         stage, "path", num_keypoints, grasping_cost_fn),
    )


def validate_official_program_numerics(directory, program: OfficialConstraintProgram,
                                       keypoints: Sequence[Sequence[float]],
                                       end_effector: Sequence[float],
                                       samples: int = 16) -> Dict[str, Any]:
    """在真实编号点附近扰动，确认每个受限 exec callable 都返回有限标量。"""
    points = np.asarray(keypoints, dtype=float)
    ee = np.asarray(end_effector, dtype=float)
    if points.shape != (len(points), 3) or ee.shape != (3,):
        raise OfficialProgramError("invalid keypoint or end-effector geometry")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(ee)):
        raise OfficialProgramError("non-finite planning geometry")
    rng = np.random.RandomState(0)
    results = []
    for stage in range(1, program.num_stages + 1):
        subgoals, paths = load_official_stage_constraints(
            directory, stage, len(points), lambda _index: 1.0)
        for category, functions in (("subgoal", subgoals), ("path", paths)):
            for index, function in enumerate(functions, start=1):
                values = []
                for sample in range(max(1, int(samples))):
                    jitter = rng.uniform(-0.01, 0.01, size=points.shape) if sample else 0.0
                    value = float(function(ee, points + jitter))
                    if not math.isfinite(value) or abs(value) > 1e6:
                        raise OfficialProgramError(
                            "stage {} {} constraint {} is numerically unsafe".format(
                                stage, category, index))
                    values.append(value)
                results.append({"stage": stage, "category": category,
                                "index": index, "current_value": values[0],
                                "sample_min": min(values), "sample_max": max(values)})
    return {
        "schema_valid": True, "ast_whitelist_valid": True,
        "numeric_stress_valid": True, "constraint_count": len(results),
        "constraint_results": results, "semantic_review_required": True,
        "planning_authorized": False,
    }
