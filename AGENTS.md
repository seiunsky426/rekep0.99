# Repository Guidelines

## Project Structure & Module Organization

ReKpiper is a ROS1 Noetic Catkin workspace. Keep every package directly under
`src/<package>`; a nested `src/src` layout fails the repository audit.
`rekpiper_*` packages cover acceptance, bringup, calibration, camera,
execution, grasping, mapping, perception, and planning; `piper*` contains the
driver, descriptions, and messages. Python packages normally use
`src/<pkg>/src/<python_pkg>/`, with nodes in `scripts/`, ROS assets in
`launch/` and `config/`, and tests in `test/`. `third_party/ReKep` is a
hash-pinned upstream snapshot: make adaptations in `rekpiper_*`, not in that
directory. Deployment contracts live in `docs/`; repository audits live in
`tools/`.

## Build, Test, and Development Commands

Target Ubuntu 20.04, ROS Noetic, and Python 3.8. Set up and build with:

```bash
python3 -m venv runtime && source runtime/bin/activate
pip install --require-hashes -r requirements.lock
source /opt/ros/noetic/setup.bash
rosdep install --from-paths src --ignore-src -r -y
catkin_make && source devel/setup.bash
```

Validate a change with `python3 tools/check_repository.py --source-root
"$PWD"`, `python3 tools/check_launch_paths.py --package-root "$PWD/src"
--dump-shadow`, `catkin_make run_tests`, and `catkin_test_results
build/test_results`. Inspect the final report; a successful test command alone
is not sufficient. Edit `requirements.in`, then regenerate `requirements.lock`
with its documented `pip-compile` command rather than editing hashes manually.

## Coding Style & Testing

Use four-space Python indentation, `snake_case` modules and functions,
`PascalCase` classes, and `UPPER_CASE` constants. Name ROS executables
`*_node.py` and packages `rekpiper_<domain>`. No formatter, linter, or coverage
threshold is configured; match nearby code. Add focused `unittest` coverage in
`src/<pkg>/test/test_<behavior>.py` and register it in that package's
`CMakeLists.txt` with `catkin_add_nosetests`. Prefer pure, hardware-free tests
for logic and safety gates.

## Safety, Configuration, and Reviews

Keep runtime data, models, calibration evidence, private keys, and `.env` files
out of Git. Do not bypass `UNCALIBRATED`, `approved: false`, or signature gates;
use `shadow` mode unless site acceptance authorizes autonomous hardware control.

This supplied archive has no usable Git history, so use concise, imperative,
scoped commits such as `planning: reject stale SDF snapshots`. PRs should state
affected packages/configuration, linked issues, validation commands and results,
and safety impact. Attach logs or replay evidence for motion changes and
screenshots for visual changes.
