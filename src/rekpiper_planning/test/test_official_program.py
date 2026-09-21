#!/usr/bin/env python3

from pathlib import Path
import tempfile
import unittest

import numpy as np

from rekpiper_planning.official_program import (
    OfficialProgramError, build_official_python_prompt,
    compute_program_sha256,
    export_stage_constraint_texts,
    generate_official_program, load_official_stage_constraints,
    parse_official_program, save_official_program,
    validate_official_program_numerics,
)


_SOURCE = '''```python
# Stage 1: approach and grasp keypoint 0.
num_stages = 2

def stage1_subgoal_constraint1(end_effector, keypoints):
    """Put the end effector at the observed grasp point."""
    return np.linalg.norm(end_effector - keypoints[0]) - 0.02

def stage2_subgoal_constraint1(end_effector, keypoints):
    target = keypoints[1] + np.array([0.0, 0.0, 0.10])
    return np.linalg.norm(keypoints[0] - target) - 0.02

def stage2_path_constraint1(end_effector, keypoints):
    return get_grasping_cost_by_keypoint_idx(0)

grasp_keypoints = [0, -1]
release_keypoints = [-1, 0]
```'''


class _TextBackend:
    def generate_text(self, _prompt, _image_path):
        return _SOURCE


class OfficialProgramTest(unittest.TestCase):
    def setUp(self):
        self.points = np.asarray([[0.1, 0.0, 0.2], [0.3, 0.1, 0.0]], dtype=float)

    def test_save_stage_files_then_restricted_exec_load(self):
        program = parse_official_program(_SOURCE, "pick and place", len(self.points))
        with tempfile.TemporaryDirectory() as directory:
            root = save_official_program(program, directory, self.points)
            self.assertTrue((root / "metadata.json").is_file())
            self.assertTrue((root / "stage1_subgoal_constraints.txt").is_file())
            self.assertTrue((root / "stage2_path_constraints.txt").is_file())
            subgoals, paths = load_official_stage_constraints(
                root, 2, len(self.points), lambda index: 0.0 if index == 0 else 1.0)
            self.assertEqual(len(subgoals), 1)
            self.assertEqual(len(paths), 1)
            self.assertLessEqual(paths[0](np.zeros(3), self.points), 0.0)
            report = validate_official_program_numerics(
                root, program, self.points, np.zeros(3), samples=3)
            self.assertTrue(report["numeric_stress_valid"])
            self.assertEqual(report["constraint_count"], 3)

    def test_generator_uses_text_backend_not_json_backend(self):
        program = generate_official_program(
            "pick and place", None, self.points, {"automatic_motion": False},
            _TextBackend())
        self.assertEqual(program.num_stages, 2)
        self.assertEqual(program.grasp_keypoints, [0, -1])

    def test_prompt_is_the_official_python_constraint_contract(self):
        prompt = build_official_python_prompt(
            "pick and place", self.points,
            {"automatic_motion": False})
        self.assertIn('Query Task: "pick and place"', prompt)
        self.assertIn("keypoints marked on the image start with index 0", prompt)
        self.assertIn("stage1_subgoal_constraint1", prompt)
        self.assertIn("grasp_keypoints", prompt)
        self.assertIn("release_keypoints", prompt)
        self.assertIn("REKPIPER LOCAL EXECUTION CONTRACT", prompt)
        self.assertIn("Do not use imports, loops", prompt)
        self.assertIn("np.linalg.norm".split(".")[-1], prompt)

    def test_metadata_records_both_prompt_contracts(self):
        program = parse_official_program(
            _SOURCE, "pick and place", len(self.points))
        metadata = program.metadata(self.points)
        self.assertEqual(len(metadata["official_prompt_sha256"]), 64)
        self.assertEqual(
            metadata["local_safety_contract_version"],
            "rekpiper_vector_ast_v1")

    def test_zero_based_stage_or_constraint_name_is_rejected(self):
        for function_name in (
                "stage0_subgoal_constraint1",
                "stage1_subgoal_constraint0"):
            source = '''
num_stages = 1
grasp_keypoints = [-1]
release_keypoints = [-1]
def {}(end_effector, keypoints):
    return np.linalg.norm(end_effector - keypoints[0])
'''.format(function_name)
            with self.assertRaisesRegex(
                    OfficialProgramError,
                    "invalid constraint function name"):
                parse_official_program(source, "bad numbering", 1)

    def test_export_stage_txt_replaces_stale_previous_program(self):
        program = parse_official_program(
            _SOURCE, "pick and place", len(self.points))
        with tempfile.TemporaryDirectory() as source_directory, \
                tempfile.TemporaryDirectory() as output_directory:
            root = save_official_program(
                program, source_directory, self.points)
            stale = Path(output_directory) / "stage8_path_constraints.txt"
            stale.write_text("stale\n", encoding="utf-8")
            unrelated = Path(output_directory) / "notes.txt"
            unrelated.write_text("keep\n", encoding="utf-8")
            exported = export_stage_constraint_texts(root, output_directory)
            self.assertFalse(stale.exists())
            self.assertTrue(unrelated.exists())
            self.assertEqual(
                [path.name for path in exported],
                ["stage1_subgoal_constraints.txt",
                 "stage2_path_constraints.txt",
                 "stage2_subgoal_constraints.txt"])
            for path in exported:
                self.assertEqual(
                    path.read_text(encoding="utf-8"),
                    (root / path.name).read_text(encoding="utf-8"))

    def test_import_and_untrusted_calls_are_rejected(self):
        malicious = '''
num_stages = 1
grasp_keypoints = [-1]
release_keypoints = [-1]
import os
'''
        with self.assertRaises(OfficialProgramError):
            parse_official_program(malicious, "bad", 1)
        malicious_call = '''
num_stages = 1
grasp_keypoints = [-1]
release_keypoints = [-1]
def stage1_subgoal_constraint1(end_effector, keypoints):
    return open("/etc/passwd").read()
'''
        with self.assertRaises(OfficialProgramError):
            parse_official_program(malicious_call, "bad", 1)

    def test_constraint_loop_is_rejected_with_actionable_error(self):
        loop = '''
num_stages = 1
grasp_keypoints = [-1]
release_keypoints = [-1]
def stage1_subgoal_constraint1(end_effector, keypoints):
    value = 0.0
    for index in range(4):
        value = value + index
    return value
'''
        with self.assertRaisesRegex(OfficialProgramError, "control flow is forbidden"):
            parse_official_program(loop, "bad loop", 4)

    def test_program_hash_covers_executed_stage_files(self):
        program = parse_official_program(
            _SOURCE, "pick and place", len(self.points))
        with tempfile.TemporaryDirectory() as directory:
            root = save_official_program(program, directory, self.points)
            before = compute_program_sha256(root)
            stage = root / "stage1_subgoal_constraints.txt"
            stage.write_text(
                stage.read_text(encoding="utf-8") + "\n",
                encoding="utf-8")
            self.assertNotEqual(before, compute_program_sha256(root))

    def test_paper_scale_ten_stage_program_is_accepted(self):
        source = "\n".join([
            "num_stages = 10",
            "grasp_keypoints = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]",
            "release_keypoints = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]",
            "def stage10_subgoal_constraint1(end_effector, keypoints):",
            "    return np.linalg.norm(end_effector - keypoints[0])",
        ])
        program = parse_official_program(source, "ten stages", 1)
        self.assertEqual(program.num_stages, 10)

    def test_official_pen_mean_axis_program_round_trips(self):
        source = '''
num_stages = 1
grasp_keypoints = [-1]
release_keypoints = [-1]
def stage1_subgoal_constraint1(end_effector, keypoints):
    pen_center = np.mean(keypoints[3:7], axis=0)
    return np.linalg.norm(end_effector - pen_center)
'''
        points = np.arange(21, dtype=float).reshape(7, 3) * 0.01
        program = parse_official_program(source, "move to pen center", 7)
        with tempfile.TemporaryDirectory() as directory:
            root = save_official_program(program, directory, points)
            functions, paths = load_official_stage_constraints(
                root, 1, 7, lambda _index: 1.0)
            self.assertFalse(paths)
            expected = np.linalg.norm(
                np.zeros(3) - np.mean(points[3:7], axis=0))
            self.assertAlmostEqual(functions[0](np.zeros(3), points), expected)

    def test_sixteen_and_thirty_two_stage_export_and_stale_cleanup(self):
        for count in (16, 32):
            source = "\n".join([
                "num_stages = {}".format(count),
                "grasp_keypoints = {}".format([-1] * count),
                "release_keypoints = {}".format([-1] * count),
                "def stage{}_subgoal_constraint1(end_effector, keypoints):".format(count),
                "    return np.linalg.norm(end_effector - keypoints[0])",
            ])
            program = parse_official_program(source, "many stages", 1)
            with tempfile.TemporaryDirectory() as directory, \
                    tempfile.TemporaryDirectory() as export_directory:
                stale = Path(directory) / "stage31_path_constraints.txt"
                stale.write_text("stale\n", encoding="utf-8")
                root = save_official_program(program, directory, self.points[:1])
                self.assertFalse(stale.exists())
                exported = export_stage_constraint_texts(
                    root, export_directory)
                self.assertEqual(
                    [item.name for item in exported],
                    ["stage{}_subgoal_constraints.txt".format(count)])

    def test_same_stage_grasp_release_is_rejected(self):
        source = '''
num_stages = 1
grasp_keypoints = [0]
release_keypoints = [0]
def stage1_subgoal_constraint1(end_effector, keypoints):
    return np.linalg.norm(end_effector - keypoints[0])
'''
        with self.assertRaisesRegex(OfficialProgramError, "both grasp and release"):
            parse_official_program(source, "invalid", 1)

    def test_uncontrolled_numpy_keyword_is_rejected(self):
        source = '''
num_stages = 1
grasp_keypoints = [-1]
release_keypoints = [-1]
def stage1_subgoal_constraint1(end_effector, keypoints):
    return np.mean(keypoints, dtype=object)
'''
        with self.assertRaises(OfficialProgramError):
            parse_official_program(source, "invalid", 1)

if __name__ == "__main__":
    unittest.main()
