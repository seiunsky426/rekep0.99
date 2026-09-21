#!/usr/bin/env python3
"""VLM ReKep constraint generation, validation and program approval."""

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import uuid

from cv_bridge import CvBridge
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

from rekpiper_msgs.msg import ReKepProgram, SceneSnapshot
from rekpiper_msgs.srv import (
    ApproveReKepProgram, ApproveReKepProgramResponse,
    GenerateReKepProgram, GenerateReKepProgramResponse,
)
from rekpiper_planning.official_program import (
    OfficialProgramError, build_official_python_prompt,
    compute_program_sha256,
    parse_official_program, save_official_program,
    validate_official_program_numerics,
    LOCAL_SAFETY_CONTRACT_VERSION, OFFICIAL_PROMPT_SHA256,
)
from rekpiper_acceptance import AcceptanceError, validate_program_approval
from rekpiper_planning.run_store import RunStore
from rekpiper_planning.vlm_program_generator import create_backend


def _point_array(snapshot):
    return np.asarray([[item.position.x, item.position.y, item.position.z]
                       for item in snapshot.keypoints.keypoints], dtype=float)


def _scene_metadata(snapshot):
    return {
        "snapshot_id": snapshot.snapshot_id,
        "instances": [{
            "rigid_group_id": int(item.rigid_group_id),
            "bounds_min": [item.bounds_min.x, item.bounds_min.y, item.bounds_min.z],
            "bounds_max": [item.bounds_max.x, item.bounds_max.y, item.bounds_max.z],
            "surface_medoid": [item.surface_medoid.x, item.surface_medoid.y,
                               item.surface_medoid.z],
        } for item in snapshot.instances.instances],
    }


class ProgramServerNode:
    def __init__(self):
        self._lock = threading.RLock()
        self._bridge = CvBridge()
        self._snapshot = None
        self._image = None
        self._proposal = None
        self._mode = str(rospy.get_param("~mode", "shadow")).strip().lower()
        self._acceptance_public_key = str(rospy.get_param(
            "~acceptance_public_key", ""))
        config = dict(rospy.get_param("~vlm"))
        self._backend = create_backend(config)
        self._model = self._backend.model
        self._capabilities = dict(rospy.get_param("~robot_capabilities", {}))
        self._store = RunStore(rospy.get_param(
            "~data_root", str(Path(__file__).resolve().parents[3] / "data")))
        self._publisher = rospy.Publisher(
            "/rekpiper/program/current", ReKepProgram, queue_size=1, latch=True)
        rospy.Subscriber("/rekpiper/perception/scene_snapshot", SceneSnapshot,
                         self._snapshot_callback, queue_size=1)
        rospy.Subscriber(rospy.get_param(
            "~annotated_image_topic", "/rekpiper/perception/candidate_image"),
            Image, self._image_callback, queue_size=1)
        rospy.Service("/rekpiper/program/generate", GenerateReKepProgram,
                      self._generate)
        rospy.Service("/rekpiper/program/approve", ApproveReKepProgram,
                      self._approve)

    def _snapshot_callback(self, message):
        with self._lock:
            self._snapshot = deepcopy(message)

    def _image_callback(self, message):
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "candidate image rejected: %s", exc)
            return
        with self._lock:
            self._image = (message.header.stamp.to_nsec(), image.copy())

    def _publish(self, proposal, approved=False, status="proposal_requires_approval"):
        message = ReKepProgram(
            header=proposal["header"], session_id=proposal["session_id"],
            snapshot_id=proposal["snapshot_id"],
            program_sha256=proposal["sha256"],
            program_directory=str(proposal["directory"]),
            num_stages=proposal["program"].num_stages,
            grasp_keypoints=proposal["program"].grasp_keypoints,
            release_keypoints=proposal["program"].release_keypoints,
            schema_valid=True, numerically_valid=True,
            approved=approved, status=status)
        self._publisher.publish(message)
        return message

    def _generate(self, request):
        try:
            instruction = str(request.instruction).strip()
            with self._lock:
                snapshot = deepcopy(self._snapshot)
                image_record = None if self._image is None else (
                    self._image[0], self._image[1].copy())
            if snapshot is None or not snapshot.valid or not snapshot.immutable_layout:
                raise OfficialProgramError("no valid immutable SceneSnapshot is available")
            if image_record is None:
                raise OfficialProgramError("annotated K0..Kn image is unavailable")
            if abs(image_record[0] - snapshot.snapshot_stamp_ns) > int(0.15 * 1e9):
                raise OfficialProgramError("annotated image and snapshot are not time-consistent")
            points = _point_array(snapshot)
            if points.shape != (len(snapshot.keypoints.keypoints), 3) or not np.all(np.isfinite(points)):
                raise OfficialProgramError("snapshot keypoints are invalid")
            session_id = "task-" + uuid.uuid4().hex
            run = self._store.create_run("programs", session_id)
            serialized_snapshot = io.BytesIO()
            snapshot.serialize(serialized_snapshot)
            snapshot_bytes = serialized_snapshot.getvalue()
            snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
            (run / "scene_snapshot.rosmsg").write_bytes(snapshot_bytes)
            image_path = run / "annotated_keypoints.png"
            if not cv2.imwrite(str(image_path), image_record[1]):
                raise OfficialProgramError("failed to persist annotated image")
            metadata = _scene_metadata(snapshot)
            prompt = build_official_python_prompt(
                instruction, points, self._capabilities, metadata)
            (run / "prompt.txt").write_text(prompt, encoding="utf-8")
            raw = self._backend.generate_text(prompt, str(image_path))
            (run / "vlm_raw_response.txt").write_text(raw, encoding="utf-8")
            program = parse_official_program(raw, instruction, len(points))
            save_official_program(program, run, points)
            validation = validate_official_program_numerics(
                run, program, points, points.mean(axis=0), samples=32)
            sha = compute_program_sha256(run)
            audit = {
                "session_id": session_id, "snapshot_id": snapshot.snapshot_id,
                "snapshot_sha256": snapshot_sha256,
                "program_sha256": sha, "vlm_model": self._model,
                "vlm_provider": self._backend.provider,
                "raw_response_file": "vlm_raw_response.txt",
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")).hexdigest(),
                "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "query_image_sha256": self._backend.last_image_sha256,
                "validation": validation, "approved": False,
                "official_rekep_prompt": True,
                "official_prompt_sha256": OFFICIAL_PROMPT_SHA256,
                "local_safety_contract_version": LOCAL_SAFETY_CONTRACT_VERSION,
            }
            RunStore.write_json(run / "audit.json", audit)
            proposal = {
                "session_id": session_id, "snapshot_id": snapshot.snapshot_id,
                "snapshot_sha256": snapshot_sha256,
                "sha256": sha, "directory": run, "program": program,
                "header": deepcopy(snapshot.header), "audit": audit,
            }
            with self._lock:
                self._proposal = proposal
            self._publish(proposal)
            response = {
                "session_id": session_id, "snapshot_id": snapshot.snapshot_id,
                "program_sha256": sha, "program_directory": str(run),
                "num_stages": program.num_stages,
                "grasp_keypoints": program.grasp_keypoints,
                "release_keypoints": program.release_keypoints,
            }
            return GenerateReKepProgramResponse(
                True, "proposal_generated_requires_program_approval",
                json.dumps(response, ensure_ascii=False, sort_keys=True),
                json.dumps(validation, ensure_ascii=False, sort_keys=True))
        except Exception as exc:
            rospy.logerr("VLM ReKep program generation rejected: %s", exc)
            return GenerateReKepProgramResponse(
                False, "{}:{}".format(type(exc).__name__, exc), "", "")

    def _approve(self, request):
        with self._lock:
            proposal = self._proposal
            if (proposal is None or request.session_id != proposal["session_id"]
                    or request.program_sha256 != proposal["sha256"]):
                return ApproveReKepProgramResponse(False, "session_or_sha_mismatch")
            if compute_program_sha256(proposal["directory"]) != proposal["sha256"]:
                return ApproveReKepProgramResponse(False, "program_files_changed_after_review")
            if self._mode == "autonomous":
                try:
                    proposal["approval_guard"] = validate_program_approval(
                        str(proposal["directory"] / "program_approval.yaml"),
                        self._acceptance_public_key,
                        proposal["session_id"], proposal["snapshot_id"],
                        proposal["sha256"], OFFICIAL_PROMPT_SHA256,
                        LOCAL_SAFETY_CONTRACT_VERSION,
                        proposal["snapshot_sha256"])
                except (AcceptanceError, OSError, ValueError) as exc:
                    return ApproveReKepProgramResponse(
                        False, "signed_program_approval_rejected:" + str(exc))
            else:
                proposal["audit"]["approved"] = True
                RunStore.write_json(
                    proposal["directory"] / "audit.json", proposal["audit"])
            self._publish(proposal, approved=True, status="program_approved_not_armed")
        return ApproveReKepProgramResponse(True, "program_approved_not_armed")


if __name__ == "__main__":
    rospy.init_node("rekep_program_server")
    ProgramServerNode()
    rospy.spin()
