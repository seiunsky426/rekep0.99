"""Frozen target provenance, directional selection and preview worker checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from rekpiper_execution.supervised_anygrasp import frozen_input
from rekpiper_execution.supervised_planning import _planning_job, template, RealtimePlanningError
from rekpiper_grasp.blue_cloud_comparison import sha256
from rekpiper_planning.official_program import parse_official_program


class DirectionalIntegrationTest(unittest.TestCase):
    def test_frozen_dual_masks_preserve_original_coordinates_and_sources(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            xyz = np.indices((15, 15)).transpose(1, 2, 0)*.004
            xyz = np.concatenate([xyz, np.full((15, 15, 1), .05)], axis=2).astype(np.float32)
            labels = np.ones((15, 15), np.uint16)
            np.savez(root/'segmentation_snapshot.npz', xyz=xyz, mask=labels, stamp_ns=1_000_000_000,
                     rs3_xyz=xyz+[.15, 0, 0], rs3_mask=labels*3, rs3_stamp_ns=1_001_000_000)
            selected = dict(stamp=1., snapshot_sha256=sha256(root/'segmentation_snapshot.npz'),
                            selection=[dict(rigid_group_id=1)], rs3_group_id=3)
            (root/'vlm_selection.json').write_text(json.dumps(selected))
            config=dict(bounds_min=[-1]*3, bounds_max=[1]*3)
            transform=np.eye(4); transform[:3,3]=[.4,.2,.3]
            with patch('rekpiper_execution.supervised_anygrasp.load_experiment',
                       return_value=({'rs1':transform},{})):
                packed, binding=frozen_input(root,config)
                self.assertEqual(set(packed['source_bits']),{1,2})
                restored=packed['points_camera'] @ transform[:3,:3].T + transform[:3,3]
                np.testing.assert_allclose(restored,packed['points_arm'],atol=1e-7)
                self.assertTrue(np.any(packed['points_arm'][:,0]>.15))
                selected['snapshot_sha256']='changed'
                (root/'vlm_selection.json').write_text(json.dumps(selected))
                with self.assertRaisesRegex(ValueError,'selection_snapshot_mismatch'):
                    frozen_input(root,config)

    def test_side_path_constrains_approach_line_instead_of_vertical(self):
        source=template(np.array([.4,0,.2]),np.array([.3,0,.2]),approach_axis=[1.,0.,0.])
        parse_official_program(source,'side approach',2)
        scope={'np':np}; exec(source,scope)
        constraint=scope['stage1_path_constraint1']
        self.assertLess(constraint(np.array([.35,0,.2]),None),0)
        self.assertGreater(constraint(np.array([.35,0,.25]),None),0)

    def test_failed_path_tries_next_model_candidate_without_motion(self):
        with tempfile.TemporaryDirectory() as name:
            first=SimpleNamespace(candidate_id='top-1',suggested_preopen_width_m=.06)
            second=SimpleNamespace(candidate_id='side-1',suggested_preopen_width_m=.05)
            planner=Mock(output=Path(name))
            planner.candidates.return_value=[first,second]
            planner.plan.side_effect=[RealtimePlanningError('Piper IK sequence failed at 12: ik_failed'),('trajectory',{})]
            job=dict(xml='',config={},system={},output=name,target=[],support=[],held_local=None,
                     grasp_matrix=None,candidate=None,stage=1,scene=None,joints=np.zeros(6),opening=0.,speed=None)
            conn=Mock()
            with patch('rekpiper_execution.supervised_planning.ExperimentPlanner',return_value=planner):
                _planning_job(conn,job)
            kind, value=conn.send.call_args.args[0]
            self.assertEqual(kind,'ok')
            self.assertEqual(value[2].candidate_id,'side-1')
            self.assertEqual(planner.plan.call_count,2)
            self.assertIn('IK sequence failed at 12',value[1]['earlier_candidate_path_failures'][0]['reason'])


    def test_initial_opening_failure_never_installs_diagnostic_path(self):
        with tempfile.TemporaryDirectory() as name:
            candidate=SimpleNamespace(candidate_id='top-1',suggested_preopen_width_m=.06)
            planner=Mock(output=Path(name),grasp_evidence=name)
            planner.candidates.return_value=[candidate]
            planner.sweep.side_effect=ValueError('robot_self_collision')
            planner.plan.return_value=('draft',{})
            job=dict(xml='',config={'extrinsics_status':'EXPERIMENTAL_PREVIEW_ONLY'},system={},
                     output=name,target=[],support=[],held_local=None,grasp_matrix=None,candidate=None,
                     stage=1,scene=None,joints=np.zeros(6),opening=0.,speed=None)
            conn=Mock()
            with patch('rekpiper_execution.supervised_planning.ExperimentPlanner',return_value=planner):
                _planning_job(conn,job)
            self.assertEqual(conn.send.call_args.args[0][0],'error')
            self.assertIn('initial_opening_sweep',conn.send.call_args.args[0][1])
            planner.plan.assert_called_once()

    def test_cancelled_planning_does_not_try_another_candidate(self):
        with tempfile.TemporaryDirectory() as name:
            candidate=SimpleNamespace(candidate_id='top-1',suggested_preopen_width_m=.06)
            planner=Mock(output=Path(name),grasp_evidence=name)
            planner.candidates.return_value=[candidate,candidate]
            planner.plan.side_effect=InterruptedError('planning_cancelled')
            job=dict(xml='',config={},system={},output=name,target=[],support=[],held_local=None,
                     grasp_matrix=None,candidate=None,stage=1,scene=None,joints=np.zeros(6),opening=.03,speed=None)
            conn=Mock()
            with patch('rekpiper_execution.supervised_planning.ExperimentPlanner',return_value=planner):
                _planning_job(conn,job)
            self.assertEqual(conn.send.call_args.args[0][0],'error')
            self.assertIn('planning_cancelled',conn.send.call_args.args[0][1])
            planner.plan.assert_called_once()


if __name__=='__main__':
    unittest.main()
