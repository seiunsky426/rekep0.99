#!/usr/bin/env python3
"""Hardware-free supervised authority and stage ordering tests."""
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np

from rekpiper_execution.supervised_authority import Lease
from rekpiper_execution.supervised_session import (Session, Speed, select_keypoints,
    select_instance_group, fixed_constraints, FIXED_TASK)
from rekpiper_execution.supervised_geometry import validate_known_points
from rekpiper_execution.supervised_planning import ExperimentPlanner, _planning_job, template
from rekpiper_planning.official_program import parse_official_program


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.session=Session()

    def install(self):
        ticket=self.session.begin_plan()
        q=np.zeros((2,6));q[1,0]=.001
        return self.session.install(ticket,q,[0.,1.],{'scene':'one'},{'goal':'test'})

    def test_preview_is_single_use_and_stage_ordered(self):
        p=self.install()
        with self.assertRaises(ValueError):
            self.session.claim(self.session.id,2,p['id'],np.zeros(6),p['binding'])
        self.session.claim(self.session.id,1,p['id'],np.zeros(6),p['binding'])
        with self.assertRaises(ValueError):
            self.session.claim(self.session.id,1,p['id'],np.zeros(6),p['binding'])
        self.session.finish()
        self.assertEqual(self.session.stage,2)
        self.assertEqual(self.session.state,'IDLE')
        p=self.install()
        self.session.claim(self.session.id,2,p['id'],np.zeros(6),p['binding'])
        self.session.finish()
        self.assertEqual(self.session.state,'WAITING_GRASP')
        with self.assertRaises(ValueError):self.session.begin_plan()
        self.session.confirm_grasp(True)
        self.assertEqual(self.session.stage,3)
        self.assertTrue(self.session.held)

    def test_changed_speed_and_stop_revoke_preview(self):
        p=self.install()
        self.session.speed=Speed(.03,.1,.5,3)
        self.session.invalidate('changed speed')
        with self.assertRaises(ValueError):
            self.session.claim(self.session.id,1,p['id'],np.zeros(6),p['binding'])
        p=self.install()
        self.session.invalidate('stopped')
        with self.assertRaises(ValueError):
            self.session.claim(self.session.id,1,p['id'],np.zeros(6),p['binding'])

    def test_vlm_only_selects_measured_different_objects(self):
        keys=[dict(id=i,valid=True,valid_depth_pixels=100,position=[i*.1,0,.2],rigid_group_id=i+1)
              for i in range(2)]
        self.assertEqual([k['id'] for k in select_keypoints('{"grasp_id":0,"target_id":1}',keys)],[0,1])
        for text in ('{"grasp_id":0,"target_id":0}',
                     '{"grasp_id":0,"target_id":8}',
                     '{"grasp_id":0,"target_id":1,"xyz":[0,0,0]}'):
            with self.assertRaises(ValueError):select_keypoints(text,keys)

    def test_rs3_vlm_group_must_exist_in_frozen_depth_mask(self):
        labels=np.zeros((20,20),dtype=np.uint16)
        labels[:15,:15]=3
        self.assertEqual(select_instance_group('{"group_id":3}',labels),3)
        for output in ('{"group_id":2}', '{"group_id":true}',
                       '{"group_id":3,"xyz":[0,0,0]}'):
            with self.assertRaises(ValueError):
                select_instance_group(output,labels)

    def test_fixed_constraints_only_bind_selected_measured_points(self):
        keys=[dict(id=i,valid=True,valid_depth_pixels=100,
                   position=[.3+i*.1,.1,.2],rigid_group_id=i+1) for i in range(2)]
        selected=select_keypoints('{"grasp_id":0,"target_id":1}',keys)
        constraints=fixed_constraints(selected)
        self.assertEqual(constraints['task'],FIXED_TASK)
        self.assertEqual(constraints['stages'],['预抓取','抓取','抬升并运输','下放释放'])
        self.assertEqual(constraints['blue_cube']['position_m'],keys[0]['position'])
        self.assertEqual(constraints['yellow_disk']['position_m'],keys[1]['position'])
        self.assertEqual(constraints['blue_cube']['rigid_group_id'],1)
        self.assertEqual(constraints['yellow_disk']['rigid_group_id'],2)

    def test_missing_grasp_source_clears_candidate_without_ik(self):
        planner=ExperimentPlanner.__new__(ExperimentPlanner)
        planner.candidate=object()
        planner.rejections=['previous candidate']
        planner.ik=Mock()
        with patch('rekpiper_execution.supervised_planning.generate_candidates',
                   side_effect=ValueError('anygrasp_unavailable')):
            with self.assertRaisesRegex(ValueError,'anygrasp_unavailable'):
                planner.candidates(Mock(),np.zeros((200,3)),np.zeros(6))
        self.assertIsNone(planner.candidate)
        self.assertEqual(planner.rejections,[])
        planner.ik.solve.assert_not_called()

    def test_missing_grasp_source_stops_worker_before_path_planning(self):
        planner=ExperimentPlanner.__new__(ExperimentPlanner)
        planner.sweep=Mock()
        planner.plan=Mock()
        connection=Mock()
        job=dict(xml='',config={},system={},output='',target=np.zeros((200,3)),
                 support=None,held_local=None,grasp_matrix=None,candidate=object(),
                 stage=1,scene=Mock(),joints=np.zeros(6),opening=.04)
        with patch('rekpiper_execution.supervised_planning.ExperimentPlanner',return_value=planner), \
                patch('rekpiper_execution.supervised_planning.generate_candidates',
                      side_effect=ValueError('anygrasp_unavailable')):
            _planning_job(connection,job)
        kind,message=connection.send.call_args.args[0]
        self.assertEqual(kind,'error')
        self.assertIn('anygrasp_unavailable',message)
        planner.sweep.assert_not_called()
        planner.plan.assert_not_called()
        connection.close.assert_called_once()

    def test_known_point_check_requires_spread_and_both_cameras(self):
        reference=np.array([[.15,-.12,.05],[.38,-.10,.05],[.16,.14,.15],[.38,.12,.15]])
        rows=[dict(base=p.tolist(),rs1=(p+[.001,0,0]).tolist(),
                   rs3=(p+[0,.001,0]).tolist()) for p in reference]
        self.assertLess(max(validate_known_points(rows)['cross_view']),.005)
        rows[0]['rs3'][0]+=.01
        with self.assertRaises(ValueError):validate_known_points(rows)
        with self.assertRaises(ValueError):validate_known_points(rows[:3])

    def test_local_template_parses(self):
        source=template(np.array([.3,.1,.12]),np.array([.3,.1,.18]),True)
        parse_official_program(source,'fixed supervised pick and place',2)


class LeaseTest(unittest.TestCase):
    def test_owner_expiry_and_preview_digest(self):
        now=[0.]
        lease=Lease('session','/owner',lambda:now[0])
        payload=dict(session_id='session',sequence=1,alive=True,executing=True,
                     operation='trajectory',trajectory_sha256='approved')
        self.assertFalse(lease.update(json.dumps(payload),'/intruder',0.))
        with self.assertRaises(ValueError):lease.check('trajectory','approved')
        self.assertTrue(lease.update(json.dumps(payload),'/owner',0.))
        lease.check('trajectory','approved')
        with self.assertRaises(ValueError):lease.check('trajectory','changed')
        self.assertFalse(lease.update(json.dumps(payload),'/owner',0.))
        now[0]=.6
        with self.assertRaises(ValueError):lease.check('trajectory','approved')


if __name__=='__main__':unittest.main()
