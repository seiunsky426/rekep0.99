import unittest
from types import SimpleNamespace
import numpy as np
import time
from rekpiper_planning.solver_deadline import bounded_solver_call
from rekpiper_planning.continuous_ik import validate_continuous_ik


class LinearIK:
    _lower = np.full(6,-3.)
    _upper = np.full(6,3.)
    position_tolerance = .01
    orientation_tolerance = .1
    def forward(self,q):
        p=np.eye(4); p[0,3]=q[0]; return p
    def _pose_errors(self,a,b):
        return abs(a[0,3]-b[0,3]),0.
    def solve(self,p,initial_joint_pos,max_iterations):
        q=np.zeros(7); q[0]=p[0,3]
        return SimpleNamespace(success=True,cspace_position=q,position_error=0.,rotation_error=0.)


class ContinuousIKTest(unittest.TestCase):
    def test_numerical_solver_deadline_interrupts_call(self):
        with self.assertRaises(TimeoutError):
            bounded_solver_call(lambda:time.sleep(.05),.005)

    def test_large_move_is_refined_and_edges_rechecked(self):
        ik=LinearIK(); goal=ik.forward([.8])
        r=validate_continuous_ik(ik,[goal],np.zeros(6))
        self.assertTrue(r['valid'])
        self.assertGreater(len(r['steps']),1)
        self.assertTrue(all(s['maximum_joint_step_rad']<=.35 for s in r['steps']))
        self.assertEqual(r['diagnostics'][0]['status'],'joint_jump')

    def test_refinement_budget_failure_never_returns_valid(self):
        r=validate_continuous_ik(LinearIK(),[LinearIK().forward([2.9])],np.zeros(6),
                                 maximum_refinement=0)
        self.assertFalse(r['valid']); self.assertEqual(r['failed_pose_index'],0)
        self.assertEqual(r['steps'],[])

    def test_fk_edge_deviation_not_hidden_by_valid_endpoints(self):
        class BadEdge(LinearIK):
            def forward(self,q):
                p=super().forward(q)
                if .04<q[0]<.06: p[2,3]=1.
                return p
            def _pose_errors(self,a,b):
                return np.linalg.norm(a[:3,3]-b[:3,3]),0.
        r=validate_continuous_ik(BadEdge(),[LinearIK().forward([.1])],np.zeros(6),
                                 maximum_refinement=0)
        self.assertFalse(r['valid'])
        self.assertEqual(r['diagnostics'][0]['status'],'fk_edge_deviation')

    def test_invalid_budgets_rejected(self):
        with self.assertRaises(ValueError):
            validate_continuous_ik(LinearIK(),[np.eye(4)],np.zeros(6),maximum_seeds=9)

    def test_trace_preserves_policy_and_records_actual_refinement_seeds(self):
        ik = LinearIK()
        poses = [ik.forward([.8])]
        plain = validate_continuous_ik(ik, poses, np.zeros(6))
        trace = []
        traced = validate_continuous_ik(ik, poses, np.zeros(6), trace=trace)
        np.testing.assert_array_equal(plain.pop('poses'), traced.pop('poses'))
        self.assertEqual(plain, traced)
        committed = [a for a in trace if a['committed']]
        self.assertEqual(len(committed), len(traced['steps']))
        for previous, following in zip(committed, committed[1:]):
            self.assertEqual(previous['joint_positions'], following['previous_joints'])
        self.assertTrue(all(a['edge_check']['samples_checked'] >= 2 for a in committed))

    def test_successful_subdivision_is_not_committed_if_original_waypoint_fails(self):
        class PartialIK(LinearIK):
            def solve(self, p, **kwargs):
                result = super().solve(p, **kwargs)
                result.success = p[0, 3] <= .4
                return result
        ik = PartialIK(); trace = []
        result = validate_continuous_ik(ik, [ik.forward([.8])], np.zeros(6), trace=trace)
        self.assertFalse(result['valid'])
        self.assertTrue(any(a['selected'] for a in trace))
        self.assertFalse(any(a['committed'] for a in trace))
        self.assertEqual(result['steps'], [])


if __name__=='__main__': unittest.main()
