"""Regression checks for residual costs, limit margins and continuous seeding."""
import unittest
from types import SimpleNamespace

import numpy as np

from rekpiper_planning.piper_path_solver import (
    PIPER_PATH_DEFAULTS, continuous_ik_cost, make_piper_path_solver)


def objective(*args, **kwargs):
    return 'original'


class OfficialStub:
    def solve(self, *args, **kwargs):
        return objective(*args, **kwargs)


class IK:
    _lower=np.full(6,-1.)
    _upper=np.full(6,1.)
    position_tolerance=.003
    orientation_tolerance=.03
    def __init__(self, results):
        self.results=iter(results)
        self.seeds=[]
    def solve(self, pose, initial_joint_pos, max_iterations):
        self.seeds.append(np.array(initial_joint_pos).copy())
        return next(self.results)


def result(q=0., success=True, pos=0., rot=0., iterations=5):
    return SimpleNamespace(cspace_position=np.full(7,q),success=success,
                           position_error=pos,rotation_error=rot,num_descents=iterations)


class PiperPathCostTest(unittest.TestCase):
    def cost(self, values, seed=None):
        ik=IK(values)
        report=continuous_ik_cost(ik,np.tile(np.eye(4),(len(values),1,1)),
                                  np.zeros(6) if seed is None else seed,PIPER_PATH_DEFAULTS)
        return report,ik

    def test_fast_failure_costs_more_than_slow_success(self):
        failed,_=self.cost([result(success=False,rot=.03965,iterations=5)])
        passed,_=self.cost([result(iterations=100)])
        self.assertGreater(failed['ik_cost'],passed['ik_cost']+999)
        self.assertGreater(failed['ik_orientation_cost'],0.)
        self.assertEqual(failed['ik_position_cost'],0.)

    def test_normalized_position_and_rotation_excess(self):
        value,_=self.cost([result(success=False,pos=.006,rot=.09)])
        self.assertAlmostEqual(value['ik_position_cost'],20.)
        self.assertAlmostEqual(value['ik_orientation_cost'],80.)

    def test_soft_limit_margin_penalizes_both_ends_without_changing_limits(self):
        for q in (-.99,.99):
            value,ik=self.cost([result(q=q)],np.full(6,q))
            self.assertGreater(value['joint_limit_cost'],0)
            np.testing.assert_array_equal(ik._upper,np.ones(6))
        center,_=self.cost([result()])
        self.assertEqual(center['joint_limit_cost'],0)

    def test_previous_valid_solution_seeds_each_sample_and_failed_one_does_not(self):
        value,ik=self.cost([result(q=.1),result(q=.2,success=False),result(q=.15)])
        np.testing.assert_allclose(ik.seeds,[np.zeros(6),np.full(6,.1),np.full(6,.1)])
        self.assertEqual(value['ik_feasible'],[True,False,True])
        self.assertEqual(value['ik_sample_count'],3)

    def test_joint_jump_is_not_a_continuous_solution(self):
        value,ik=self.cost([result(q=.5),result(q=.1)])
        self.assertEqual(value['ik_feasible'],[False,True])
        np.testing.assert_array_equal(ik.seeds[1],np.zeros(6))

    def test_solver_objective_is_instance_local_and_settings_are_checked(self):
        modules=SimpleNamespace(path_solver=SimpleNamespace(PathSolver=OfficialStub))
        baseline=OfficialStub.solve.__globals__['objective']
        solver=make_piper_path_solver({},IK([]),np.zeros(7),modules,lambda x:x)
        self.assertIs(OfficialStub.solve.__globals__['objective'],baseline)
        self.assertIsNot(solver.solve.__func__.__globals__['objective'],baseline)
        self.assertEqual(OfficialStub().solve(),'original')
        with self.assertRaises(ValueError):
            make_piper_path_solver({'joint_limit_margin_deg':0},IK([]),np.zeros(7),modules,None)


if __name__=='__main__':
    unittest.main()
