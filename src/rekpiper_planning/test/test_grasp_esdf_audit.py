import unittest
from unittest.mock import Mock
import numpy as np

from rekpiper_planning.grasp_esdf_audit import GraspESDFAudit, query_grid


def grid(value=-.1):
    return dict(bounds_min=np.array([-.1]*3), bounds_max=np.array([.1]*3),
                distances_m=np.full((21,21,21),value), observed=np.ones((21,21,21),bool))


class GraspESDFAuditTest(unittest.TestCase):
    def setUp(self):
        self.sampler=Mock()
        self.sampler.kinematics.forward.return_value=np.eye(4)
        self.audit=GraspESDFAudit(self.sampler,grid(0),grid(),[[0,.02,0]])

    def check(self, label='link7', point=(0,.02,0), contact=True):
        return self.audit.check(np.array([point]),np.zeros(1),np.array([label]),np.eye(4),.04,contact)

    def test_only_local_inner_pad_can_contact_target(self):
        self.assertTrue(self.check()['passed'])
        self.assertFalse(self.check('gripper_base')['passed'])
        self.assertFalse(self.check(point=(0,.02,-.04))['passed'])
        self.assertFalse(self.check(contact=False)['passed'])

    def test_target_removal_does_not_exempt_other_obstacle(self):
        self.audit.target_removed_grid=grid(0)
        self.assertFalse(self.check()['passed'])

    def test_unknown_in_either_map_is_not_contact(self):
        self.audit.target_removed_grid['observed'][:]=False
        self.assertFalse(self.check()['passed'])
        self.audit.target_removed_grid=grid()
        self.audit.grid['observed'][:]=False
        self.assertFalse(self.check()['passed'])

    def test_table_not_exempted_by_target_contact(self):
        self.audit.table=dict(table_plane_normal=[0,0,1],table_plane_offset_m=0,
            table_footprint_xy_m=[[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]])
        result=self.check()
        self.assertFalse(result['passed'])
        self.assertEqual(result['table_collision_count'],1)

    def test_sweep_detects_obstacle_between_endpoints(self):
        self.audit.pose=Mock(side_effect=lambda pose,width,contact:dict(passed=not(-.034<pose[2,3]<-.026)))
        result=self.audit.grasp_sweep(np.eye(4),.04)
        self.assertFalse(result['passed'])
        self.assertEqual(result['phase'],'approach')
        self.assertGreater(result['offset_m'],.026)

    def test_full_arm_uses_each_joint_state(self):
        self.sampler.contact_samples.return_value=(np.array([[0,0,.05]]),np.zeros(1),np.array(['link2']))
        self.audit.grid=grid()
        for q in (np.zeros(6),np.ones(6)):
            self.audit.joints(q,.04)
        self.assertEqual(self.sampler.contact_samples.call_count,2)
        np.testing.assert_equal(self.sampler.contact_samples.call_args[0][0],np.ones(6))

    def test_point_outside_actual_bounds_is_unknown_despite_rounding(self):
        _,known=query_grid(np.array([[.101,0,0]]),np.zeros(1),grid())
        self.assertFalse(known[0])

    def test_nonfinite_geometry_rejected(self):
        with self.assertRaises(ValueError):
            self.check(point=(np.nan,0,0))

    def test_nonfinite_esdf_is_unknown(self):
        _,known=query_grid(np.zeros((1,3)),np.zeros(1),grid(float('nan')))
        self.assertFalse(known[0])

    def test_occupied_diagnostic_respects_positive_occupied_convention(self):
        self.audit.grid=grid()
        self.assertEqual(self.check(contact=False)['occupied_sample_count'],0)
        self.audit.grid=grid(.01)
        self.assertEqual(self.check(contact=False)['occupied_sample_count'],1)

    def test_relocated_target_blocks_non_pad_finger_contact(self):
        self.audit.grid=grid()
        result=self.audit.check(np.array([[0,.02,-.04]]),np.zeros(1),np.array(['link7']),
            np.eye(4),.04,True,target_points=np.array([[-.01,.01,-.05],[.01,.03,-.03]]))
        self.assertFalse(result['passed'])
        self.assertEqual(result['moved_target_collision_count'],1)


if __name__=='__main__':
    unittest.main()
