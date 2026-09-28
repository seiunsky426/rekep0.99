"""Synthetic direction, table sweep and contact evidence checks without hardware."""
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rekpiper_grasp.anygrasp_adapter import CameraGrasp
from rekpiper_grasp.directional_grasp import direction_queries, direction_checks, DirectionalAudit, load_gripper


class DirectionalGraspTest(unittest.TestCase):
    def test_directions_follow_tilted_table_not_camera_z(self):
        normal=Rotation.from_euler('xy',[7,4],degrees=True).apply([0,0,1])
        top,cone=direction_queries(normal,'top')
        np.testing.assert_allclose(top[0],-normal)
        sides,_=direction_queries(normal,'side')
        self.assertEqual(len(sides),12)
        np.testing.assert_allclose(np.array(sides)@normal,0,atol=1e-12)
        self.assertEqual(cone,15.)
        rotation=Rotation.from_euler('xyz',[30,50,10],degrees=True).as_matrix()
        for d in top+sides:
            np.testing.assert_allclose(rotation@(rotation.T@d),d,atol=1e-12)

    def test_side_rejects_vertical_closing_even_with_horizontal_approach(self):
        checks,_=direction_checks([1,0,0],[0,0,1],[0,0,1],'side')
        self.assertTrue(checks['approach_direction'])
        self.assertFalse(checks['closing_parallel_to_table'])
        checks,_=direction_checks([0,0,1],[0,1,0],[0,0,1],'top')
        self.assertFalse(checks['approach_direction'])

    def test_palm_table_collision_and_missing_contacts_are_separate(self):
        x,y=np.meshgrid(np.linspace(-.01,.01,15),np.linspace(-.01,.01,15))
        target=np.c_[x.ravel(),y.ravel(),np.full(x.size,.04)]
        class Gripper:
            maximum_opening_m=.07
            usable_depth_m=.07
            def meshes(self,pose,width):
                local=np.array([[.09,0,0],[0,.03,0]])
                return [('palm',SimpleNamespace(vertices=local@pose[:3,:3].T+pose[:3,3]))]
        audit=DirectionalAudit(target,[0,0,1],0.,Gripper())
        grasp=CameraGrasp(np.array([0,0,.04]),np.eye(3),.04,.01,.8,'rs1')
        result=audit.audit(grasp,np.eye(4),'side','test')
        self.assertFalse(result['geometry_pass'])
        self.assertFalse(result['checks']['piper_table_clearance'])
        self.assertFalse(result['contact_supported'])
        self.assertIn('preopen_goal/palm',result['part_clearances_m'])
        self.assertIn('closed_goal/palm',result['part_clearances_m'])
        self.assertFalse(result['planning_authorized'])

    def test_real_urdf_clearance_matches_all_vertices_and_includes_pregrasp(self):
        root=Path(__file__).resolve().parents[3]
        gripper=load_gripper((root/'src/piper_description/urdf/piper_description.urdf').read_text())
        x,y=np.meshgrid(np.linspace(-.01,.01,15),np.linspace(-.01,.01,15))
        target=np.c_[x.ravel(),y.ravel(),np.full(x.size,.30)]
        audit=DirectionalAudit(target,[0,0,1],0.,gripper)
        rotation=Rotation.from_euler('y',90,degrees=True).as_matrix()
        grasp=CameraGrasp(np.array([0,0,.30]),rotation,.04,.02,.8,'rs1')
        result=audit.audit(grasp,np.eye(4),'top','real')
        self.assertTrue(result['geometry_pass'],result)
        pose=np.asarray(result['piper_pose_arm_base'])
        direct=min(m.vertices[:,2].min() for _,m in gripper.meshes(pose,result['preopen_width_m']))
        checked=min(v for k,v in result['part_clearances_m'].items() if k.startswith('preopen_goal/'))
        self.assertAlmostEqual(direct,checked)
        self.assertEqual(len(result['part_clearances_m']),9)
        self.assertFalse(result['contact_supported'])


if __name__=='__main__':unittest.main()
