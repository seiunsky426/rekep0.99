import unittest
import numpy as np
from rekpiper_planning.motion_backend import MotionBackend, MotionBackendError


def robot_xml():
    xml='<robot name="test"><link name="base"/>'
    for i in range(1,7):
        xml+='<link name="l{}"/>'.format(i)
        xml+='''<joint name="joint{}" type="revolute"><parent link="{}"/>
          <child link="l{}"/><axis xyz="0 0 1"/><limit lower="-2" upper="2"
          effort="1" velocity="1"/></joint>'''.format(i,'base' if i==1 else 'l'+str(i-1),i)
    return xml+'</robot>'


class MotionBackendTest(unittest.TestCase):
    def setUp(self): self.backend=MotionBackend(robot_xml())
    def tearDown(self): self.backend.close()
    def test_exact_path_uses_checked_edges(self):
        visited=[]
        def check(q): visited.append(q.copy()); return True
        result=self.backend.plan(np.full(6,-2.),np.full(6,2.),np.zeros(6),
                                 np.full(6,.1),check,.5,7)
        np.testing.assert_allclose(result[0],0,atol=1e-9)
        np.testing.assert_allclose(result[-1],.1,atol=1e-9)
        self.assertLessEqual(np.max(np.abs(np.diff(result,axis=0))),.02000001)
        self.assertGreater(len(visited),2)
    def test_no_path_fails_closed(self):
        with self.assertRaises(MotionBackendError):
            self.backend.plan(np.full(6,-2.),np.full(6,2.),np.zeros(6),
                              np.full(6,.1),lambda q:False,.05,7)
    def test_callback_exception_is_not_swallowed_as_success(self):
        def check(q): raise ValueError('bad_scene')
        with self.assertRaisesRegex(MotionBackendError,'bad_scene'):
            self.backend.plan(np.full(6,-2.),np.full(6,2.),np.zeros(6),
                              np.full(6,.1),check,.05,7)

    def test_thin_obstacle_between_valid_endpoints_is_not_skipped(self):
        with self.assertRaises(MotionBackendError):
            self.backend.plan(np.full(6,-2.),np.full(6,2.),np.zeros(6),
                np.full(6,.1),lambda q:not .045<q[0]<.055,.10,3)


if __name__=='__main__': unittest.main()
