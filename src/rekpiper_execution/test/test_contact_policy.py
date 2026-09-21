import unittest
from types import SimpleNamespace
import numpy as np
from rekpiper_execution.contact_policy import validate_preopen,validate_support_geometry,TargetContactPolicy


class ContactPolicyTest(unittest.TestCase):
    def test_contact_permission_never_covers_palm_or_neighbor(self):
        class IK:
            def forward(self,q): return np.eye(4)
            def _pose_errors(self,a,b): return 0.,0.
        target=np.zeros((150,3)); target[:75,0]=.02; target[75:,0]=-.02
        contacts=[SimpleNamespace(x=.02,y=0.,z=0.),SimpleNamespace(x=-.02,y=0.,z=0.)]
        scene=np.vstack([target,[[.2,.2,.2]]])
        policy=TargetContactPolicy(IK(),SimpleNamespace(contact_points=contacts),target,scene,
                                   ['finger'],.04,np.eye(4))
        result=policy.allowed(np.zeros(6),np.array([[.02,0,0],[.02,0,0],[.2,.2,.2]]),
                              np.full(3,.001),np.array(['finger','palm','finger']))
        self.assertEqual(result.tolist(),[True,False,False])

    def test_preopen_requires_measured_opening(self):
        with self.assertRaisesRegex(RuntimeError,'preopen'):
            validate_preopen(SimpleNamespace(success=True,final_opening_m=.01),.05)
        validate_preopen(SimpleNamespace(success=True,final_opening_m=.05),.05)
    def support(self):
        x,y=np.meshgrid(np.linspace(-.06,.06,10),np.linspace(-.06,.06,10))
        return np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    def test_supported_release(self):
        obj=self.support()*.2; obj[:,2]=.001
        self.assertTrue(validate_support_geometry(obj,self.support(),.001)['supported'])
    def test_two_centimeter_air_gap_rejected(self):
        obj=self.support()*.2; obj[:,2]=.02
        with self.assertRaisesRegex(ValueError,'conflicts'):
            validate_support_geometry(obj,self.support(),.001)
    def test_wrong_support_or_uncertain_geometry_rejected(self):
        obj=self.support()*.2; obj[:,0]+=.1
        with self.assertRaisesRegex(ValueError,'outside'):
            validate_support_geometry(obj,self.support(),.001)
        with self.assertRaises(ValueError):
            validate_support_geometry(obj,self.support(),.02)


if __name__=='__main__': unittest.main()
