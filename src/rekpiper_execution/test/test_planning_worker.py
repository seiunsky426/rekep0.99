import unittest
import time
from rekpiper_execution.planning_worker import PlanningWorker, validate_returned_context


class PlanningWorkerTest(unittest.TestCase):
    def test_success(self):
        worker=PlanningWorker(1.)
        worker.start('a',{},lambda:42)
        result=None
        for _ in range(100):
            result=worker.poll()
            if result is not None: break
            time.sleep(.005)
        worker.cancel()
        self.assertEqual(result,42)
    def test_timeout_cancels_child(self):
        worker=PlanningWorker(.02)
        worker.start('a',{},lambda:time.sleep(1))
        time.sleep(.03)
        with self.assertRaisesRegex(RuntimeError,'timeout'): worker.poll()
        self.assertIsNone(worker.process)
    def test_error_propagates(self):
        def fail(): raise ValueError('bad scene')
        worker=PlanningWorker(1.)
        worker.start('a',{},fail)
        time.sleep(.03)
        with self.assertRaisesRegex(RuntimeError,'bad scene'): worker.poll()
    def test_changed_identity_and_joint_state_rejected(self):
        old=dict(identity=('object',),joints=[0]*6,keypoints=[[0,0,0]])
        for new in [dict(old,identity=('other',)),dict(old,joints=[.1]*6)]:
            with self.assertRaises(RuntimeError): validate_returned_context(old,new)


if __name__=='__main__': unittest.main()
