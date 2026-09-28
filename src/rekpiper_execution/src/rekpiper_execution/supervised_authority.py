"""Short-lived supervisor leases, separate from signed autonomous releases."""
import json
import threading
import time


class Lease:
    def __init__(self, session, owner, clock=time.monotonic):
        self.session, self.owner, self.clock = session, owner, clock
        self.sequence = -1
        self.received = -float('inf')
        self.data = {}
        self.lock = threading.RLock()

    def update(self, text, caller, ros_age):
        value = json.loads(text)
        with self.lock:
            if (caller != self.owner or value.get('session_id') != self.session
                    or not 0 <= ros_age <= .5
                    or type(value.get('sequence')) is not int
                    or value['sequence'] <= self.sequence):
                return False
            self.sequence = value['sequence']
            self.data = value
            self.received = self.clock()
            return True

    def check(self, operation='connected', trajectory_sha256=None):
        with self.lock:
            if self.clock()-self.received > .5 or not self.data.get('alive'):
                raise ValueError('supervised_session_lease_expired')
            if operation != 'connected':
                if not self.data.get('executing') or self.data.get('operation') != operation:
                    raise ValueError('supervised_segment_not_authorized')
                if trajectory_sha256 is not None and self.data.get('trajectory_sha256') != trajectory_sha256:
                    raise ValueError('supervised_trajectory_not_the_reviewed_preview')
            return dict(self.data)


class RosLease:
    def __init__(self):
        import rospy
        from std_msgs.msg import String
        session = rospy.get_param('~supervised_session_id', '')
        if not session:
            raise ValueError('supervised_session_id_required')
        self.lease = Lease(session, '/rekpiper/supervised/session')
        self.subscriber = rospy.Subscriber('/rekpiper/supervised/lease', String,
                                            self.callback, queue_size=1)
        deadline = time.monotonic()+3.
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            try:
                self.check()
                return
            except ValueError:
                time.sleep(.02)
        raise ValueError('supervised_session_owner_unavailable')

    def callback(self, message):
        import rospy
        try:
            value = json.loads(message.data)
            self.lease.update(message.data, message._connection_header.get('callerid'),
                              rospy.Time.now().to_sec()-float(value['stamp']))
        except (ValueError, KeyError, TypeError):
            return

    def check(self, operation='connected', trajectory_sha256=None):
        return self.lease.check(operation, trajectory_sha256)
