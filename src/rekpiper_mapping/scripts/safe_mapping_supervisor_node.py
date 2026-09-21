#!/usr/bin/env python3
"""Public fail-closed facade for one guarded safe nvblox mapper instance."""

import copy
import threading

import rospy

from rekpiper_msgs.msg import SafeMappingStatus
from rekpiper_msgs.srv import RebuildMap, RebuildMapResponse


class SafeMappingSupervisorNode:
    def __init__(self):
        self._lock = threading.Lock()
        self._last = None
        self._last_received = rospy.Time(0)
        self._timeout = float(rospy.get_param("~mapper_status_timeout_s", 0.5))
        self._publisher = rospy.Publisher(
            "/rekpiper/mapping/safe_status", SafeMappingStatus, queue_size=1, latch=True)
        self._internal_service_name = rospy.get_param(
            "~internal_rebuild_service", "/rekpiper/mapping/rebuild_map")
        self._rebuild_client = rospy.ServiceProxy(
            self._internal_service_name, RebuildMap, persistent=False)
        rospy.Subscriber("/rekpiper/mapping/raw_safe_status", SafeMappingStatus,
                         self._status_callback, queue_size=1)
        rospy.Service("/rekpiper/mapping/rebuild_safe_tsdf", RebuildMap,
                      self._rebuild)
        rospy.Timer(rospy.Duration(0.1), self._watchdog)
        self._publish_unsafe("waiting_for_mapper_safe_status")

    def _status_callback(self, message):
        with self._lock:
            self._last = copy.deepcopy(message)
            self._last_received = rospy.Time.now()
            self._publisher.publish(message)

    def _publish_unsafe(self, reason):
        message = SafeMappingStatus()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = "base_link"
        message.state = SafeMappingStatus.PAUSED_UNSAFE
        message.state_name = "PAUSED_UNSAFE"
        message.map_query_allowed = False
        message.planning_safe = False
        message.reason = str(reason)
        self._publisher.publish(message)

    def _watchdog(self, _event):
        with self._lock:
            if ((rospy.Time.now() - self._last_received).to_sec() > self._timeout):
                self._publish_unsafe("mapper_safe_status_stale")

    def _rebuild(self, request):
        try:
            rospy.wait_for_service(self._internal_service_name, timeout=0.5)
            response = self._rebuild_client(request.reason)
            return RebuildMapResponse(response.success,
                                      response.map_generation_uuid,
                                      response.message)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            self._publish_unsafe("mapper_rebuild_unavailable")
            return RebuildMapResponse(False, "", str(exc))


if __name__ == "__main__":
    rospy.init_node("safe_mapping_supervisor", anonymous=False)
    SafeMappingSupervisorNode()
    rospy.spin()
