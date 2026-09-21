cmake_minimum_required(VERSION 3.0.2)
project(rekpiper_perception)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  message_filters
  piper_msgs
  rekpiper_acceptance
  rekpiper_camera
  rekpiper_msgs
  rospy
  sensor_msgs
  std_msgs
  tf2_ros
  visualization_msgs
)

catkin_python_setup()
catkin_package(CATKIN_DEPENDS
  cv_bridge geometry_msgs message_filters piper_msgs rekpiper_acceptance rekpiper_camera
  rekpiper_msgs rospy sensor_msgs std_msgs tf2_ros visualization_msgs)

catkin_install_python(PROGRAMS
  scripts/task_perception_node.py
  scripts/scene_snapshot_node.py
  scripts/snapshot_object_registrar_node.py
  scripts/dual_dino_tracker_node.py
  scripts/multicamera_object_tracker_node.py
  scripts/object_registry_node.py
  scripts/grasp_state_monitor_node.py
  scripts/keypoint_trajectory_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_grasp_state.py)
  catkin_add_nosetests(test/test_object_registry.py)
  catkin_add_nosetests(test/test_official_keypoint_adapter.py)
  catkin_add_nosetests(test/test_scene_change.py)
  catkin_add_nosetests(test/test_keypoint_tracking.py)
  catkin_add_nosetests(test/test_tracking_geometry.py)
  catkin_add_nosetests(test/test_task_geometry.py)
  catkin_add_nosetests(test/test_sam_arbitration.py)
  catkin_add_nosetests(test/test_sam_source.py)
  catkin_add_nosetests(test/test_snapshot_contract.py)
  catkin_add_nosetests(test/test_manual_tracking.py)
endif()
