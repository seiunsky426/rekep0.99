cmake_minimum_required(VERSION 3.0.2)
project(rekpiper_mapping)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  diagnostic_msgs
  geometry_msgs
  message_filters
  rekpiper_acceptance
  rekpiper_camera
  rekpiper_msgs
  rospy
  sensor_msgs
  std_msgs
  tf2_ros
)
catkin_python_setup()
catkin_package(CATKIN_DEPENDS
  cv_bridge diagnostic_msgs geometry_msgs message_filters rekpiper_acceptance rekpiper_camera
  rekpiper_msgs rospy sensor_msgs std_msgs tf2_ros
)

catkin_install_python(PROGRAMS
  scripts/curobo_robot_segmenter_node.py
  scripts/nvblox_mapping_node.py
  scripts/safe_mapping_supervisor_node.py
  scripts/sdf_grid_snapshot_node.py
  scripts/mapping_launch_guard_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_session_guard.py)
  catkin_add_nosetests(test/test_frame_preparation.py)
  catkin_add_nosetests(test/test_gripper_obstacle_range.py)
  catkin_add_nosetests(test/test_safe_map_state.py)
  catkin_add_nosetests(test/test_sdf_conventions.py)
  catkin_add_nosetests(test/test_mapping_input_queue.py)
  catkin_add_nosetests(test/test_stepwise_capture.py)
endif()
