cmake_minimum_required(VERSION 3.0.2)
project(rekpiper_grasp)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  message_filters
  rekpiper_planning
  rekpiper_msgs
  rekpiper_perception
  rospy
  sensor_msgs
  std_msgs
  tf2_ros
)

catkin_python_setup()

catkin_package(CATKIN_DEPENDS
  cv_bridge geometry_msgs message_filters rekpiper_planning rekpiper_msgs
  rekpiper_perception rospy sensor_msgs std_msgs tf2_ros
)

catkin_install_python(PROGRAMS
  scripts/keypoint_anygrasp_selector_node.py
  scripts/keypoint_anygrasp_node.py
  scripts/anygrasp_runtime_preflight.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS
  scripts/bootstrap_anygrasp_runtime.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_grasp_geometry.py)
  catkin_add_nosetests(test/test_blue_cloud_comparison.py)
  catkin_add_nosetests(test/test_directional_grasp.py)
  catkin_add_nosetests(test/test_target_geometry.py)
  catkin_add_nosetests(test/test_horizontal_grasp.py)
  catkin_add_nosetests(test/test_selection.py)
endif()
