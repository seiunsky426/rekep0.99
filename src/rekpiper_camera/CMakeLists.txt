cmake_minimum_required(VERSION 3.0.2)
project(rekpiper_camera)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  diagnostic_msgs
  geometry_msgs
  message_filters
  rekpiper_acceptance
  rospy
  sensor_msgs
  std_msgs
  tf2_ros
)

catkin_python_setup()

catkin_package(CATKIN_DEPENDS rekpiper_acceptance)

catkin_install_python(PROGRAMS
  scripts/rgbd_projection_node.py
  scripts/workspace_filter_node.py
  scripts/dual_workspace_cloud_fusion_node.py
  scripts/dual_extrinsics_publisher_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_projection.py)
  catkin_add_nosetests(test/test_workspace.py)
  catkin_add_nosetests(test/test_cloud_fusion.py)
  catkin_add_nosetests(test/test_dual_extrinsics_publisher.py)
  catkin_add_nosetests(test/test_observation_nodes.py)
endif()
