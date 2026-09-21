cmake_minimum_required(VERSION 3.0.2)
project(rekpiper_calibration)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  actionlib
  control_msgs
  geometry_msgs
  message_filters
  rekpiper_camera
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf2_ros
  trajectory_msgs
)

catkin_python_setup()
catkin_package(CATKIN_DEPENDS
  actionlib control_msgs cv_bridge geometry_msgs message_filters rekpiper_camera rospy sensor_msgs
  std_msgs std_srvs tf2_ros trajectory_msgs)

catkin_install_python(PROGRAMS
  scripts/approve_camera_extrinsics.py
  scripts/dual_aruco18_capture_node.py
  scripts/solve_dual_aruco18_dataset.py
  scripts/moved_checkerboard_crosscheck_node.py
  scripts/camera_validation_node.py
  scripts/calibration_preview_node.py
  scripts/manual_teaching_recorder_node.py
  scripts/calibration_motion_node.py
  scripts/calibration_motion_client.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_camera_validation.py)
  catkin_add_nosetests(test/test_checkerboard_calibration.py)
  catkin_add_nosetests(test/test_eye_to_hand.py)
  catkin_add_nosetests(test/test_robot_world_handeye.py)
  catkin_add_nosetests(test/test_trajectory_preview.py)
  catkin_add_nosetests(test/test_workspace_bounds.py)
  catkin_add_nosetests(test/test_teaching_checks.py)
  catkin_add_nosetests(test/test_calibration_motion.py)
endif()
