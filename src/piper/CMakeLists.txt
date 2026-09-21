cmake_minimum_required(VERSION 3.0.2)
project(piper)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs piper_msgs rekpiper_acceptance rospy sensor_msgs std_msgs std_srvs tf)
catkin_package(CATKIN_DEPENDS
  geometry_msgs piper_msgs rekpiper_acceptance rospy sensor_msgs std_msgs std_srvs tf)

catkin_install_python(PROGRAMS
  scripts/piper_ctrl_single_node.py
  scripts/piper_joint_state_readonly_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_piper_joint_state_readonly.py)
endif()
