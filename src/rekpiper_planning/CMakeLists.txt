cmake_minimum_required(VERSION 3.0.2)
project(rekpiper_planning)
add_compile_options(-std=c++14)
find_package(ompl REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  rekpiper_msgs
  rekpiper_acceptance
  rospy
  sensor_msgs
  std_msgs
  tf2_ros
  trajectory_msgs
  visualization_msgs
)
catkin_python_setup()

include_directories(${OMPL_INCLUDE_DIRS} ${moveit_core_INCLUDE_DIRS}
  ${moveit_ros_planning_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
add_library(rekpiper_motion_backend SHARED src/motion_backend.cpp)
target_link_libraries(rekpiper_motion_backend ${OMPL_LIBRARIES}
  ${moveit_core_LIBRARIES} ${moveit_ros_planning_LIBRARIES} ${catkin_LIBRARIES})
install(TARGETS rekpiper_motion_backend LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

catkin_package(CATKIN_DEPENDS
  cv_bridge geometry_msgs rekpiper_msgs rekpiper_acceptance rospy
  sensor_msgs std_msgs tf2_ros trajectory_msgs visualization_msgs
)

catkin_install_python(PROGRAMS
  scripts/approve_paper_real_weights.py
  scripts/approve_rekep_program.py
  scripts/calibrate_paper_real_weights.py
  scripts/capture_stepwise_sdf.py
  scripts/m8_offline_preview_node.py
  scripts/m8_three_stage_preview_node.py
  scripts/program_server_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_offline_observation.py)
  catkin_add_nosetests(test/test_continuous_ik.py)
  catkin_add_nosetests(test/test_ik_path_diagnostics.py)
  catkin_add_nosetests(test/test_motion_backend.py DEPENDENCIES rekpiper_motion_backend)
  catkin_add_nosetests(test/test_official_program.py)
  catkin_add_nosetests(test/test_realtime_stage_entry.py)
  catkin_add_nosetests(test/test_official_upstream.py)
  catkin_add_nosetests(test/test_paper_real_solver.py)
  catkin_add_nosetests(test/test_piper_collision_sampling.py)
  catkin_add_nosetests(test/test_piper_geometry_model.py)
  catkin_add_nosetests(test/test_piper_urdf_ik.py)
  catkin_add_nosetests(test/test_trajectory_audit.py)
  catkin_add_nosetests(test/test_planning_snapshot.py)
  catkin_add_nosetests(test/test_grasp_preview.py)
  catkin_add_nosetests(test/test_vlm_backend.py)
endif()
