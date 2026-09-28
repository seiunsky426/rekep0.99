cmake_minimum_required(VERSION 3.0.2)
project(rekpiper_execution)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  control_msgs
  geometry_msgs
  message_filters
  interactive_markers
  visualization_msgs
  piper_msgs
  rekpiper_acceptance
  rekpiper_planning
  rekpiper_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf2_ros
  trajectory_msgs
)

catkin_python_setup()

catkin_package(CATKIN_DEPENDS
  actionlib control_msgs geometry_msgs message_filters interactive_markers visualization_msgs piper_msgs rekpiper_acceptance rekpiper_planning
  rekpiper_msgs rospy sensor_msgs std_msgs std_srvs tf2_ros trajectory_msgs
)

catkin_install_python(PROGRAMS
  scripts/approve_runtime_performance.py
  scripts/closed_loop_node.py
  scripts/collect_runtime_performance.py
  scripts/gripper_joint_state_adapter_node.py
  scripts/hardware_hold_stream_node.py
  scripts/piper_trajectory_bridge_node.py
  scripts/piper_gripper_action_node.py
  scripts/software_trajectory_sink_node.py
  scripts/verify_piper_ready.py
  scripts/supervised_workbench_node.py
  scripts/supervised_panel_node.py
  scripts/dual_mask_cloud_review_node.py
  scripts/rs3_anygrasp_review_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_planning_worker.py)
  catkin_add_nosetests(test/test_contact_policy.py)
  catkin_add_nosetests(test/test_gripper.py)
  catkin_add_nosetests(test/test_gripper_joint_state_adapter.py)
  catkin_add_nosetests(test/test_grasp_approach.py)
  catkin_add_nosetests(test/test_gripper_close.py)
  catkin_add_nosetests(test/test_lifecycle_contract.py)
  catkin_add_nosetests(test/test_coordinator.py)
  catkin_add_nosetests(test/test_closed_loop_construction.py)
  catkin_add_nosetests(test/test_trajectory_validation.py)
  catkin_add_nosetests(test/test_rate_monitor.py)
  catkin_add_nosetests(test/test_runtime_acceptance.py)
  catkin_add_nosetests(test/test_runtime_evidence.py)
  catkin_add_nosetests(test/test_supervised_session.py)
  catkin_add_nosetests(test/test_supervised_anygrasp.py)
  catkin_add_nosetests(test/test_supervised_workbench.py)
endif()
