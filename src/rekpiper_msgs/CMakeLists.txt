cmake_minimum_required(VERSION 3.0.2)
project(rekpiper_msgs)

find_package(catkin REQUIRED COMPONENTS
  actionlib_msgs
  geometry_msgs
  message_generation
  sensor_msgs
  std_msgs
  trajectory_msgs
)

add_action_files(FILES
  CommandGripper.action
  ExecuteReKep.action
)

add_message_files(FILES
  GraspCandidate.msg
  GraspCandidateArray.msg
  InstanceGeometry.msg
  InstanceGeometryArray.msg
  Keypoint3D.msg
  Keypoint3DArray.msg
  ObjectSeed.msg
  SceneSnapshot.msg
  SDFGrid.msg
  ReKepProgram.msg
  ReKepHorizon.msg
  ClosedLoopStatus.msg
  SafeMappingStatus.msg
  TrackedObject.msg
  TrackedObjectArray.msg
  TrackedObjectCloud.msg
  TrackedObjectCloudArray.msg
)

add_service_files(FILES
  ApproveReKepProgram.srv
  CaptureSDFSnapshot.srv
  GenerateReKepProgram.srv
  ObjectCommand.srv
  QuerySDF.srv
  RebuildMap.srv
  RegisterObject.srv
  ResolveObjectState.srv
)

generate_messages(DEPENDENCIES actionlib_msgs geometry_msgs sensor_msgs std_msgs trajectory_msgs)

catkin_package(CATKIN_DEPENDS actionlib_msgs geometry_msgs message_runtime sensor_msgs std_msgs trajectory_msgs)
