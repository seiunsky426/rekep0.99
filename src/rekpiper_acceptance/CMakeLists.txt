cmake_minimum_required(VERSION 3.0.2)
project(rekpiper_acceptance)

find_package(catkin REQUIRED COMPONENTS rospy)
catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS
  scripts/generate_acceptance_key.py
  scripts/sign_acceptance_artifact.py
  scripts/verify_release_bundle.py
  scripts/install_release_counter.py
  scripts/create_release_bundle.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY schema
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_signatures.py)
endif()
