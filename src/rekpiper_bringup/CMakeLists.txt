cmake_minimum_required(VERSION 3.0.2)
project(rekpiper_bringup)
find_package(catkin REQUIRED)
get_filename_component(REKPIPER_SOURCE_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
catkin_add_env_hooks(99.rekpiper SHELLS sh DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks)
catkin_package()
catkin_install_python(PROGRAMS
  scripts/preflight_node.py
  scripts/preflight_cli.py
  scripts/start_rekpiper.py
  scripts/hardware_supervisor_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
# A relocated install space is a complete source-archive deployment surface.
# Catkin owns install/setup.bash; the environment hook below derives paths
# from the relocated prefix instead of embedding the build machine path.
install(DIRECTORY ${REKPIPER_SOURCE_ROOT}/third_party
                  ${REKPIPER_SOURCE_ROOT}/docs
                  ${REKPIPER_SOURCE_ROOT}/tools
        DESTINATION ${CMAKE_INSTALL_PREFIX}
        USE_SOURCE_PERMISSIONS
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE)
install(FILES ${REKPIPER_SOURCE_ROOT}/requirements.in
              ${REKPIPER_SOURCE_ROOT}/requirements.lock
              ${REKPIPER_SOURCE_ROOT}/requirements.platform.in
              ${REKPIPER_SOURCE_ROOT}/requirements.platform.lock
              ${REKPIPER_SOURCE_ROOT}/requirements.models.in
              ${REKPIPER_SOURCE_ROOT}/requirements.models.lock
        DESTINATION ${CMAKE_INSTALL_PREFIX})
if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_production_surface.py)
  catkin_add_nosetests(test/test_cross_node_contracts.py)
  catkin_add_nosetests(test/test_repository_layout.py)
endif()
