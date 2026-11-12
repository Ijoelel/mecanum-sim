cmake_minimum_required(VERSION 3.8)
project(mecanum_ros2)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    urdf
    meshes
    worlds
    launch
    config
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/mecanum_controller.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME mecanum_controller
)

ament_package()
