cmake_minimum_required(VERSION 3.0.2)
project(sep23)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)  # Eigen is ~100x slower unoptimised
endif()
add_compile_options(-std=c++17 -Wall -Wextra)

set(DEPS geometry_msgs message_filters roscpp sensor_msgs std_msgs std_srvs tf2_eigen tf2_ros urdf visualization_msgs)
find_package(catkin REQUIRED COMPONENTS ${DEPS} message_generation)
find_package(ompl REQUIRED)
find_package(Eigen3 REQUIRED)

add_message_files(FILES PickState.msg)
generate_messages(DEPENDENCIES std_msgs)
catkin_package(CATKIN_DEPENDS ${DEPS} message_runtime)

include_directories(include ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${OMPL_INCLUDE_DIRS})

add_library(${PROJECT_NAME} src/arm.cpp src/bpl.cpp src/cloud.cpp src/collision.cpp src/follow.cpp src/fsm.cpp src/park.cpp src/plan.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OMPL_LIBRARIES})

foreach(node driver pick)
  add_executable(${node} nodes/${node}.cpp)
  target_link_libraries(${node} ${PROJECT_NAME})
  add_dependencies(${node} ${PROJECT_NAME}_generate_messages_cpp)
endforeach()

if(CATKIN_ENABLE_TESTING)
  find_package(kdl_parser REQUIRED)
  catkin_add_gtest(${PROJECT_NAME}_test test/test_sep23.cpp)
  target_include_directories(${PROJECT_NAME}_test PRIVATE ${kdl_parser_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME} ${kdl_parser_LIBRARIES})
  target_compile_definitions(${PROJECT_NAME}_test PRIVATE SEP23_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
endif()
