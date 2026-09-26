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
find_package(OpenCV REQUIRED COMPONENTS core imgproc video calib3d)

add_message_files(FILES PickState.msg)
generate_messages(DEPENDENCIES std_msgs)
catkin_package(CATKIN_DEPENDS ${DEPS} message_runtime)

include_directories(include ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${OMPL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_library(${PROJECT_NAME} src/arm.cpp src/bpl.cpp src/cloud.cpp src/collision.cpp src/driver.cpp src/follow.cpp src/fsm.cpp src/park.cpp src/pick.cpp src/plan.cpp src/track.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OMPL_LIBRARIES} ${OpenCV_LIBRARIES})

# driver and pick are ROS-free; roswrapper.cpp is all the ROS1 around them, built once per executable.
foreach(node driver pick)
  string(TOUPPER ${node} NODE)
  add_executable(${node} src/roswrapper.cpp)
  target_compile_definitions(${node} PRIVATE SEP23_${NODE})
  target_link_libraries(${node} ${PROJECT_NAME})
  add_dependencies(${node} ${PROJECT_NAME}_generate_messages_cpp)
endforeach()

catkin_install_python(PROGRAMS exp/synthetic.py exp/occluder.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

if(CATKIN_ENABLE_TESTING)
  find_package(kdl_parser REQUIRED)
  catkin_add_gtest(sanity test/sanity.cpp)  # does each part give the right answer: catkin test sep23
  target_include_directories(sanity PRIVATE ${kdl_parser_INCLUDE_DIRS})
  target_link_libraries(sanity ${PROJECT_NAME} ${kdl_parser_LIBRARIES})
  target_compile_definitions(sanity PRIVATE SEP23_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_executable(runtime test/runtime.cpp)  # how long each stage takes: devel/lib/sep23/runtime
  target_link_libraries(runtime ${PROJECT_NAME})
  target_compile_definitions(runtime PRIVATE SEP23_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
endif()
