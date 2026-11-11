cmake_minimum_required(VERSION 3.10)
project(dyna_rc)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(DYNA_RC_BUILD_BASELINES "Build the bundled RaPlace and ReFeree adapters" OFF)

find_package(catkin REQUIRED COMPONENTS
  roscpp rospy sensor_msgs geometry_msgs nav_msgs std_msgs tf pcl_ros pcl_conversions)
find_package(PCL 1.10 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

catkin_package(INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp rospy sensor_msgs geometry_msgs nav_msgs std_msgs tf pcl_ros pcl_conversions
  DEPENDS PCL)
include_directories(include ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(dyna_rc_scancontext STATIC src/Scancontext.cpp)
target_link_libraries(dyna_rc_scancontext PUBLIC ${catkin_LIBRARIES} ${PCL_LIBRARIES} Eigen3::Eigen)

add_executable(place_recognition src/place_recognition.cpp)
add_executable(seq_place_recognition src/seq_place_recognition.cpp src/seq_relocalization_enhanced.cpp)
add_executable(non_neighbor_precompute src/non_neighbor_precompute.cpp)
foreach(node place_recognition seq_place_recognition non_neighbor_precompute)
  target_link_libraries(${node} dyna_rc_scancontext)
endforeach()

add_executable(radar_ego_motion src/radar_ego_motion.cpp)
target_compile_definitions(radar_ego_motion PRIVATE PCL_NO_PRECOMPILE)
target_link_libraries(radar_ego_motion ${catkin_LIBRARIES} ${PCL_LIBRARIES} Eigen3::Eigen)
add_executable(pointcloud_converter_node src/pointcloud_converter_node.cpp)
target_link_libraries(pointcloud_converter_node ${catkin_LIBRARIES})
add_executable(gt_publisher src/gt_publisher.cpp)
target_link_libraries(gt_publisher ${catkin_LIBRARIES})

set(DYNA_RC_NODES place_recognition seq_place_recognition non_neighbor_precompute
  radar_ego_motion pointcloud_converter_node gt_publisher)
if(DYNA_RC_BUILD_BASELINES)
  find_package(OpenCV 4 REQUIRED)
  add_library(raplace_core STATIC third_party/raplace/raplace.cpp)
  target_include_directories(raplace_core PUBLIC third_party/raplace ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(raplace_core PUBLIC ${OpenCV_LIBRARIES})
  add_executable(raplace_place_recognition src/raplace_place_recognition.cpp)
  target_link_libraries(raplace_place_recognition raplace_core ${catkin_LIBRARIES} ${PCL_LIBRARIES})

  add_library(referee_core STATIC third_party/referee/referee.cpp)
  target_include_directories(referee_core PUBLIC third_party/referee)
  target_link_libraries(referee_core PUBLIC Eigen3::Eigen)
  add_executable(referee_place_recognition src/referee_place_recognition.cpp)
  target_include_directories(referee_place_recognition PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(referee_place_recognition referee_core ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
  list(APPEND DYNA_RC_NODES raplace_place_recognition referee_place_recognition)
endif()

foreach(node ${DYNA_RC_NODES})
  add_dependencies(${node} ${catkin_EXPORTED_TARGETS})
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    target_link_libraries(${node} stdc++fs)
  endif()
endforeach()
install(TARGETS ${DYNA_RC_NODES} RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY include/ DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION})
install(DIRECTORY launch config docs DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(FILES README.md LICENSE_NOTICE.md THIRD_PARTY_NOTICES.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY third_party/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/third_party
  FILES_MATCHING PATTERN LICENSE PATTERN SOURCE.txt)
catkin_install_python(PROGRAMS scripts/localization_recall_recorder.py
  scripts/localization_recall_comparison.py scripts/localization_error_histograms.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(PROGRAMS scripts/print_environment.sh DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_ntu_pointcloud_conversion tests/test_ntu_pointcloud_conversion.cpp)
  if(TARGET test_ntu_pointcloud_conversion)
    target_link_libraries(test_ntu_pointcloud_conversion ${catkin_LIBRARIES})
  endif()
  catkin_add_gtest(test_runtime_timing tests/test_runtime_timing.cpp)
  catkin_add_gtest(test_seq_static_mode tests/test_seq_static_mode.cpp src/seq_relocalization_enhanced.cpp)
  if(TARGET test_seq_static_mode)
    target_link_libraries(test_seq_static_mode dyna_rc_scancontext)
  endif()
  if(DYNA_RC_BUILD_BASELINES)
    catkin_add_gtest(test_raplace tests/test_raplace.cpp)
    if(TARGET test_raplace)
      target_link_libraries(test_raplace raplace_core ${PCL_LIBRARIES})
    endif()
  endif()
  find_package(rostest REQUIRED)
  catkin_install_python(PROGRAMS tests/test_radar_preprocessing.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
  add_rostest(tests/radar_preprocessing.test DEPENDENCIES radar_ego_motion)
endif()
