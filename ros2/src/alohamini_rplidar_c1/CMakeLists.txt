cmake_minimum_required(VERSION 3.16)
project(alohamini_rplidar_c1)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(Threads REQUIRED)

file(GLOB RPLIDAR_SDK_SRC CONFIGURE_DEPENDS
  "sdk/src/arch/linux/*.cpp"
  "sdk/src/dataunpacker/*.cpp"
  "sdk/src/dataunpacker/unpacker/*.cpp"
  "sdk/src/hal/*.cpp"
  "sdk/src/*.cpp"
)

add_library(rplidar_sdk STATIC ${RPLIDAR_SDK_SRC})
target_compile_features(rplidar_sdk PUBLIC cxx_std_17)
target_include_directories(rplidar_sdk
  PUBLIC
    sdk/include
    sdk/src
)
target_link_libraries(rplidar_sdk PUBLIC Threads::Threads)

add_executable(rplidar_c1_node src/rplidar_c1_node.cpp)
target_compile_features(rplidar_c1_node PUBLIC cxx_std_17)
ament_target_dependencies(rplidar_c1_node rclcpp sensor_msgs std_srvs)
target_link_libraries(rplidar_c1_node rplidar_sdk)

add_executable(rplidar_c1_scan_test src/rplidar_c1_scan_test.cpp)
target_compile_features(rplidar_c1_scan_test PUBLIC cxx_std_17)
ament_target_dependencies(rplidar_c1_scan_test rclcpp sensor_msgs)

install(TARGETS
  rplidar_c1_node
  rplidar_c1_scan_test
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY udev
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  README.md
  SLAMTEC_SDK_LICENSE
  SOURCE.json
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
