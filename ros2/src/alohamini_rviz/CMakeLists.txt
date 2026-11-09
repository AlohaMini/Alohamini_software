cmake_minimum_required(VERSION 3.8)
project(alohamini_rviz)

find_package(ament_cmake REQUIRED)
find_package(rviz_default_plugins REQUIRED)
find_package(pluginlib REQUIRED)

add_library(${PROJECT_NAME} SHARED src/image_display.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(${PROJECT_NAME} rviz_default_plugins pluginlib)
pluginlib_export_plugin_description_file(rviz_common plugins.xml)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_image_shutdown test/test_image_shutdown.cpp
    SKIP_LINKING_MAIN_LIBRARIES TIMEOUT 10 ENV QT_QPA_PLATFORM=offscreen)
  target_link_libraries(test_image_shutdown ${PROJECT_NAME})
  ament_target_dependencies(test_image_shutdown rviz_default_plugins)
endif()

ament_package()
