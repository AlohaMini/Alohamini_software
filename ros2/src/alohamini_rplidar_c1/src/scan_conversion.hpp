// Adapted from Slamtec rplidar_ros/src/node.cpp; see SLAMTEC_SDK_LICENSE.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sl_lidar.h"

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;

double node_angle_deg(const sl_lidar_response_measurement_node_hq_t & node)
{
  return static_cast<double>(node.angle_z_q14) * 90.0 / 16384.0;
}

float node_range_m(const sl_lidar_response_measurement_node_hq_t & node)
{
  if (node.dist_mm_q2 == 0) {
    return std::numeric_limits<float>::infinity();
  }
  return static_cast<float>(node.dist_mm_q2) / 4.0F / 1000.0F;
}

float node_intensity(const sl_lidar_response_measurement_node_hq_t & node)
{
  return static_cast<float>(node.quality >> 2);
}

}  // namespace

inline std::optional<sensor_msgs::msg::LaserScan> make_scan(
  const std::vector<sl_lidar_response_measurement_node_hq_t> & nodes,
  size_t count, const rclcpp::Time & stamp, double scan_duration,
  bool angle_compensate, int angle_compensate_multiple, bool inverted,
  double scan_frequency, float max_distance, const std::string & frame_id)
{
    if (count < 2) {
      return std::nullopt;
    }

    std::vector<sl_lidar_response_measurement_node_hq_t> compensated;
    double angle_min = 0.0;
    double angle_max = 360.0;
    const sl_lidar_response_measurement_node_hq_t * scan_nodes = nodes.data();
    size_t scan_count = count;

    if (angle_compensate) {
      scan_count = static_cast<size_t>(360 * angle_compensate_multiple);
      compensated.assign(scan_count, sl_lidar_response_measurement_node_hq_t{});
      int angle_offset = 0;
      for (size_t i = 0; i < count; ++i) {
        if (nodes[i].dist_mm_q2 == 0) {
          continue;
        }
        const int angle_value = static_cast<int>(node_angle_deg(nodes[i]) * angle_compensate_multiple);
        angle_offset = std::min(angle_offset, angle_value);
        for (int j = 0; j < angle_compensate_multiple; ++j) {
          int index = angle_value - angle_offset + j;
          index = std::clamp(index, 0, static_cast<int>(scan_count) - 1);
          compensated[static_cast<size_t>(index)] = nodes[i];
        }
      }
      scan_nodes = compensated.data();
    } else {
      size_t first = 0;
      while (first < count && nodes[first].dist_mm_q2 == 0) {
        ++first;
      }
      size_t last = count;
      while (last > first && nodes[last - 1].dist_mm_q2 == 0) {
        --last;
      }
      if (first < last) {
        scan_nodes = nodes.data() + first;
        scan_count = last - first;
        angle_min = node_angle_deg(nodes[first]);
        angle_max = node_angle_deg(nodes[last - 1]);
      }
    }

    if (scan_count < 2) {
      return std::nullopt;
    }

    sensor_msgs::msg::LaserScan message;
    message.header.stamp = stamp;
    message.header.frame_id = frame_id;
    const bool reversed = angle_max > angle_min;
    if (reversed) {
      message.angle_min = static_cast<float>((180.0 - angle_max) * kDegToRad);
      message.angle_max = static_cast<float>((180.0 - angle_min) * kDegToRad);
    } else {
      message.angle_min = static_cast<float>((180.0 - angle_min) * kDegToRad);
      message.angle_max = static_cast<float>((180.0 - angle_max) * kDegToRad);
    }
    message.angle_increment =
      (message.angle_max - message.angle_min) / static_cast<float>(scan_count - 1);
    message.scan_time = static_cast<float>(
      scan_duration > 0.0 ? scan_duration : 1.0 / scan_frequency);
    message.time_increment = message.scan_time / static_cast<float>(scan_count - 1);
    message.range_min = 0.15F;
    message.range_max = max_distance;
    message.ranges.resize(scan_count);
    message.intensities.resize(scan_count);

    const bool reverse_data = (!inverted && reversed) || (inverted && !reversed);
    for (size_t i = 0; i < scan_count; ++i) {
      const size_t output_index = reverse_data ? scan_count - 1 - i : i;
      message.ranges[output_index] = node_range_m(scan_nodes[i]);
      message.intensities[output_index] = node_intensity(scan_nodes[i]);
    }
    return message;
}
