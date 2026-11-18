/*
 * Copyright (c) 2009 - 2014 RoboPeak Team
 * Copyright (c) 2014 - 2018 Shanghai Slamtec Co., Ltd.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * 
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*
 * ROS 2 scan monitor migrated from rplidar_ros/src/client.cpp.
 * It is deliberately small: it verifies that the driver is publishing live,
 * ordered LaserScan messages before the robot is driven or used for mapping.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

namespace
{
constexpr double kPi = 3.14159265358979323846;
}

class RplidarC1ScanTest final : public rclcpp::Node
{
public:
  RplidarC1ScanTest()
  : Node("rplidar_c1_scan_test"), scan_count_(0)
  {
    scan_topic_ = declare_parameter<std::string>("scan_topic", "/scan");
    report_period_sec_ = declare_parameter<double>("report_period_sec", 1.0);
    startup_timeout_sec_ = declare_parameter<double>("startup_timeout_sec", 10.0);
    if (report_period_sec_ <= 0.0) {
      report_period_sec_ = 1.0;
    }
    if (startup_timeout_sec_ < 0.0) {
      startup_timeout_sec_ = 0.0;
    }
    last_report_ = now();
    subscription_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic_, rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::LaserScan::SharedPtr message) {
        on_scan(*message);
      });
    if (startup_timeout_sec_ > 0.0) {
      startup_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(startup_timeout_sec_)),
        [this]() {
          if (!received_scan_) {
            RCLCPP_ERROR(
              get_logger(), "No LaserScan received on %s within %.1f seconds",
              scan_topic_.c_str(), startup_timeout_sec_);
            passed_ = false;
            rclcpp::shutdown();
          }
          startup_timer_->cancel();
        });
    }
    RCLCPP_INFO(get_logger(), "Monitoring %s", scan_topic_.c_str());
  }

private:
  void on_scan(const sensor_msgs::msg::LaserScan & message)
  {
    if (message.header.frame_id.empty() || message.ranges.size() < 2 ||
      !std::isfinite(message.angle_min) || !std::isfinite(message.angle_max) ||
      !std::isfinite(message.angle_increment) || message.angle_increment <= 0.0F ||
      !std::isfinite(message.range_min) || !std::isfinite(message.range_max) ||
      message.range_min <= 0.0F || message.range_max <= message.range_min) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Received an invalid LaserScan on %s; waiting for a valid C1 scan", scan_topic_.c_str());
      return;
    }
    received_scan_ = true;
    ++scan_count_;
    size_t valid_count = 0;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = 0.0F;
    for (const float range : message.ranges) {
      if (std::isfinite(range) && range >= message.range_min && range <= message.range_max) {
        ++valid_count;
        minimum = std::min(minimum, range);
        maximum = std::max(maximum, range);
      }
    }

    const auto current_time = now();
    if ((current_time - last_report_).seconds() < report_period_sec_) {
      return;
    }
    const double elapsed = (current_time - last_report_).seconds();
    const double frequency = elapsed > 0.0 ? static_cast<double>(scan_count_) / elapsed : 0.0;
    const double angle_min_deg = message.angle_min * 180.0 / kPi;
    const double angle_max_deg = message.angle_max * 180.0 / kPi;
    RCLCPP_INFO(
      get_logger(),
      "scan ok: frame=%s points=%zu valid=%zu rate=%.2f Hz angle=[%.1f, %.1f] deg range=[%.3f, %.3f] m",
      message.header.frame_id.c_str(), message.ranges.size(), valid_count, frequency,
      angle_min_deg, angle_max_deg,
      std::isfinite(minimum) ? minimum : 0.0F, maximum);
    scan_count_ = 0;
    last_report_ = current_time;
  }

public:
  bool passed() const {return passed_;}

  std::string scan_topic_;
  double report_period_sec_{1.0};
  double startup_timeout_sec_{10.0};
  size_t scan_count_;
  bool received_scan_{false};
  bool passed_{true};
  rclcpp::Time last_report_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
  rclcpp::TimerBase::SharedPtr startup_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  const auto node = std::make_shared<RplidarC1ScanTest>();
  rclcpp::spin(node);
  const bool passed = node->passed();
  rclcpp::shutdown();
  return passed ? 0 : 1;
}
