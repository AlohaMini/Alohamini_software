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
 * ROS 2 adapter for the Slamtec RPLIDAR SDK.
 *
 * The SDK is vendored in this package so the AlohaMini workspace has no build
 * time or runtime dependency on the original ROS 1 rplidar_ros checkout.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cerrno>
#include <unistd.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_srvs/srv/empty.hpp"
#include "sl_lidar.h"

#include "scan_conversion.hpp"

class RplidarC1Node final : public rclcpp::Node
{
public:
  RplidarC1Node()
  : Node("rplidar_c1_node"), running_(true)
  {
    serial_port_ = declare_parameter<std::string>("serial_port", "/dev/lidar_c1");
    serial_baudrate_ = declare_parameter<int>("serial_baudrate", 460800);
    frame_id_ = declare_parameter<std::string>("frame_id", "laser_frame");
    scan_topic_ = declare_parameter<std::string>("scan_topic", "/scan");
    inverted_ = declare_parameter<bool>("inverted", false);
    angle_compensate_ = declare_parameter<bool>("angle_compensate", true);
    scan_frequency_ = declare_parameter<double>("scan_frequency", 10.0);
    scan_mode_ = declare_parameter<std::string>("scan_mode", "Standard");
    initial_reset_ = declare_parameter<bool>("initial_reset", false);
    grab_timeout_ms_ = declare_parameter<int>("grab_timeout_ms", 2000);
    reconnect_failures_ = declare_parameter<int>("reconnect_failures", 3);

    if (serial_baudrate_ <= 0 || !std::isfinite(scan_frequency_) ||
      scan_frequency_ <= 0.0 || scan_frequency_ > 100.0 ||
      grab_timeout_ms_ <= 0 || grab_timeout_ms_ > 5000 || frame_id_.empty() || reconnect_failures_ < 1) {
      throw std::invalid_argument("invalid parameters: baudrate > 0, frequency in (0, 100] Hz, timeout in [1, 5000] ms and nonempty frame required");
    }

    scan_publisher_ = create_publisher<sensor_msgs::msg::LaserScan>(
      scan_topic_, rclcpp::SensorDataQoS());
    start_service_ = create_service<std_srvs::srv::Empty>(
      "~/start_motor",
      [this](const std::shared_ptr<std_srvs::srv::Empty::Request>,
      std::shared_ptr<std_srvs::srv::Empty::Response>) {
        std::lock_guard<std::mutex> lock(driver_mutex_);
        scanning_.store(true);
        if (driver_ == nullptr || !driver_->isConnected()) {
          RCLCPP_WARN(get_logger(), "RPLIDAR will reconnect before starting");
          return;
        }
        auto result = driver_->setMotorSpeed(motor_speed_);
        if (SL_IS_FAIL(result)) {
          RCLCPP_ERROR(get_logger(), "RPLIDAR motor start failed: 0x%08x", result);
          return;
        }
        // setMotorSpeed(0) stops the active scan on C1. Restart the scan
        // command as well so this service restores the /scan data stream.
        if (use_typical_scan_) {
          result = driver_->startScan(false, true, 0, &scan_mode_info_);
        } else {
          result = driver_->startScanExpress(false, selected_scan_mode_, 0, &scan_mode_info_);
        }
        if (SL_IS_FAIL(result)) {
          RCLCPP_ERROR(get_logger(), "RPLIDAR scan restart failed: 0x%08x", result);
        }
      });
    stop_service_ = create_service<std_srvs::srv::Empty>(
      "~/stop_motor",
      [this](const std::shared_ptr<std_srvs::srv::Empty::Request>,
      std::shared_ptr<std_srvs::srv::Empty::Response>) {
        std::lock_guard<std::mutex> lock(driver_mutex_);
        scanning_.store(false);
        if (driver_ == nullptr) {
          return;
        }
        driver_->stop();
        const auto result = driver_->setMotorSpeed(0);
        if (SL_IS_FAIL(result)) {
          RCLCPP_ERROR(get_logger(), "RPLIDAR motor stop failed: 0x%08x", result);
        }
      });

    try {
      connect_and_start();
      scan_thread_ = std::thread(&RplidarC1Node::scan_loop, this);
    } catch (...) {
      cleanup_driver();
      throw;
    }
  }

  ~RplidarC1Node() override
  {
    running_.store(false);
    if (scan_thread_.joinable()) {
      scan_thread_.join();
    }

    std::lock_guard<std::mutex> lock(driver_mutex_);
    cleanup_driver();
  }

private:
  void connect_and_start()
  {
    if (::access(serial_port_.c_str(), F_OK) != 0) {
      throw std::runtime_error(
              "RPLIDAR serial port does not exist: " + serial_port_ + " (" + std::strerror(errno) + ")");
    }
    if (::access(serial_port_.c_str(), R_OK | W_OK) != 0) {
      throw std::runtime_error(
              "RPLIDAR serial port is not readable/writable: " + serial_port_ + " (" + std::strerror(errno) + ")");
    }

    driver_ = *sl::createLidarDriver();
    if (driver_ == nullptr) {
      throw std::runtime_error("failed to create RPLIDAR driver");
    }

    channel_ = *sl::createSerialPortChannel(serial_port_, serial_baudrate_);
    if (channel_ == nullptr) {
      delete driver_;
      driver_ = nullptr;
      throw std::runtime_error("failed to create RPLIDAR serial channel for " + serial_port_);
    }

    auto result = driver_->connect(channel_);
    if (SL_IS_FAIL(result)) {
      throw std::runtime_error("failed to connect RPLIDAR on " + serial_port_);
    }
    RCLCPP_INFO(get_logger(), "Connected to RPLIDAR on %s at %d baud", serial_port_.c_str(), serial_baudrate_);

    sl_lidar_response_device_info_t device_info{};
    result = driver_->getDeviceInfo(device_info);
    if (SL_IS_FAIL(result)) {
      throw std::runtime_error("failed to read RPLIDAR device information");
    }
    RCLCPP_INFO(
      get_logger(), "RPLIDAR model=%u firmware=%u.%02u hardware=%u",
      static_cast<unsigned>(device_info.model),
      static_cast<unsigned>(device_info.firmware_version >> 8),
      static_cast<unsigned>(device_info.firmware_version & 0xff),
      static_cast<unsigned>(device_info.hardware_version));

    sl_lidar_response_device_health_t health{};
    result = driver_->getHealth(health);
    if (SL_IS_FAIL(result) || health.status == SL_LIDAR_STATUS_ERROR) {
      throw std::runtime_error("RPLIDAR health check failed");
    }

    if (initial_reset_) {
      RCLCPP_INFO(get_logger(), "Resetting RPLIDAR");
      result = driver_->reset();
      if (SL_IS_FAIL(result)) {
        throw std::runtime_error("RPLIDAR reset failed");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Match the supplied example: C1 (major model 4) uses PWM; only the
    // later ToF models tune RPM after the first complete scan.
    scan_frequency_tuning_pending_ = (device_info.model >> 4) > 5;
    motor_speed_ = scan_frequency_tuning_pending_ ?
      static_cast<sl_u16>(scan_frequency_ * 60.0) : 600;
    if (!scan_frequency_tuning_pending_) {
      result = driver_->setMotorSpeed(600);
      if (SL_IS_FAIL(result)) {
        throw std::runtime_error("failed to start RPLIDAR motor");
      }
    }

    if (scan_mode_.empty()) {
      use_typical_scan_ = true;
      result = driver_->startScan(false, true, 0, &scan_mode_info_);
    } else {
      use_typical_scan_ = false;
      std::vector<sl::LidarScanMode> modes;
      result = driver_->getAllSupportedScanModes(modes);
      if (SL_IS_OK(result)) {
        sl_u16 selected_mode = static_cast<sl_u16>(-1);
        for (const auto & mode : modes) {
          if (scan_mode_ == mode.scan_mode) {
            selected_mode = mode.id;
            break;
          }
        }
        if (selected_mode == static_cast<sl_u16>(-1)) {
          RCLCPP_ERROR(get_logger(), "RPLIDAR scan mode '%s' is not supported", scan_mode_.c_str());
          for (const auto & mode : modes) {
            RCLCPP_ERROR(
              get_logger(), "  supported mode: %s (max %.1f m, %.1f kS/s)",
              mode.scan_mode, mode.max_distance, 1000.0F / mode.us_per_sample);
          }
          result = SL_RESULT_OPERATION_FAIL;
        } else {
          selected_scan_mode_ = selected_mode;
          result = driver_->startScanExpress(false, selected_mode, 0, &scan_mode_info_);
        }
      }
    }

    if (SL_IS_FAIL(result)) {
      throw std::runtime_error("failed to start RPLIDAR scan");
    }

    if (!std::isfinite(scan_mode_info_.us_per_sample) || scan_mode_info_.us_per_sample <= 0 ||
      !std::isfinite(scan_mode_info_.max_distance) || scan_mode_info_.max_distance <= 0.15F) {
      throw std::runtime_error("invalid scan mode metadata received from RPLIDAR");
    }
    points_per_circle_ = std::clamp(
      static_cast<int>(std::min(8192.0, 1000000.0 / scan_mode_info_.us_per_sample / scan_frequency_)), 360, 8192);
    angle_compensate_multiple_ = std::max(
      1, static_cast<int>(std::ceil(static_cast<double>(points_per_circle_) / 360.0)));
    RCLCPP_INFO(
      get_logger(), "RPLIDAR scan mode=%s sample_rate=%.1f kHz max_range=%.1f m frequency=%.1f Hz",
      scan_mode_info_.scan_mode,
      1000.0F / scan_mode_info_.us_per_sample,
      scan_mode_info_.max_distance,
      scan_frequency_);
  }

  void cleanup_driver()
  {
    if (driver_ != nullptr) {
      if (driver_->isConnected()) {
        driver_->setMotorSpeed(0);
        driver_->stop();
        driver_->disconnect();
      }
      delete driver_;
      driver_ = nullptr;
    }
    if (channel_ != nullptr) {
      delete channel_;
      channel_ = nullptr;
    }
  }

  void scan_loop()
  {
    int failures = 0;
    while (running_.load() && rclcpp::ok()) {
      if (!scanning_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }
      if (driver_ == nullptr) {
        try {
          std::lock_guard<std::mutex> lock(driver_mutex_);
          if (!scanning_.load()) {continue;}
          connect_and_start();
          failures = 0;
        } catch (const std::exception & error) {
          {
            std::lock_guard<std::mutex> lock(driver_mutex_);
            cleanup_driver();
          }
          RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "RPLIDAR reconnect: %s", error.what());
          // Bounded, interruptible backoff; no busy retry on a disconnected USB port.
          for (int i = 0; i < 20 && running_.load() && rclcpp::ok(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
          }
          continue;
        }
      }
      std::vector<sl_lidar_response_measurement_node_hq_t> nodes(8192);
      size_t count = nodes.size();
      const rclcpp::Time start_time = now();

      sl_result result;
      {
        std::lock_guard<std::mutex> lock(driver_mutex_);
        if (!scanning_.load()) {continue;}
        result = driver_->grabScanDataHq(nodes.data(), count, static_cast<sl_u32>(grab_timeout_ms_));
      }

      const rclcpp::Duration scan_duration = now() - start_time;
      if (SL_IS_FAIL(result)) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 5000, "RPLIDAR scan read failed: 0x%08x", result);
        if (scanning_.load() && ++failures >= reconnect_failures_) {
          std::lock_guard<std::mutex> lock(driver_mutex_);
          cleanup_driver();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      failures = 0;
      if (count < 2) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "RPLIDAR returned fewer than two scan points");
        continue;
      }

      {
        std::lock_guard<std::mutex> lock(driver_mutex_);
        result = driver_->ascendScanData(nodes.data(), count);
      }
      if (SL_IS_FAIL(result)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "RPLIDAR scan ordering failed: 0x%08x", result);
        continue;
      }

      if (scan_frequency_tuning_pending_) {
        std::lock_guard<std::mutex> lock(driver_mutex_);
        if (!scanning_.load()) {continue;}
        result = driver_->setMotorSpeed(static_cast<sl_u16>(scan_frequency_ * 60.0));
        if (SL_IS_FAIL(result)) {
          RCLCPP_WARN(get_logger(), "Failed to tune RPLIDAR scan frequency to %.1f Hz", scan_frequency_);
        }
        scan_frequency_tuning_pending_ = false;
      }

      if (running_.load() && scanning_.load() && rclcpp::ok()) {
        auto message = make_scan(nodes, count, start_time, scan_duration.seconds(),
          angle_compensate_, angle_compensate_multiple_, inverted_, scan_frequency_,
          scan_mode_info_.max_distance, frame_id_);
        if (message) {
          scan_publisher_->publish(*message);
        }
      }
    }
  }

  std::string serial_port_;
  int serial_baudrate_{0};
  std::string frame_id_;
  std::string scan_topic_;
  bool inverted_{false};
  bool angle_compensate_{true};
  double scan_frequency_{10.0};
  std::string scan_mode_;
  bool initial_reset_{false};
  int grab_timeout_ms_{2000};
  int reconnect_failures_{3};

  int points_per_circle_{360};
  int angle_compensate_multiple_{1};
  bool scan_frequency_tuning_pending_{false};
  bool use_typical_scan_{false};
  sl_u16 selected_scan_mode_{0};
  sl_u16 motor_speed_{600};
  sl::ILidarDriver * driver_{nullptr};
  sl::IChannel * channel_{nullptr};
  sl::LidarScanMode scan_mode_info_{};
  std::atomic<bool> running_;
  std::atomic<bool> scanning_{true};
  std::mutex driver_mutex_;
  std::thread scan_thread_;

  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_publisher_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr start_service_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr stop_service_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<RplidarC1Node>();
    rclcpp::spin(node);
  } catch (const std::exception & error) {
    fprintf(stderr, "rplidar_c1_node: %s\n", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
