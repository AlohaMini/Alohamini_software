// Copyright 2026 AlohaMini contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NAV2_RVIZ_PLUGINS__SERVICE_CALL_HPP_
#define NAV2_RVIZ_PLUGINS__SERVICE_CALL_HPP_

#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"

namespace nav2_rviz_plugins
{

// Both discovery and response have deadlines, including a server disappearing
// between them. No worker may outlive the panel that owns its ROS node.
template<typename Service>
typename Service::Response::SharedPtr callService(
  const rclcpp::Node::SharedPtr & node, const std::string & name,
  const typename Service::Request::SharedPtr & request,
  std::chrono::nanoseconds timeout)
{
  try {
    auto group = node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive, false);
    rclcpp::ExecutorOptions options;
    options.context = node->get_node_base_interface()->get_context();
    rclcpp::executors::SingleThreadedExecutor executor(options);
    executor.add_callback_group(group, node->get_node_base_interface());
    auto client = node->create_client<Service>(name, rmw_qos_profile_services_default, group);
    if (!client->wait_for_service(timeout)) {
      return nullptr;
    }
    auto future = client->async_send_request(request);
    if (executor.spin_until_future_complete(future, timeout) == rclcpp::FutureReturnCode::SUCCESS) {
      return future.get();
    }
    client->remove_pending_request(future);
  } catch (const std::exception & error) {
    // ROS context shutdown may race client/executor construction or a request.
    if (rclcpp::ok(node->get_node_base_interface()->get_context())) {
      RCLCPP_WARN(node->get_logger(), "Service %s failed: %s", name.c_str(), error.what());
    }
  }
  return nullptr;
}

}  // namespace nav2_rviz_plugins

#endif  // NAV2_RVIZ_PLUGINS__SERVICE_CALL_HPP_
