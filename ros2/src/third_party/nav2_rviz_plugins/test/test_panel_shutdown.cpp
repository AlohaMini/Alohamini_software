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

#include <QApplication>
#include <QTest>

#include <chrono>
#include <memory>
#include <thread>

#include "gtest/gtest.h"
#include "nav2_rviz_plugins/nav2_panel.hpp"

using namespace std::chrono_literals;

class PanelShutdown : public ::testing::Test
{
protected:
  void SetUp() override {rclcpp::init(0, nullptr);}
  void TearDown() override
  {
    rclcpp::shutdown();
    QApplication::processEvents();
  }

  void destroyPromptly(std::unique_ptr<nav2_rviz_plugins::Nav2Panel> panel)
  {
    const auto start = std::chrono::steady_clock::now();
    panel.reset();
    EXPECT_LT(std::chrono::steady_clock::now() - start, 2500ms);
  }
};

TEST_F(PanelShutdown, NoNav2ServerAndRosStillRunning)
{
  auto panel = std::make_unique<nav2_rviz_plugins::Nav2Panel>();
  QTest::qWait(100);  // Start the state machine and its status worker.
  destroyPromptly(std::move(panel));
  EXPECT_TRUE(rclcpp::ok());
}

TEST_F(PanelShutdown, RosShutdownWhileStatusWorkerWaits)
{
  auto panel = std::make_unique<nav2_rviz_plugins::Nav2Panel>();
  QTest::qWait(100);
  rclcpp::shutdown();
  destroyPromptly(std::move(panel));
}

TEST_F(PanelShutdown, FinishedStatusWorkerIsStillOwned)
{
  auto node = rclcpp::Node::make_shared("panel_shutdown_services");
  auto active = [](const std_srvs::srv::Trigger::Request::SharedPtr,
      std_srvs::srv::Trigger::Response::SharedPtr response) {response->success = true;};
  auto nav = node->create_service<std_srvs::srv::Trigger>(
    "lifecycle_manager_navigation/is_active", active);
  auto loc = node->create_service<std_srvs::srv::Trigger>(
    "lifecycle_manager_localization/is_active", active);
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() {executor.spin();});
  auto panel = std::make_unique<nav2_rviz_plugins::Nav2Panel>();
  QTest::qWait(1500);  // Deliver finished and queued GUI status signals.
  bool active_label = false;
  for (const auto * label : panel->findChildren<QLabel *>()) {
    active_label |= label->text().contains("color=green>active");
  }
  EXPECT_TRUE(active_label);
  destroyPromptly(std::move(panel));
  executor.cancel();
  spinner.join();
}

TEST_F(PanelShutdown, PendingButtonRequestsWithoutServers)
{
  auto panel = std::make_unique<nav2_rviz_plugins::Nav2Panel>();
  QTest::qWait(100);
  EXPECT_TRUE(QMetaObject::invokeMethod(panel.get(), "onStartup", Qt::DirectConnection));
  EXPECT_TRUE(QMetaObject::invokeMethod(panel.get(), "onPause", Qt::DirectConnection));
  EXPECT_TRUE(QMetaObject::invokeMethod(panel.get(), "onResume", Qt::DirectConnection));
  EXPECT_TRUE(QMetaObject::invokeMethod(panel.get(), "onShutdown", Qt::DirectConnection));
  destroyPromptly(std::move(panel));
  EXPECT_TRUE(rclcpp::ok());
}

TEST_F(PanelShutdown, DiscoveredServiceWithoutResponseHasDeadline)
{
  auto server = rclcpp::Node::make_shared("panel_unresponsive_server");
  // Deliberately never spin the server: discovery succeeds, response cannot.
  auto service = server->create_service<std_srvs::srv::Trigger>(
    "panel_unresponsive", [](const std_srvs::srv::Trigger::Request::SharedPtr,
    std_srvs::srv::Trigger::Response::SharedPtr) {});
  auto client = rclcpp::Node::make_shared("panel_bounded_client");
  auto probe = client->create_client<std_srvs::srv::Trigger>("panel_unresponsive");
  ASSERT_TRUE(probe->wait_for_service(2s));
  const auto start = std::chrono::steady_clock::now();
  EXPECT_EQ(
    nav2_rviz_plugins::callService<std_srvs::srv::Trigger>(
      client, "panel_unresponsive", std::make_shared<std_srvs::srv::Trigger::Request>(),
      100ms), nullptr);
  EXPECT_LT(std::chrono::steady_clock::now() - start, 500ms);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  QApplication app(argc, argv);
  return RUN_ALL_TESTS();
}
