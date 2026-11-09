// Copyright 2026 AlohaMini contributors
// SPDX-License-Identifier: Apache-2.0

#include <QApplication>
#include <memory>
#include "gtest/gtest.h"
#include "alohamini_rviz/image_display.hpp"

class TestImage : public alohamini_rviz::ImageDisplay
{
public:
  // Subscription lifetime does not require an Ogre render target.
  TestImage()
  : ImageDisplay(std::unique_ptr<rviz_default_plugins::displays::ROSImageTextureIface>()) {}

  using ImageDisplay::unsubscribe;

  std::weak_ptr<image_transport::SubscriberFilter> connectFilter()
  {
    subscription_ = std::make_shared<image_transport::SubscriberFilter>();
    subscription_callback_ = subscription_->registerCallback(
      [](const sensor_msgs::msg::Image::ConstSharedPtr &) {});
    return subscription_;
  }
};

TEST(ImageShutdown, RepeatedDisconnectThenDestruction)
{
  TestImage image;
  for (int i = 0; i < 10; ++i) {
    auto filter = image.connectFilter();
    EXPECT_FALSE(filter.expired());
    image.unsubscribe();
    EXPECT_TRUE(filter.expired());
    image.unsubscribe();
  }
}

TEST(ImageShutdown, DestructionWithActiveConnection)
{
  std::weak_ptr<image_transport::SubscriberFilter> filter;
  {
    TestImage image;
    filter = image.connectFilter();
  }
  EXPECT_TRUE(filter.expired());
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  QApplication app(argc, argv);
  return RUN_ALL_TESTS();
}
