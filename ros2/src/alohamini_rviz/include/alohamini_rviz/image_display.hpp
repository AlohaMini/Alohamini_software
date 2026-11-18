// Copyright 2026 AlohaMini contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef ALOHAMINI_RVIZ__IMAGE_DISPLAY_HPP_
#define ALOHAMINI_RVIZ__IMAGE_DISPLAY_HPP_

#include "rviz_default_plugins/displays/image/image_display.hpp"

namespace alohamini_rviz
{

// Humble's ImageTransportDisplay disconnects the same Connection again after
// destroying its SubscriberFilter. Clear that connection while it is valid.
class ImageDisplay : public rviz_default_plugins::displays::ImageDisplay
{
public:
  using rviz_default_plugins::displays::ImageDisplay::ImageDisplay;
  ~ImageDisplay() override {unsubscribe();}

protected:
  void unsubscribe() override
  {
    if (subscription_) {
      subscription_callback_.disconnect();
    }
    subscription_callback_ = message_filters::Connection();
    subscription_.reset();
  }

  void onDisable() override
  {
    unsubscribe();
    reset();
  }
};

}  // namespace alohamini_rviz
#endif  // ALOHAMINI_RVIZ__IMAGE_DISPLAY_HPP_
