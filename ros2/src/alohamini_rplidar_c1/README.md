# RPLIDAR C1

Standalone C1 USB driver, defaulting to `/dev/lidar_c1`, 460800, and `laser_frame`, publishing `/scan`.

```bash
ros2 launch alohamini_rplidar_c1 rplidar_c1.launch.py serial_port:=/dev/lidar_c1
```

Use `alohamini_sensors` and `alohamini_navigation` for full-robot sensors and mapping; see platform `docs/ros2-navigation.md` for setup/deployment. This package opens no motor-control serial ports and publishes no base motion commands.

`~/stop_motor` and `~/start_motor` are `std_srvs/srv/Empty` services. A stopped scan remains stopped. By default, three consecutive acquisition failures clean up and reconnect, with 2 s between retries. Initial connection failure exits nonzero; sensor launch restarts it at 2 s intervals.

Fill in the actual USB serial number in the `udev/` device-name template. Set `inverted`, `scan_mode`, `scan_frequency`, and other parameters for the actual installation. Message timestamps come from the driver host; hardware clock synchronization is not provided.

See `SOURCE.json` for provenance. The Slamtec SDK is retained in `sdk/`; see `SLAMTEC_SDK_LICENSE` for its license.
