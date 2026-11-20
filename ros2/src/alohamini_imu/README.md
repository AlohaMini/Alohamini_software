# Hiwonder USB IMU

Supports the `55 55 ID LEN DATA SUM` protocol, defaulting to `/dev/imu_usb`, 115200, and `imu_link`.

```bash
ros2 launch alohamini_imu imu.launch.py serial_port:=/dev/imu_usb
ros2 run alohamini_imu imu_data_test
```

The driver exclusively opens the serial port and reads the current range, report frequency, and report switches without rewriting device settings. Automatic reporting and acceleration/angular-velocity output must already be enabled. It publishes `/imu/data_raw` (m/s², rad/s) and `/imu/temperature`; without a valid quaternion, orientation is marked unavailable.

Timestamps use the receiving host's clock, not hardware-synchronized time. Acceleration includes gravity-related specific force and cannot directly represent planar motion acceleration. Adjust nominal covariance through launch arguments. Valid-data timeout, backlog, or serial errors cause exit; the integrated sensor launch handles restarting.

This driver publishes no mounting TF and opens no motor ports. See platform `docs/ros2-navigation.md` for mounting transforms, EKF, and recording. Protocol sources are in `SOURCE.json`.
