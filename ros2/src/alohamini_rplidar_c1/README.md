# RPLIDAR C1

C1 的独立 USB 驱动，默认 `/dev/lidar_c1`、460800、`laser_frame`，发布 `/scan`。

```bash
ros2 launch alohamini_rplidar_c1 rplidar_c1.launch.py serial_port:=/dev/lidar_c1
```

整机传感器与建图使用 `alohamini_sensors`、`alohamini_navigation`；配置与部署见平台 `docs/ros2-navigation.md`。本包不打开电机控制串口，也不发布底盘运动命令。

`~/stop_motor`、`~/start_motor` 为 `std_srvs/srv/Empty` 服务。停止扫描后保持停止；默认连续 3 次采集失败后清理连接并重连，断连重试间隔 2 秒。首次连接失败返回非零退出码，由传感器启动文件按 2 秒间隔重启。

`udev/` 中的设备名模板须填写实际 USB 序列号。`inverted`、`scan_mode`、`scan_frequency` 等参数以实际安装为准。消息时间戳来自驱动主机，未提供设备硬件时钟同步。

来源见 `SOURCE.json`；Slamtec SDK 保留在 `sdk/`，许可证见 `SLAMTEC_SDK_LICENSE`。
