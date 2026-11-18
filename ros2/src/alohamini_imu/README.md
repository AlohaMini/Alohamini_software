# 幻尔 USB IMU

支持 `55 55 ID LEN DATA SUM` 协议，默认 `/dev/imu_usb`、115200、`imu_link`。

```bash
ros2 launch alohamini_imu imu.launch.py serial_port:=/dev/imu_usb
ros2 run alohamini_imu imu_data_test
```

驱动以独占串口读取设备当前量程、报告频率与报告开关，不改写设备配置。设备须已启用自动上报和加速度／角速度。发布 `/imu/data_raw`（m/s²、rad/s）与 `/imu/temperature`；没有有效四元数时标记 orientation 不可用。

数据使用接收主机时间，不代表硬件同步时间。加速度包含重力对应比力，不能直接作为平面运动加速度；名义协方差可通过启动参数调整。无有效数据超时、积压或串口错误时退出，传感器整合启动文件负责重启。

本驱动不发布安装 TF、不打开电机串口。安装变换、EKF 和录制入口见平台 `docs/ros2-navigation.md`。协议来源见 `SOURCE.json`。
