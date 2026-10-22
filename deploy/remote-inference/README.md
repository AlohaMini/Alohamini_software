# PI05 云端推理部署

树莓派直接发送图像与状态到云端 GPU，接收动作块并在本地执行。PC 仅用于登录管理，不转发或编解码图像。通信使用 ZeroMQ；通过 SSH 隧道或 WireGuard 保护连接，不向公网开放推理端口 `5560`。

云端和树莓派使用相同版本的 Alohamini。云端需具备可运行 PI05 的 CUDA 环境、`pyzmq` 与 checkpoint；树莓派先启动本地 Host，完成校准和升降归零，并退出其他控制客户端。Host 地址保持 `127.0.0.1`，不开放 Host 的控制与状态端口到公网。

在项目根目录、各自现有 Conda 环境中安装；保留与 GPU 匹配的 PyTorch 环境，不另建或强制替换 CUDA/PyTorch：

```bash
# 云端：激活已有 PI05 GPU 环境后执行
pip install -e '.[pi05,zmq]'
```

```bash
# 树莓派：复用运行 Host 的环境
pip install -e '.[host]'
```

以下命令中的 `CHECKPOINT`、`TASK`、`PORT`、`USER`、`CLOUD` 均需替换为实际值。

更换租用实例时，SSH 方式更新登录地址与端口；WireGuard 方式还需更新树莓派配置中的云端公钥与 `Endpoint`。树莓派密钥可继续使用，新云端配置填入该树莓派的公钥。

## SSH 隧道

适合云端已有 SSH 登录入口的情况。云端先启动推理服务：

```bash
alohamini inference serve --bind tcp://127.0.0.1:5560 --policy.path CHECKPOINT --device cuda --chunk-size 32 --task "TASK"
```

等待云端显示 `Remote policy server listening on tcp://127.0.0.1:5560`，再启动树莓派推理客户端；加载模型期间提前运行客户端可能触发握手超时。

树莓派主动连接云端，并保持此终端运行：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -L 127.0.0.1:5560:127.0.0.1:5560 -p PORT USER@CLOUD
```

`PORT` 是租用平台提供的外部 SSH 端口。首次连接需核对云端 SSH 主机指纹。SSH 服务须允许本地端口转发；转发目标 `127.0.0.1:5560` 位于 SSH 服务端所在的网络环境，需能访问推理服务。[OpenSSH 转发说明](https://man.openbsd.org/ssh.1)

树莓派另开终端运行：

```bash
alohamini inference run --server tcp://127.0.0.1:5560 --host 127.0.0.1 --robot_model alohamini2pro --fps 30
```

`ExitOnForwardFailure` 检查转发监听是否建立，不检查云端推理服务是否可用。SSH 保活也不替代动作有效期或控制 watchdog。[OpenSSH 配置说明](https://man.openbsd.org/ssh_config.5)

停止时先按 `Ctrl-C` 退出树莓派推理客户端，再退出 SSH 隧道和云端服务。若树莓派本地 `5560` 已被占用，可将 `-L` 的第一个端口和 `--server` 端口一起改为其他空闲端口，云端目标端口仍为 `5560`。

## WireGuard

适合两端可创建 WireGuard 接口、云端可接收 UDP 的情况。租用容器可能缺少 root 或网络管理权限、WireGuard 内核支持、用户态实现所需的 `/dev/net/tun`，或没有 UDP 映射；无法配置时使用上面的 SSH 路径。先按所在系统安装 [WireGuard 工具](https://www.wireguard.com/install/)。

两端分别在本机生成密钥，仅执行一次；命令拒绝覆盖已有密钥：

```bash
(
    umask 077
    set -eC
    mkdir -p "$HOME/.config/alohamini/wireguard"
    chmod 700 "$HOME/.config/alohamini/wireguard"
    wg genkey > "$HOME/.config/alohamini/wireguard/private.key"
    wg pubkey < "$HOME/.config/alohamini/wireguard/private.key" > "$HOME/.config/alohamini/wireguard/public.key"
)
```

只交换 `public.key`。私钥及填好私钥的配置保留在各自机器、仓库之外，权限设为 `600`。[WireGuard 密钥生成说明](https://www.wireguard.com/quickstart/)

首次配置时，将对应模板复制到本机配置目录：

```bash
# 云端执行
cp -n deploy/remote-inference/wireguard-cloud.conf.example "$HOME/.config/alohamini/wireguard/wg-infer.conf"
chmod 600 "$HOME/.config/alohamini/wireguard/wg-infer.conf"
```

```bash
# 树莓派执行
cp -n deploy/remote-inference/wireguard-pi.conf.example "$HOME/.config/alohamini/wireguard/wg-infer.conf"
chmod 600 "$HOME/.config/alohamini/wireguard/wg-infer.conf"
```

在各自本机编辑 `wg-infer.conf`，填入本机私钥、对端公钥；树莓派还需将 `CLOUD_PUBLIC_HOST` 替换为云端可达的公网地址。模板使用云端 `10.88.0.1/32`、树莓派 `10.88.0.2/32`，只允许对端 `/32`，不设置 NAT、转发或互联网默认路由。若已有网络使用这些地址，先为两端选择其他不冲突的地址，并同步修改配置和命令。[wg-quick 路由说明](https://git.zx2c4.com/wireguard-tools/about/src/man/wg-quick.8)

云端安全组和防火墙允许 UDP `51820`；推理 TCP `5560` 仅允许来自 WireGuard 接口上的树莓派。若平台映射的外部 UDP 端口不同，将树莓派 `Endpoint` 的端口改为该外部端口，云端 `ListenPort` 保持内部监听端口。树莓派的 `PersistentKeepalive = 25` 用于维持 NAT 映射。[WireGuard 配置说明](https://git.zx2c4.com/wireguard-tools/about/src/man/wg.8)

两端分别启动：

```bash
sudo wg-quick up "$HOME/.config/alohamini/wireguard/wg-infer.conf"
sudo wg show wg-infer
```

树莓派可用 `ping -c 3 10.88.0.1` 检查目标连通性；防火墙禁用 ICMP 时，不能仅凭 ping 失败认定隧道不可用。检查握手、路由与服务监听后，云端启动：

```bash
alohamini inference serve --bind tcp://10.88.0.1:5560 --policy.path CHECKPOINT --device cuda --chunk-size 32 --task "TASK"
```

等待云端显示 `Remote policy server listening on tcp://10.88.0.1:5560` 后，树莓派运行：

```bash
alohamini inference run --server tcp://10.88.0.1:5560 --host 127.0.0.1 --robot_model alohamini2pro --fps 30
```

停止推理客户端和服务后，两端按需关闭隧道：

```bash
sudo wg-quick down "$HOME/.config/alohamini/wireguard/wg-infer.conf"
```

## 连接检查

SSH 隧道已建立但客户端连接失败时，检查云端推理进程是否仍在运行，以及 SSH 服务与推理服务是否位于相同网络环境。WireGuard 没有握手时，检查公钥、云端 UDP 入口和 `Endpoint`；已有握手但推理失败时，检查 `/32` 路由、云端绑定地址与接口防火墙。可在云端用 `ss -ltn '( sport = :5560 )'` 查看推理监听地址。

隧道不保证推理时延。`--fps 30` 是树莓派执行频率，`--chunk-size 32` 是请求的动作块长度，均不代表云端能持续及时返回；链路中断和过期结果由运行时处理，不通过放宽控制 watchdog 补偿网络延迟。

执行频率必须与 checkpoint 保存的数据频率一致；动作块长度不能超过模型的执行窗口。

## 运行限制

通道和动作调度可复用，目前仅提供平台原生 PI05 后端，未实现 RTC 训练或前缀 inpainting。迟到的动作块按树莓派时间跳过已经过去的前缀；可用动作耗尽、Host 会话或控制参考变化后会停止，需要显式重新运行。尚无实际云端广域网与机器人组合的性能保证。

`run` 默认每回合运行 `90` 秒；可选 `--episode_time`、`--rpc-timeout-s`、`--max-prediction-age-s`、`--ack-timeout-s` 分别控制回合时长、云端请求超时、预测最大年龄和 Host 确认超时。预测最大年龄默认 `1` 秒，确认超时默认 `0.5` 秒。

若需要固定部分当前关节，在云端 `serve` 和树莓派 `run` 两端传入相同的 `--fixed-current FIELD ...`；可选择 `base`、`lift_axis`、`arm_left`、`arm_right` 或具体 Host 状态字段名。树莓派从新鲜 Host 反馈捕获固定值。
