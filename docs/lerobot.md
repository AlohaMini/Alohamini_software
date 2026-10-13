# LeRobot 兼容

平台使用 LeRobot v3 数据集。ACT、AM-ACT、Diffusion、FastWAM、SmolVLA 和 π0.5
共用数据读取接口，支持训练、验证和离线推理；各算法仍有自己的 state、语言和动作表示要求。

外部 v3 数据缺少平台元信息时，加 `--dataset.allow_unannotated=true`。
`--state=recorded` 使用原 state 向量，`--state=none` 使用纯视觉。
视频、任务文本和数值字段按原 episode 配对，缺少的标定不会自动补齐。
真机推理须满足目标机器的动作字段、单位和标定要求。

数据使用方式见[训练与部署](training.md)。需要选择字段或另存数据时，
使用[导出命令](alohamini.md#导出与选择字段)。

数据格式与模型权重格式分开：平台训练出的 checkpoint 可用于 `alohamini evaluate`；
外部 LeRobot checkpoint 仅按对应算法支持的基座格式加载，不能因数据同为 v3 就直接互换权重。
