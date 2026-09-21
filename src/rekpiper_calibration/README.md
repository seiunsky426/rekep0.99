# 双固定相机标定

20 姿态自动标定设计与已实现的 RViz 离线预览见
[M2_AUTOMATIC_CALIBRATION_WORKFLOW.md](../../docs/M2_AUTOMATIC_CALIBRATION_WORKFLOW.md)。
预览不会控制真机；`calibration_motion_node.py` 和工作区顶层工具提供独立的
操作者监护运动流程，不要求签名文件。

真机链路要求 `base_link -> rs1_link` 与 `base_link -> rs3_link` 两个独立、
刚性固定且经过交叉验证的变换。使用本包的 ArUco/棋盘格采集、求解与交叉检查节点完成标定，
不要手工猜测或沿用其他安装位置的矩阵。

验收至少包含：

- 相机序列号与文件身份一致；
- 独立验证集的重投影 RMS 不超过 0.8 px；
- 三个以上工作区位置的双相机三维点中位误差不超过 5 mm、P95 不超过 8 mm；
- 深度尺度、彩色对齐、时间同步和固定支架复测通过；
- 原始采集、求解参数、残差和操作者签字报告可追溯。

通过后，用求解结果替换
`../rekpiper_camera/config/active_fixed_camera_extrinsics/{rs1,rs3}_extrinsics.yaml`，写入真实矩阵、
四元数与报告路径，并同时设置：

```yaml
status: ACCEPTED
publish_tf_allowed: true
precision_operation_allowed: true
base_absolute_accuracy: VERIFIED
```

默认模板是 `UNCALIBRATED`，生产 TF 节点会直接拒绝。仅做标定诊断时可显式
启用非精度候选；该状态永远不能通过机械臂执行门。

采集、求解、验证和预览节点不发布关节命令。只有显式启动
`calibration_motion.launch execute:=true` 时，标定运动节点才会独占 CAN、使能并发布
当前会话中的低速关节目标。

标定时需要的 `base_link -> link6` 只能由独立的只读遥测入口提供：

```bash
roslaunch piper calibration_joint_state_readonly.launch can_port:=can0
```

该入口用 `piper_init=False` 打开 SDK，只发布 `/joint_states_single` 和机器人
TF，不注册订阅者或服务；若标定会话期间 SocketCAN 的 TX 计数发生任何增加，
节点会立即失败退出。不要同时启动正式 Piper 控制节点。
