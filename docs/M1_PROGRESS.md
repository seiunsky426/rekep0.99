# M1 机器人反馈与几何模型进度

日期：2026-09-11  
工程根目录：`/home/zhengsihze/Rekep_v1.0-main`  
当前结论：**4/5 完成；M1.5 等待现场核对。**

## 安全边界

本轮只运行 `piper_joint_state_readonly_node.py` 或直接以
`C_PiperInterface_V2.ConnectPort(piper_init=False)` 接收周期反馈。没有启动
`piper_ctrl_single_node.py`，没有创建运动/夹爪服务，没有调用 SDK 控制函数。
每次实机采集都比较 `/sys/class/net/can0/statistics/tx_packets`；累计计数始终为
0。`system.yaml` 中五项正式验收门仍为 `false`，现有工作区仍是
`UNVERIFIED`，没有授权自主运动。

## 子目标状态

### M1.1 只读反馈：完成

- `can0`：`UP`、`ERROR-ACTIVE`、1,000,000 bit/s。
- `/joint_states_single` 连续采集 80 帧，消息时间戳频率 100.629 Hz。
- 关节名固定为 `joint1..joint6`；SDK 的 0.001° 乘
  `pi/180000` 后输出 rad，所有值有限。
- 同一采集窗口 CAN `RX +33846`、`TX +0`。
- 节点拒绝非 UP/ERROR-ACTIVE/1 Mbps 的 CAN、陈旧反馈和任何 TX 计数变化。
- 同一节点还把周期性 0x2A1 状态帧转换为只读 `/arm_status`，供控制模式、
  故障、关节限位、通信异常和实体急停核对使用。

### M1.2 关节与夹爪适配：完成

- 只读节点新增被动 `GetArmGripperMsgs()`；0.001 mm 总行程乘 `1e-6`
  后以 `gripper` 写入 `/joint_states_single`。
- 本次实机原始开度为 `-0.000070 m`，处于允许的 1 mm 零位负偏差内；
  适配器将其钳位为 0，并输出 `joint7=0`、`joint8=-0`。
- 适配后的关节顺序固定为 `joint1..joint8`，六轴保持原值，正开度按
  `(+opening/2, -opening/2)` 分到两指。
- 新增回归测试覆盖乱序输入、rad/m 单位、总开度拆分及速度符号。
- rs1、rs3 双视角彩色帧与同次夹爪反馈的最大时间差为 55 ms；两视角中
  夹爪均处于闭合位置，与 `-0.000070 m` 的零位反馈和对称零位模型一致。
  同步视觉采集期间 CAN TX 增量仍为 0。

### M1.3 FK 与 TCP：完成

- `robot_state_publisher` 现在读取适配后的 `/joint_states`，实时生成
  `base_link → link6/gripper_base/rekep_tcp/link7/link8`。
- 固件被动末端反馈与相同关节状态下 URDF `link6` 的差异为
  `0.000067051 m`、`0.000055704 rad`，验证了六轴顺序、符号、单位和
  当前 DH/轴向约定。
- `link6 → gripper_base` 为零固定变换；`gripper_base → rekep_tcp` 为
  局部 +Z 方向 `0.1358 m`。固件端点到 `rekep_tcp` 的实测 TF 距离为
  `0.135812820 m`。
- 135.8 mm 与 AgileX 官方 Noetic URDF 中两指根部的轴向距离一致；
  `rekep_tcp` 是本工程在该夹爪中心面增加的固定规划坐标系。

### M1.4 IK 与碰撞几何：完成

- 使用工程实际 URDF、STL 网格和 `base_link → rekep_tcp` 六轴链完成
  3 组非零位 FK→IK 往返。
- 最大位置误差 `0.000004114 m`，最大旋转误差 `0.000070417 rad`；
  所有解都位于本工程关节限位内。
- 4 m 目标被拒绝，残余位置误差约 3.238 m。
- 碰撞采样覆盖 `base_link`、`link1..link6`、`gripper_base` 共 8 个刚性
  部分；夹爪以完整 ±35 mm 行程的保守扫掠体计入 `gripper_base`。
  当前姿态共 991 个有限采样点，采样球半径 0.0173205 m。
- 本工程把 joint1 正向上限保持为 2.168 rad，低于 AgileX 官方
  2.618 rad 物理上限；这是执行侧的保守现场上限，不扩大运动范围。

### M1.5 现场运动条件：未完成

被动状态已读到：`TEACHING_MODE`、`NORMAL`、`MOVE_J`、静止、错误码 0，
20 帧 `/arm_status` 中六轴限位/通信故障位均为 False，低速状态帧完整后
六轴使能位均为 True。CAN 适配器为 bytewerk candleLight/`gs_usb`，序列号
`002100344148570D20343133`。120 秒急停监测没有收到
`EMERGENCY_STOP`，因此不能把实体急停标为已验证。仍需记录：

1. Piper 机身铭牌型号和序列号，以及控制器固件版本；
2. 实体急停按下时 `arm_status=1`，复位后回到正常且机械臂没有运动；
3. 以 `base_link` 表示的现场允许运动边界、桌面高度、禁入区和现场操作者；
4. M1 阶段保持示教/停止条件；自主 CAN 控制继续禁止。

官方说明还指出：固件早于 `S-V1.6-3` 时应使用旧 DH/URDF，之后使用当前
URDF。本轮固件 FK 与当前 URDF 高精度吻合，但正式身份记录仍必须填写版本号。

## 代码和证据

- 只读反馈：`src/piper/scripts/piper_joint_state_readonly_node.py`
- 只读整链启动：`src/piper/launch/calibration_joint_state_readonly.launch`
- 夹爪适配：`src/rekpiper_execution/scripts/gripper_joint_state_adapter_node.py`
- 真实几何回归：`src/rekpiper_planning/test/test_piper_geometry_model.py`
- 实机证据目录：`runtime/evidence/m1_20260911_readonly_geometry/`
- 汇总报告：`geometry_model_report.json`；逐文件校验：`SHA256SUMS`
- 现场基线候选：`site_motion_conditions_baseline.yaml`，状态保持
  `HOLD_PENDING_PHYSICAL_ESTOP_AND_SITE_IDENTITY`

验证结果：13 个 Catkin 包构建通过；Piper、execution、planning 相关测试
通过；汇总为 216 tests、0 errors、0 failures、0 skipped；仓库审计和 launch
展开检查均通过。测试结论覆盖软件与本轮只读实机反馈，不替代正式签名验收。
