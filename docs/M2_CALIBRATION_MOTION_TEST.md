# Piper标定运动测试：连接、保持、试动、分段

## 推荐：单命令人工监护流程

`run_supervised_calibration.py` 将服务端和客户端封装为一次运行：连接CAN后自动失能并确认
`000000`，等待操作者回车，使能并确认`111111`、保持当前位置、执行1°试动，随后P1到P20
每点等待回车，结束或按`q`时失能。它校验准备时绑定的计划、URDF和代码哈希。

```bash
python3 tools/run_supervised_calibration.py \
  --session "$motion_dir/session.yaml"
```

当前提供代码和离线测试，实机使能、停止和轨迹跟踪尚未测试。现有20点仍是PREVIEW_ONLY。
新执行器只接受当前准备好的标定会话，不修改正式抓取执行器的release、地图或相机验收状态。
它不自动采集标定样本；到点后检查两相机READY，失败则停止，不继续下一段。

## 当前计划与速度

起点使用操作者明确指定的当前实测姿态。保留原始角度；沿用执行模块已有0.01 rad
反馈边界容差，把边界附近的测量值归到合法指令边界，不扩大URDF命令限位。
2026-09-11的起点原始角度为 `[1.619,-0.254,0.328,-6.092,26.225,-0.837]` 度；
合法起始指令为 `[1.619,0,0,-6.092,26.225,-0.837]` 度。
使能保持可能包含最大0.328°的零位边界修正，并非绝对零位移。

| 设置 | 默认及当前硬上限 | 含义 |
|---|---:|---|
| SDK速度百分比 | 5% | `MotionCtrl_2(0x01,0x01,5,0)`，不是5°/s |
| 关节速度 | 3°/s | 五次关节插值的最大目标速度 |
| 关节加速度 | 6°/s² | 用于计算最小段时长 |
| 关节jerk | 12°/s³ | 用于计算最小段时长 |
| link6速度 | 10 mm/s | 根据FK采样估计并加入25%时间余量 |
| 指令频率 | 50 Hz | 不按RViz播放速度直接发指令 |
| link6工作范围 | X 0–70 cm，Y ±40 cm，Z 5–70 cm | 高度70厘米与关节角限位无关 |

每段采用 `s(u)=10u³−15u⁴+6u⁵`，`q=q_start+s(u)*(q_goal−q_start)`，
端点目标速度、加速度为零。末端限速基于采样；实际控制器跟踪能力需现场测试。
试动沿起点到预览P1方向，最大单关节变化1°；不是立即运动到P1。
试动后从其终点连接到P1，再按预览P2…P20逐段运行，每段停留1秒。

## 终端A：环境与CAN

以下命令仅检查/启用CAN网卡，不使能电机。已UP的CAN不会被关闭或重配。

```bash
cd /home/zhengsihze/Rekep_v1.0-main
source setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311
python tools/connect_calibration_can.py
```

SDK连接在执行器内部为 `C_PiperInterface_V2(can_name="can0")` 和
`ConnectPort(piper_init=False)`；构造前必须通过会话文件与代码哈希检查。
不应另外运行一份SDK控制脚本争用CAN。

## 先生成当前起点的待审会话

此步骤需要现有只读反馈在线、机械臂静止约1秒。每次使用新的目录名，旧会话不覆盖。
速度可以降低；修改速度或起点后需要生成新会话。

```bash
motion_dir="$PWD/runtime/evidence/m2_20260911_base_frame_validation/motion_$(date -u +%Y%m%dT%H%M%SZ)"
python3 tools/prepare_calibration_motion.py \
  --plan "$PWD/runtime/evidence/m2_20260911_base_frame_validation/taught12_preview_20260911/plan.yaml" \
  --output-dir "$motion_dir" \
  --driver-speed-percent 5 --velocity-deg-s 3 --tip-speed-mm-s 10
rosrun rekpiper_calibration calibration_motion_client.py check --session "$motion_dir/session.yaml"
```

输出21段：trial以及point_01…point_20。`check`核对计划、URDF、速度限制和代码哈希。
原12个示教点、8个插值候选和正式相机外参记录保持不变。

## 切换控制节点

会话生成并核对通过后，停止原只读反馈launch及manual_teaching_recorder；不要关闭相机/采集器。
原只读launch中Piper节点是required，结束后其关节适配器和TF也会退出。
新launch会提供替代的真实反馈、关节适配器和TF。不得用关闭只读TX检查来混跑两套驱动。

## 终端B：启动控制节点

`motion_dir`须指向刚刚从当前真实关节反馈生成的会话。先执行check；失败时不要启动硬件节点。

```bash
source /home/zhengsihze/Rekep_v1.0-main/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311
export ROS_HOME=/tmp/rekpiper_calibration_motion_ros
mkdir -p "$ROS_HOME"
rosrun rekpiper_calibration calibration_motion_client.py check \
  --session "$motion_dir/session.yaml" && \
roslaunch rekpiper_calibration calibration_motion.launch \
  session:="$motion_dir/session.yaml" execute:=true
```

启动只连接CAN并发布反馈，不自动使能。若旧反馈节点仍在线，启动被拒绝。
这套测试使用 `/calibration_motion/*` 服务，不使用此前不存在的 `/enable_srv`。

## 终端C：使能保持 → 试动 → 分段

同样source工程环境、设置11311，并设置与终端B一致的`motion_dir`。

```bash
# 1. 显式使能；内部先失能并确认000000，再使能确认111111并保持起点2秒。
rosrun rekpiper_calibration calibration_motion_client.py enable-hold --session "$motion_dir/session.yaml"

# 2. 单次小步试动；先观察运动、停止和双相机识别，再进行下一条。
rosrun rekpiper_calibration calibration_motion_client.py trial --session "$motion_dir/session.yaml"

# 3. 连接到预览P1；success后再运行后续段。
rosrun rekpiper_calibration calibration_motion_client.py point --point 1 --session "$motion_dir/session.yaml"

# 4. 每次明确启动一个区间；不要一次粘贴运行全部命令。
rosrun rekpiper_calibration calibration_motion_client.py range --first 2 --last 5 --session "$motion_dir/session.yaml"
rosrun rekpiper_calibration calibration_motion_client.py range --first 6 --last 10 --session "$motion_dir/session.yaml"
rosrun rekpiper_calibration calibration_motion_client.py range --first 11 --last 15 --session "$motion_dir/session.yaml"
rosrun rekpiper_calibration calibration_motion_client.py range --first 16 --last 20 --session "$motion_dir/session.yaml"
```

每个区间内客户端逐段发送 `FollowJointTrajectory` action，等待成功才发下一段。
执行器只接受准备会话中恰好匹配的下一段，不接受任意关节目标、跳点、改变时间或并发运动。
本测试不触发 `capture_pose`，也不会把20个运动目标计为已采集20个标定样本。

## 单独准备停止终端

```bash
source /home/zhengsihze/Rekep_v1.0-main/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311
rosservice call /calibration_motion/stop "{}"
```

停止发送控制器停止帧，并等待新鲜反馈确认；确认失败时返回false，使用现场实体电源开关。
它不等于机械式急停，也不自动断使能，避免把重力下垂误认为安全停止。
Ctrl-C取消轨迹也会请求停止。停止、故障或超时锁住本次进程，不自动reset、不自动恢复。
重新启动前须恢复现场并从实际当前位置重新生成会话。

事件日志为会话目录下 `motion_events_*.jsonl`，记录保持、每段开始/完成、实际角度与停止原因。
