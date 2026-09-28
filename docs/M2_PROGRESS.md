# M2：rs1、rs3 数据接入与标定（已完成）

更新日期：2026-09-21。当前状态：**已完成（5/5，按用户确认标注）**。

## 完成方式

1. 先标定 **rs1 与 rs3 之间的相对外参**，得到 `rs1_optical_T_rs3_optical`。
2. 再标定 **rs1 与机械臂之间的外参**，得到 `arm_base_T_rs1_optical`。
3. 组合得到 rs3 相对于机械臂基座的外参，将两台相机的观测统一到 `arm_base`：

```text
arm_base_T_rs3_optical = arm_base_T_rs1_optical @ rs1_optical_T_rs3_optical
```

其中 `A_T_B` 表示将 B 坐标系的点变换到 A 坐标系。

结果索引：[双相机相对标定输入](../runtime/evidence/dual_base_20260921_rs3_update/runtime_sources/stereo_result.yaml)、[rs1 手眼标定输入](../runtime/evidence/dual_base_20260921_rs3_update/runtime_sources/rs1_handeye.yaml)、[组合外参与点云证据](../runtime/evidence/dual_base_20260921_rs3_update/)。

## 历史核对记录（2026-09-11）

以下保留当时的检查结果、缺口和采集条件。旧子项编号沿用当时定义；当前完成状态及完成方式以上文为准。

当时状态：**1/5 完成**。M2.1 已由实时设备与数据验证通过；M2.2 的两份内参拟合已通过，但米制深度验证尚未完成；M2.3–M2.5 仍需独立基座验证，不能发布正式相机 TF。

### 历史 M2.1：设备与数据源

现场枚举确认两台设备均为 Intel RealSense D435，固件 5.17.3.10：

- rs1：`346522071783`
- rs3：`934222070377`

使用正式工程环境和 640×360@30 Hz 彩色配置，同时开启 848×480@30 Hz 深度并对齐到彩色。每台采集 90 组同步 RGB、对齐深度和 CameraInfo：

| 指标 | rs1 | rs3 |
|---|---:|---:|
| RGB 实测频率 | 29.980 Hz | 29.981 Hz |
| 对齐深度实测频率 | 29.980 Hz | 29.981 Hz |
| RGB/深度最大时间差 | 0 ms | 0 ms |
| 平均有效深度比例 | 67.58% | 71.03% |
| 实际 optical frame | `rs1_color_optical_frame` | `rs3_color_optical_frame` |

验证器现会核对实时 serial 和 optical frame，并记录消息中的真实 frame，不再写固定占位名称。正式相机、运行和标定 launch 的彩色默认配置已统一为 640×360@30 Hz。

证据：[rs1 实时报告](../runtime/evidence/m2_20260911T021200Z/rs1_live_rgbd_final.json)、[rs3 实时报告](../runtime/evidence/m2_20260911T021200Z/rs3_live_rgbd_final.json)、[rs1 现场图像](../runtime/evidence/m2_20260911T021200Z/rs1_live.jpg)、[rs3 现场图像](../runtime/evidence/m2_20260911T021200Z/rs3_live.jpg)。相机会话结束后无 ROS/RealSense 残留进程，CAN TX 始终为 0。

### 历史 M2.2–M2.5：当时缺口

| 子目标 | 已有证据 | 仍需完成 |
|---|---|---|
| M2.2 内参与深度 | rs1/rs3 的 640×360 内参记录 RMS 分别为 0.2371 px、0.2435 px；实时 RGB-D 尺寸及深度有效率通过 | 两份记录均标记未独立验证；需用当前实时 CameraInfo、100 mm 标记和对齐深度检查米制尺度与配准 |
| M2.3 rs1 外参 | 20 样本内部一致性通过，平移 RMS 6.02 mm、旋转 RMS 0.899° | 需要独立 validation poses；确认 `arm_base` 与 `base_link` 的物理关系 |
| M2.4 rs3 外参 | 20 样本内部一致性通过，平移 RMS 6.40 mm、旋转 RMS 0.994° | 同上；不能继承 rs2，也不能只凭内部 PASS 激活 |
| M2.5 正式 TF | 两路驱动内部 `rs*_link → rs*_color_optical_frame` 已实时存在 | 需要至少 6 个独立双相机验证姿态，达到中位误差 ≤5 mm、P95 ≤8 mm，并核对绝对基座位置后才能审批 |

静态记录审计见 [static_calibration_audit.json](../runtime/evidence/m2_20260911T021200Z/static_calibration_audit.json)。现有外参仍保持 `UNCALIBRATED`、`publish_tf_allowed: false`、`precision_operation_allowed: false`。

### 历史基座坐标系外参复核

本次只选用最新且内部通过的两份原始结果，并分别复制到不可变证据目录：

- rs1：`20260910T100042Z_346522071783.yaml`，20 样本，平移 RMS 6.02 mm、旋转 RMS 0.899°；
- rs3：`20260910T123031Z_934222070377.yaml`，20 样本，平移 RMS 6.40 mm、旋转 RMS 0.994°。

复算确认两份 `base_T_link × link_T_optical = base_T_optical`，平移闭环误差均小于 `2e-16 m`，旋转矩阵合法。M1 已用同一姿态核对 Piper 控制器末端反馈和本工程 `base_link → link6` FK，位置差 0.067 mm、姿态差 0.0032°，因此把原结果中的机械臂安装基座 `arm_base` 映射为工程 `base_link` 有数值依据。该映射仍须由新的多姿态数据验证绝对位置。

两次独立标定反推出的固定 `link6_T_marker` 相差 10.6 mm、1.06°，大于正式双相机 5/8 mm 门槛，不能仅凭各自的内部 PASS 激活。候选矩阵、原始文件副本和复算报告见 [基座复核目录](../runtime/evidence/m2_20260911_base_frame_validation/)。候选均保持 `publish_tf_allowed: false`。

实时预检已确认 rs1、rs3 序列号、RGB-D、CameraInfo、相机内部 TF 和只读 `base_link → link6` 链路可启动；顺序启动相机可避免 USB 初始化竞争。随后已采集 6 个互不重复的双相机 validation poses，三轴位置跨度为 174.6、205.9、99.4 mm，最大姿态跨度为 54.0°，满足独立验证集的数量和覆盖范围要求。

现有外参候选没有通过该验证集：按原标定文件内参计算时，双相机基座标记位置差的中位数为 8.26 mm、P95 为 18.91 mm；按当前驱动发布的 CameraInfo 计算时，中位数为 21.21 mm、P95 为 34.38 mm，均超过 5/8 mm 门槛。当前在线 CameraInfo 与对齐深度的标记中心一致性优于旧文件内参，因此不能用切换回旧内参掩盖外参误差。正式结论为 `FAIL_RECALIBRATION_REQUIRED`，rs1、rs3 候选保持禁止发布；完整数据集和指标见 [独立验证报告](../runtime/evidence/m2_20260911_base_frame_validation/external_candidate_validation_report.yaml)。

这 6 个姿态已封存为独立验证集，不参与下一次求解。下一步使用当前实时 CameraInfo、对齐深度和 `base_link → link6` 采集至少 18 个新的 optimization poses，重新求解 rs1、rs3 外参，再用这 6 个固定姿态复验。当前在线会话只读取反馈，不发送 Piper 运动命令。

### 当时继续采集所需现场条件

标定节点要求至少 18 个 optimization poses 和 6 个互不重复的 validation poses。每个姿态由 50 帧 RGB-D、实时 `base_link → link6` 和 `/joint_states_single` 组成，全程只读。

当前 `can0` 已为 UP、ERROR-ACTIVE、1 Mbps。若重启或接口变为 DOWN，启动只读反馈前由有 sudo 权限的操作者执行：

```bash
sudo ip link set can0 up type can bitrate 1000000
```

正式图像为 640×360，标记检测门槛已按分辨率修正为最短边至少 30 px、距图像边缘至少 30 px。继续前需在安全的人工/示教模式下调整机械臂姿态，让标记正面同时朝向两台相机，并使两路至少 12/25 个深度网格点有效。ReKpiper 不会为标定发送运动命令。
