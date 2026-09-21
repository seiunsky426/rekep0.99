# M3：统一三维观测测试

## 完成状态：M3.3、M3.4 已按用户确认标记完成

2026-09-12，用户确认当前工作区过滤与双视角真彩显示效果，并明确要求将
M3.3、M3.4 标为完成、开始 M4。完成范围为当前场景的观测处理功能：工作区
XYZ `[-0.1,-0.45,-0.1]` 至 `[0.75,0.55,0.6]` m，真实过滤/融合输出和 RViz
检查通过。下面保存的物理错层、深度噪声、持续时延和 M2 标定限制不变，
不将此次确认写成独立物理精度或机器人执行验收。

- [x] M3.3 工作区过滤。
- [x] M3.4 双视角融合与真彩观测显示。

随后执行 M4 时 RS1 发生 USB 断流，用户重新插拔后恢复；收尾检查还发现
RS3 时间戳超前约 0.963 s。定向重启 RS3 驱动、重建时间基准后，20 s 收到
92 个融合输出，RS3 RGB 平均接收延迟约 0.039 s，深度曝光仍为 1 ms。
见 [M4 执行与恢复记录](M4_PROGRESS.md) 及
[融合恢复证据](../runtime/evidence/m4_20260912T091408Z/final_camera_check.json)。

## 最新：2026-09-12 工作区 X 上限收至 0.75 m

按用户要求，当前观测工作区 XYZ 最小为 **[-0.1,-0.45,-0.1] m**，最大为
**[0.75,0.55,0.6] m**。仅改变 X 上限，RS3 保持 1 ms、gain 16、300 mW，
外参仍为昨天 18 点在线内参 PARK 版。已重新加载 M3 观测节点；相机继续运行。
两路各 10 帧及 10 个融合输出的完整检查通过，包含像素对应、边界和新鲜度。
该短窗口通过不替代之前持续时延、噪声和物理精度验收。

当前配置：[recognition_workspace.yaml](../runtime/evidence/m3_roi075_20260912T085633Z/recognition_workspace.yaml)；
实测报告：[live_observation_report.json](../runtime/evidence/m3_roi075_20260912T085633Z/live_observation_report.json)。
用户随后要求保留 RGB 查看融合；已打开 RViz，显示
`/rekpiper/camera/fused/points_rgb_visualization`。该真彩可视化流将时间配对、
按当前工作区裁剪的两路原始 XYZRGB 点拼接，保留每点颜色，不做体素平均；
既有 3 mm 来源色融合话题继续运行。见 [RViz 配置与实测截图](../runtime/evidence/m3_rgb_rviz_20260912T090445Z/README.md)。
以下记录保留，表述此前较大的 X 范围。

## 2026-09-12 RS3 改为 1 ms 并保留生效

按用户要求将 RS3 深度曝光改为 **1000 μs**；实际帧元数据确认生效，gain=16、
laser_power=300 mW、自动深度曝光关闭。继续使用昨天 18 点在线内参 PARK 版
外参与同一 ROI；master 仍为 11321。

两路各 30 帧的投影和过滤检查通过，平均保留 rs1 **110,097** 点、rs3
**121,248** 点。独立 30 帧中，RS3 桌面 15000 像素区域有效深度覆盖为
**99.85%**，中央 6400 像素为 **99.09%**；中央有效像素的时间标准差中位数
仍为 **21.69 mm**。覆盖改善不代表绝对精度通过。

首轮完整检查因 **2 帧接收延迟超过 0.6 s** 保持 false。随后约 20 秒独立
监测收到 78 个融合输出，平均 **153,012 点、3.89 Hz**，接收延迟中位数
0.456 s、最大 0.504 s；节点记录 18 次处理后过期和 1 次输入过期拒绝。
10 对点云桌面高度差的跨帧中位数为 **7.04 mm**，逐帧 P95 的跨帧中位数
为 **19.58 mm**。共同网格集合随有效深度变化，仍是诊断比较，几何验收待完成。

新证据：[曝光前后图](../runtime/evidence/m3_18pts_1ms_20260912T084613Z/rs3_2ms_vs_1ms.png)、
[双路过滤与融合图](../runtime/evidence/m3_18pts_1ms_20260912T084613Z/observation_result.png)、
[完整说明](../runtime/evidence/m3_18pts_1ms_20260912T084613Z/README.md)。
以下 2 ms 记录保留，表述的是此前状态。

## 2026-09-12：使用昨天 18 点 PARK 外参的实测

随后用户指出 RS3 深度覆盖未复现昨日改善。逐帧元数据确认 2 ms、gain 16、
300 mW 已生效，但新曝光对照中同一桌面图像区域在 2 ms 下仅 56%–59%
有效，1 ms 下为 99.83%，返回 2 ms 后再次下降。说明昨日曝光设置不适合
今天的场景；具体光学原因尚缺今日红外图像证据。试验后恢复 2 ms，未改变
下方 M3 实测记录。详见 [曝光与掩膜原因诊断](../runtime/evidence/rs3_exposure_20260912T084232Z/README.md)。

用户明确选择 **2026-09-11 在线内参、PARK 初始化版**。来源为
`/home/zhengsihze/eyetohandauto_ws/records/diagnostic_18pts_20260911T121808.128810Z/park_initialized/`，
求解报告生成于北京时间 20:24:09；14 个拟合点、4 个验证点，缺 P19/P20。
12 个归档文件的 SHA256 全部核对通过。两台实际设备序列号为 rs1
`346522071783`、rs3 `934222070377`，均为 USB 3.2、640×360 RGB 和对齐深度。

两套矩阵原样复制到本轮独立观测配置，通过现有 `PROVISIONAL_DIAGNOSTIC`
接口用于 master **11321**；原始 `publish_tf_allowed: false` 结果和包内活动配置
不变，观测副本 `precision_operation_allowed: false`。实时投影采用在线
CameraInfo（D=0），没有采用文件内参版，没有附加 ICP 或外参修正。
沿用上轮桌面观测 ROI：XYZ 最小 `[-0.1,-0.45,-0.1] m`，最大
`[1.1,0.55,0.6] m`；RS3 深度曝光 2,000 μs、gain=16、laser_power=300 mW。

| 检查 | 本轮结果 |
|---|---|
| 工作区过滤 | 两路各 30 帧的像素、NaN、边界与坐标不变检查通过；平均保留 rs1 109,473 点、rs3 80,593 点 |
| 同步与投影 | 同相机 RGB-D 时间差均为 0；TF 与所选外参一致；最大数值投影差约 1.35e-7 m，不是物理精度 |
| 首轮完整检查 | `data_contract_pass: false`：两路各 30 帧、13 个有效融合输出，另有 1 帧接收延迟超过 0.6 s |
| 随后独立 20 秒轻量监测 | 91 帧，平均 124,641 点，4.51 Hz；接收延迟中位数 0.251 s、最大 0.470 s，无接收超 0.6 s 帧；它不替代首轮完整检查 |
| 融合配对与拒绝 | 最大配对时差 17.88 ms，已消费时间戳严格递增；记录 3 次 `STALE_AFTER_FUSION`、1 次 `STALE_OR_INVALID_INPUT`，未放宽 0.5 s 节点新鲜度限制 |
| 重叠面 | 10 对实测点云，同一深度数据用新旧外参分别计算：高度差中位数的跨帧中位数 17.46→15.96 mm，逐帧 P95 的跨帧中位数 58.65→44.58 mm；仍有明显错层 |

重叠面沿用 X=0.1～0.6、Y=−0.15～0.25、Z=±0.08 m 的原始点云高度带，
每个 20 mm XY 网格要求两路各至少 10 点。新旧外参的共同有效网格数可能不同，
且高度带可能包含物体、边缘和深度噪声；上述数值是诊断比较，不是已知物理点的
绝对位置误差，也不能直接与昨天不同场景的 11.93 mm 比较。

**结论：18 点外参已用于实时过滤和融合；M3.3 的现场边界、M3.4 的空间配准
验收仍未完成，持续稳定 5 Hz 也未达到。** 本轮仅启动相机和观测节点。

本轮证据：[结果图](../runtime/evidence/m3_18pts_20260912T083133Z/observation_result.png)、
[同数据新旧外参对比图](../runtime/evidence/m3_18pts_20260912T083133Z/extrinsics_overlap_comparison.png)、
[指标与逐对几何比较](../runtime/evidence/m3_18pts_20260912T083133Z/summary.json)、
[首轮完整报告](../runtime/evidence/m3_18pts_20260912T083133Z/live_observation_report.json)、
[20 秒状态记录](../runtime/evidence/m3_18pts_20260912T083133Z/steady_observation.json)、
[配置、来源和复测说明](../runtime/evidence/m3_18pts_20260912T083133Z/README.md)。

## 2026-09-11 记录（保留；下文“当前”指当日会话）

核对日期：2026-09-11，桌面范围调整后的双路复测约 17:01。按用户要求，采用
**9 月 10 日的 rs1/rs3 外参**推进观测，未重算或修正矩阵。
**两台相机均已恢复，M3 四部分的数据链路实测通过。** M3.1 同步/数值投影和
M3.2 点图契约已完成；M3.3 的现场 ROI、M3.4 的重叠面几何验收仍未完成。
双路点云存在可见错层，不能把数据契约 PASS 当作定位精度验收。

## 按用户要求保留桌面及其物体

用户确认桌面上的圆盘、方块等需要保留。已在独立 master 11321 启用新的
观测范围：最小 XYZ `[-0.1, -0.45, -0.1] m`，最大 `[1.1, 0.55, 0.6] m`。
原 X=0.7 m 上限确实裁掉了这些物体附近的有效点。在同一张新点图上比较两种
ROI，rs1 黄色圆盘诊断矩形内保留点由 1,804 增至 6,576，粉色方块附近由
1,256 增至 6,822，均保留了矩形内全部有效点。这些矩形是人工选取的图像区域，
不是精确的物体分割或识别准确率统计。M3 掩膜表示几何工作区，桌面和区内
机械臂等也会为白色。

RS3 中央缺失还涉及深度采集。红外中央桌面诊断区的左右图饱和像素比例原为
99.15% / 94.86%（灰度 ≥250），短曝光下均降至 0。最终采用 **手动曝光
2,000 μs、gain=16、laser_power=300 mW**；关闭的是深度传感器自动曝光，
彩色流的曝光没有调整。该档 30 帧的中央有效深度覆盖率为 99.997%，原自动
曝光复测约 0.073%；这支持红外过曝是本场景深度缺失的重要原因。新增深度
来自相机实测，没有启用插值或补洞。检查红外亮度、曝光及投射器设置的方法
参考 [RealSense 官方调参说明](https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/)。

新一轮 M3 验证：两路各 30 帧、10 个融合输出通过；RGB-D 时间差均为 0，
双路平均保留点数为 **108,739 / 120,912**。融合平均 148,843 点，实测约
**2.93 Hz**，低于配置的 5 Hz 目标；最大配对时差 24.31 ms，最大接收延迟
599.73 ms。节点仍在计算完成后检查 0.5 s 新鲜度，验证器接收端允许 0.6 s；
该轮接收延迟接近上限，不能将其描述为稳定 5 Hz。新样本中央 6,400 个像素
全部有效且在 ROI 内，原样本对应区域全部深度为 0。
随后 6 秒状态检查收到 26 个 READY，同时出现过输入过期和融合完成后过期
的拒绝状态；这些帧被丢弃，没有通过放宽新鲜度限制增加输出。

这些改动解决了观测覆盖问题；所选曝光的中央有效像素深度时间标准差中位数
仍约 **12.37 mm**（30 帧中至少 20 帧有效的像素），不代表绝对定位精度。
9 月 10 日矩阵、原始结果和 M2 的独立验证失败结论不变，物理配准验收仍待完成。

本轮证据：[前后掩膜图](../runtime/evidence/m3_table_20260911T085053Z/table_roi_before_after.png)、
[融合俯视/侧视图](../runtime/evidence/m3_table_20260911T085053Z/table_fused_views.png)、
[指标汇总](../runtime/evidence/m3_table_20260911T085053Z/summary.json)、
[逐帧验证](../runtime/evidence/m3_table_20260911T085053Z/live_observation_report.json)、
[曝光比较](../runtime/evidence/m3_table_20260911T085053Z/infrared_probe/exposure_comparison.png)、
[短曝光统计](../runtime/evidence/m3_table_20260911T085053Z/short_exposure_probe/depth_probe.json)。

## 首轮严格同步复测结果（16:40，保留记录）

| 部分 | 实现与验证 | 当前现场结论 |
|---|---|---|
| M3.1 同步与投影 | RGB、对齐深度、在线 K、深度采集时刻 TF；拒绝时差、错 frame、TF 缺失和不支持的深度编码 | 两路各 30 帧通过，RGB-D 最大时差均为 0；绝对定位仍受 M2 外参精度限制 |
| M3.2 有组织点图 | FLOAT32 XYZ、height=360、width=640、point_step=12、row_step=7680；无效像素全 NaN | 两路各 30 帧的尺寸、像素对应和 NaN 检查通过 |
| M3.3 工作区过滤 | 保留 H×W 布局与采集时间；边界外置 NaN；保留点坐标与原图逐元素一致 | 两路通过，平均保留 78,628 / 29,077 点；现场 ROI 仍为候选 |
| M3.4 双视角融合 | 时间队列配对、3 mm 体素、25 ms 容差；0.5 s 新鲜度限制；防重复消费、迟到旧帧及处理后过期 | 实测 13 个融合输出通过，约 4.61 Hz；重叠面存在错层，几何验收未通过 |

rs1/rs3 平均有效深度像素为 143,017 / 135,669（各 230,400 个像素）。每帧
分布于整张图的最多 128 个有效像素，独立复算最大数值差分别为
`1.563e-7 m` / `1.330e-7 m`，证明软件投影与选定矩阵一致；这些数值
**不是实际空间定位精度**。两路点图检查延迟中位数为 0.332 / 0.328 s。

融合平均每帧 62,739 点（62,127～63,180），最大双路时间差 23.26 ms，
最大接收延迟 365.64 ms；两路已消费时间戳均递增，未重复使用旧帧。
原 50 ms 同相机容差允许 30 Hz 下相邻帧配对，现已收紧为 15 ms；复测两路
各 30 帧的 RGB、深度和 CameraInfo 时间戳完全一致。

首轮严格同步证据：[逐帧 JSON](../runtime/evidence/m3_20260911T083450Z_dual/isolated/strict_sync/live_observation_report.json)、
[指标汇总](../runtime/evidence/m3_20260911T083450Z_dual/isolated/strict_sync/summary.json)、
[双路图像/ROI/融合图](../runtime/evidence/m3_20260911T083450Z_dual/isolated/strict_sync/dual_observation.png)。
首轮单路记录仍保留在 [原证据目录](../runtime/evidence/m3_20260911T080139Z/)。

## 重叠面的实际限制

使用相差 10.18 ms 的两路原始点图，检查共同 XY 区域的桌面高度带，避免
ROI 下限提前裁掉错层点。将 X=0.1～0.6 m、Y=−0.15～0.25 m、Z=±0.08 m
内的点按 20 mm XY 网格取各自中位高度，保留两路各至少 10 点的 140 个共同
网格；高度差绝对值中位数为 **11.93 mm**，P95 为 **50.68 mm**。

该结果是桌面附近点云的诊断比较，可能包含局部物体或边缘，不能解释为独立
物理点的绝对定位误差。图中仍能看到错层，原 M2 外参独立验证失败结论保留。
因此 M3.4 的融合数据接口已经可用，但“重叠面无明显错层”的验收项仍未完成。
见 [重叠面报告](../runtime/evidence/m3_20260911T083450Z_dual/isolated/strict_sync/overlap_report.json)
和 [可重跑脚本](../runtime/evidence/m3_20260911T083450Z_dual/isolated/strict_sync/analyze_overlap.py)。

## RS3 掩膜中央为何为黑色

白色要求原始深度有效且变换后的三维点位于识别工作区内。对上述图像的原始
数组逐像素核查：中央 `u=280..359, v=160..239` 的 6,400 个像素，深度全部
为 0，因此投影为 NaN、掩膜为黑色。像素 `(320,220)` 即为一个例子；
`(450,280)` 的深度为 954 mm，基座坐标约为 `(0.152,-0.018,-0.003) m`，
位于工作区内，所以为白色。

在更大的中央桌面区域 `u=250..399, v=190..289` 中，共 15,000 个像素：
13,350 个（89.00%）深度无效，504 个（3.36%）低于 Z 下限 −17 mm，
1,146 个（7.64%）保留。该区域没有因 XY 越界被剔除的点。因此中央大块黑色
首先来自深度缺失，不能仅归因于外参或 ROI 边界。后续红外曝光对照见本文
开头；扩大 ROI 本身不会恢复深度为 0 的像素。

见 [原因分色图](../runtime/evidence/m3_20260911T083450Z_dual/isolated/strict_sync/rs3_mask_causes.png)
和 [像素诊断记录](../runtime/evidence/m3_20260911T083450Z_dual/isolated/strict_sync/rs3_mask_diagnosis.json)。

## 外参与工作区依据

原始结果及归档 `SHA256SUMS` 已重新核对，原件与旧审批文件均保留：

| 相机 | 原结果 | SHA256 |
|---|---|---|
| rs1 / 346522071783 | `20260910T100042Z_346522071783.yaml` | `e8bbd250940117bd1c28a0813d4cce168e5fa4bb9009fe2ffed781f1c61697d4` |
| rs3 / 934222070377 | `20260910T123031Z_934222070377.yaml` | `2e0f21399773efaae697c8509e49cf5500d44e8a1c6b36d807fa3fc0b84f869a` |

测试副本与原始副本位于 [当前独立会话](../runtime/evidence/m3_table_20260911T085053Z/)，
与首轮副本的矩阵相同。
采用现有 `PROVISIONAL_DIAGNOSTIC` 接口，测试副本允许诊断 TF，
`precision_operation_allowed: false`；包内活动外参仍为 `UNCALIBRATED`。
TF 链为 `base_link → rs*_link → rs*_color_optical_frame`，其中相机内部链由
驱动发布；没有直接给 optical frame 添加第二个父节点。

`arm_base` 到工程 `base_link` 的名称对应沿用 M1 数值复核依据。
10 日外参在独立多姿态验证中，使用在线 CameraInfo 的双相机位置差中位数
21.21 mm、P95 34.38 mm，原 `FAIL_RECALIBRATION_REQUIRED` 结论保留。
本次使用在线 CameraInfo 投影对齐深度，不切换旧文件内参。

首轮识别范围依据 M2 的现场 XY/上边界和约 −7 mm 桌面高度，额外保留桌面
下方 10 mm。用户确认桌面上的物体需要保留后，当前观测副本扩大如下：

| 范围 | 最小 XYZ（m） | 最大 XYZ（m） |
|---|---|---|
| 首轮识别 ROI | `[0, -0.4, -0.017]` | `[0.7, 0.4, 0.6]` |
| 当前桌面观测 ROI | `[-0.1, -0.45, -0.1]` | `[1.1, 0.55, 0.6]` |
| 工程规划边界 | `[-0.2, -0.6, -0.07]` | `[0.8, 0.6, 0.8]` |
| 地图边界 | `[-0.8, -0.8, -0.07]` | `[0.8, 0.8, 1.5]` |

首轮 ROI 包含在规划和地图边界内；当前观测 ROI 的 X 上限和 Z 下限超出
这两类边界。扩展只用于观测，未经现场重新测量，不等同于运动安全边界，
不能用来替代下游工作区约束。包内历史 ROI、规划和地图配置保持原值。

## 连接恢复记录

RS3 原驱动报 `RS2_USB_STATUS_NO_DEVICE`；定向重启后报 `HW not ready`，再尝试
同序列号设备复位仍无图像，随后 USB 枚举中只剩 rs1。保持相机安装位置不动，
恢复 RS3 USB 连接后先确认序列号和 RGB-D 帧。已有 rs1 驱动继续复用。
首轮新增的临时观测与 RS3 重连进程已结束，原有 rs1 与 ROS master 保留。

### 16:27 USB 重连后的复测

用户重新连接 USB 后，实际设备信息服务确认唯一在线设备仍为 rs1
`346522071783`，其 `firmware_update_id` 为 `428623020421`。USB 描述符中的
这个固件更新 ID 不能替代用于外参绑定的设备序列号。rs3 `934222070377` 仍未
被驱动识别，日志明确报告 requested device NOT found。

rs1 自动重连后原私有参数缺失，输出曾退回 640×480、
`camera_color_optical_frame`；已定向重启恢复为 640×360、
`rs1_color_optical_frame` 和 `16UC1` 对齐深度。新的 60 秒观测窗口再次完成
rs1 30 帧校验，rs3 0 帧、融合 0 帧，`data_contract_pass: false`。
本次 CAN TX 为 283→283，未发送机器人控制。

证据保存在独立的 [重连复测目录](../runtime/evidence/m3_20260911T082734Z_retest/)，
包括 [输入恢复记录](../runtime/evidence/m3_20260911T082734Z_retest/input_recovery.json)
和 [逐帧报告](../runtime/evidence/m3_20260911T082734Z_retest/live_observation_report.json)。
本次只读相机与 M3 观测进程保留，rs3 驱动等待指定序列号设备连接。
继续时先查 `rosnode list`，复用现有节点，避免重复启动；仍需确认两台相机
同时连接，再完成双路实测。

### 16:34 双路恢复及 ROS 会话隔离

RS3 已由 `device_info` 服务确认序列号 `934222070377`，固件更新 ID 为
`935523022795`。两台相机均发布 640×360 RGB、对齐深度和正确 optical frame。
期间共享 master 11311 被另一个回零会话重启，原观测节点因此丢失注册和参数；
本任务仅重启自己管理的相机/M3 节点，将它们迁入 **11321 的独立只读 master**。
另一会话的机械臂节点未被操作。当前独立会话没有机器人控制节点。

## 查看与复测当前会话

当前相机、TF、投影、过滤和融合已在运行。以下查看/复测命令均使用 11321；
11311 上其他任务的 ROS 图不能代表本次 M3 状态。

所有终端先执行：

```bash
cd /home/zhengsihze/Rekep_v1.0-main
source setup.bash
export ROS_MASTER_URI=http://localhost:11321
export ROS_HOME="$PWD/runtime/data/ros/m3_dual_11321"
```

查看实时融合（READY 与等待下一对帧的状态会交替出现，以实际输出为准）：

```bash
rostopic hz /rekpiper/camera/fused/points_base
rostopic echo -n 1 /rekpiper/camera/fused/status
```

完整重启时才依次启动 `roscore -p 11321`、两台相机和 M3；已有节点时不要重复
启动。两台相机分别使用以下命令，在不同终端顺序启动并确认前一路出图：

```bash
roslaunch rekpiper_camera d435_rgbd.launch \
  camera:=rs1 serial_no:=346522071783 initial_reset:=false \
  width:=640 height:=360 fps:=30 enable_depth:=true align_depth:=true

roslaunch realsense2_camera rs_camera.launch \
  camera:=rs3 tf_prefix:=rs3 serial_no:=934222070377 initial_reset:=false \
  color_width:=640 color_height:=360 color_fps:=30 \
  depth_width:=848 depth_height:=480 depth_fps:=30 \
  enable_depth:=true align_depth:=true enable_pointcloud:=false \
  enable_infra1:=true enable_infra2:=true \
  infra_width:=848 infra_height:=480 infra_fps:=30
```

RS3 的红外流用于这次曝光诊断。相机重启后，在已加载同一 ROS 环境的另一
终端重新应用本轮深度参数；当前运行的驱动已应用，无需重复操作：

```bash
rosrun dynamic_reconfigure dynparam set /rs3/stereo_module \
  '{enable_auto_exposure: false, exposure: 2000, laser_power: 300.0, gain: 16}'
```

有效设置及两台真实序列号见 [运行状态](../runtime/evidence/m3_table_20260911T085053Z/final_state.json)。
试验各阶段均在结束后恢复原设置，最后才显式启用所选候选参数。原配置为
自动曝光、exposure 配置值 8500、laser_power=150、gain=16，保存在
[原始参数记录](../runtime/evidence/m3_table_20260911T085053Z/depth_controls_original.json)。
自动曝光下的配置值不等于逐帧实际曝光。

终端 2 启动四部分观测（同一 master 下不要重复启动同名节点）：

```bash
M3_SESSION="$PWD/runtime/evidence/m3_table_20260911T085053Z"
roslaunch rekpiper_camera m3_observation_test.launch \
  rs1_extrinsics:="$M3_SESSION/rs1_diagnostic_extrinsics.yaml" \
  rs3_extrinsics:="$M3_SESSION/rs3_diagnostic_extrinsics.yaml" \
  recognition_workspace:="$M3_SESSION/recognition_workspace.yaml"
```

终端 3 在新目录保存复测证据，避免覆盖本轮失败/部分完成记录：

```bash
M3_SOURCE="$PWD/runtime/evidence/m3_table_20260911T085053Z"
M3_RETEST="$PWD/runtime/evidence/m3_$(date -u +%Y%m%dT%H%M%SZ)_retest"
mkdir -p "$M3_RETEST"
cp -a "$M3_SOURCE/raw_sources" "$M3_RETEST/"
cp "$M3_SOURCE/session.json" "$M3_SOURCE/recognition_workspace.yaml" "$M3_RETEST/"
python tools/check_m3_observation.py --session "$M3_RETEST" --frames 30 --timeout 60
```

验证器订阅 `/rekpiper/camera/{rs1,rs3}/points_base`、`points_recognition` 和
`/rekpiper/camera/fused/points_base`，检查两路各 30 帧及至少 10 个融合输出与
配对状态。报告只有在这些检查全通过时才写 `data_contract_pass: true`；物理
验收字段仍保持 false。RViz 使用 `base_link`，显示两路 ROI 及融合点云；融合
来源颜色为 rs1 红、rs3 青、共用体素白。还需检查重叠面错层，并完成独立位置验收。

## 软件验证

- 相机包：26 项测试通过，包括深度单位、错 frame/时差/TF 拒绝、像素布局、ROI
  以及重复帧、迟到帧、节流消费、融合期间过期的行为测试。
- 仓库审计与 launch 路径/Shadow 参数展开通过。
- `catkin_make run_tests` 后检查汇总：**245 tests, 0 errors, 0 failures, 0 skipped**。
- 最新 15 ms 同步改动后，仓库/launch 审计、26 项相机测试和 245 项 Catkin
  测试重新通过。本任务没有启动 Piper 控制节点，M3 ROS 图位于独立 master。
- 桌面追加测试只调整本次观测参数并补充文档；双路现场逐帧验证、仓库审计及
  source 环境后的 launch 审计再次通过，代码测试沿用上述最后一次 245 项结果。
- 首轮 CAN TX 为 148→148；重连复测为 283→283。此次最终会话不能沿用这些
  旧数值证明宿主机无运动，因为另一 ROS master 曾运行独立的回零任务。

当前已完成候选外参下的 M3.1/M3.2，以及 M3.3/M3.4 的数据处理验证。
M3.3 的现场边界与 M3.4 的空间配准验收仍受 M2 和现场几何验证约束。
