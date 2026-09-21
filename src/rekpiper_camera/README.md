# rekpiper_camera

该包只负责 ReKep 在线 RGB-D 数据面：双 D435 启动与同步、彩色/深度反投影、
工作区裁剪、双视角点云融合，以及发布已经验收的固定相机外参。

正式入口是 `dual_rgbd_runtime.launch`。活动外参位于
`config/active_fixed_camera_extrinsics/`；默认状态为 `UNCALIBRATED`，因此不会
发布可用于精密操作的 TF。采集、求解和交叉检查统一放在
`rekpiper_calibration`，本包不包含机械臂控制、GUI、数据录放或重复标定流程。

M3 的临时观测入口为 `m3_observation_test.launch`，复用正在运行的 rs1/rs3
驱动，显式接收外部 `PROVISIONAL_DIAGNOSTIC` 外参副本，依次启动 TF、
投影、工作区过滤和融合。正式活动外参与精密操作审批不随观测测试改变。
9 月 10 日候选的测试结果、现场限制和可复制命令见
[M3 进度](../../docs/M3_PROGRESS.md)。

投影保留在线 CameraInfo 和深度采集时间，要求三路消息属于同一 optical
frame；默认同步容差 15 ms，避免 30 Hz 下相邻视频帧错配。拒绝非零畸变系数；
`16UC1` 按配置毫米尺度换算，`32FC1` 按米处理。
`points_base` 与 `points_recognition` 都保持 H×W×3，坏深度和 ROI 外像素为
NaN。融合输出使用 3 mm 体素和 25 ms 双路时间容差，拒绝超过 0.5 s 的云，
并防止已消费帧或迟到旧帧再次输出。融合的 RGB 字段表示相机来源颜色，
并非原图的真实颜色；抓取节点读取其中 XYZ。
