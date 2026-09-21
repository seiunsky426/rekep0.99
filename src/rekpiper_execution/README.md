# ReKep-Piper execution

该包只负责把已批准的 ReKep 短时关节前缀发送给 Piper，并在每个 20 Hz
控制周期检查关节反馈、Piper 状态、精确相机外参和当前安全地图 generation。
任一检查失败都会取消 action、撤销硬件授权并调用停止服务。

生产入口、验收步骤和任务调用方式见仓库根目录 `README.md`。
