"""Piper 的本地 URDF 数值 IK，供官方 ReKep 优化器调用。

官方 ``SubgoalSolver`` 和 ``PathSolver`` 只要求 IK 对象提供
``solve(target_pose_homo, max_iterations, initial_joint_pos)``。本模块用
URDF 前向运动学和 SciPy 最小二乘实现该接口；它不连接外部规划场景或
任何 ROS 控制接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from urdf_parser_py.urdf import URDF


@dataclass(frozen=True)
class PiperIKResult:
    """与官方 ReKep 优化器所需字段兼容的 IK 结果。"""

    success: bool
    num_descents: int
    position_error: float
    rotation_error: float
    cspace_position: np.ndarray
    status: str


@dataclass(frozen=True)
class _ChainJoint:
    """从父 link 到子 link 的一段 URDF 关节。"""

    name: str
    parent: str
    child: str
    joint_type: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def _transform(xyz, rpy) -> np.ndarray:
    """创建 URDF ``origin`` 对应的齐次变换矩阵。"""
    result = np.eye(4, dtype=float)
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def _rotation_about_axis(axis, angle) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = Rotation.from_rotvec(np.asarray(axis) * angle).as_matrix()
    return result


class PiperURDFIKSolver:
    """Piper 本地数值运动学求解器。

    该类只依赖 ``robot_description`` 的 URDF 文本。IK 是局部数值求解，
    因而以当前关节状态作种子；这与官方 ReKep 在每轮重规划中传入当前
    ``initial_joint_pos`` 的用法一致。
    """

    backend_name = "piper_urdf_numerical_ik"

    def __init__(
            self,
            base_frame: str,
            tip_frame: str,
            joint_names: Iterable[str],
            robot_description_param: str = "/robot_description",
            robot_description_file: str = "",
            position_tolerance: float = 0.01,
            orientation_tolerance: float = 0.10,
    ):
        import rospy

        xml = ""
        if rospy.has_param(robot_description_param):
            xml = str(rospy.get_param(robot_description_param))
        if not xml and robot_description_file:
            xml = Path(robot_description_file).expanduser().read_text(
                encoding="utf-8")
        if not xml:
            raise RuntimeError(
                "piper_urdf_unavailable: load {} or set robot_description_file"
                .format(robot_description_param))
        self._init_from_xml(
            xml, base_frame, tip_frame, joint_names, position_tolerance,
            orientation_tolerance)

    @classmethod
    def from_urdf_xml(
            cls,
            xml: str,
            base_frame: str,
            tip_frame: str,
            joint_names: Iterable[str],
            position_tolerance: float = 0.01,
            orientation_tolerance: float = 0.10,
    ):
        """创建纯 Python 测试或离线诊断可用的求解器。"""
        instance = cls.__new__(cls)
        instance._init_from_xml(
            xml, base_frame, tip_frame, joint_names, position_tolerance,
            orientation_tolerance)
        return instance

    def _init_from_xml(
            self, xml, base_frame, tip_frame, joint_names,
            position_tolerance, orientation_tolerance):
        self.base_frame = str(base_frame)
        self.tip_frame = str(tip_frame)
        self.joint_names = list(joint_names)
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        if not self.joint_names:
            raise ValueError("Piper IK requires at least one active joint")
        if self.position_tolerance <= 0.0 or self.orientation_tolerance <= 0.0:
            raise ValueError("Piper IK tolerances must be positive")

        robot = URDF.from_xml_string(xml)
        by_child = {joint.child: joint for joint in robot.joints}
        reversed_joints = []
        cursor = self.tip_frame
        while cursor != self.base_frame:
            joint = by_child.get(cursor)
            if joint is None:
                raise ValueError(
                    "URDF has no chain from {} to {}".format(
                        self.base_frame, self.tip_frame))
            reversed_joints.append(joint)
            cursor = joint.parent
            if len(reversed_joints) > len(robot.joints):
                raise ValueError("URDF link graph contains a cycle")

        self._chain = []
        active_names = []
        limits = {}
        for joint in reversed(reversed_joints):
            origin = joint.origin
            xyz = np.asarray(origin.xyz if origin is not None else [0, 0, 0],
                             dtype=float)
            rpy = np.asarray(origin.rpy if origin is not None else [0, 0, 0],
                             dtype=float)
            active = joint.type in ("revolute", "continuous")
            if active:
                active_names.append(joint.name)
                if joint.type == "continuous":
                    lower, upper = -np.pi, np.pi
                elif joint.limit is None:
                    raise ValueError("revolute joint {} has no limits".format(
                        joint.name))
                else:
                    lower, upper = float(joint.limit.lower), float(joint.limit.upper)
                limits[joint.name] = (lower, upper)
            else:
                lower = upper = 0.0
            self._chain.append(_ChainJoint(
                name=joint.name,
                parent=joint.parent,
                child=joint.child,
                joint_type=joint.type,
                origin=_transform(xyz, rpy),
                axis=np.asarray(joint.axis if joint.axis is not None else [0, 0, 1],
                                dtype=float),
                lower=lower,
                upper=upper,
            ))

        if active_names != self.joint_names:
            raise ValueError(
                "URDF active chain joints {} do not match configured joints {}"
                .format(active_names, self.joint_names))
        self._lower = np.asarray([limits[name][0] for name in self.joint_names])
        self._upper = np.asarray([limits[name][1] for name in self.joint_names])

    @staticmethod
    def description_available(robot_description_param: str,
                              robot_description_file: str = "") -> bool:
        """轻量检查模型是否可供本地 IK 使用。"""
        import rospy

        if rospy.has_param(robot_description_param):
            return bool(str(rospy.get_param(robot_description_param)))
        return bool(robot_description_file and Path(
            robot_description_file).expanduser().is_file())

    def forward(self, joints: Iterable[float]) -> np.ndarray:
        """计算 ``base_frame -> tip_frame`` 的前向运动学。"""
        values = np.asarray(joints, dtype=float).reshape(-1)
        if values.size < len(self.joint_names) or not np.all(np.isfinite(values)):
            raise ValueError("Piper FK joint seed is invalid")
        mapping = dict(zip(self.joint_names, values[:len(self.joint_names)]))
        pose = np.eye(4, dtype=float)
        for joint in self._chain:
            pose = pose @ joint.origin
            if joint.joint_type in ("revolute", "continuous"):
                pose = pose @ _rotation_about_axis(joint.axis, mapping[joint.name])
        return pose

    def link_transforms(self, joints: Iterable[float]) -> dict[str, np.ndarray]:
        """Return base-frame transforms for every link on the active chain."""
        values = np.asarray(joints, dtype=float).reshape(-1)
        if values.size < len(self.joint_names) or not np.all(np.isfinite(values)):
            raise ValueError("Piper FK joint state is invalid")
        mapping = dict(zip(self.joint_names, values[:len(self.joint_names)]))
        pose = np.eye(4, dtype=float)
        transforms = {self.base_frame: pose.copy()}
        for joint in self._chain:
            pose = pose @ joint.origin
            if joint.joint_type in ("revolute", "continuous"):
                pose = pose @ _rotation_about_axis(
                    joint.axis, mapping[joint.name])
            transforms[joint.child] = pose.copy()
        return transforms

    def _forward(self, joints: np.ndarray) -> np.ndarray:
        """内部 FK；单独保留避免改变公开 ``forward`` 的输入校验。"""
        mapping = dict(zip(self.joint_names, joints))
        pose = np.eye(4, dtype=float)
        for joint in self._chain:
            pose = pose @ joint.origin
            if joint.joint_type in ("revolute", "continuous"):
                pose = pose @ _rotation_about_axis(joint.axis, mapping[joint.name])
        return pose

    @staticmethod
    def _pose_errors(actual: np.ndarray, target: np.ndarray):
        position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        relative = actual[:3, :3].T @ target[:3, :3]
        rotation = float(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec()))
        return position, rotation

    def solve(
            self,
            target_pose_homo,
            max_iterations: int = 20,
            initial_joint_pos: Optional[Iterable[float]] = None,
            position_tolerance: Optional[float] = None,
            orientation_tolerance: Optional[float] = None,
            position_weight: float = 1.0,
            orientation_weight: float = 0.05,
            **_unused,
    ) -> PiperIKResult:
        """求解局部数值 IK，并返回官方 ReKep 兼容结果。"""
        target = np.asarray(target_pose_homo, dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("Piper IK target must be a finite 4x4 pose")
        if initial_joint_pos is None:
            seed = 0.5 * (self._lower + self._upper)
        else:
            seed = np.asarray(initial_joint_pos, dtype=float).reshape(-1)
            if seed.size < len(self.joint_names) or not np.all(np.isfinite(seed)):
                raise ValueError("Piper IK seed does not contain finite arm joints")
            seed = seed[:len(self.joint_names)]
        seed = np.clip(seed, self._lower, self._upper)
        pos_tol = self.position_tolerance if position_tolerance is None else float(
            position_tolerance)
        rot_tol = self.orientation_tolerance if orientation_tolerance is None else float(
            orientation_tolerance)

        def residual(joints):
            actual = self._forward(joints)
            position_delta = actual[:3, 3] - target[:3, 3]
            relative = actual[:3, :3].T @ target[:3, :3]
            rotation_delta = Rotation.from_matrix(relative).as_rotvec()
            return np.r_[float(position_weight) * position_delta,
                         float(orientation_weight) * rotation_delta]

        result = least_squares(
            residual, seed, bounds=(self._lower, self._upper), method="trf",
            max_nfev=max(1, int(max_iterations)), xtol=1e-5, ftol=1e-5,
            gtol=1e-5)
        actual = self._forward(result.x)
        position_error, rotation_error = self._pose_errors(actual, target)
        success = bool(
            np.all(np.isfinite(result.x))
            and position_error <= pos_tol
            and rotation_error <= rot_tol)
        return PiperIKResult(
            success=success,
            num_descents=int(min(max(0, result.nfev), max(1, int(max_iterations)))),
            position_error=position_error,
            rotation_error=rotation_error,
            cspace_position=np.r_[result.x, 0.0],
            status="ok" if success else "ik_residual_exceeded",
        )

    def validate_pose_sequence(self, poses, initial_joint_pos, trace=None) -> dict:
        """按顺序验证官方规划器产生的笛卡尔路径是否全程可达。"""
        from .continuous_ik import validate_continuous_ik
        return validate_continuous_ik(self, poses, initial_joint_pos, trace=trace)

    def diagnose_pose_sequence(self, poses, initial_joint_pos) -> dict:
        """Legacy residual-only scan for comparisons, never execution acceptance."""
        seed = np.asarray(initial_joint_pos, dtype=float)
        steps = []
        for index, pose in enumerate(np.asarray(poses, dtype=float)):
            result = self.solve(pose, initial_joint_pos=seed)
            steps.append({
                "pose_index": int(index),
                "success": bool(result.success),
                "status": result.status,
                "position_error": float(result.position_error),
                "rotation_error": float(result.rotation_error),
                "joint_positions": result.cspace_position[:-1].tolist(),
            })
            if not result.success:
                return {"valid": False, "reason": "ik_failed", "steps": steps}
            seed = result.cspace_position[:-1]
        return {"valid": True, "reason": "ok", "steps": steps}
