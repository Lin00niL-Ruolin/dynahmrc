"""
DynaHMRC: Dynamic Heterogeneous Multi-Robot Collaboration via Large Language Models
基于 BestMan 仿真平台的异构多机器人动态协作框架

本模块实现了论文《DynaHMRC: Dynamic Heterogeneous Multi-Robot Collaboration via Large Language Models》
的核心算法和系统集成。

主要组件:
    - DynaHMRCSystem: 主系统集成类，封装整个协作流程
    - BestManAdapter: BestMan API 适配层
    - RobotFactory: 机器人动态工厂
    - Heterogeneous robots: 三类异构机器人（ArmRobot, MobileBase, MobileManipulator）
"""

from .system import DynaHMRCSystem
from .integration.bestman_adapter import BestManAdapter, ExecutionFeedback
from .integration.robot_factory import RobotFactory
from .robots.arm_robot import ArmRobot
from .robots.mobile_base import MobileBase
from .robots.mobile_manipulator import MobileManipulator

__version__ = "1.0.0"
__author__ = "DynaHMRC Team"

__all__ = [
    "DynaHMRCSystem",
    "BestManAdapter",
    "ExecutionFeedback",
    "RobotFactory",
    "ArmRobot",
    "MobileBase",
    "MobileManipulator",
]
