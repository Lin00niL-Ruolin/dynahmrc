"""
DynaHMRC Integration Module
集成层：桥接 dyna_hmrc_web 的 LLM 逻辑与 BestMan 仿真平台
"""

from .bestman_adapter import BestManAdapter, ExecutionFeedback, ActionType
from .robot_factory import RobotFactory

__all__ = [
    "BestManAdapter",
    "ExecutionFeedback",
    "ActionType",
    "RobotFactory",
]
