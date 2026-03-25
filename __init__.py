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
    - Tasks: 三类任务（PackObjects, MakeSandwich, SortSolids）
    - Evaluation: 评估指标和实验运行器
"""

__version__ = "1.0.0"
__author__ = "DynaHMRC Team"

# 延迟导入 - 只在需要时导入，避免循环依赖
# 核心系统组件
__all__ = [
    # 核心系统
    "DynaHMRCSystem",
    "BestManAdapter",
    "ExecutionFeedback",
    "RobotFactory",
    # 机器人类型
    "ArmRobot",
    "MobileBase",
    "MobileManipulator",
    # 任务类型
    "BaseTask",
    "TaskVariation",
    "PackObjectsTask",
    "MakeSandwichTask",
    "SortSolidsTask",
    # 评估模块
    "MetricsCollector",
    "MetricsFormatter",
    "TaskMetrics",
    "RobotMetrics",
    "ExperimentRunner",
    "ExperimentConfig",
    "TaskVariationInjector",
    # 四阶段协作模块
    "FourStageCollaboration",
    "CollaborationPhase",
    "CollaborationManager",
    "CollaborationResult",
    "RobotAgent",
    "MemoryModule",
]


def __getattr__(name):
    """延迟导入，避免循环依赖"""
    # 核心系统
    if name == "DynaHMRCSystem":
        from .system import DynaHMRCSystem
        return DynaHMRCSystem
    if name == "BestManAdapter":
        from .integration.bestman_adapter import BestManAdapter
        return BestManAdapter
    if name == "ExecutionFeedback":
        from .integration.bestman_adapter import ExecutionFeedback
        return ExecutionFeedback
    if name == "RobotFactory":
        from .integration.robot_factory import RobotFactory
        return RobotFactory
    
    # 机器人类型
    if name == "ArmRobot":
        from .robots.arm_robot import ArmRobot
        return ArmRobot
    if name == "MobileBase":
        from .robots.mobile_base import MobileBase
        return MobileBase
    if name == "MobileManipulator":
        from .robots.mobile_manipulator import MobileManipulator
        return MobileManipulator
    
    # 任务类型
    if name == "BaseTask":
        from .tasks.base_task import BaseTask
        return BaseTask
    if name == "TaskVariation":
        from .tasks.base_task import TaskVariation
        return TaskVariation
    if name == "PackObjectsTask":
        from .tasks.pack_objects import PackObjectsTask
        return PackObjectsTask
    if name == "MakeSandwichTask":
        from .tasks.make_sandwich import MakeSandwichTask
        return MakeSandwichTask
    if name == "SortSolidsTask":
        from .tasks.sort_solids import SortSolidsTask
        return SortSolidsTask
    
    # 评估模块
    if name in ["MetricsCollector", "MetricsFormatter", "TaskMetrics", "RobotMetrics"]:
        from .evaluation import metrics
        return getattr(metrics, name)
    if name in ["ExperimentRunner", "ExperimentConfig", "TaskVariationInjector"]:
        from .evaluation import experiment_runner
        return getattr(experiment_runner, name)
    
    # 四阶段协作模块
    if name == "FourStageCollaboration":
        from .core.collaboration import FourStageCollaboration
        return FourStageCollaboration
    if name == "CollaborationPhase":
        from .core.collaboration import CollaborationPhase
        return CollaborationPhase
    if name == "CollaborationManager":
        from .core.collaboration import CollaborationManager
        return CollaborationManager
    if name == "CollaborationResult":
        from .core.collaboration import CollaborationResult
        return CollaborationResult
    if name == "RobotAgent":
        from .core.robot_agent import RobotAgent
        return RobotAgent
    if name == "MemoryModule":
        from .core.memory import MemoryModule
        return MemoryModule
    
    raise AttributeError(f"module 'dynahmrc' has no attribute '{name}'")
