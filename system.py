"""
system.py - DynaHMRC 主系统集成类
封装整个异构多机器人协作流程

实现论文中的核心算法:
- Algorithm 1: Dynamic Replanning（动态重规划）
- Algorithm 2: Task Allocation（任务分配）
- Algorithm 3: Conflict Resolution（冲突解决）
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# 导入 BestMan 核心组件
from Env.Client import Client
from Visualization.Visualizer import Visualizer
# 注：不直接使用 Config.load_config，因为它使用相对路径
# 我们将在 _init_pybullet 中自己加载配置

# 导入 dyna_hmrc_web 的 LLM 逻辑层
from Dyna_hmrc_web.dynahmrc_web.dynahmrc.coordinator import (
    DynaHMRC_Coordinator, ExecutionResult, ExecutionStatus
)
from Dyna_hmrc_web.dynahmrc_web.dynahmrc.utils.llm_api import (
    create_llm_client, BaseLLMClient
)

# 导入 DynaHMRC 集成层
from .integration.robot_factory import RobotFactory
from .integration.bestman_adapter import BestManAdapter, ExecutionFeedback


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLANNING = "replanning"


@dataclass
class SystemConfig:
    """系统配置数据类"""
    scene_config: Dict[str, Any]
    robot_configs: List[Dict[str, Any]]
    llm_config: Dict[str, Any]
    enable_replanning: bool = True
    max_replan_attempts: int = 3
    execution_interval: float = 0.1


class DynaHMRCSystem:
    """
    DynaHMRC 主系统类
    
    核心职责:
    1. 初始化 BestMan 场景和多机器人
    2. 集成 LLM 协调器进行任务规划
    3. 执行动态任务分配和监控
    4. 处理异常和触发重规划
    
    使用示例:
        >>> system = DynaHMRCSystem(scene_config, robot_configs, llm_config)
        >>> result = system.execute_task("把箱子搬到桌子上")
        >>> print(result)
    """
    
    def __init__(
        self,
        scene_config: Dict[str, Any],
        robot_configs: List[Dict[str, Any]],
        llm_config: Dict[str, Any],
        enable_visualization: bool = True
    ):
        """
        初始化 DynaHMRC 系统
        
        Args:
            scene_config: 场景配置
                {
                    "config_path": "Config/default.yaml",
                    "gui": True,
                    "objects": [...]
                }
            robot_configs: 机器人配置列表
                [
                    {
                        "robot_id": "robot_1",
                        "robot_type": "mobile_manipulator",
                        "robot_model": "panda_on_segbot",
                        "init_position": [0, 0, 0],
                        "capabilities": ["navigation", "manipulation"]
                    },
                    ...
                ]
            llm_config: LLM 配置
                {
                    "provider": "kimi",  # 或 "openai", "mock"
                    "api_key": "...",
                    "model": "kimi-k2.5",
                    "temperature": 0.3
                }
            enable_visualization: 是否启用可视化
        """
        self.scene_config = scene_config
        self.robot_configs = robot_configs
        self.llm_config = llm_config
        self.enable_visualization = enable_visualization
        
        # 核心组件
        self.client: Optional[Client] = None
        self.visualizer: Optional[Visualizer] = None
        self.robot_factory: Optional[RobotFactory] = None
        self.bestman_adapter: Optional[BestManAdapter] = None
        self.coordinator: Optional[DynaHMRC_Coordinator] = None
        
        # 状态管理
        self.is_initialized = False
        self.execution_history: List[Dict] = []
        self.current_task_status: Dict[str, TaskStatus] = {}
        
        # 统计信息
        self.start_time: Optional[float] = None
        self.replan_count = 0
        
        print("[DynaHMRCSystem] 系统实例已创建")
    
    def _ensure_working_directory(self):
        """确保工作目录是项目根目录"""
        import os
        # 获取 dynahmrc 包的路径
        current_file = os.path.abspath(__file__)
        # dynahmrc/system.py -> 项目根目录
        root_dir = os.path.dirname(os.path.dirname(current_file))
        
        # 如果当前目录不是根目录，切换到根目录
        if os.getcwd() != root_dir:
            os.chdir(root_dir)
            print(f"[DynaHMRCSystem] 切换工作目录到: {root_dir}")
    
    def initialize(self) -> bool:
        """
        初始化系统（初始化 BestMan 场景和机器人）
        
        Returns:
            是否初始化成功
        """
        try:
            print("[DynaHMRCSystem] 开始初始化...")
            
            # 确保工作目录正确（Config/utils.py 使用相对路径）
            self._ensure_working_directory()
            
            # 1. 初始化 PyBullet 客户端和可视化器
            self._init_pybullet()
            
            # 2. 加载场景
            self._load_scene()
            
            # 3. 初始化机器人工厂并创建机器人
            self._init_robots()
            
            # 4. 初始化 BestMan 适配器
            self.bestman_adapter = BestManAdapter(
                self.robot_factory.get_all_robots()
            )
            
            # 5. 初始化 LLM 协调器
            self._init_coordinator()
            
            self.is_initialized = True
            print("[DynaHMRCSystem] 初始化完成")
            return True
            
        except Exception as e:
            print(f"[DynaHMRCSystem] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_config_directly(self, config_path: str = "Config/default.yaml"):
        """
        直接加载配置文件（避免 Config.utils.load_config 的相对路径问题）
        """
        from yacs.config import CfgNode as CN
        
        # 获取当前文件所在目录（dynahmrc/）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 项目根目录
        root_dir = os.path.dirname(current_dir)
        
        # 构建绝对路径
        if config_path.startswith("Config"):
            default_config_path = os.path.join(root_dir, "Config", "default.yaml")
            user_config_path = os.path.join(root_dir, config_path) if config_path else None
        else:
            default_config_path = os.path.join(root_dir, "Config", "default.yaml")
            user_config_path = config_path if os.path.isabs(config_path) else os.path.join(root_dir, config_path)
        
        # 加载默认配置
        with open(default_config_path, "r") as f:
            cfg = CN.load_cfg(f)
        
        # 合并用户配置
        if user_config_path and os.path.exists(user_config_path) and user_config_path != default_config_path:
            cfg.merge_from_file(user_config_path)
        
        return cfg
    
    def _init_pybullet(self):
        """初始化 PyBullet"""
        config_path = self.scene_config.get("config_path", "Config/default.yaml")
        cfg = self._load_config_directly(config_path)
        
        # 覆盖 GUI 设置
        if not self.enable_visualization:
            cfg.Client.enable_GUI = False
        
        self.client = Client(cfg.Client)
        self.visualizer = Visualizer(self.client, cfg.Visualizer)
        
        print(f"[DynaHMRCSystem] PyBullet 客户端已创建 (GUI={cfg.Client.enable_GUI})")
    
    def _load_scene(self):
        """加载场景物体"""
        objects = self.scene_config.get("objects", [])
        
        for obj in objects:
            obj_name = obj.get("name", "object")
            model_path = obj.get("model_path")
            position = obj.get("position", [0, 0, 0])
            orientation = obj.get("orientation", [0, 0, 0, 1])
            
            if model_path:
                self.client.load_object(
                    obj_name=obj_name,
                    model_path=model_path,
                    object_position=position,
                    object_orientation=orientation
                )
        
        print(f"[DynaHMRCSystem] 已加载 {len(objects)} 个场景物体")
    
    def _init_robots(self):
        """初始化机器人"""
        self.robot_factory = RobotFactory(self.client, self.visualizer)
        
        for config in self.robot_configs:
            robot_id = config["robot_id"]
            robot_type = config["robot_type"]
            robot_model = config["robot_model"]
            init_position = config.get("init_position", [0, 0, 0])
            init_orientation = config.get("init_orientation", [0, 0, 0, 1])
            
            self.robot_factory.create_robot(
                robot_id=robot_id,
                robot_type=robot_type,
                robot_model=robot_model,
                init_position=init_position,
                init_orientation=init_orientation
            )
        
        print(f"[DynaHMRCSystem] 已创建 {len(self.robot_configs)} 个机器人")
    
    def _init_coordinator(self):
        """初始化 LLM 协调器"""
        # 创建 LLM 客户端
        llm_client = create_llm_client(**self.llm_config)
        
        # 创建 dyna_hmrc_web 的 BaseRobot 包装
        base_robots = []
        for robot_id, robot in self.robot_factory.get_all_robots().items():
            from Dyna_hmrc_web.dynahmrc_web.dynahmrc.coordinator import BaseRobot
            
            base_robot = BaseRobot(
                robot_id=robot_id,
                robot_type=robot.robot_type,
                capabilities=robot.capabilities
            )
            # 同步初始状态
            state = robot.get_state()
            base_robot.update_state(
                position=state.get("position"),
                orientation=state.get("orientation"),
                is_busy=state.get("is_busy", False)
            )
            base_robots.append(base_robot)
        
        # 创建协调器
        self.coordinator = DynaHMRC_Coordinator(
            robots=base_robots,
            llm_client=llm_client,
            enable_replanning=self.llm_config.get("enable_replanning", True),
            max_replan_attempts=self.llm_config.get("max_replan_attempts", 3)
        )
        
        print("[DynaHMRCSystem] LLM 协调器已初始化")
    
    def execute_task(self, natural_language_task: str) -> ExecutionResult:
        """
        执行自然语言任务（主入口）
        
        实现 Algorithm 1: Dynamic Replanning
        
        Args:
            natural_language_task: 自然语言描述的任务
                例如: "把 A 区的 3 个箱子搬到 B 区的手推车上"
        
        Returns:
            ExecutionResult 执行结果
        """
        if not self.is_initialized:
            if not self.initialize():
                return ExecutionResult(
                    success=False,
                    message="系统初始化失败",
                    execution_time=0.0
                )
        
        print(f"\n{'='*60}")
        print(f"[DynaHMRCSystem] 开始执行任务: {natural_language_task}")
        print(f"{'='*60}\n")
        
        self.start_time = time.time()
        self.replan_count = 0
        
        # 阶段 1: 使用 LLM 生成任务规划
        plan = self._generate_task_plan(natural_language_task)
        
        if "error" in plan:
            return ExecutionResult(
                success=False,
                message=f"任务规划失败: {plan['error']}",
                execution_time=time.time() - self.start_time
            )
        
        print(f"[DynaHMRCSystem] 任务规划完成: {len(plan.get('task_decomposition', []))} 个子任务")
        
        # 阶段 2: 执行任务（带监控和重规划）
        result = self._execute_with_monitoring(plan)
        
        total_time = time.time() - self.start_time
        result.execution_time = total_time
        result.replan_count = self.replan_count
        
        print(f"\n{'='*60}")
        print(f"[DynaHMRCSystem] 任务执行结束")
        print(f"  结果: {'成功' if result.success else '失败'}")
        print(f"  耗时: {total_time:.2f} 秒")
        print(f"  重规划次数: {self.replan_count}")
        print(f"{'='*60}\n")
        
        return result
    
    def _generate_task_plan(self, task: str) -> Dict[str, Any]:
        """
        生成任务规划
        
        使用 LLM 分析任务并生成协作规划方案
        """
        # 构建场景描述
        scene_info = self._build_scene_description()
        
        # 构建机器人能力描述
        robot_info = self._build_robot_description()
        
        # 调用协调器生成规划
        plan = self.coordinator.generate_task_plan(task, context={
            "scene_info": scene_info,
            "robot_info": robot_info
        })
        
        return plan
    
    def _build_scene_description(self) -> str:
        """构建场景描述"""
        # 从场景配置中提取物体信息
        objects = self.scene_config.get("objects", [])
        
        if not objects:
            return "场景暂无特定物体。"
        
        desc_parts = ["场景中的物体："]
        for obj in objects:
            name = obj.get("name", "unknown")
            pos = obj.get("position", [0, 0, 0])
            obj_type = obj.get("type", "object")
            desc_parts.append(f"- {name} ({obj_type}): 位置({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        return "\n".join(desc_parts)
    
    def _build_robot_description(self) -> str:
        """构建机器人能力描述"""
        desc_parts = ["可用机器人："]
        
        for robot_id, robot in self.robot_factory.get_all_robots().items():
            state = robot.get_state()
            caps = ", ".join(robot.capabilities)
            busy = "忙碌" if state.get("is_busy") else "空闲"
            pos = state.get("position", [0, 0, 0])
            desc_parts.append(
                f"- {robot_id} ({robot.robot_type}): "
                f"能力[{caps}], 状态[{busy}], 位置[{pos[0]:.2f}, {pos[1]:.2f}]"
            )
        
        return "\n".join(desc_parts)
    
    def _execute_with_monitoring(self, plan: Dict[str, Any]) -> ExecutionResult:
        """
        执行任务并监控（支持动态重规划）
        
        Algorithm 1: Dynamic Replanning
        """
        decomposition = plan.get("task_decomposition", [])
        assignment = plan.get("robot_assignment", {})
        execution_sequence = plan.get("execution_sequence", [])
        
        completed_tasks = []
        failed_tasks = []
        
        for task_id in execution_sequence:
            # 查找任务详情
            task_info = None
            for t in decomposition:
                if t.get("id") == task_id:
                    task_info = t
                    break
            
            if not task_info:
                continue
            
            robot_id = assignment.get(task_id)
            if not robot_id:
                failed_tasks.append(task_id)
                continue
            
            # 更新状态
            self.current_task_status[task_id] = TaskStatus.EXECUTING
            
            # 执行子任务
            success = self._execute_subtask(robot_id, task_info)
            
            if success:
                completed_tasks.append(task_id)
                self.current_task_status[task_id] = TaskStatus.COMPLETED
                print(f"[DynaHMRCSystem] 子任务完成: {task_id}")
            else:
                failed_tasks.append(task_id)
                self.current_task_status[task_id] = TaskStatus.FAILED
                print(f"[DynaHMRCSystem] 子任务失败: {task_id}")
                
                # 检查是否需要重规划
                if self.coordinator.enable_replanning and self.replan_count < self.coordinator.max_replan_attempts:
                    print(f"[DynaHMRCSystem] 触发重规划 (第 {self.replan_count + 1} 次)")
                    
                    # 重规划
                    replan_success = self._replan_after_failure(
                        task_id, robot_id, plan, completed_tasks
                    )
                    
                    if replan_success:
                        self.replan_count += 1
                        # 重试当前任务
                        success = self._execute_subtask(robot_id, task_info)
                        if success:
                            completed_tasks.append(task_id)
                            failed_tasks.remove(task_id)
                            self.current_task_status[task_id] = TaskStatus.COMPLETED
        
        success = len(failed_tasks) == 0
        
        return ExecutionResult(
            success=success,
            message="任务完成" if success else "部分任务失败",
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            execution_time=0.0,  # 由调用者填充
            replan_count=self.replan_count
        )
    
    def _execute_subtask(self, robot_id: str, task_info: Dict) -> bool:
        """
        执行子任务
        
        将高层任务转换为具体的 BestMan API 调用
        """
        description = task_info.get("description", "")
        
        print(f"[DynaHMRCSystem] 执行子任务 [{robot_id}]: {description}")
        
        # 解析任务类型和参数
        action, params = self._parse_task_description(description)
        
        # 通过适配器执行
        feedback = self.bestman_adapter.execute_action(robot_id, action, params)
        
        # 记录执行历史
        self.execution_history.append({
            "timestamp": time.time(),
            "robot_id": robot_id,
            "task": task_info,
            "action": action,
            "params": params,
            "feedback": feedback.to_dict()
        })
        
        # 打印反馈
        print(f"  -> {feedback.to_llm_string()}")
        
        return feedback.success
    
    def _parse_task_description(self, description: str) -> Tuple[str, Dict]:
        """
        解析任务描述，提取动作和参数
        
        这是一个简化的实现，实际可以使用 NLP 或 LLM 进行更精确的解析
        """
        description_lower = description.lower()
        
        # 导航相关
        if any(kw in description_lower for kw in ["导航", "移动", "前往", "移到", "navigate", "move to"]):
            # 尝试提取位置信息
            return "navigate", {"target": [0, 0, 0]}  # 简化实现
        
        # 抓取相关
        if any(kw in description_lower for kw in ["抓取", "拿起", "pick", "grasp"]):
            return "pick", {"object_id": 0, "object_name": "object"}
        
        # 放置相关
        if any(kw in description_lower for kw in ["放置", "放下", "place", "put"]):
            return "place", {"target": [0, 0, 0]}
        
        # 运输相关
        if any(kw in description_lower for kw in ["运输", "搬运", "transport", "carry"]):
            return "transport", {"object_id": 0, "source": [0, 0, 0], "target": [0, 0, 0]}
        
        # 默认
        return "wait", {"duration": 1.0}
    
    def _replan_after_failure(
        self,
        failed_task_id: str,
        failed_robot_id: str,
        current_plan: Dict,
        completed_tasks: List[str]
    ) -> bool:
        """
        失败后重规划
        
        根据失败原因调整规划策略
        """
        print(f"[DynaHMRCSystem] 重规划: 任务 {failed_task_id} 失败")
        
        # 获取失败原因
        failure_reason = self._analyze_failure(failed_task_id)
        
        # 尝试重新分配任务给其他机器人
        available_robots = [
            rid for rid in self.robot_factory.get_all_robots().keys()
            if rid != failed_robot_id
        ]
        
        if available_robots:
            # 尝试分配给另一个机器人
            new_robot_id = available_robots[0]
            print(f"[DynaHMRCSystem] 重新分配任务 {failed_task_id} 给 {new_robot_id}")
            
            # 更新分配
            current_plan["robot_assignment"][failed_task_id] = new_robot_id
            return True
        
        return False
    
    def _analyze_failure(self, task_id: str) -> str:
        """分析失败原因"""
        # 从执行历史中查找失败信息
        for record in reversed(self.execution_history):
            if record.get("task", {}).get("id") == task_id:
                feedback = record.get("feedback", {})
                if not feedback.get("success"):
                    return feedback.get("error", {}).get("code", "UNKNOWN")
        
        return "UNKNOWN"
    
    def get_system_state(self) -> Dict[str, Any]:
        """获取系统当前状态"""
        return {
            "initialized": self.is_initialized,
            "robots": {
                rid: robot.get_state()
                for rid, robot in self.robot_factory.get_all_robots().items()
            } if self.robot_factory else {},
            "task_status": {
                tid: status.value
                for tid, status in self.current_task_status.items()
            },
            "execution_history_count": len(self.execution_history),
            "replan_count": self.replan_count
        }
    
    def emergency_stop(self):
        """系统紧急停止"""
        print("[DynaHMRCSystem] 紧急停止!")
        
        if self.robot_factory:
            for robot_id, robot in self.robot_factory.get_all_robots().items():
                robot.emergency_stop()
    
    def shutdown(self):
        """关闭系统"""
        print("[DynaHMRCSystem] 关闭系统...")
        
        if self.client:
            self.client.disconnect()
        
        self.is_initialized = False
        print("[DynaHMRCSystem] 系统已关闭")
