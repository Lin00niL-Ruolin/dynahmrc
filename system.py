"""
system.py - DynaHMRC 主系统集成类
封装整个异构多机器人协作流程

实现论文中的核心算法:
- Algorithm 1: Dynamic Replanning（动态重规划）
- Algorithm 2: Task Allocation（任务分配）
- Algorithm 3: Conflict Resolution（冲突解决）
- Four-Stage Collaboration: Self-Description -> Task Allocation -> Leader Election -> Execution
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

# 导入评估模块
from .evaluation.metrics import MetricsCollector, TaskMetrics

# 导入四阶段协作模块
from .core.collaboration import FourStageCollaboration, CollaborationResult
from .core.robot_agent import RobotAgent

# 导入路径规划模块
from .utils.path_planning import PathPlanner


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
            scene_config: 场景配置，支持两种方式：
                1. 使用scene_path加载JSON场景文件（优先）：
                {
                    "config_path": "Config/default.yaml",
                    "gui": True,
                    "scene_path": "Asset/Scene/your_scene.json"
                }
                2. 使用objects列表逐个定义物体：
                {
                    "config_path": "Config/default.yaml",
                    "gui": True,
                    "objects": [
                        {
                            "name": "object_name",
                            "model_path": "path/to/model.urdf",
                            "position": [0, 0, 0],
                            "orientation": [0, 0, 0, 1],
                            "scale": 1
                        }
                    ]
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
        
        # 场景物体缓存（在BestManAdapter初始化前加载）
        self._scene_objects_cache: List[Dict] = []
        
        # 统计信息
        self.start_time: Optional[float] = None
        self.replan_count = 0
        
        # 评估指标收集器
        self.metrics_collector: Optional[MetricsCollector] = None
        self.current_task_metrics: Optional[TaskMetrics] = None
        self.action_count = 0
        self.communication_count = 0
        
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
            self.adapter = self.bestman_adapter  # 别名，方便协作框架使用
            
            # 4.1 注册场景物体到 BestManAdapter
            self._register_scene_objects_to_adapter()
            
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
    
    def _get_project_root(self) -> str:
        """获取项目根目录"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(current_dir)
    
    def _load_scene(self):
        """加载场景物体"""
        import json
        
        # 优先检查是否有scene_path，如果有则加载JSON场景文件
        scene_path = self.scene_config.get("scene_path")
        
        if scene_path:
            root_dir = self._get_project_root()
            if not os.path.isabs(scene_path):
                abs_scene_path = os.path.join(root_dir, scene_path)
            else:
                abs_scene_path = scene_path
            
            if os.path.exists(abs_scene_path):
                # 手动解析JSON并加载物体，避免Client.create_scene的路径问题
                print(f"[DynaHMRCSystem] 正在加载场景文件: {abs_scene_path}")
                try:
                    with open(abs_scene_path, "r") as f:
                        scene_data = json.load(f)
                    
                    loaded_count = 0
                    for obj in scene_data:
                        # 支持多种字段名: object_name 或 obj_name
                        obj_name = obj.get("object_name") or obj.get("obj_name", "object")
                        model_path = obj.get("model_path")
                        position = obj.get("object_position", [0, 0, 0])
                        orientation = obj.get("object_orientation", [0, 0, 0, 1])
                        scale = obj.get("scale", 1)
                        fixed_base = obj.get("fixed_base", False)
                        
                        # 处理 orientation 中的字符串表达式（如 'math.pi / 2'）
                        if isinstance(orientation, list):
                            import math
                            parsed_orientation = []
                            for val in orientation:
                                if isinstance(val, str):
                                    try:
                                        # 安全地评估字符串表达式
                                        parsed_val = eval(val, {"__builtins__": {}}, {"math": math})
                                        parsed_orientation.append(parsed_val)
                                    except:
                                        parsed_orientation.append(0.0)
                                else:
                                    parsed_orientation.append(float(val))
                            orientation = parsed_orientation
                        
                        if model_path:
                            # 将相对路径转换为绝对路径
                            if not os.path.isabs(model_path):
                                abs_model_path = os.path.join(root_dir, model_path)
                            else:
                                abs_model_path = model_path
                            
                            # 检查文件是否存在
                            if not os.path.exists(abs_model_path):
                                print(f"[DynaHMRCSystem] 警告: 模型文件不存在: {abs_model_path}")
                                continue
                            
                            obj_id = self.client.load_object(
                                obj_name=obj_name,
                                model_path=abs_model_path,
                                object_position=position,
                                object_orientation=orientation,
                                scale=scale,
                                fixed_base=fixed_base
                            )
                            loaded_count += 1
                            
                            # 缓存物体信息，稍后注册到 BestManAdapter
                            self._scene_objects_cache.append({
                                'name': obj_name,
                                'id': obj_id,
                                'type': obj.get('object_type', 'unknown'),
                                'position': position,
                                'orientation': orientation
                            })
                    
                    print(f"[DynaHMRCSystem] 已从场景文件加载 {loaded_count} 个物体")
                    return
                except Exception as e:
                    print(f"[DynaHMRCSystem] 加载场景文件失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[DynaHMRCSystem] 警告: 场景文件不存在: {abs_scene_path}")
        
        # 如果没有scene_path或加载失败，则使用objects列表逐个加载
        objects = self.scene_config.get("objects", [])
        root_dir = self._get_project_root()
        
        for obj in objects:
            obj_name = obj.get("name", "object")
            model_path = obj.get("model_path")
            position = obj.get("position", [0, 0, 0])
            orientation = obj.get("orientation", [0, 0, 0, 1])
            scale = obj.get("scale", 1)
            
            if model_path:
                # 将相对路径转换为绝对路径
                if not os.path.isabs(model_path):
                    abs_model_path = os.path.join(root_dir, model_path)
                else:
                    abs_model_path = model_path
                
                # 检查文件是否存在
                if not os.path.exists(abs_model_path):
                    print(f"[DynaHMRCSystem] 警告: 模型文件不存在: {abs_model_path}")
                    continue
                
                self.client.load_object(
                    obj_name=obj_name,
                    model_path=abs_model_path,
                    object_position=position,
                    object_orientation=orientation,
                    scale=scale
                )
        
        print(f"[DynaHMRCSystem] 已加载 {len(objects)} 个场景物体")
    
    def _register_scene_objects_to_adapter(self):
        """将缓存的场景物体注册到 BestManAdapter"""
        if not self.bestman_adapter:
            print("[DynaHMRCSystem] 警告: BestManAdapter 未初始化，无法注册场景物体")
            return
        
        if not self._scene_objects_cache:
            print("[DynaHMRCSystem] 没有缓存的场景物体需要注册")
            return
        
        print(f"[DynaHMRCSystem] 正在注册 {len(self._scene_objects_cache)} 个场景物体到 BestManAdapter...")
        print(f"[DynaHMRCSystem] 缓存内容: {[obj['name'] for obj in self._scene_objects_cache]}")
        registered_count = 0
        for obj_info in self._scene_objects_cache:
            try:
                print(f"[DynaHMRCSystem] 注册物体: {obj_info['name']} (ID: {obj_info['id']}, Type: {obj_info.get('type', 'unknown')})")
                self.bestman_adapter.register_scene_object(
                    obj_name=obj_info['name'],
                    obj_id=obj_info['id'],
                    obj_type=obj_info.get('type', 'unknown')
                )
                registered_count += 1
            except Exception as e:
                print(f"[DynaHMRCSystem] 注册物体 {obj_info['name']} 失败: {e}")
        
        print(f"[DynaHMRCSystem] 成功注册 {registered_count} 个场景物体到 BestManAdapter")
        # 清空缓存
        self._scene_objects_cache.clear()
    
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
        # 转换配置参数以适配 create_llm_client
        llm_config = self.llm_config.copy()
        provider = llm_config.pop("provider", "mock")
        
        # 将 provider 映射为 client_type
        client_type = "kimi" if provider == "kimi" else "mock"
        
        # 移除不需要的参数（这些参数由协调器管理，不传递给 LLM 客户端）
        llm_config.pop("enable_replanning", None)
        llm_config.pop("max_replan_attempts", None)
        llm_config.pop("temperature", None)
        llm_config.pop("model", None)
        
        llm_client = create_llm_client(client_type=client_type, **llm_config)
        
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
    
    def execute_task(self, natural_language_task: str, max_steps: int = 100,
                     task_type: str = "generic", variation: str = "static") -> Dict[str, Any]:
        """
        执行自然语言任务（主入口）
        
        实现 Algorithm 1: Dynamic Replanning
        
        Args:
            natural_language_task: 自然语言描述的任务
                例如: "把 A 区的 3 个箱子搬到 B 区的手推车上"
            max_steps: 最大执行步数
            task_type: 任务类型 (pack_objects, make_sandwich, sort_solids)
            variation: 任务变化类型 (static, cto, irz, anc, rec)
        
        Returns:
            Dict 包含执行结果和评估指标
        """
        if not self.is_initialized:
            if not self.initialize():
                return {
                    'success': False,
                    'message': "系统初始化失败",
                    'steps': 0,
                    'communications': 0,
                    'partial_success': 0.0
                }
        
        print(f"\n{'='*60}")
        print(f"[DynaHMRCSystem] 开始执行任务: {natural_language_task}")
        print(f"{'='*60}\n")
        
        # 初始化评估指标
        self.start_time = time.time()
        self.action_count = 0
        self.communication_count = 0
        
        # 获取机器人团队信息
        robot_team = list(self.robot_factory.get_all_robots().keys())
        robot_types = {
            rid: robot.robot_type 
            for rid, robot in self.robot_factory.get_all_robots().items()
        }
        
        # 开始指标收集
        if self.metrics_collector:
            self.metrics_collector.start_task(
                task_id=f"task_{int(time.time())}",
                task_type=task_type,
                variation=variation,
                robot_team=robot_team,
                robot_types=robot_types
            )
        self.replan_count = 0
        
        # 阶段 1: 使用 LLM 生成任务规划
        plan = self._generate_task_plan(natural_language_task)
        
        if "error" in plan:
            total_time = time.time() - self.start_time
            result = {
                'success': False,
                'message': f"任务规划失败: {plan['error']}",
                'steps': 0,
                'communications': 0,
                'partial_success': 0.0,
                'execution_time': total_time,
                'replan_count': 0
            }
            # 记录失败结果
            if self.metrics_collector:
                self.metrics_collector.end_task(success=False, partial_success=0.0)
            return result
        
        print(f"[DynaHMRCSystem] 任务规划完成: {len(plan.get('task_decomposition', []))} 个子任务")
        
        # 阶段 2: 执行任务（带监控和重规划）
        exec_result = self._execute_with_monitoring(plan, max_steps=max_steps)
        
        total_time = time.time() - self.start_time
        
        # 构建结果字典
        result = {
            'success': exec_result.success,
            'message': exec_result.message,
            'steps': self.action_count,
            'communications': self.communication_count,
            'partial_success': 1.0 if exec_result.success else 0.5,  # 简化计算
            'execution_time': total_time,
            'replan_count': self.replan_count,
            'completed_tasks': exec_result.completed_tasks,
            'failed_tasks': exec_result.failed_tasks
        }
        
        # 结束指标收集
        if self.metrics_collector:
            self.metrics_collector.end_task(
                success=exec_result.success,
                partial_success=result['partial_success']
            )
        
        print(f"\n{'='*60}")
        print(f"[DynaHMRCSystem] 任务执行结束")
        print(f"  结果: {'成功' if result['success'] else '失败'}")
        print(f"  耗时: {total_time:.2f} 秒")
        print(f"  动作步数: {self.action_count}")
        print(f"  通信次数: {self.communication_count}")
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
    
    def _execute_with_monitoring(self, plan: Dict[str, Any], max_steps: int = 100) -> ExecutionResult:
        """
        执行任务并监控（支持动态重规划）
        
        Algorithm 1: Dynamic Replanning
        """
        decomposition = plan.get("task_decomposition", [])
        assignment = plan.get("robot_assignment", {})
        execution_sequence = plan.get("execution_sequence", [])
        
        completed_tasks = []
        failed_tasks = []
        step_count = 0
        
        for task_id in execution_sequence:
            # 检查步数限制
            if step_count >= max_steps:
                print(f"[DynaHMRCSystem] 达到最大步数限制 ({max_steps})")
                break
            
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
            step_count += 1
            self.action_count += 1
            
            # 记录动作
            if self.metrics_collector and self.metrics_collector.current_task:
                self.metrics_collector.record_action(
                    robot_id=robot_id,
                    action_type=task_info.get("type", "unknown"),
                    action_details=task_info
                )
            
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
                    
                    # 记录重规划
                    if self.metrics_collector:
                        self.metrics_collector.record_replan()
                    
                    # 重规划
                    replan_success = self._replan_after_failure(
                        task_id, robot_id, plan, completed_tasks
                    )
                    
                    if replan_success:
                        self.replan_count += 1
                        # 重试当前任务
                        success = self._execute_subtask(robot_id, task_info)
                        self.action_count += 1
                        
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
    
    def execute_task_with_four_stages(self, natural_language_task: str, 
                                       max_steps: int = 100,
                                       task_type: str = "generic", 
                                       variation: str = "static") -> Dict[str, Any]:
        """
        使用四阶段协作流程执行任务
        
        Four-Stage Collaboration:
        1. Self-Description: 每个机器人自我介绍
        2. Task Allocation: 任务分配和领导竞选
        3. Leader Election: 投票选举领导者
        4. Closed-Loop Execution: 闭环执行
        
        Args:
            natural_language_task: 自然语言描述的任务
            max_steps: 最大执行步数
            task_type: 任务类型
            variation: 任务变化类型
            
        Returns:
            Dict 包含执行结果和评估指标
        """
        if not self.is_initialized:
            if not self.initialize():
                return {
                    'success': False,
                    'message': "系统初始化失败",
                    'steps': 0,
                    'communications': 0,
                    'partial_success': 0.0
                }
        
        print(f"\n{'='*60}")
        print(f"[DynaHMRCSystem] 开始四阶段协作任务: {natural_language_task}")
        print(f"{'='*60}\n")
        
        # 初始化评估指标
        self.start_time = time.time()
        self.action_count = 0
        self.communication_count = 0
        
        # 获取机器人团队信息
        robot_team = list(self.robot_factory.get_all_robots().keys())
        robot_types = {
            rid: robot.robot_type 
            for rid, robot in self.robot_factory.get_all_robots().items()
        }
        
        # 开始指标收集
        if self.metrics_collector:
            self.metrics_collector.start_task(
                task_id=f"task_{int(time.time())}",
                task_type=task_type,
                variation=variation,
                robot_team=robot_team,
                robot_types=robot_types
            )
        
        try:
            # 创建 RobotAgent 实例
            robot_agents = self._create_robot_agents()
            
            # 创建四阶段协作框架
            collaboration = FourStageCollaboration(
                robots=robot_agents,
                max_execution_steps=max_steps,
                enable_communication=True,
                enable_visualization=self.enable_visualization
            )
            
            # 设置 BestManAdapter 用于执行真实动作和获取场景信息
            if hasattr(self, 'adapter') and self.adapter:
                collaboration.set_adapter(self.adapter)
                print(f"[DynaHMRCSystem] BestManAdapter 已设置到协作框架")
            
            # 更新机器人位置用于可视化
            self._update_robot_positions_for_visualization(collaboration)
            
            # 运行四阶段协作
            collab_result = collaboration.run_collaboration(natural_language_task)
            
            total_time = time.time() - self.start_time
            
            # 更新统计信息
            self.action_count = collab_result.execution_steps
            
            # 构建结果字典
            result = {
                'success': collab_result.success,
                'message': collab_result.message,
                'steps': collab_result.execution_steps,
                'communications': self.communication_count,
                'partial_success': 1.0 if collab_result.success else 0.5,
                'execution_time': total_time,
                'replan_count': 0,  # 四阶段框架中重规划是内部的
                'leader': collab_result.leader_name,
                'task_plan': collab_result.task_plan,
                'robot_assignments': collab_result.robot_assignments
            }
            
            # 结束指标收集
            if self.metrics_collector:
                self.metrics_collector.end_task(
                    success=collab_result.success,
                    partial_success=result['partial_success']
                )
            
            print(f"\n{'='*60}")
            print(f"[DynaHMRCSystem] 四阶段协作完成")
            print(f"  结果: {'成功' if result['success'] else '失败'}")
            print(f"  领导者: {result['leader']}")
            print(f"  耗时: {total_time:.2f} 秒")
            print(f"  执行步数: {result['steps']}")
            print(f"{'='*60}\n")
            
            return result
            
        except Exception as e:
            total_time = time.time() - self.start_time
            print(f"[DynaHMRCSystem] 四阶段协作失败: {e}")
            import traceback
            traceback.print_exc()
            
            result = {
                'success': False,
                'message': f"四阶段协作失败: {str(e)}",
                'steps': 0,
                'communications': 0,
                'partial_success': 0.0,
                'execution_time': total_time
            }
            
            if self.metrics_collector:
                self.metrics_collector.end_task(success=False, partial_success=0.0)
            
            return result
    
    def _create_robot_agents(self) -> List[RobotAgent]:
        """
        从 RobotFactory 创建 RobotAgent 实例
        
        Returns:
            List of RobotAgent instances
        """
        robot_agents = []
        
        # 创建共享的路径规划器实例
        client_id = self.client.client_id if hasattr(self, 'client') and self.client else 0
        shared_path_planner = PathPlanner(client_id=client_id)
        print(f"[DynaHMRCSystem] 创建共享路径规划器")
        
        for robot_id, robot in self.robot_factory.get_all_robots().items():
            # 获取 LLM 客户端
            llm_client = self.llm_client if hasattr(self, 'llm_client') else None
            
            if not llm_client:
                llm_config = self.llm_config.copy()
                provider = llm_config.pop("provider", "mock")
        
                # 将 provider 映射为 client_type
                client_type = "kimi" if provider == "kimi" else "mock"
                
                # 创建默认 LLM 客户端
                llm_client = create_llm_client(client_type=client_type, **llm_config)
            
            # 映射机器人类型
            robot_type_map = {
                'mobile_manipulator': 'MobileManipulation',
                'manipulator': 'Manipulator',
                'mobile_base': 'Mobile',
                'drone': 'Drone'
            }
            
            agent_type = robot_type_map.get(robot.robot_type, 'MobileManipulation')
            
            # 创建 RobotAgent
            agent = RobotAgent(
                name=robot_id,
                robot_type=agent_type,
                capabilities=robot.capabilities,
                llm_client=llm_client,
                avatar="🤖"
            )
            
            # 设置共享的路径规划器
            agent.set_path_planner(shared_path_planner)
            
            # 如果机器人有 set_path_planner 方法（如 MobileManipulator），也设置给它
            if hasattr(robot, 'set_path_planner'):
                robot.set_path_planner(shared_path_planner)
                print(f"[DynaHMRCSystem] 为 {robot_id} 设置共享路径规划器")
            
            robot_agents.append(agent)
            print(f"[DynaHMRCSystem] 创建 RobotAgent: {robot_id} ({agent_type})")
        
        return robot_agents
    
    def _update_robot_positions_for_visualization(self, collaboration):
        """
        更新机器人位置用于气泡框可视化
        
        Args:
            collaboration: FourStageCollaboration instance
        """
        if not self.enable_visualization:
            return
        
        try:
            for robot_id, robot in self.robot_factory.get_all_robots().items():
                # 获取机器人当前位置
                position = None
                
                # 尝试不同的方法获取位置
                if hasattr(robot, 'get_base_position'):
                    position = robot.get_base_position()
                elif hasattr(robot, 'get_position'):
                    position = robot.get_position()
                elif hasattr(robot, 'base_position'):
                    position = robot.base_position
                elif hasattr(robot, 'pose'):
                    pose = robot.pose
                    if hasattr(pose, 'position'):
                        position = pose.position
                    elif isinstance(pose, (list, tuple)):
                        position = pose[:3]
                
                # 如果找到位置，更新到可视化器
                if position and len(position) >= 3:
                    collaboration.set_robot_position(
                        robot_id, 
                        (float(position[0]), float(position[1]), float(position[2]))
                    )
                    print(f"[DynaHMRCSystem] Set position for {robot_id}: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
                else:
                    # 使用默认位置
                    collaboration.set_robot_position(robot_id, (0.0, 0.0, 0.5))
                    print(f"[DynaHMRCSystem] Using default position for {robot_id}")
                    
        except Exception as e:
            print(f"[DynaHMRCSystem] 更新机器人位置失败: {e}")
            import traceback
            traceback.print_exc()
    
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
    
    def set_metrics_collector(self, collector: MetricsCollector):
        """设置评估指标收集器"""
        self.metrics_collector = collector
        print("[DynaHMRCSystem] 已设置评估指标收集器")
    
    def record_communication(self, from_robot: str, to_robot: str, 
                            message_type: str, content: str):
        """记录机器人间的通信"""
        self.communication_count += 1
        if self.metrics_collector and self.metrics_collector.current_task:
            self.metrics_collector.record_communication(
                from_robot=from_robot,
                to_robot=to_robot,
                message_type=message_type,
                content=content
            )
