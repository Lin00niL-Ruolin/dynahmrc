"""
warehouse_task.py - 仓储协作场景

场景描述:
仓库中有货架（shelves）、箱子（boxes）、手推车（trolleys）。
任务示例: "把 A 区的 3 个箱子搬到 B 区的手推车上"

涉及的异构协作:
- mobile_base: 导航到 A 区、B 区（运输路径规划）
- arm: 从货架 pick 箱子（精细操作）
- mobile_manipulator: transport 箱子并 place 到手推车上（复合任务）
- LLM 协调器: 动态分配任务、处理冲突（如路径冲突时的重规划）
"""

import os
from typing import Dict, List, Any, Optional
import numpy as np

from ..system import DynaHMRCSystem, ExecutionResult


class WarehouseTaskScenario:
    """
    仓储协作场景类
    
    实现一个典型的仓库搬运任务，展示异构多机器人的协作能力。
    """
    
    # 默认场景配置
    DEFAULT_SCENE_CONFIG = {
        "config_path": "Config/default.yaml",
        "gui": True,
        "objects": [
            # A 区货架
            {
                "name": "shelf_A",
                "type": "shelf",
                "model_path": "Asset/Scene/Object/Kitchen_world_models/Box/100129/mobility.urdf",
                "position": [1.5, 0, 0],
                "orientation": [0, 0, 0, 1]
            },
            # B 区手推车
            {
                "name": "trolley_B",
                "type": "trolley",
                "model_path": "Asset/Scene/Object/Kitchen_world_models/Box/100154/mobility.urdf",
                "position": [-1.5, 0, 0],
                "orientation": [0, 0, 0, 1]
            },
            # 箱子 1
            {
                "name": "box_1",
                "type": "box",
                "model_path": "Asset/Scene/Object/Kitchen_world_models/004_sugar_box/decomp.obj",
                "position": [1.5, 0, 1.0],
                "orientation": [0, 0, 0, 1]
            },
            # 箱子 2
            {
                "name": "box_2",
                "type": "box",
                "model_path": "Asset/Scene/Object/Kitchen_world_models/004_sugar_box/decomp.obj",
                "position": [1.5, 0.3, 1.0],
                "orientation": [0, 0, 0, 1]
            },
            # 箱子 3
            {
                "name": "box_3",
                "type": "box",
                "model_path": "Asset/Scene/Object/Kitchen_world_models/004_sugar_box/decomp.obj",
                "position": [1.5, -0.3, 1.0],
                "orientation": [0, 0, 0, 1]
            },
        ]
    }
    
    # 默认机器人配置
    DEFAULT_ROBOT_CONFIGS = [
        {
            "robot_id": "mobile_base_1",
            "robot_type": "mobile_base",
            "robot_model": "segbot",
            "init_position": [0, 1.5, 0],
            "init_orientation": [0, 0, 0, 1],
            "capabilities": ["navigation", "transport"]
        },
        {
            "robot_id": "arm_1",
            "robot_type": "arm",
            "robot_model": "panda",
            "init_position": [2.0, 0, 0],
            "init_orientation": [0, 0, 0, 1],
            "capabilities": ["manipulation", "perception"]
        },
        {
            "robot_id": "mobile_manipulator_1",
            "robot_type": "mobile_manipulator",
            "robot_model": "panda_on_segbot",
            "init_position": [0, -1.5, 0],
            "init_orientation": [0, 0, 0, 1],
            "capabilities": ["navigation", "manipulation", "transport", "perception"]
        }
    ]
    
    # 默认 LLM 配置
    DEFAULT_LLM_CONFIG = {
        "provider": "mock",  # 默认使用 mock，避免需要真实 API key
        "model": "mock",
        "temperature": 0.3,
        "enable_replanning": True,
        "max_replan_attempts": 3
    }
    
    def __init__(
        self,
        scene_config: Optional[Dict] = None,
        robot_configs: Optional[List[Dict]] = None,
        llm_config: Optional[Dict] = None,
        enable_visualization: bool = True
    ):
        """
        初始化仓储协作场景
        
        Args:
            scene_config: 场景配置（可选，使用默认配置）
            robot_configs: 机器人配置列表（可选，使用默认配置）
            llm_config: LLM 配置（可选，使用默认配置）
            enable_visualization: 是否启用可视化
        """
        self.scene_config = scene_config or self.DEFAULT_SCENE_CONFIG.copy()
        self.robot_configs = robot_configs or self.DEFAULT_ROBOT_CONFIGS.copy()
        self.llm_config = llm_config or self.DEFAULT_LLM_CONFIG.copy()
        self.enable_visualization = enable_visualization
        
        # DynaHMRC 系统实例
        self.system: Optional[DynaHMRCSystem] = None
        
        # 场景状态
        self.boxes = ["box_1", "box_2", "box_3"]
        self.source_zone = "A 区"
        self.target_zone = "B 区"
        self.target_trolley = "trolley_B"
        
        print("[WarehouseTask] 仓储协作场景已创建")
    
    def setup(self) -> bool:
        """
        设置场景
        
        Returns:
            是否设置成功
        """
        print("\n" + "="*60)
        print("[WarehouseTask] 设置仓储协作场景")
        print("="*60 + "\n")
        
        try:
            # 创建 DynaHMRC 系统
            self.system = DynaHMRCSystem(
                scene_config=self.scene_config,
                robot_configs=self.robot_configs,
                llm_config=self.llm_config,
                enable_visualization=self.enable_visualization
            )
            
            # 初始化系统
            if not self.system.initialize():
                print("[WarehouseTask] 系统初始化失败")
                return False
            
            print("\n[WarehouseTask] 场景设置完成")
            print(f"  - 箱子数量: {len(self.boxes)}")
            print(f"  - 源区域: {self.source_zone}")
            print(f"  - 目标区域: {self.target_zone}")
            print(f"  - 机器人数量: {len(self.robot_configs)}")
            
            return True
            
        except Exception as e:
            print(f"[WarehouseTask] 场景设置失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_task(
        self,
        task_description: Optional[str] = None,
        num_boxes: int = 3
    ) -> ExecutionResult:
        """
        运行仓储任务
        
        Args:
            task_description: 任务描述（可选，使用默认描述）
            num_boxes: 要搬运的箱子数量
        
        Returns:
            ExecutionResult 执行结果
        """
        if self.system is None:
            if not self.setup():
                return ExecutionResult(
                    success=False,
                    message="场景设置失败",
                    execution_time=0.0
                )
        
        # 构建任务描述
        if task_description is None:
            task_description = self._build_default_task_description(num_boxes)
        
        print("\n" + "="*60)
        print("[WarehouseTask] 开始执行任务")
        print(f"  任务: {task_description}")
        print("="*60 + "\n")
        
        # 执行任务
        result = self.system.execute_task(task_description)
        
        # 打印结果
        self._print_result(result)
        
        return result
    
    def _build_default_task_description(self, num_boxes: int) -> str:
        """构建默认任务描述"""
        return (
            f"把 {self.source_zone} 的 {num_boxes} 个箱子 "
            f"({', '.join(self.boxes[:num_boxes])}) "
            f"搬到 {self.target_zone} 的 {self.target_trolley} 上"
        )
    
    def _print_result(self, result: ExecutionResult):
        """打印执行结果"""
        print("\n" + "="*60)
        print("[WarehouseTask] 任务执行结果")
        print("="*60)
        print(f"  成功: {result.success}")
        print(f"  消息: {result.message}")
        print(f"  完成任务数: {len(result.completed_tasks)}")
        print(f"  失败任务数: {len(result.failed_tasks)}")
        print(f"  执行时间: {result.execution_time:.2f} 秒")
        print(f"  重规划次数: {result.replan_count}")
        
        if result.completed_tasks:
            print(f"\n  已完成的任务:")
            for task in result.completed_tasks:
                print(f"    - {task}")
        
        if result.failed_tasks:
            print(f"\n  失败的任务:")
            for task in result.failed_tasks:
                print(f"    - {task}")
        
        print("="*60 + "\n")
    
    def get_scene_info(self) -> Dict[str, Any]:
        """获取场景信息"""
        return {
            "scene_type": "warehouse",
            "boxes": self.boxes,
            "source_zone": self.source_zone,
            "target_zone": self.target_zone,
            "target_trolley": self.target_trolley,
            "robots": [
                {
                    "id": cfg["robot_id"],
                    "type": cfg["robot_type"],
                    "capabilities": cfg["capabilities"]
                }
                for cfg in self.robot_configs
            ]
        }
    
    def get_system_state(self) -> Dict[str, Any]:
        """获取系统状态"""
        if self.system:
            return self.system.get_system_state()
        return {"initialized": False}
    
    def emergency_stop(self):
        """紧急停止"""
        if self.system:
            self.system.emergency_stop()
    
    def shutdown(self):
        """关闭场景"""
        if self.system:
            self.system.shutdown()
            self.system = None
        print("[WarehouseTask] 场景已关闭")


def run_warehouse_demo():
    """
    运行仓储任务演示
    
    这是一个可以直接运行的示例函数
    """
    print("\n" + "="*70)
    print(" DynaHMRC 仓储协作场景演示")
    print("="*70)
    print("\n场景描述:")
    print("  - 仓库中有货架（A区）和手推车（B区）")
    print("  - 3 个箱子需要搬运")
    print("  - 使用异构机器人协作完成任务")
    print("\n机器人配置:")
    print("  1. mobile_base_1: 移动基座（负责导航和运输）")
    print("  2. arm_1: 固定机械臂（负责抓取）")
    print("  3. mobile_manipulator_1: 移动操作复合机器人（全能型）")
    print("\n" + "="*70 + "\n")
    
    # 创建场景
    scenario = WarehouseTaskScenario(
        enable_visualization=True  # 启用 GUI 可视化
    )
    
    try:
        # 运行任务
        result = scenario.run_task(num_boxes=3)
        
        # 等待用户查看结果
        input("\n按 Enter 键关闭...")
        
    except KeyboardInterrupt:
        print("\n[WarehouseTask] 用户中断")
    finally:
        scenario.shutdown()
    
    return result if 'result' in dir() else None


if __name__ == "__main__":
    # 设置工作目录
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    run_warehouse_demo()
