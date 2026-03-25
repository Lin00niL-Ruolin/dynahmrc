"""
assembly_task.py - 装配任务场景

场景描述:
模拟一个简单的装配线，需要将多个零件组装成产品。
涉及不同类型的装配操作，展示异构机器人在精细操作中的协作。

任务示例: "将零件 A、B、C 装配到底座上"

涉及的异构协作:
- arm: 精细的零件定位和装配操作
- mobile_manipulator: 搬运零件和辅助定位
- mobile_base: 在装配线之间运输零件
"""

import os
from typing import Dict, List, Any, Optional
import numpy as np

from ..system import DynaHMRCSystem, ExecutionResult


class AssemblyTaskScenario:
    """
    装配任务场景类
    
    实现一个典型的装配线任务，展示异构多机器人在精细操作中的协作能力。
    """
    
    # 默认场景配置
    DEFAULT_SCENE_CONFIG = {
        "config_path": "Config/default.yaml",
        "gui": True,
        "objects": [
            # 装配底座
            {
                "name": "base_platform",
                "type": "platform",
                "model_path": "Asset/Scene/Object/Kitchen_world_models/Box/100162/mobility.urdf",
                "position": [0, 0, 0],
                "orientation": [0, 0, 0, 1]
            },
            # 零件 A
            {
                "name": "part_A",
                "type": "part",
                "model_path": "Asset/Scene/Object/Kitchen_world_models/Bottle/3558/mobility.urdf",
                "position": [1.0, 1.0, 0.5],
                "orientation": [0, 0, 0, 1]
            },
            # 零件 B
            {
                "name": "part_B",
                "type": "part",
                "model_path": "Asset/Scene/Object/Kitchen_world_models/Bottle/3574/mobility.urdf",
                "position": [1.2, 1.0, 0.5],
                "orientation": [0, 0, 0, 1]
            },
            # 零件 C
            {
                "name": "part_C",
                "type": "part",
                "model_path": "Asset/Scene/Object/Kitchen_world_models/Bottle/3614/mobility.urdf",
                "position": [1.4, 1.0, 0.5],
                "orientation": [0, 0, 0, 1]
            },
            # 零件存储区
            {
                "name": "storage_rack",
                "type": "rack",
                "model_path": "Asset/Scene/Object/Kitchen_world_models/Box/100189/mobility.urdf",
                "position": [1.2, 1.5, 0],
                "orientation": [0, 0, 0, 1]
            },
        ]
    }
    
    # 默认机器人配置
    DEFAULT_ROBOT_CONFIGS = [
        {
            "robot_id": "precision_arm_1",
            "robot_type": "arm",
            "robot_model": "panda",
            "init_position": [0.5, 0, 0],
            "init_orientation": [0, 0, 0, 1],
            "capabilities": ["manipulation", "perception"]
        },
        {
            "robot_id": "precision_arm_2",
            "robot_type": "arm",
            "robot_model": "panda",
            "init_position": [-0.5, 0, 0],
            "init_orientation": [0, 0, 1, 0],  # 朝向另一侧
            "capabilities": ["manipulation", "perception"]
        },
        {
            "robot_id": "helper_mobile_manipulator",
            "robot_type": "mobile_manipulator",
            "robot_model": "panda_on_segbot",
            "init_position": [1.2, 2.0, 0],
            "init_orientation": [0, 0, -1, 0],  # 朝向装配区
            "capabilities": ["navigation", "manipulation", "transport", "perception"]
        },
        {
            "robot_id": "logistics_mobile_base",
            "robot_type": "mobile_base",
            "robot_model": "segbot",
            "init_position": [2.0, 1.0, 0],
            "init_orientation": [0, 0, -0.707, 0.707],
            "capabilities": ["navigation", "transport"]
        }
    ]
    
    # 默认 LLM 配置
    DEFAULT_LLM_CONFIG = {
        "provider": "mock",
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
        初始化装配任务场景
        
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
        self.parts = ["part_A", "part_B", "part_C"]
        self.base_platform = "base_platform"
        self.assembly_sequence = ["part_A", "part_B", "part_C"]
        
        print("[AssemblyTask] 装配任务场景已创建")
    
    def _ensure_working_directory(self):
        """确保工作目录是项目根目录"""
        # 获取 dynahmrc 包的路径
        current_file = os.path.abspath(__file__)
        # dynahmrc/scenarios/assembly_task.py -> 项目根目录
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        
        # 如果当前目录不是根目录，切换到根目录
        if os.getcwd() != root_dir:
            os.chdir(root_dir)
            print(f"[AssemblyTask] 切换工作目录到: {root_dir}")
    
    def setup(self) -> bool:
        """
        设置场景
        
        Returns:
            是否设置成功
        """
        print("\n" + "="*60)
        print("[AssemblyTask] 设置装配任务场景")
        print("="*60 + "\n")
        
        # 确保工作目录正确
        self._ensure_working_directory()
        
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
                print("[AssemblyTask] 系统初始化失败")
                return False
            
            print("\n[AssemblyTask] 场景设置完成")
            print(f"  - 零件数量: {len(self.parts)}")
            print(f"  - 装配底座: {self.base_platform}")
            print(f"  - 装配顺序: {' -> '.join(self.assembly_sequence)}")
            print(f"  - 机器人数量: {len(self.robot_configs)}")
            
            return True
            
        except Exception as e:
            print(f"[AssemblyTask] 场景设置失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_task(
        self,
        task_description: Optional[str] = None
    ) -> ExecutionResult:
        """
        运行装配任务
        
        Args:
            task_description: 任务描述（可选，使用默认描述）
        
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
            task_description = self._build_default_task_description()
        
        print("\n" + "="*60)
        print("[AssemblyTask] 开始执行任务")
        print(f"  任务: {task_description}")
        print("="*60 + "\n")
        
        # 执行任务
        result = self.system.execute_task(task_description)
        
        # 打印结果
        self._print_result(result)
        
        return result
    
    def _build_default_task_description(self) -> str:
        """构建默认任务描述"""
        parts_str = "、".join(self.parts)
        return f"将零件 {parts_str} 按顺序装配到 {self.base_platform} 上"
    
    def _print_result(self, result: ExecutionResult):
        """打印执行结果"""
        print("\n" + "="*60)
        print("[AssemblyTask] 任务执行结果")
        print("="*60)
        print(f"  成功: {result.success}")
        print(f"  消息: {result.message}")
        print(f"  完成任务数: {len(result.completed_tasks)}")
        print(f"  失败任务数: {len(result.failed_tasks)}")
        print(f"  执行时间: {result.execution_time:.2f} 秒")
        print(f"  重规划次数: {result.replan_count}")
        
        if result.completed_tasks:
            print(f"\n  已完成的装配步骤:")
            for i, task in enumerate(result.completed_tasks, 1):
                print(f"    {i}. {task}")
        
        if result.failed_tasks:
            print(f"\n  失败的步骤:")
            for task in result.failed_tasks:
                print(f"    - {task}")
        
        print("="*60 + "\n")
    
    def get_scene_info(self) -> Dict[str, Any]:
        """获取场景信息"""
        return {
            "scene_type": "assembly",
            "parts": self.parts,
            "base_platform": self.base_platform,
            "assembly_sequence": self.assembly_sequence,
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
        print("[AssemblyTask] 场景已关闭")


def run_assembly_demo():
    """
    运行装配任务演示
    
    这是一个可以直接运行的示例函数
    """
    print("\n" + "="*70)
    print(" DynaHMRC 装配任务场景演示")
    print("="*70)
    print("\n场景描述:")
    print("  - 装配底座位于中心")
    print("  - 3 个零件需要按顺序装配")
    print("  - 使用异构机器人协作完成精细装配")
    print("\n机器人配置:")
    print("  1. precision_arm_1: 精密机械臂（主要装配）")
    print("  2. precision_arm_2: 精密机械臂（辅助装配）")
    print("  3. helper_mobile_manipulator: 移动操作机器人（搬运零件）")
    print("  4. logistics_mobile_base: 物流机器人（运输）")
    print("\n" + "="*70 + "\n")
    
    # 创建场景
    scenario = AssemblyTaskScenario(
        enable_visualization=True
    )
    
    try:
        # 运行任务
        result = scenario.run_task()
        
        # 等待用户查看结果
        input("\n按 Enter 键关闭...")
        
    except KeyboardInterrupt:
        print("\n[AssemblyTask] 用户中断")
    finally:
        scenario.shutdown()
    
    return result if 'result' in dir() else None


if __name__ == "__main__":
    # 设置工作目录
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    run_assembly_demo()
