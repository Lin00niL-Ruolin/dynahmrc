"""
run_demo.py - DynaHMRC 一键运行入口

提供简单易用的接口来运行 DynaHMRC 系统的各种演示场景。

使用方法:
    # 运行仓储场景
    python -m dynahmrc.run_demo --scenario warehouse
    
    # 运行装配场景
    python -m dynahmrc.run_demo --scenario assembly
    
    # 运行自定义任务
    python -m dynahmrc.run_demo --task "把红色箱子搬到桌子上"
    
    # 禁用可视化（用于测试）
    python -m dynahmrc.run_demo --scenario warehouse --no-gui

示例代码:
    >>> from dynahmrc.run_demo import run_warehouse_demo, run_assembly_demo
    >>> result = run_warehouse_demo()
    >>> print(f"任务完成: {result.success}")
"""

import os
import sys
import argparse
import json
from typing import Optional, Dict, Any


def setup_pythonpath():
    """设置 Python 路径，确保可以导入所有模块"""
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    # 添加到路径
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    
    # 添加 BestMan 目录
    bestman_dir = os.path.join(root_dir, 'BestMan')
    if os.path.exists(bestman_dir) and bestman_dir not in sys.path:
        sys.path.insert(0, bestman_dir)
    
    return root_dir


def run_warehouse_demo(
    enable_gui: bool = True,
    llm_provider: str = "mock",
    task_description: Optional[str] = None
) -> Any:
    """
    运行仓储协作场景演示
    
    场景描述:
        仓库中有货架、箱子和手推车。
        任务：把 A 区的 3 个箱子搬到 B 区的手推车上。
    
    Args:
        enable_gui: 是否启用 GUI 可视化
        llm_provider: LLM 提供商 ("kimi", "openai", "mock")
        task_description: 自定义任务描述（可选）
    
    Returns:
        ExecutionResult 执行结果
    
    示例:
        >>> result = run_warehouse_demo(enable_gui=True, llm_provider="mock")
        >>> print(f"成功: {result.success}, 耗时: {result.execution_time:.2f}s")
    """
    from dynahmrc.scenarios.warehouse_task import WarehouseTaskScenario
    
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
    scenario = WarehouseTaskScenario(enable_visualization=enable_gui)
    
    try:
        # 运行任务
        result = scenario.run_task(num_boxes=3)
    except KeyboardInterrupt:
        print("\n[WarehouseTask] 用户中断")
        result = None
    finally:
        scenario.shutdown()
    
    return result


def run_assembly_demo(
    enable_gui: bool = True,
    llm_provider: str = "mock"
) -> Any:
    """
    运行装配任务场景演示
    
    场景描述:
        模拟装配线，需要将零件组装成产品。
        涉及：搬运零件、定位装配、质量检查。
    
    Args:
        enable_gui: 是否启用 GUI 可视化
        llm_provider: LLM 提供商 ("kimi", "openai", "mock")
    
    Returns:
        ExecutionResult 执行结果
    
    示例:
        >>> result = run_assembly_demo(enable_gui=True)
        >>> print(f"成功: {result.success}")
    """
    from dynahmrc.scenarios.assembly_task import AssemblyTaskScenario
    
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
    scenario = AssemblyTaskScenario(enable_visualization=enable_gui)
    
    try:
        # 运行任务
        result = scenario.run_task()
    except KeyboardInterrupt:
        print("\n[AssemblyTask] 用户中断")
        result = None
    finally:
        scenario.shutdown()
    
    return result


def run_custom_task(
    task_description: str,
    robot_config: Optional[Dict] = None,
    scene_config: Optional[Dict] = None,
    llm_config: Optional[Dict] = None,
    enable_gui: bool = True
) -> Any:
    """
    运行自定义任务
    
    使用 DynaHMRCSystem 执行任意自然语言描述的任务。
    
    Args:
        task_description: 自然语言任务描述
        robot_config: 机器人配置（可选，使用默认配置）
        scene_config: 场景配置（可选，使用默认配置）
        llm_config: LLM 配置（可选，使用默认配置）
        enable_gui: 是否启用 GUI 可视化
    
    Returns:
        ExecutionResult 执行结果
    
    示例:
        >>> result = run_custom_task("把箱子搬到桌子上")
        >>> print(f"任务完成: {result.success}")
    """
    from dynahmrc.system import DynaHMRCSystem
    
    # 默认机器人配置
    default_robot_config = [
        {
            "robot_id": "mobile_manipulator_1",
            "robot_type": "mobile_manipulator",
            "robot_model": "panda_on_segbot",
            "init_position": [0.5, 0.5, 0.0],
            "init_orientation": [0, 0, 0, 1],
            "capabilities": ["navigation", "manipulation", "transport", "perception"]
        }
    ]
    
    # 默认场景配置
    default_scene_config = {
        "config_path": "Config/default.yaml",
        "gui": enable_gui,
        "objects": [
            {
                "name": "table",
                "model_path": "Asset/Scene/Object/Kitchen_world_models/Table/102379/mobility.urdf",
                "position": [1.0, 0, 0],
                "orientation": [0, 0, 0, 1],
                "type": "furniture"
            },
            {
                "name": "box",
                "model_path": "Asset/Scene/Object/Kitchen_world_models/Box/100129/mobility.urdf",
                "position": [0.5, 0.5, 0.5],
                "orientation": [0, 0, 0, 1],
                "type": "object"
            }
        ]
    }
    
    # 默认 LLM 配置
    default_llm_config = {
        "provider": "mock",
        "model": "mock-model",
        "temperature": 0.3,
        "enable_replanning": True,
        "max_replan_attempts": 3
    }
    
    # 合并配置
    robots = robot_config or default_robot_config
    scene = scene_config or default_scene_config
    llm = llm_config or default_llm_config
    
    # 创建系统并执行任务
    system = DynaHMRCSystem(scene, robots, llm, enable_visualization=enable_gui)
    result = system.execute_task(task_description)
    
    # 关闭系统
    system.shutdown()
    
    return result


def list_available_scenarios():
    """列出所有可用场景"""
    scenarios = {
        "warehouse": {
            "name": "仓储协作场景",
            "description": "仓库中搬运箱子到手推车",
            "robots": ["mobile_base", "arm", "mobile_manipulator"],
            "difficulty": "medium"
        },
        "assembly": {
            "name": "装配任务场景",
            "description": "装配线上零件组装",
            "robots": ["arm", "mobile_manipulator"],
            "difficulty": "hard"
        }
    }
    return scenarios


def print_system_info():
    """打印系统信息"""
    print("=" * 60)
    print("DynaHMRC: Dynamic Heterogeneous Multi-Robot Collaboration")
    print("基于 BestMan 仿真平台的异构多机器人动态协作框架")
    print("=" * 60)
    print()
    print("可用场景:")
    scenarios = list_available_scenarios()
    for key, info in scenarios.items():
        print(f"  - {key}: {info['name']}")
        print(f"    描述: {info['description']}")
        print(f"    机器人: {', '.join(info['robots'])}")
        print(f"    难度: {info['difficulty']}")
        print()


def main():
    """主入口函数"""
    # 设置 Python 路径
    setup_pythonpath()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="DynaHMRC 演示运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行仓储场景
  python -m dynahmrc.run_demo --scenario warehouse
  
  # 运行装配场景（无 GUI）
  python -m dynahmrc.run_demo --scenario assembly --no-gui
  
  # 使用 Kimi LLM 运行
  python -m dynahmrc.run_demo --scenario warehouse --llm kimi
  
  # 运行自定义任务
  python -m dynahmrc.run_demo --task "把红色箱子搬到桌子上"
        """
    )
    
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["warehouse", "assembly"],
        help="选择要运行的场景"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        help="自定义任务描述（与 --scenario 互斥）"
    )
    
    parser.add_argument(
        "--llm",
        type=str,
        choices=["mock", "kimi", "openai"],
        default="mock",
        help="LLM 提供商 (默认: mock)"
    )
    
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="禁用 GUI 可视化"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="显示系统信息并退出"
    )
    
    args = parser.parse_args()
    
    # 显示信息
    if args.info:
        print_system_info()
        return
    
    # 检查参数
    if not args.scenario and not args.task:
        print("错误: 必须指定 --scenario 或 --task")
        print()
        print_system_info()
        parser.print_help()
        return
    
    # 运行场景
    enable_gui = not args.no_gui
    
    try:
        if args.scenario == "warehouse":
            print(f"运行仓储场景 (GUI={enable_gui}, LLM={args.llm})...")
            result = run_warehouse_demo(
                enable_gui=enable_gui,
                llm_provider=args.llm
            )
        
        elif args.scenario == "assembly":
            print(f"运行装配场景 (GUI={enable_gui}, LLM={args.llm})...")
            result = run_assembly_demo(
                enable_gui=enable_gui,
                llm_provider=args.llm
            )
        
        elif args.task:
            print(f"运行自定义任务: {args.task}")
            result = run_custom_task(
                task_description=args.task,
                enable_gui=enable_gui
            )
        
        # 打印结果
        print()
        print("=" * 60)
        print("执行结果:")
        print(f"  成功: {result.success}")
        print(f"  消息: {result.message}")
        print(f"  耗时: {result.execution_time:.2f} 秒")
        print(f"  完成任务: {len(result.completed_tasks)}")
        print(f"  失败任务: {len(result.failed_tasks)}")
        print(f"  重规划次数: {result.replan_count}")
        print("=" * 60)
        
        # 返回码
        return 0 if result.success else 1
        
    except KeyboardInterrupt:
        print("\n用户中断")
        return 130
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
