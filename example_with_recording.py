"""
使用示例：带有录像和日志记录功能的 DynaHMRC 系统

这个示例展示了如何启用录像和日志记录功能。
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynahmrc import DynaHMRCSystem


def main():
    """主函数"""

    # 场景配置
    scene_config = {
        "config_path": "Config/default.yaml",
        "gui": True,
        "scene_path": "Asset/Scene/Kitchen.json"  # 使用厨房场景
    }

    # 机器人配置
    robot_configs = [
        {
            "robot_id": "alice",
            "robot_type": "mobile_manipulator",
            "robot_model": "panda_on_segbot",
            "init_position": [1.2, 2.9, 0.0],
            "init_orientation": [0.0, 0.0, 1.0, 0.0],
            "capabilities": ["navigation", "manipulation", "pick", "place"]
        },
        {
            "robot_id": "bob",
            "robot_type": "mobile_manipulator",
            "robot_model": "panda_on_segbot",
            "init_position": [4.2, 2.9, 0.0],
            "init_orientation": [0.0, 0.0, 1.0, 0.0],
            "capabilities": ["navigation", "manipulation", "pick", "place"]
        }
    ]

    # LLM 配置
    llm_config = {
        "provider": "kimi",  # 或 "mock" 用于测试
        "api_key": os.environ.get("KIMI_API_KEY", ""),
        "model": "kimi-k2.5",
        "temperature": 0.3,
        "enable_replanning": True,
        "max_replan_attempts": 3
    }

    # 创建系统实例 - 启用录像和日志
    print("="*60)
    print("创建 DynaHMRC 系统（启用录像和日志）")
    print("="*60)

    system = DynaHMRCSystem(
        scene_config=scene_config,
        robot_configs=robot_configs,
        llm_config=llm_config,
        enable_visualization=True,   # 启用可视化
        enable_recording=True,        # 启用录像（第四阶段自动开始/停止）
        enable_logging=True           # 启用LLM交互日志
    )

    # 初始化系统
    if not system.initialize():
        print("系统初始化失败！")
        return

    try:
        # 执行任务
        task = "把柠檬从桌子上搬到柜台上"

        print("\n" + "="*60)
        print(f"执行任务: {task}")
        print("="*60)

        result = system.execute_task(
            natural_language_task=task,
            max_steps=50,
            task_type="pack_objects",
            variation="static"
        )

        # 打印结果
        print("\n" + "="*60)
        print("执行结果")
        print("="*60)
        print(f"成功: {result['success']}")
        print(f"消息: {result['message']}")
        print(f"步数: {result['steps']}")
        print(f"通信次数: {result['communications']}")
        print(f"执行时间: {result['execution_time']:.2f}秒")
        print(f"领导者: {result.get('leader', 'N/A')}")

        if result.get('robot_assignments'):
            print("\n任务分配:")
            for robot, tasks in result['robot_assignments'].items():
                print(f"  {robot}: {tasks}")

        print("\n" + "="*60)
        print("输出文件位置:")
        print("  - 录像: outputs/recordings/*.pkl")
        print("  - 日志: outputs/logs/llm_interaction_*.txt")
        print("="*60)

    except KeyboardInterrupt:
        print("\n用户中断执行")

    finally:
        # 关闭系统（会自动保存录像和日志）
        system.shutdown()


if __name__ == "__main__":
    main()
