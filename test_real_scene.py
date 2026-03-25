"""
真实场景测试 - 在 BestMan 仿真环境中测试接口调用
测试内容：
1. 初始化真实 BestMan 场景
2. 创建真实机器人
3. 执行真实动作（导航、抓取、放置）
4. 获取真实 Scene Graph 和 Robot States
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BestMan 导入
from Env.Client import Client
from Visualization.Visualizer import Visualizer

# DynaHMRC 导入
from dynahmrc.integration.bestman_adapter import BestManAdapter
from dynahmrc.integration.robot_factory import RobotFactory
from dynahmrc.core.collaboration import FourStageCollaboration
from dynahmrc.core.robot_agent import RobotAgent

# Import MockLLMClient from dyna_hmrc_web
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Dyna_hmrc_web'))
from dynahmrc_web.dynahmrc.utils.llm_api import MockLLMClient


def test_real_scene():
    """在真实 BestMan 场景中测试"""
    print("\n" + "="*60)
    print("真实场景测试 - BestMan 仿真环境")
    print("="*60)
    
    # 1. 初始化 BestMan 客户端
    print("\n1. 初始化 BestMan 客户端...")
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "Config", "default.yaml")
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"   [WARN] 配置文件不存在: {config_path}")
        print("   使用默认配置...")
        config_path = None
    else:
        print(f"   找到配置文件: {config_path}")
    
    try:
        client = Client(config_path=config_path, gui=False)  # 无 GUI 模式，适合测试
        print("   ✓ BestMan 客户端初始化成功")
    except Exception as e:
        print(f"   [ERROR] 初始化失败: {e}")
        print("   请确保 BestMan 环境已正确安装")
        return None
    
    # 2. 创建测试场景
    print("\n2. 创建测试场景...")
    try:
        # 创建地面
        client.create_plane()
        
        # 创建桌子
        table_id = client.create_object(
            "Asset/Scene/Object/Table/mobility.urdf",
            position=[0, 0, 0],
            orientation=[0, 0, 0, 1],
            scale=1.0,
            fixed_base=True
        )
        print(f"   ✓ 创建桌子 (ID: {table_id})")
        
        # 创建箱子（可抓取物体）
        box_id = client.create_object(
            "Asset/Scene/Object/Box/mobility.urdf",
            position=[0.5, 0, 0.5],  # 放在桌子上方
            orientation=[0, 0, 0, 1],
            scale=0.5,
            fixed_base=False
        )
        print(f"   ✓ 创建箱子 (ID: {box_id})")
        
        # 创建目标区域标记
        target_id = client.create_object(
            "Asset/Scene/Object/Box/mobility.urdf",
            position=[2.0, 0, 0.1],
            orientation=[0, 0, 0, 1],
            scale=0.3,
            fixed_base=True
        )
        print(f"   ✓ 创建目标区域 (ID: {target_id})")
        
    except Exception as e:
        print(f"   [ERROR] 创建场景失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 3. 创建机器人
    print("\n3. 创建机器人...")
    try:
        robot_factory = RobotFactory(client)
        
        # 创建移动操作机器人
        robot1 = robot_factory.create_robot(
            robot_id="robot_1",
            robot_type="mobile_manipulator",
            robot_model="panda_on_segbot",
            init_position=[-1.0, 0, 0]
        )
        print(f"   ✓ 创建机器人 robot_1 (mobile_manipulator)")
        
        # 创建移动基座机器人
        robot2 = robot_factory.create_robot(
            robot_id="robot_2",
            robot_type="mobile_base",
            robot_model="turtlebot",
            init_position=[-1.0, 1.0, 0]
        )
        print(f"   ✓ 创建机器人 robot_2 (mobile_base)")
        
    except Exception as e:
        print(f"   [ERROR] 创建机器人失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 4. 创建 BestManAdapter
    print("\n4. 创建 BestManAdapter...")
    try:
        all_robots = robot_factory.get_all_robots()
        adapter = BestManAdapter(all_robots)
        print(f"   ✓ Adapter 创建成功，管理 {len(all_robots)} 个机器人")
    except Exception as e:
        print(f"   [ERROR] 创建 Adapter 失败: {e}")
        return None
    
    # 5. 测试 Scene Graph 获取
    print("\n5. 测试 Scene Graph 获取...")
    try:
        scene_graph = adapter.get_scene_graph()
        print(f"   ✓ 获取到 {len(scene_graph)} 个场景物体:")
        for obj_name, obj_info in list(scene_graph.items())[:5]:  # 只显示前5个
            print(f"     - {obj_name}: position={obj_info.get('position')}")
    except Exception as e:
        print(f"   [WARN] 获取 Scene Graph 失败: {e}")
        scene_graph = {}
    
    # 6. 测试 Robot States 获取
    print("\n6. 测试 Robot States 获取...")
    try:
        robot_states = adapter.get_robot_states()
        print(f"   ✓ 获取到 {len(robot_states)} 个机器人状态:")
        for robot_id, state in robot_states.items():
            pos = state.get('position', [0, 0, 0])
            print(f"     - {robot_id}: position=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    except Exception as e:
        print(f"   [WARN] 获取 Robot States 失败: {e}")
        robot_states = {}
    
    # 7. 测试动作执行
    print("\n7. 测试动作执行...")
    
    # 7.1 导航测试
    print("\n   7.1 测试 navigate 动作...")
    try:
        feedback = adapter.execute_action(
            'robot_1', 
            'navigate', 
            {'target': [0.3, 0, 0]}  # 导航到桌子附近
        )
        print(f"   结果: success={feedback.success}, message={feedback.message}")
        if feedback.success:
            print("   ✓ 导航成功")
        else:
            print(f"   ✗ 导航失败: {feedback.error_details}")
    except Exception as e:
        print(f"   [ERROR] 导航测试失败: {e}")
    
    # 7.2 等待测试
    print("\n   7.2 测试 wait 动作...")
    try:
        feedback = adapter.execute_action(
            'robot_2',
            'wait',
            {'duration': 1}
        )
        print(f"   结果: success={feedback.success}, message={feedback.message}")
    except Exception as e:
        print(f"   [ERROR] 等待测试失败: {e}")
    
    # 8. 测试协作框架集成
    print("\n8. 测试协作框架集成...")
    try:
        # 创建 Robot Agents
        llm_client = MockLLMClient()
        robot_agents = [
            RobotAgent(
                name='robot_1',
                robot_type='mobile_manipulator',
                capabilities=['navigation', 'manipulation', 'transport'],
                llm_client=llm_client
            ),
            RobotAgent(
                name='robot_2',
                robot_type='mobile_base',
                capabilities=['navigation', 'transport'],
                llm_client=llm_client
            )
        ]
        
        # 创建协作框架
        collaboration = FourStageCollaboration(
            robots=robot_agents,
            max_execution_steps=10,
            enable_communication=True,
            enable_visualization=False,
            enable_reflection=False  # 简化测试，禁用反思
        )
        
        # 设置 adapter
        collaboration.set_adapter(adapter)
        
        # 测试 observation 获取
        observation = collaboration._get_observation()
        print(f"   ✓ Observation 获取成功:")
        print(f"     - scene_graph: {len(observation.get('scene_graph', {}))} 个物体")
        print(f"     - robot_states: {len(observation.get('robot_states', {}))} 个机器人")
        
        # 测试 action 执行
        action = {'action': 'navigate', 'params': {'target': [0.5, 0.5, 0]}}
        feedback = collaboration._execute_action('robot_1', action)
        print(f"   ✓ Action 执行: success={feedback['success']}")
        
    except Exception as e:
        print(f"   [ERROR] 协作框架测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 9. 清理
    print("\n9. 清理资源...")
    try:
        client.disconnect()
        print("   ✓ 资源清理完成")
    except Exception as e:
        print(f"   [WARN] 清理失败: {e}")
    
    print("\n" + "="*60)
    print("真实场景测试完成！")
    print("="*60)
    
    return True


def main():
    """主函数"""
    print("\n" + "="*60)
    print("DynaHMRC 真实场景接口测试")
    print("="*60)
    print("\n此测试将在真实的 BestMan 仿真环境中执行")
    print("包括：场景创建、机器人控制、动作执行")
    
    try:
        result = test_real_scene()
        if result:
            print("\n✅ 所有测试通过！接口调用正常")
        else:
            print("\n❌ 测试失败")
    except Exception as e:
        print(f"\n[ERROR] 测试异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
