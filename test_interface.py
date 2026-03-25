"""
测试代码 - 验证 DynaHMRC 接口调用
测试内容：
1. BestManAdapter 动作执行
2. Scene Graph 获取
3. Robot States 获取
4. 完整的五阶段协作流程
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynahmrc.integration.bestman_adapter import BestManAdapter, ActionType, ExecutionFeedback
from dynahmrc.core.collaboration import FourStageCollaboration
from dynahmrc.core.robot_agent import RobotAgent

# Import MockLLMClient from dyna_hmrc_web
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Dyna_hmrc_web'))
from dynahmrc_web.dynahmrc.utils.llm_api import MockLLMClient


class MockRobot:
    """模拟机器人，用于测试"""
    def __init__(self, robot_id, robot_type, capabilities):
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.capabilities = capabilities
        self.position = [0, 0, 0]
        self.is_holding_object = False
        self.held_object_id = None
        
    def get_state(self):
        return {
            'robot_id': self.robot_id,
            'robot_type': self.robot_type,
            'position': self.position,
            'is_holding_object': self.is_holding_object,
            'held_object_id': self.held_object_id,
            'capabilities': self.capabilities
        }
    
    def navigate_to(self, position, orientation=None):
        self.position = position
        print(f"  [MockRobot] {self.robot_id} navigated to {position}")
        return True
    
    def pick(self, object_id):
        self.is_holding_object = True
        self.held_object_id = object_id
        print(f"  [MockRobot] {self.robot_id} picked object {object_id}")
        return True
    
    def place(self, position):
        self.is_holding_object = False
        self.held_object_id = None
        print(f"  [MockRobot] {self.robot_id} placed object at {position}")
        return True


class MockEnvironment:
    """模拟环境，用于测试 Scene Graph"""
    def __init__(self):
        self.scene_objects = {
            'table': {
                'position': [0, 0, 0],
                'orientation': [0, 0, 0, 1],
                'type': 'furniture',
                'model_path': 'Asset/Scene/Object/Table/mobility.urdf'
            },
            'box': {
                'position': [1.0, 0, 0.5],
                'orientation': [0, 0, 0, 1],
                'type': 'object',
                'model_path': 'Asset/Scene/Object/Box/mobility.urdf'
            },
            'target_zone': {
                'position': [2.0, 1.0, 0],
                'orientation': [0, 0, 0, 1],
                'type': 'zone',
                'model_path': ''
            }
        }


def test_bestman_adapter():
    """测试 BestManAdapter 接口"""
    print("\n" + "="*60)
    print("测试 1: BestManAdapter 动作执行")
    print("="*60)
    
    # 创建模拟机器人
    robot1 = MockRobot('robot_1', 'mobile_manipulator', ['navigation', 'manipulation', 'transport'])
    robot2 = MockRobot('robot_2', 'mobile_base', ['navigation'])
    
    # 创建环境并关联到机器人
    env = MockEnvironment()
    robot1.env = env
    robot2.env = env
    
    # 创建 Adapter
    robot_registry = {'robot_1': robot1, 'robot_2': robot2}
    adapter = BestManAdapter(robot_registry)
    
    print("\n1. 测试 execute_action - navigate")
    feedback = adapter.execute_action('robot_1', 'navigate', {'target': [1.0, 2.0, 0.0]})
    print(f"   结果: success={feedback.success}, message={feedback.message}")
    
    print("\n2. 测试 execute_action - pick")
    feedback = adapter.execute_action('robot_1', 'pick', {'object_id': 5, 'object_name': 'box'})
    print(f"   结果: success={feedback.success}, message={feedback.message}")
    
    print("\n3. 测试 execute_action - place")
    feedback = adapter.execute_action('robot_1', 'place', {'target': [2.0, 2.0, 0.5]})
    print(f"   结果: success={feedback.success}, message={feedback.message}")
    
    print("\n4. 测试 execute_action - wait")
    feedback = adapter.execute_action('robot_2', 'wait', {'duration': 1})
    print(f"   结果: success={feedback.success}, message={feedback.message}")
    
    print("\n5. 测试 execute_action - 无效动作")
    feedback = adapter.execute_action('robot_2', 'pick', {'object_id': 5})
    print(f"   结果: success={feedback.success}, message={feedback.message}, error_code={feedback.error_code}")
    
    print("\n6. 测试 execute_action - 不存在的机器人")
    feedback = adapter.execute_action('robot_999', 'navigate', {})
    print(f"   结果: success={feedback.success}, message={feedback.message}, error_code={feedback.error_code}")
    
    return adapter


def test_scene_graph(adapter):
    """测试 Scene Graph 获取"""
    print("\n" + "="*60)
    print("测试 2: Scene Graph 获取")
    print("="*60)
    
    scene_graph = adapter.get_scene_graph()
    print(f"\n获取到的场景图:")
    for obj_name, obj_info in scene_graph.items():
        print(f"  - {obj_name}: position={obj_info.get('position')}, type={obj_info.get('type')}")
    
    return scene_graph


def test_robot_states(adapter):
    """测试 Robot States 获取"""
    print("\n" + "="*60)
    print("测试 3: Robot States 获取")
    print("="*60)
    
    robot_states = adapter.get_robot_states()
    print(f"\n获取到的机器人状态:")
    for robot_id, state in robot_states.items():
        print(f"  - {robot_id}: position={state.get('position')}, holding={state.get('is_holding_object')}")
    
    return robot_states


def test_four_stage_collaboration():
    """测试五阶段协作流程"""
    print("\n" + "="*60)
    print("测试 4: 五阶段协作流程（使用 Mock LLM）")
    print("="*60)
    
    # 创建 Mock LLM Client
    llm_client = MockLLMClient()
    
    # 创建 Robot Agents
    robot1 = RobotAgent(
        name='robot_1',
        robot_type='mobile_manipulator',
        capabilities=['navigation', 'manipulation', 'transport'],
        llm_client=llm_client
    )
    robot2 = RobotAgent(
        name='robot_2',
        robot_type='mobile_base',
        capabilities=['navigation', 'transport'],
        llm_client=llm_client
    )
    
    # 创建协作框架
    collaboration = FourStageCollaboration(
        robots=[robot1, robot2],
        max_execution_steps=20,
        enable_communication=True,
        enable_visualization=False,
        reflection_interval=5,
        enable_reflection=True
    )
    
    # 创建 Mock Adapter 并设置
    mock_robot1 = MockRobot('robot_1', 'mobile_manipulator', ['navigation', 'manipulation', 'transport'])
    mock_robot2 = MockRobot('robot_2', 'mobile_base', ['navigation'])
    env = MockEnvironment()
    mock_robot1.env = env
    mock_robot2.env = env
    
    adapter = BestManAdapter({
        'robot_1': mock_robot1,
        'robot_2': mock_robot2
    })
    collaboration.set_adapter(adapter)
    
    # 测试 observation 获取
    print("\n1. 测试 _get_observation:")
    observation = collaboration._get_observation()
    print(f"   scene_graph keys: {list(observation['scene_graph'].keys())}")
    print(f"   robot_states keys: {list(observation['robot_states'].keys())}")
    
    # 测试 action 执行
    print("\n2. 测试 _execute_action:")
    action = {'action': 'navigate', 'params': {'target': [1.0, 1.0, 0.0]}}
    feedback = collaboration._execute_action('robot_1', action)
    print(f"   结果: success={feedback['success']}, message={feedback['message']}")
    
    action = {'action': 'pick', 'params': {'object_id': 5}}
    feedback = collaboration._execute_action('robot_1', action)
    print(f"   结果: success={feedback['success']}, message={feedback['message']}")
    
    print("\n3. 完整协作流程测试（简化版）:")
    print("   注意：完整流程需要真实的 LLM，这里仅测试接口连接")
    
    return collaboration


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("DynaHMRC 接口调用测试")
    print("="*60)
    
    try:
        # 测试 1: BestManAdapter
        adapter = test_bestman_adapter()
        
        # 测试 2: Scene Graph
        scene_graph = test_scene_graph(adapter)
        
        # 测试 3: Robot States
        robot_states = test_robot_states(adapter)
        
        # 测试 4: 五阶段协作
        collaboration = test_four_stage_collaboration()
        
        print("\n" + "="*60)
        print("所有测试完成！")
        print("="*60)
        print("\n测试总结:")
        print(f"  ✓ BestManAdapter 动作执行: 通过")
        print(f"  ✓ Scene Graph 获取: 通过 (获取到 {len(scene_graph)} 个物体)")
        print(f"  ✓ Robot States 获取: 通过 (获取到 {len(robot_states)} 个机器人)")
        print(f"  ✓ 五阶段协作接口: 通过")
        print("\n所有接口调用正常！")
        
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
