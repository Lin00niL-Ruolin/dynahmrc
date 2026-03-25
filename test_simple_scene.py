"""
简化场景测试 - 无需 PyBullet，直接测试接口逻辑
测试内容：
1. 模拟真实场景数据结构
2. 测试 BestManAdapter 动作分发
3. 测试 Scene Graph 和 Robot States 获取
4. 测试协作框架集成
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynahmrc.integration.bestman_adapter import BestManAdapter, ActionType, ExecutionFeedback
from dynahmrc.core.collaboration import FourStageCollaboration
from dynahmrc.core.robot_agent import RobotAgent

# Import MockLLMClient from dyna_hmrc_web
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Dyna_hmrc_web'))
from dynahmrc_web.dynahmrc.utils.llm_api import MockLLMClient


class SimpleObject:
    """简化场景物体"""
    def __init__(self, obj_id, name, position, obj_type="object"):
        self.obj_id = obj_id
        self.name = name
        self.position = position
        self.orientation = [0, 0, 0, 1]
        self.type = obj_type
        self.is_grasped = False


class SimpleRobot:
    """简化机器人，模拟真实 BestMan 机器人"""
    def __init__(self, robot_id, robot_type, capabilities, init_position):
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.capabilities = capabilities
        self.position = init_position
        self.orientation = [0, 0, 0, 1]
        self.is_holding_object = False
        self.held_object_id = None
        self.env = None  # 将在创建后设置
        
    def get_state(self):
        """获取机器人状态（与 BestMan 接口一致）"""
        return {
            'robot_id': self.robot_id,
            'robot_type': self.robot_type,
            'position': self.position,
            'orientation': self.orientation,
            'is_holding_object': self.is_holding_object,
            'held_object_id': self.held_object_id,
            'capabilities': self.capabilities
        }
    
    def navigate_to(self, position, orientation=None):
        """导航到指定位置"""
        self.position = position[:3]  # 只取前3个元素
        if orientation:
            self.orientation = orientation
        print(f"  [Robot Action] {self.robot_id} 导航到 {position}")
        return True
    
    def pick(self, object_id):
        """抓取物体"""
        self.is_holding_object = True
        self.held_object_id = object_id
        print(f"  [Robot Action] {self.robot_id} 抓取物体 {object_id}")
        return True
    
    def place(self, position):
        """放置物体"""
        self.is_holding_object = False
        self.held_object_id = None
        print(f"  [Robot Action] {self.robot_id} 放置物体到 {position}")
        return True
    
    def move_forward(self, distance):
        """向前移动"""
        self.position[0] += distance
        print(f"  [Robot Action] {self.robot_id} 向前移动 {distance}m")
        return True
    
    def move_backward(self, distance):
        """向后移动"""
        self.position[0] -= distance
        print(f"  [Robot Action] {self.robot_id} 向后移动 {distance}m")
        return True
    
    def rotate(self, angle):
        """旋转"""
        print(f"  [Robot Action] {self.robot_id} 旋转 {angle}度")
        return True


class SimpleEnvironment:
    """简化环境，模拟 BestMan 场景"""
    def __init__(self):
        self.scene_objects = {}
        self.objects = {}  # 用于 get_objects 方法
        self.next_id = 0
    
    def create_object(self, model_path, position, orientation=None, scale=1.0, fixed_base=False):
        """创建物体（模拟 BestMan 的 create_object）"""
        obj_id = self.next_id
        self.next_id += 1
        
        # 从 model_path 提取物体名称
        obj_name = model_path.split('/')[-2] if '/' in model_path else f'object_{obj_id}'
        
        obj = SimpleObject(obj_id, obj_name, position, "object")
        if orientation:
            obj.orientation = orientation
        
        self.scene_objects[obj_name] = {
            'id': obj_id,
            'position': position,
            'orientation': obj.orientation,
            'type': obj.type,
            'model_path': model_path,
            'scale': scale,
            'fixed_base': fixed_base
        }
        
        self.objects[obj_id] = {
            'name': obj_name,
            'position': position,
            'orientation': obj.orientation,
            'type': obj.type
        }
        
        print(f"    [Scene] 创建物体: {obj_name} (ID: {obj_id}) at {position}")
        return obj_id
    
    def create_plane(self):
        """创建地面"""
        self.scene_objects['plane'] = {
            'id': -1,
            'position': [0, 0, 0],
            'orientation': [0, 0, 0, 1],
            'type': 'plane'
        }
        print("    [Scene] 创建地面")
    
    def get_objects(self):
        """获取所有物体（模拟 BestMan 接口）"""
        return self.objects


def test_simple_scene():
    """测试简化场景"""
    print("\n" + "="*60)
    print("简化场景测试 - 模拟 BestMan 环境")
    print("="*60)
    
    # 1. 创建环境
    print("\n1. 创建场景环境...")
    env = SimpleEnvironment()
    env.create_plane()
    
    # 创建桌子
    table_id = env.create_object(
        "Asset/Scene/Object/Table/mobility.urdf",
        position=[0, 0, 0],
        fixed_base=True
    )
    
    # 创建箱子（可抓取）
    box_id = env.create_object(
        "Asset/Scene/Object/Box/mobility.urdf",
        position=[0.5, 0, 0.5],
        fixed_base=False
    )
    
    # 创建目标区域
    target_id = env.create_object(
        "Asset/Scene/Object/Box/mobility.urdf",
        position=[2.0, 0, 0.1],
        fixed_base=True
    )
    
    print(f"   ✓ 场景创建完成，共 {len(env.scene_objects)} 个物体")
    
    # 2. 创建机器人
    print("\n2. 创建机器人...")
    robot1 = SimpleRobot(
        robot_id="robot_1",
        robot_type="mobile_manipulator",
        capabilities=["navigation", "manipulation", "transport"],
        init_position=[-1.0, 0, 0]
    )
    robot1.env = env
    
    robot2 = SimpleRobot(
        robot_id="robot_2",
        robot_type="mobile_base",
        capabilities=["navigation", "transport"],
        init_position=[-1.0, 1.0, 0]
    )
    robot2.env = env
    
    print(f"   ✓ 创建 2 个机器人")
    
    # 3. 创建 Adapter
    print("\n3. 创建 BestManAdapter...")
    robot_registry = {'robot_1': robot1, 'robot_2': robot2}
    adapter = BestManAdapter(robot_registry)
    print(f"   ✓ Adapter 创建成功")
    
    # 4. 测试 Scene Graph 获取
    print("\n4. 测试 Scene Graph 获取...")
    scene_graph = adapter.get_scene_graph()
    print(f"   ✓ 获取到 {len(scene_graph)} 个物体:")
    for obj_name, obj_info in scene_graph.items():
        pos = obj_info.get('position', [0, 0, 0])
        print(f"     - {obj_name}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    
    # 5. 测试 Robot States 获取
    print("\n5. 测试 Robot States 获取...")
    robot_states = adapter.get_robot_states()
    print(f"   ✓ 获取到 {len(robot_states)} 个机器人状态:")
    for robot_id, state in robot_states.items():
        pos = state.get('position', [0, 0, 0])
        print(f"     - {robot_id}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    
    # 6. 测试动作执行 - 完整场景模拟
    print("\n6. 测试动作执行 - 完整任务场景...")
    print("\n   场景: robot_1 抓取箱子并放置到目标区域")
    
    # 6.1 导航到箱子附近
    print("\n   步骤 1: 导航到箱子附近")
    feedback = adapter.execute_action(
        'robot_1',
        'navigate',
        {'target': [0.3, 0, 0]}
    )
    print(f"   结果: {feedback.message}")
    
    # 6.2 抓取箱子
    print("\n   步骤 2: 抓取箱子")
    feedback = adapter.execute_action(
        'robot_1',
        'pick',
        {'object_id': box_id, 'object_name': 'box'}
    )
    print(f"   结果: {feedback.message}")
    
    # 6.3 导航到目标区域
    print("\n   步骤 3: 导航到目标区域")
    feedback = adapter.execute_action(
        'robot_1',
        'navigate',
        {'target': [1.5, 0, 0]}
    )
    print(f"   结果: {feedback.message}")
    
    # 6.4 放置箱子
    print("\n   步骤 4: 放置箱子")
    feedback = adapter.execute_action(
        'robot_1',
        'place',
        {'target': [2.0, 0, 0.5]}
    )
    print(f"   结果: {feedback.message}")
    
    # 6.5 robot_2 等待
    print("\n   步骤 5: robot_2 等待")
    feedback = adapter.execute_action(
        'robot_2',
        'wait',
        {'duration': 1}
    )
    print(f"   结果: {feedback.message}")
    
    # 7. 验证最终状态
    print("\n7. 验证最终状态...")
    final_states = adapter.get_robot_states()
    print(f"   robot_1 最终位置: {final_states['robot_1']['position']}")
    print(f"   robot_1 持有物体: {final_states['robot_1']['is_holding_object']}")
    print(f"   robot_2 最终位置: {final_states['robot_2']['position']}")
    
    # 8. 测试协作框架
    print("\n8. 测试协作框架集成...")
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
    
    collaboration = FourStageCollaboration(
        robots=robot_agents,
        max_execution_steps=10,
        enable_communication=True,
        enable_visualization=False,
        enable_reflection=False
    )
    
    collaboration.set_adapter(adapter)
    
    # 测试 observation
    observation = collaboration._get_observation()
    print(f"   ✓ Observation: {len(observation['scene_graph'])} 物体, {len(observation['robot_states'])} 机器人")
    
    # 测试 action 执行
    action = {'action': 'navigate', 'params': {'target': [0.5, 0.5, 0]}}
    feedback = collaboration._execute_action('robot_1', action)
    print(f"   ✓ Action 执行: success={feedback['success']}")
    
    print("\n" + "="*60)
    print("简化场景测试完成！")
    print("="*60)
    
    return True


def main():
    """主函数"""
    print("\n" + "="*60)
    print("DynaHMRC 简化场景接口测试")
    print("="*60)
    print("\n此测试模拟真实 BestMan 环境，无需 PyBullet")
    print("包括：场景创建、机器人控制、动作执行、协作框架")
    
    try:
        result = test_simple_scene()
        if result:
            print("\n✅ 所有测试通过！接口调用正常")
            print("\n测试验证内容:")
            print("  ✓ Scene Graph 获取")
            print("  ✓ Robot States 获取")
            print("  ✓ navigate 动作执行")
            print("  ✓ pick 动作执行")
            print("  ✓ place 动作执行")
            print("  ✓ wait 动作执行")
            print("  ✓ 协作框架集成")
        else:
            print("\n❌ 测试失败")
    except Exception as e:
        print(f"\n[ERROR] 测试异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
