"""
动作执行测试 - 测试单个动作指令是否能正确驱动机器人运动
测试内容：
1. 初始化 BestMan 仿真环境
2. 创建机器人和场景
3. 直接调用 Adapter 执行各种动作
4. 验证机器人是否正确响应
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BestMan 导入
from Env.Client import Client
from Visualization.Visualizer import Visualizer

# DynaHMRC 导入
from dynahmrc.integration.bestman_adapter import BestManAdapter, ActionType
from dynahmrc.integration.robot_factory import RobotFactory


def load_config(config_path: str = "Config/default.yaml"):
    """加载配置文件"""
    from yacs.config import CfgNode as CN
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    with open(config_path, "r") as f:
        cfg = CN.load_cfg(f)
    
    return cfg


class ActionExecutionTester:
    """动作执行测试器"""
    
    def __init__(self):
        self.client = None
        self.visualizer = None
        self.robot_factory = None
        self.adapter = None
        self.robots = {}
        
    def setup_environment(self):
        """设置仿真环境"""
        print("\n" + "="*60)
        print("1. 初始化 BestMan 仿真环境")
        print("="*60)
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "Config", "default.yaml")
        
        try:
            cfg = load_config(config_path)
            cfg.Client.enable_GUI = True
            cfg.Client.enable_Debug = True
            
            self.client = Client(cfg.Client)
            print("   ✓ BestMan 客户端初始化成功")
            
            self.visualizer = Visualizer(self.client, cfg.Visualizer)
            print("   ✓ Visualizer 初始化成功")
            
        except Exception as e:
            print(f"   [ERROR] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def setup_scene(self):
        """设置测试场景"""
        print("\n" + "="*60)
        print("2. 创建测试场景")
        print("="*60)
        
        try:
            # 创建桌子
            table_id = self.client.load_object(
                obj_name="table",
                model_path="Asset/Scene/Object/URDF_models/furniture_table_rectangle_high/table.urdf",
                object_position=[0, 0, 0],
                object_orientation=[0, 0, 0, 1],
                scale=1.0,
                fixed_base=True
            )
            print(f"   ✓ 创建桌子 (ID: {table_id})")
            
            # 创建箱子
            box_id = self.client.load_object(
                obj_name="box",
                model_path="Asset/Scene/Object/URDF_models/cracker_box/model.urdf",
                object_position=[0.5, 0, 0.5],
                object_orientation=[0, 0, 0, 1],
                scale=1.0,
                fixed_base=False
            )
            print(f"   ✓ 创建箱子 (ID: {box_id})")
            
            # 创建目标区域
            target_id = self.client.load_object(
                obj_name="target_zone",
                model_path="Asset/Scene/Object/URDF_models/clear_box/model.urdf",
                object_position=[2.0, 0, 0.1],
                object_orientation=[0, 0, 0, 1],
                scale=1.0,
                fixed_base=True
            )
            print(f"   ✓ 创建目标区域 (ID: {target_id})")
            
            return {
                "table": table_id,
                "box": box_id,
                "target_zone": target_id
            }
            
        except Exception as e:
            print(f"   [ERROR] 创建场景失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def setup_robots(self):
        """设置机器人"""
        print("\n" + "="*60)
        print("3. 创建机器人")
        print("="*60)
        
        try:
            self.robot_factory = RobotFactory(self.client, self.visualizer)
            
            # 创建移动操作机器人
            robot1 = self.robot_factory.create_robot(
                robot_id="test_mobile_manipulator",
                robot_type="mobile_manipulator",
                robot_model="panda_on_segbot",
                init_position=[-1.0, 0, 0]
            )
            print(f"   ✓ 创建机器人 test_mobile_manipulator")
            
            # 创建移动基座机器人
            robot2 = self.robot_factory.create_robot(
                robot_id="test_mobile_base",
                robot_type="mobile_base",
                robot_model="segbot",
                init_position=[-1.0, 1.0, 0]
            )
            print(f"   ✓ 创建机器人 test_mobile_base")
            
            return self.robot_factory.get_all_robots()
            
        except Exception as e:
            print(f"   [ERROR] 创建机器人失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def setup_adapter(self, scene_objects):
        """设置 BestManAdapter"""
        print("\n" + "="*60)
        print("4. 创建 BestManAdapter")
        print("="*60)
        
        try:
            all_robots = self.robot_factory.get_all_robots()
            self.adapter = BestManAdapter(all_robots, self.client)
            
            # 注册场景物体
            for obj_name, obj_id in scene_objects.items():
                obj_type = "furniture" if obj_name in ["table", "target_zone"] else "graspable"
                self.adapter.register_scene_object(obj_name, obj_id, obj_type)
            
            print(f"   ✓ Adapter 创建成功，管理 {len(all_robots)} 个机器人")
            print(f"   ✓ 注册 {len(self.adapter.scene_objects)} 个场景物体")
            
            return True
            
        except Exception as e:
            print(f"   [ERROR] 创建 Adapter 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_navigate_action(self, robot_id, target, description=""):
        """测试导航动作"""
        print(f"\n   [测试导航] {robot_id} -> {target} ({description})")
        
        try:
            # 获取机器人当前位置
            robot = self.adapter.robot_registry.get(robot_id)
            if robot:
                state = robot.get_state()
                start_pos = state.get('position', [0, 0, 0])
                print(f"      起始位置: {start_pos}")
            
            # 执行导航动作
            params = {"target": target}
            feedback = self.adapter.execute_action(robot_id, "navigate", params)
            
            print(f"      结果: {'✓ 成功' if feedback.success else '✗ 失败'}")
            print(f"      消息: {feedback.message}")
            
            if robot:
                state = robot.get_state()
                end_pos = state.get('position', [0, 0, 0])
                print(f"      结束位置: {end_pos}")
            
            return feedback.success
            
        except Exception as e:
            print(f"      [ERROR] {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_pick_action(self, robot_id, object_name):
        """测试抓取动作"""
        print(f"\n   [测试抓取] {robot_id} -> {object_name}")
        
        try:
            params = {"object_id": object_name, "object_name": object_name}
            feedback = self.adapter.execute_action(robot_id, "pick", params)
            
            print(f"      结果: {'✓ 成功' if feedback.success else '✗ 失败'}")
            print(f"      消息: {feedback.message}")
            
            return feedback.success
            
        except Exception as e:
            print(f"      [ERROR] {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_place_action(self, robot_id, location):
        """测试放置动作"""
        print(f"\n   [测试放置] {robot_id} -> {location}")
        
        try:
            params = {"location": location}
            feedback = self.adapter.execute_action(robot_id, "place", params)
            
            print(f"      结果: {'✓ 成功' if feedback.success else '✗ 失败'}")
            print(f"      消息: {feedback.message}")
            
            return feedback.success
            
        except Exception as e:
            print(f"      [ERROR] {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_communicate_action(self, robot_id, to_robot, message, broadcast=False):
        """测试通信动作"""
        print(f"\n   [测试通信] {robot_id} -> {to_robot if not broadcast else 'all'}: {message[:30]}...")
        
        try:
            params = {
                "to": to_robot,
                "message": message,
                "broadcast": broadcast
            }
            feedback = self.adapter.execute_action(robot_id, "communicate", params)
            
            print(f"      结果: {'✓ 成功' if feedback.success else '✗ 失败'}")
            print(f"      消息: {feedback.message}")
            
            return feedback.success
            
        except Exception as e:
            print(f"      [ERROR] {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_wait_action(self, robot_id, duration=1.0):
        """测试等待动作"""
        print(f"\n   [测试等待] {robot_id} 等待 {duration} 秒")
        
        try:
            params = {"duration": duration}
            feedback = self.adapter.execute_action(robot_id, "wait", params)
            
            print(f"      结果: {'✓ 成功' if feedback.success else '✗ 失败'}")
            print(f"      消息: {feedback.message}")
            
            return feedback.success
            
        except Exception as e:
            print(f"      [ERROR] {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_action_tests(self):
        """运行所有动作测试"""
        print("\n" + "="*60)
        print("5. 执行动作测试")
        print("="*60)
        
        results = []
        
        # 5.1 测试 MobileManipulator 的导航
        print("\n5.1 测试 MobileManipulator 导航")
        results.append(("MobileManipulator 导航到目标区域", 
                       self.test_navigate_action("test_mobile_manipulator", [1.5, 0.0, 0.0], "目标区域")))
        
        results.append(("MobileManipulator 导航到箱子附近",
                       self.test_navigate_action("test_mobile_manipulator", [0.0, 0.0, 0.0], "箱子附近")))
        
        # 5.2 测试 MobileBase 的导航
        print("\n5.2 测试 MobileBase 导航")
        results.append(("MobileBase 导航到目标区域",
                       self.test_navigate_action("test_mobile_base", [1.5, 1.0, 0.0], "目标区域")))
        
        # 5.3 测试抓取动作
        print("\n5.3 测试抓取动作")
        results.append(("MobileManipulator 抓取箱子",
                       self.test_pick_action("test_mobile_manipulator", "box")))
        
        # 5.4 测试放置动作
        print("\n5.4 测试放置动作")
        results.append(("MobileManipulator 放置到目标区域",
                       self.test_place_action("test_mobile_manipulator", [2.0, 0.0, 0.5])))
        
        # 5.5 测试通信动作
        print("\n5.5 测试通信动作")
        results.append(("MobileManipulator 发送消息",
                       self.test_communicate_action("test_mobile_manipulator", "test_mobile_base", 
                                                   "Hello from mobile manipulator!")))
        
        results.append(("MobileBase 广播消息",
                       self.test_communicate_action("test_mobile_base", "", 
                                                   "Broadcast from mobile base!", broadcast=True)))
        
        # 5.6 测试等待动作
        print("\n5.6 测试等待动作")
        results.append(("MobileManipulator 等待",
                       self.test_wait_action("test_mobile_manipulator", 1.0)))
        
        results.append(("MobileBase 等待",
                       self.test_wait_action("test_mobile_base", 1.0)))
        
        # 汇总结果
        print("\n" + "="*60)
        print("6. 测试结果汇总")
        print("="*60)
        
        passed = sum(1 for _, result in results if result)
        failed = len(results) - passed
        
        print(f"\n   总测试数: {len(results)}")
        print(f"   通过: {passed}")
        print(f"   失败: {failed}")
        
        print("\n   详细结果:")
        for test_name, result in results:
            status = "✓" if result else "✗"
            print(f"      {status} {test_name}")
        
        return failed == 0
    
    def run(self):
        """运行完整测试"""
        print("\n" + "="*70)
        print("动作执行测试")
        print("="*70)
        
        # 1. 设置环境
        if not self.setup_environment():
            print("\n[FAILED] 环境设置失败")
            return False
        
        # 2. 设置场景
        scene_objects = self.setup_scene()
        if scene_objects is None:
            print("\n[FAILED] 场景设置失败")
            return False
        
        # 3. 设置机器人
        robots = self.setup_robots()
        if robots is None:
            print("\n[FAILED] 机器人设置失败")
            return False
        
        # 4. 设置 Adapter
        if not self.setup_adapter(scene_objects):
            print("\n[FAILED] Adapter 设置失败")
            return False
        
        # 5. 运行动作测试
        success = self.run_action_tests()
        
        # 6. 清理
        print("\n" + "="*60)
        print("7. 测试完成，清理资源")
        print("="*60)
        
        # 保持仿真运行一段时间以便观察
        print("   保持仿真运行 3 秒以便观察...")
        import time
        time.sleep(3)
        
        # 断开连接
        if self.client:
            self.client.disconnect()
            print("   ✓ 断开 BestMan 连接")
        
        print("\n" + "="*70)
        if success:
            print("[SUCCESS] 所有动作测试通过！")
        else:
            print("[FAILED] 部分动作测试失败")
        print("="*70)
        
        return success


def main():
    """主函数"""
    tester = ActionExecutionTester()
    success = tester.run()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
