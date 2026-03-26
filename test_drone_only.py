"""
无人机单独测试 - 测试导航、抓取和放置功能
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BestMan 导入
from Env.Client import Client
from Visualization.Visualizer import Visualizer
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


class DroneTester:
    """无人机功能测试器"""
    
    def __init__(self):
        self.client = None
        self.visualizer = None
        self.robot_factory = None
        self.drone = None
        self.scene_objects = {}
        
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
            # 创建地面标记（作为参考点）
            ground_id = self.client.load_object(
                obj_name="ground",
                model_path="Asset/Scene/Object/URDF_models/clear_box/model.urdf",
                object_position=[0, 0, 0],
                object_orientation=[0, 0, 0, 1],
                scale=5.0,
                fixed_base=True
            )
            print(f"   ✓ 创建地面参考 (ID: {ground_id})")
            
            # 创建桌子
            table_id = self.client.load_object(
                obj_name="table",
                model_path="Asset/Scene/Object/URDF_models/furniture_table_rectangle_high/table.urdf",
                object_position=[2.0, 0, 0],
                object_orientation=[0, 0, 0, 1],
                scale=1.0,
                fixed_base=True
            )
            print(f"   ✓ 创建桌子 (ID: {table_id})")
            
            # 创建可抓取物体（小箱子）
            box_id = self.client.load_object(
                obj_name="small_box",
                model_path="Asset/Scene/Object/URDF_models/cracker_box/model.urdf",
                object_position=[2.0, 0, 1.0],
                object_orientation=[0, 0, 0, 1],
                scale=0.5,
                fixed_base=False
            )
            print(f"   ✓ 创建小箱子 (ID: {box_id})")
            
            # 创建目标放置区域
            target_id = self.client.load_object(
                obj_name="target_zone",
                model_path="Asset/Scene/Object/URDF_models/clear_box/model.urdf",
                object_position=[-2.0, 0, 0.1],
                object_orientation=[0, 0, 0, 1],
                scale=1.0,
                fixed_base=True
            )
            print(f"   ✓ 创建目标区域 (ID: {target_id})")
            
            self.scene_objects = {
                "ground": ground_id,
                "table": table_id,
                "small_box": box_id,
                "target_zone": target_id
            }
            
            return True
            
        except Exception as e:
            print(f"   [ERROR] 创建场景失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def setup_drone(self):
        """设置无人机"""
        print("\n" + "="*60)
        print("3. 创建无人机")
        print("="*60)
        
        try:
            self.robot_factory = RobotFactory(self.client, self.visualizer)
            
            # 创建无人机
            self.drone = self.robot_factory.create_robot(
                robot_id="test_drone",
                robot_type="drone",
                robot_model="drone",
                init_position=[0.0, 0.0, 1.5]  # 空中初始位置
            )
            print(f"   ✓ 创建无人机 test_drone")
            print(f"   ✓ 无人机类型: {self.drone.robot_type}")
            print(f"   ✓ 无人机能力: {self.drone.capabilities}")
            
            return True
            
        except Exception as e:
            print(f"   [ERROR] 创建无人机失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_navigation(self):
        """测试导航功能"""
        print("\n" + "="*60)
        print("4. 测试导航功能")
        print("="*60)
        
        try:
            # 测试点1: 飞到桌子上方
            print("\n   [测试 4.1] 导航到桌子上方...")
            target1 = [2.0, 0.0, 2.0]  # 桌子上方2米高
            success1 = self.drone.navigate_to(target1)
            print(f"   {'✓' if success1 else '✗'} 导航到 {target1}: {'成功' if success1 else '失败'}")
            time.sleep(1)
            
            # 测试点2: 飞到目标区域上方
            print("\n   [测试 4.2] 导航到目标区域上方...")
            target2 = [-2.0, 0.0, 2.0]  # 目标区域上方2米高
            success2 = self.drone.navigate_to(target2)
            print(f"   {'✓' if success2 else '✗'} 导航到 {target2}: {'成功' if success2 else '失败'}")
            time.sleep(1)
            
            # 测试点3: 飞到箱子附近（准备抓取）
            print("\n   [测试 4.3] 导航到箱子附近...")
            target3 = [2.0, 0.0, 1.2]  # 箱子附近，高度1.2米
            success3 = self.drone.navigate_to(target3)
            print(f"   {'✓' if success3 else '✗'} 导航到 {target3}: {'成功' if success3 else '失败'}")
            time.sleep(1)
            
            return success1 and success2 and success3
            
        except Exception as e:
            print(f"   [ERROR] 导航测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_pick_place(self):
        """测试抓取和放置功能"""
        print("\n" + "="*60)
        print("5. 测试抓取和放置功能")
        print("="*60)
        
        try:
            # 获取箱子ID
            box_id = self.scene_objects.get("small_box")
            if box_id is None:
                print("   [ERROR] 找不到箱子物体")
                return False
            
            # 测试抓取
            print("\n   [测试 5.1] 抓取箱子...")
            print(f"   箱子ID: {box_id}")
            
            # 先飞到箱子正上方
            print("   先导航到箱子正上方...")
            box_pos = [2.0, 0.0, 1.2]
            self.drone.navigate_to(box_pos)
            time.sleep(0.5)
            
            # 执行抓取
            pick_success = self.drone.pick(object_id=box_id)
            print(f"   {'✓' if pick_success else '✗'} 抓取箱子: {'成功' if pick_success else '失败'}")
            
            if not pick_success:
                print("   [警告] 抓取失败，跳过后续放置测试")
                return False
            
            time.sleep(1)
            
            # 测试放置
            print("\n   [测试 5.2] 放置箱子到目标区域...")
            
            # 先飞到目标区域上方
            target_pos = [-2.0, 0.0, 2.0]
            print(f"   先导航到目标区域上方 {target_pos}...")
            self.drone.navigate_to(target_pos)
            time.sleep(0.5)
            
            # 执行放置
            place_location = [-2.0, 0.0, 1.0]  # 目标区域，高度1米
            place_success = self.drone.place(location=place_location)
            print(f"   {'✓' if place_success else '✗'} 放置箱子到 {place_location}: {'成功' if place_success else '失败'}")
            
            return pick_success and place_success
            
        except Exception as e:
            print(f"   [ERROR] 抓取/放置测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_get_state(self):
        """测试状态获取"""
        print("\n" + "="*60)
        print("6. 测试状态获取")
        print("="*60)
        
        try:
            state = self.drone.get_state()
            print(f"   机器人ID: {state.get('robot_id')}")
            print(f"   机器人类型: {state.get('robot_type')}")
            print(f"   当前位置: {state.get('position')}")
            print(f"   当前朝向: {state.get('orientation')}")
            print(f"   是否忙碌: {state.get('is_busy')}")
            print(f"   是否持有物体: {state.get('is_holding_object')}")
            print(f"   持有物体ID: {state.get('held_object_id')}")
            
            return True
            
        except Exception as e:
            print(f"   [ERROR] 状态获取失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("开始无人机功能测试")
        print("="*60)
        
        # 设置环境
        if not self.setup_environment():
            print("\n[FAILED] 环境设置失败")
            return False
        
        # 设置场景
        if not self.setup_scene():
            print("\n[FAILED] 场景设置失败")
            return False
        
        # 设置无人机
        if not self.setup_drone():
            print("\n[FAILED] 无人机设置失败")
            return False
        
        # 等待场景稳定
        print("\n   等待场景稳定...")
        time.sleep(2)
        
        results = {}
        
        # 测试状态获取
        results['state'] = self.test_get_state()
        
        # 测试导航
        results['navigation'] = self.test_navigation()
        
        # 测试抓取和放置
        results['pick_place'] = self.test_pick_place()
        
        # 最终状态
        print("\n" + "="*60)
        print("7. 最终状态")
        print("="*60)
        self.test_get_state()
        
        # 打印测试报告
        print("\n" + "="*60)
        print("测试报告")
        print("="*60)
        for test_name, success in results.items():
            status = "✓ 通过" if success else "✗ 失败"
            print(f"   {test_name:20s}: {status}")
        
        all_passed = all(results.values())
        print(f"\n   总体结果: {'全部通过' if all_passed else '部分失败'}")
        
        # 保持仿真运行
        print("\n   按 Ctrl+C 退出仿真...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n   退出仿真")
        
        return all_passed


def main():
    """主函数"""
    tester = DroneTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
