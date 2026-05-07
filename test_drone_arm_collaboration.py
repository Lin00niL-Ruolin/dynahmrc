"""
无人机与固定机械臂协作测试
==========================

任务场景：
1. 无人机(Lucy)从初始位置飞到物品(cup)位置
2. 无人机抓取物品
3. 无人机将物品放置到客厅桌子上
4. 固定机械臂(Bob)从桌子上抓取物品
5. 固定机械臂将物品放置到托盘

异构协作演示：
- Lucy (Drone): 具备空中导航和轻量抓取能力
- Bob (Arm): 具备精确的桌面操作能力
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


class DroneArmCollaborationTest:
    """无人机与固定机械臂协作测试器"""
    
    def __init__(self):
        self.client = None
        self.visualizer = None
        self.robot_factory = None
        
        # 机器人
        self.drone = None      # Lucy - 无人机
        self.arm_robot = None  # Bob - 固定机械臂
        
        # 场景物体
        self.scene_objects = {}
        
    def setup_environment(self):
        """初始化仿真环境"""
        print("\n" + "="*70)
        print("步骤 1: 初始化 BestMan 仿真环境")
        print("="*70)
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "Config", "default.yaml")
        
        try:
            cfg = load_config(config_path)
            cfg.Client.enable_GUI = True
            cfg.Client.enable_Debug = True
            
            self.client = Client(cfg.Client)
            print("   [OK] BestMan 客户端初始化成功")
            
            self.visualizer = Visualizer(self.client, cfg.Visualizer)
            print("   [OK] Visualizer 初始化成功")
            
        except Exception as e:
            print(f"   [ERROR] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def setup_scene(self):
        """创建测试场景"""
        print("\n" + "="*70)
        print("步骤 2: 创建协作场景")
        print("="*70)
        
        try:
            # 1. 创建客厅桌子
            table_id = self.client.load_object(
                obj_name="living_room_table",
                model_path="Asset/Scene/Object/URDF_models/furniture_table_rectangle_high/table.urdf",
                object_position=[0, 0, 0],
                object_orientation=[0, 0, 0, 1],
                scale=1.5,
                fixed_base=True
            )
            self.scene_objects['table'] = table_id
            print(f"   [OK] 创建客厅桌子 (ID: {table_id})")

            # 2. 创建柜子
            cabinet_id = self.client.load_object(
                obj_name="cabinet",
                model_path="Asset/Scene/Object/URDF_models/cabinate/cabinate_Dynahmrc.urdf",
                object_position=[-3.0, 2.0, 0],
                object_orientation=[0.5, 0.5, 0.5, 0.5],
                scale=1.0,
                fixed_base=True
            )
            self.scene_objects['cabinet'] = cabinet_id
            print(f"   [OK] 创建柜子 (ID: {cabinet_id})")
            
            # 3. 创建托盘
            tray_id = self.client.load_object(
                obj_name="tray",
                model_path="Asset/Scene/Object/URDF_models/clear_box/model.urdf",
                object_position=[0.2, 0.2, 1.4],
                object_orientation=[0, 0, 0, 1],
                scale=2.2,
                fixed_base=True
            )
            self.scene_objects['tray'] = tray_id
            print(f"   [OK] 创建托盘 (ID: {tray_id})")

            # 4. 创建物品（杯子）- 使用黄色杯子模型
            cup_id = self.client.load_object(
                obj_name="cup",
                model_path="Asset/Scene/Object/URDF_models/yellow_cup/model.urdf",
                object_position=[-2.9, 1.99, 2.1],
                object_orientation=[0, 0, 0, 1],
                scale=1.5,
                fixed_base=False
            )
            self.scene_objects['cup'] = cup_id
            print(f"   创建黄色杯子 (ID: {cup_id}) 在位置 [-2.9, 1.99, 3.1]")

            # 等待场景稳定
            for _ in range(100):
                self.client.run(1)
            
            return True
            
        except Exception as e:
            print(f"   [ERROR] 创建场景失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def setup_robots(self):
        """创建机器人"""
        print("\n" + "="*70)
        print("步骤 3: 创建协作机器人")
        print("="*70)
        
        try:
            self.robot_factory = RobotFactory(self.client, self.visualizer)
            
            # 1. 创建无人机 Lucy
            print("\n   [创建] 无人机 Lucy...")
            self.drone = self.robot_factory.create_robot(
                robot_id="Lucy",
                robot_type="drone",
                robot_model="drone",
                init_position=[0.0,0.0,0.0]
            )
            print(f"   [OK] 无人机 Lucy 创建成功")
            print(f"        位置: [0.0,0.0,0.0]")   
            print(f"        能力: navigation, pick, place, perception")
            
            # 2. 创建固定机械臂 Bob
            print("\n   [创建] 固定机械臂 Bob...")
            self.arm_robot = self.robot_factory.create_robot(
                robot_id="Bob",
                robot_type="arm",
                robot_model="panda",
                init_position=[-1.0, 0, 1.5]
            )
            print(f"   [OK] 固定机械臂 Bob 创建成功")
            print(f"        基座位置: [-1.3, 0, 1.3]")  
            print(f"        能力: manipulation, perception (固定基座，不可移动)")
            
            # 等待机器人初始化
            time.sleep(0.5)

            
            return True
            
        except Exception as e:
            print(f"   [ERROR] 创建机器人失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase1_drone_pick_and_place(self):
        """
        阶段 1: 无人机拾取并放置到桌子
        ==============================
        Lucy (Drone) 执行任务：
        1. 导航到物品位置
        2. 抓取物品
        3. 导航到桌子上方
        4. 放置物品
        """
        print("\n" + "="*70)
        print("阶段 1: 无人机 Lucy 执行任务 (Pick & Place to Table)")
        print("="*70)
        
        cup_id = self.scene_objects['cup']
        item_pos = [-2.9, 1.99, 2.1]
        table_pos = [0, 0, 0]
        
        # 步骤 1.1: 导航到物品上方
        print("\n   [1.1] Lucy 导航到物品上方...")
        approach_pos = [item_pos[0], item_pos[1], item_pos[2]]
        success, msg = self.drone.navigate_to(approach_pos)
        if success:
            print(f"   [OK] 导航成功: {msg}")
        else:
            print(f"   [ERROR] 导航失败: {msg}")
            return False
        
        # 步骤 1.2: 抓取物品
        print("\n   [1.2] Lucy 抓取物品...")
        success, msg = self.drone.pick("cup", item_pos)
        if success:
            print(f"   [OK] 抓取成功: {msg}")
        else:
            print(f"   [ERROR] 抓取失败: {msg}")
            return False
        
        # 等待抓取稳定
        time.sleep(1.0)
        
        # 步骤 1.3: 抬升到安全高度
        print("\n   [1.3] Lucy 抬升到安全高度...")
        safe_pos = [item_pos[0], item_pos[1], 4.0]
        success, msg = self.drone.navigate_to(safe_pos)
        if success:
            print(f"   [OK] 抬升成功")
        else:
            print(f"   [WARN] 抬升警告: {msg}")
        
        # 步骤 1.4: 导航到桌子上方
        print("\n   [1.4] Lucy 导航到桌子上方...")
        # 桌子高度约0.8米，放置高度约0.9米
        place_pos = [table_pos[0], table_pos[1], table_pos[2] + 0.1]
        hover_pos = [place_pos[0], place_pos[1], place_pos[2] + 0.4]
        
        success, msg = self.drone.navigate_to(hover_pos)
        if success:
            print(f"   [OK] 到达桌子上方: {msg}")
        else:
            print(f"   [ERROR] 导航失败: {msg}")
            return False
        
        # 步骤 1.5: 下降并放置
        print("\n   [1.5] Lucy 下降并放置物品...")
        success, msg = self.drone.place(place_pos)
        if success:
            print(f"   [OK] 放置成功: {msg}")
        else:
            print(f"   [ERROR] 放置失败: {msg}")
            return False
        
        # 步骤 1.6: 抬升离开
        print("\n   [1.6] Lucy 离开桌子...")
        leave_pos = [table_pos[0] + 1.0, table_pos[1], 1.5]
        success, msg = self.drone.navigate_to(leave_pos)
        if success:
            print(f"   [OK] Lucy 已离开工作区域")
        else:
            print(f"   [WARN] 离开警告: {msg}")
        
        print("\n   [DONE] 阶段 1 完成: 物品已放置在桌子上")
        return True
    
    def phase2_arm_pick_and_place(self):
        """
        阶段 2: 固定机械臂从桌子拾取并放置到托盘
        ===========================================
        Bob (Arm) 执行任务：
        1. 检测桌子上的物品
        2. 抓取物品
        3. 移动物品到托盘
        4. 放置物品
        """
        print("\n" + "="*70)
        print("阶段 2: 固定机械臂 Bob 执行任务 (Pick from Table & Place to Tray)")
        print("="*70)
        
        cup_id = self.scene_objects['cup']
        table_pos = [0, 0, 0]
        tray_pos = [0.0, 0.2, 1.0]
        
        # 等待物品稳定（无人机放置后可能有晃动）
        print("\n   [等待] 等待物品稳定...")
        for _ in range(50):
            self.client.run(1)
        print("   [OK] 物品已稳定")
        
        # 步骤 2.1: 获取物品当前位置
        print("\n   [2.1] Bob 检测物品位置...")
        try:
            obj_pos, obj_orn = self.client.get_object_pose(cup_id)
            print(f"   [OK] 检测到物品位置: {obj_pos}")
        except:
            # 备用：使用桌子中心位置
            obj_pos = [table_pos[0], table_pos[1], table_pos[2] + 0.05]
            print(f"   [INFO] 使用预估位置: {obj_pos}")
        
        # 步骤 2.2: 抓取物品
        print("\n   [2.2] Bob 抓取物品...")
        success = self.arm_robot.pick(cup_id)
        if success:
            print(f"   [OK] 抓取成功")
        else:
            print(f"   [ERROR] 抓取失败")
            return False
        
        # 等待抓取稳定
        time.sleep(0.5)
        for _ in range(50):
            self.client.run(1)
        
        # 步骤 2.3: 移动物品到托盘
        print("\n   [2.3] Bob 移动物品到托盘...")
        # 托盘上的放置位置（稍微高一点，确保物品落入托盘）
        tray_place_pos = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.15]
        
        # 先移动到预放置位置（托盘上方）
        pre_place_pos = [tray_place_pos[0], tray_place_pos[1], tray_place_pos[2] + 0.1]
        success = self.arm_robot.move_to_position(pre_place_pos)
        if success:
            print(f"   [OK] 到达预放置位置")
        else:
            print(f"   [WARN] 移动到预放置位置失败，继续尝试放置")
        
        # 步骤 2.4: 放置物品到托盘
        print("\n   [2.4] Bob 放置物品到托盘...")
        success = self.arm_robot.place(tray_place_pos)
        if success:
            print(f"   [OK] 放置成功")
        else:
            print(f"   [ERROR] 放置失败")
            return False
        
        # 步骤 2.5: 机械臂复位
        print("\n   [2.5] Bob 复位...")
        # 回到初始姿态
        home_pos = [-0.5, 0, 2.4]
        self.arm_robot.move_to_position(home_pos)
        print(f"   [OK] Bob 已复位")
        
        print("\n   [DONE] 阶段 2 完成: 物品已从桌子转移到托盘")
        return True
    
    def run_collaboration(self):
        """运行完整协作任务"""
        print("\n" + "#"*70)
        print("# " + " "*20 + "异构多机器人协作任务")
        print("# " + " "*15 + "无人机 + 固定机械臂协作演示")
        print("#"*70)
        
        print("\n任务描述:")
        print("  - Lucy (Drone): 空中运输，从物品位置 -> 客厅桌子")
        print("  - Bob (Arm):  桌面操作，从客厅桌子 -> 托盘")
        print("  - 物品: yellow_cup (黄色杯子)")
        
        # 阶段 1: 无人机任务
        if not self.phase1_drone_pick_and_place():
            print("\n   [FAILED] 阶段 1 失败，协作任务中断")
            return False
        
        # 阶段过渡
        print("\n" + "-"*70)
        print("阶段过渡: Lucy 完成空中运输，Bob 准备桌面操作")
        print("-"*70)
        time.sleep(1)
        
        # 阶段 2: 机械臂任务
        if not self.phase2_arm_pick_and_place():
            print("\n   [FAILED] 阶段 2 失败，协作任务中断")
            return False
        
        # 任务完成
        print("\n" + "#"*70)
        print("# " + " "*22 + "协作任务完成!")
        print("#"*70)
        print("\n任务总结:")
        print("  [✓] Lucy (Drone): 成功抓取物品并放置到桌子")
        print("  [✓] Bob (Arm):    成功从桌子抓取物品并放置到托盘")
        print("  [✓] 异构协作:     空中运输 + 桌面操作的完美配合")
        
        return True
    
    def cleanup(self):
        """清理资源"""
        print("\n" + "="*70)
        print("清理资源")
        print("="*70)
        
        if self.client:
            try:
                self.client.disconnect()
                print("   [OK] 已断开仿真连接")
            except:
                pass


def main():
    """主函数"""
    test = DroneArmCollaborationTest()
    
    try:
        # 1. 初始化环境
        if not test.setup_environment():
            print("环境初始化失败")
            return 1
        
        # 2. 创建场景
        if not test.setup_scene():
            print("场景创建失败")
            return 1
        
        # 3. 创建机器人
        if not test.setup_robots():
            print("机器人创建失败")
            return 1
        
        # 4. 运行协作任务
        success = test.run_collaboration()
        
        # 5. 保持仿真运行（便于观察结果）
        if success:
            print("\n保持仿真运行 5 秒以便观察结果...")
            time.sleep(5)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n用户中断")
        return 1
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        test.cleanup()


if __name__ == "__main__":
    exit(main())
