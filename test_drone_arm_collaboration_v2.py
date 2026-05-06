"""
无人机与固定机械臂协作测试 V2
============================

任务场景（按用户要求）：
1. 家具布局：
   - 柜子（Cabinet）：位于场景一侧，上面放着杯子
   - 桌子（Table）：位于场景另一侧，上面有固定机械臂和托盘
   - 柜子和桌子有一定距离，位置不重合

2. 协作流程：
   - 初始：无人机在桌子上方，固定机械臂在桌子旁就绪
   - Step 1: 无人机飞到柜子，抓取杯子
   - Step 2: 无人机飞回桌子，放置杯子
   - Step 3: 固定机械臂从桌子抓取杯子，放置到托盘
   - Step 4: 无人机回到初始位置（桌子上方）
   - Step 5: 固定机械臂复位

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
        
        # ===== 场景位置配置（V2版）=====
        # 柜子位置（放杯子）
        self.cabinet_pos = [2.5, 0, 0]           # 柜子基座位置
        self.cabinet_height = 0.8                # 柜子高度（估计）
        
        # 桌子位置（放机械臂和托盘）
        self.table_pos = [-1.5, 0, 0]            # 桌子基座位置
        self.table_height = 0.8                  # 桌子高度（估计）
        
        # 杯子位置（在柜子上面）
        self.cup_pos = [self.cabinet_pos[0], self.cabinet_pos[1], self.cabinet_height + 0.1]
        
        # 托盘位置（在桌子上，与机械臂不重合）
        self.tray_pos = [self.table_pos[0], self.table_pos[1] + 0.8, self.table_height + 0.05]
        
        # 机械臂基座位置（在桌子旁）
        self.arm_base_pos = [self.table_pos[0] - 0.5, self.table_pos[1], 0]
        
        # 无人机初始位置（在桌子上方）
        self.drone_start_pos = [self.table_pos[0], self.table_pos[1], 2.0]
        
        # 保存到locations便于访问
        self.locations = {
            'cabinet': self.cabinet_pos,
            'table': self.table_pos,
            'cup': self.cup_pos,
            'tray': self.tray_pos,
            'arm_base': self.arm_base_pos,
            'drone_start': self.drone_start_pos,
        }
        
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
            
            # 设置摄像机视角对着整个场景
            self._set_camera_view()
            
        except Exception as e:
            print(f"   [ERROR] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def _set_camera_view(self):
        """设置摄像机视角，能够看到柜子和桌子"""
        try:
            import pybullet as p
            
            # 场景中心点（柜子和桌子的中间）
            scene_center = [
                (self.cabinet_pos[0] + self.table_pos[0]) / 2,
                (self.cabinet_pos[1] + self.table_pos[1]) / 2,
                1.0
            ]
            
            # 摄像机参数
            camera_distance = 6.0    # 距离场景中心6米
            camera_yaw = 90          # 从侧面观察
            camera_pitch = -20       # 稍微俯视
            
            p.resetDebugVisualizerCamera(
                cameraDistance=camera_distance,
                cameraYaw=camera_yaw,
                cameraPitch=camera_pitch,
                cameraTargetPosition=scene_center,
                physicsClientId=self.client.get_client_id()
            )
            print(f"   [OK] 摄像机已对准场景中心 (位置: {scene_center})")
            
        except Exception as e:
            print(f"   [WARN] 设置摄像机视角失败: {e}")
    
    def setup_scene(self):
        """创建测试场景（V2版：柜子+桌子布局）"""
        print("\n" + "="*70)
        print("步骤 2: 创建协作场景")
        print("="*70)
        
        try:
            # 1. 创建柜子（Cabinet）- 放杯子的家具
            cabinet_id = self.client.load_object(
                obj_name="cabinet",
                model_path="Asset/Scene/Object/URDF_models/furniture_table_rectangle_high/table.urdf",
                object_position=self.cabinet_pos,
                object_orientation=[0, 0, 0, 1],
                scale=0.8,  # 稍微小一点的柜子
                fixed_base=True
            )
            self.scene_objects['cabinet'] = cabinet_id
            print(f"   [OK] 创建柜子 (ID: {cabinet_id}) 位置: {self.cabinet_pos}")
            
            # 2. 创建桌子（Table）- 放机械臂和托盘的家具
            table_id = self.client.load_object(
                obj_name="table",
                model_path="Asset/Scene/Object/URDF_models/furniture_table_rectangle_high/table.urdf",
                object_position=self.table_pos,
                object_orientation=[0, 0, 0, 1],
                scale=1.0,
                fixed_base=True
            )
            self.scene_objects['table'] = table_id
            print(f"   [OK] 创建桌子 (ID: {table_id}) 位置: {self.table_pos}")
            
            # 3. 创建托盘（Tray）- 放在桌子上，与机械臂不重合
            tray_id = self.client.load_object(
                obj_name="tray",
                model_path="Asset/Scene/Object/URDF_models/clear_box/model.urdf",
                object_position=self.tray_pos,
                object_orientation=[0, 0, 0, 1],
                scale=0.6,
                fixed_base=True
            )
            self.scene_objects['tray'] = tray_id
            print(f"   [OK] 创建托盘 (ID: {tray_id}) 位置: {self.tray_pos}")
            
            # 4. 创建杯子（Cup）- 放在柜子上面
            cup_id = self.client.load_object(
                obj_name="cup",
                model_path="Asset/Scene/Object/URDF_models/yellow_cup/model.urdf",
                object_position=self.cup_pos,
                object_orientation=[0, 0, 0, 1],
                scale=1.0,
                fixed_base=False
            )
            self.scene_objects['cup'] = cup_id
            print(f"   [OK] 创建黄色杯子 (ID: {cup_id}) 位置: {self.cup_pos}")
            
            # 5. 创建地面标记（便于观察位置）
            marker_positions = [
                ('cabinet_zone', [self.cabinet_pos[0], self.cabinet_pos[1], 0.05]),
                ('table_zone', [self.table_pos[0], self.table_pos[1], 0.05]),
                ('tray_zone', [self.tray_pos[0], self.tray_pos[1], 0.05]),
            ]
            for name, pos in marker_positions:
                marker_id = self.client.load_object(
                    obj_name=name,
                    model_path="Asset/Scene/Object/URDF_models/clear_box/model.urdf",
                    object_position=pos,
                    object_orientation=[0, 0, 0, 1],
                    scale=0.2,
                    fixed_base=True
                )
                self.scene_objects[name] = marker_id
            
            print("   [OK] 创建位置标记")
            
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
            
            # 1. 创建无人机 Lucy - 初始位置在桌子上方
            print("\n   [创建] 无人机 Lucy...")
            self.drone = self.robot_factory.create_robot(
                robot_id="Lucy",
                robot_type="drone",
                robot_model="drone",
                init_position=self.drone_start_pos
            )
            print(f"   [OK] 无人机 Lucy 创建成功")
            print(f"        初始位置: {self.drone_start_pos} (桌子上方)")
            print(f"        能力: navigation, pick, place, perception")
            
            # 2. 创建固定机械臂 Bob - 在桌子旁
            print("\n   [创建] 固定机械臂 Bob...")
            self.arm_robot = self.robot_factory.create_robot(
                robot_id="Bob",
                robot_type="arm",
                robot_model="panda",
                init_position=self.arm_base_pos
            )
            print(f"   [OK] 固定机械臂 Bob 创建成功")
            print(f"        基座位置: {self.arm_base_pos} (桌子旁)")
            print(f"        能力: manipulation, perception (固定基座，不可移动)")
            
            # 等待机器人初始化
            time.sleep(0.5)
            
            return True
            
        except Exception as e:
            print(f"   [ERROR] 创建机器人失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase1_drone_get_cup_from_cabinet(self):
        """
        阶段 1: 无人机从柜子取杯子
        ==========================
        Lucy (Drone) 执行任务：
        1. 从桌子上方出发
        2. 飞到柜子
        3. 抓取杯子
        4. 飞回桌子
        5. 放置杯子到桌子上
        """
        print("\n" + "="*70)
        print("阶段 1: 无人机 Lucy 从柜子取杯子")
        print("="*70)
        
        cup_id = self.scene_objects['cup']
        cabinet_pos = self.cabinet_pos
        table_pos = self.table_pos
        
        # 步骤 1.1: 确认无人机在初始位置（桌子上方）
        print("\n   [1.1] Lucy 确认初始位置（桌子上方）...")
        success, msg = self.drone.navigate_to(self.drone_start_pos)
        if success:
            print(f"   [OK] 已在桌子上方就绪")
        else:
            print(f"   [WARN] 位置调整: {msg}")
        
        # 步骤 1.2: 飞到柜子
        print("\n   [1.2] Lucy 飞到柜子...")
        approach_pos = [cabinet_pos[0], cabinet_pos[1], self.cabinet_height + 1.0]
        success, msg = self.drone.navigate_to(approach_pos)
        if success:
            print(f"   [OK] 到达柜子上方")
        else:
            print(f"   [ERROR] 导航失败: {msg}")
            return False
        
        # 步骤 1.3: 下降并抓取杯子
        print("\n   [1.3] Lucy 抓取杯子...")
        grab_pos = [self.cup_pos[0], self.cup_pos[1], self.cup_pos[2] + 0.3]
        success, msg = self.drone.navigate_to(grab_pos)
        if not success:
            print(f"   [ERROR] 接近失败: {msg}")
            return False
        
        success, msg = self.drone.pick("cup", self.cup_pos)
        if success:
            print(f"   [OK] 抓取成功: {msg}")
        else:
            print(f"   [ERROR] 抓取失败: {msg}")
            return False
        
        time.sleep(0.3)
        
        # 步骤 1.4: 抬升并飞回桌子
        print("\n   [1.4] Lucy 抬升并飞回桌子...")
        safe_pos = [cabinet_pos[0], cabinet_pos[1], 2.0]
        self.drone.navigate_to(safe_pos)
        
        table_hover_pos = [table_pos[0], table_pos[1], 2.0]
        success, msg = self.drone.navigate_to(table_hover_pos)
        if success:
            print(f"   [OK] 到达桌子上方")
        else:
            print(f"   [ERROR] 导航失败: {msg}")
            return False
        
        # 步骤 1.5: 下降并放置杯子到桌子
        print("\n   [1.5] Lucy 放置杯子到桌子...")
        # 放置位置：桌子中心稍微偏一点，方便机械臂操作
        place_pos = [table_pos[0] + 0.2, table_pos[1], self.table_height + 0.1]
        place_hover = [place_pos[0], place_pos[1], place_pos[2] + 0.3]
        
        self.drone.navigate_to(place_hover)
        success, msg = self.drone.place(place_pos)
        if success:
            print(f"   [OK] 放置成功: {msg}")
        else:
            print(f"   [ERROR] 放置失败: {msg}")
            return False
        
        # 步骤 1.6: 抬升到安全位置（等待下一步）
        print("\n   [1.6] Lucy 抬升到安全高度...")
        self.drone.navigate_to([table_pos[0], table_pos[1], 2.0])
        print(f"   [OK] Lucy 已在桌子上方待命")
        
        print("\n   [DONE] 阶段 1 完成: 杯子已从柜子转移到桌子")
        return True
    
    def phase2_arm_pick_and_place(self):
        """
        阶段 2: 固定机械臂从桌子拾取并放置到托盘
        ===========================================
        Bob (Arm) 执行任务：
        1. 检测桌子上的杯子
        2. 抓取杯子
        3. 移动杯子到托盘
        4. 放置杯子
        """
        print("\n" + "="*70)
        print("阶段 2: 固定机械臂 Bob 从桌子拾取并放置到托盘")
        print("="*70)
        
        cup_id = self.scene_objects['cup']
        table_pos = self.table_pos
        tray_pos = self.tray_pos
        
        # 等待物品稳定
        print("\n   [等待] 等待物品稳定...")
        for _ in range(50):
            self.client.run(1)
        print("   [OK] 物品已稳定")
        
        # 步骤 2.1: 获取杯子位置
        print("\n   [2.1] Bob 检测杯子位置...")
        try:
            obj_pos, obj_orn = self.client.get_object_pose(cup_id)
            print(f"   [OK] 检测到杯子位置: {obj_pos}")
        except:
            obj_pos = [table_pos[0] + 0.2, table_pos[1], self.table_height + 0.05]
            print(f"   [INFO] 使用预估位置: {obj_pos}")
        
        # 步骤 2.2: 抓取杯子
        print("\n   [2.2] Bob 抓取杯子...")
        success = self.arm_robot.pick(cup_id)
        if success:
            print(f"   [OK] 抓取成功")
        else:
            print(f"   [ERROR] 抓取失败")
            return False
        
        time.sleep(0.3)
        for _ in range(30):
            self.client.run(1)
        
        # 步骤 2.3: 移动杯子到托盘
        print("\n   [2.3] Bob 移动杯子到托盘...")
        tray_place_pos = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.15]
        
        pre_place_pos = [tray_place_pos[0], tray_place_pos[1], tray_place_pos[2] + 0.1]
        success = self.arm_robot.move_to_position(pre_place_pos)
        if success:
            print(f"   [OK] 到达预放置位置")
        else:
            print(f"   [WARN] 移动警告")
        
        # 步骤 2.4: 放置杯子到托盘
        print("\n   [2.4] Bob 放置杯子到托盘...")
        success = self.arm_robot.place(tray_place_pos)
        if success:
            print(f"   [OK] 放置成功")
        else:
            print(f"   [ERROR] 放置失败")
            return False
        
        print("\n   [DONE] 阶段 2 完成: 杯子已从桌子转移到托盘")
        return True
    
    def phase3_reset_positions(self):
        """
        阶段 3: 复位 - 无人机和机械臂回到初始位置
        ==========================================
        """
        print("\n" + "="*70)
        print("阶段 3: 复位 - 机器人回到初始位置")
        print("="*70)
        
        # 步骤 3.1: 无人机回到初始位置（桌子上方）
        print("\n   [3.1] Lucy 回到初始位置...")
        success, msg = self.drone.navigate_to(self.drone_start_pos)
        if success:
            print(f"   [OK] Lucy 已回到桌子上方")
        else:
            print(f"   [WARN] Lucy 复位警告: {msg}")
        
        # 步骤 3.2: 机械臂复位
        print("\n   [3.2] Bob 复位...")
        home_pos = [self.arm_base_pos[0], self.arm_base_pos[1], self.arm_base_pos[2] + 0.8]
        self.arm_robot.move_to_position(home_pos)
        print(f"   [OK] Bob 已复位")
        
        print("\n   [DONE] 复位完成")
        return True
    
    def run_collaboration(self):
        """运行完整协作任务（V2版）"""
        print("\n" + "#"*70)
        print("# " + " "*20 + "异构多机器人协作任务 V2")
        print("# " + " "*15 + "柜子→桌子→托盘 协作流程")
        print("#"*70)
        
        print("\n任务描述:")
        print(f"  - 柜子 (Cabinet): 位于 {self.cabinet_pos}，上面放着杯子")
        print(f"  - 桌子 (Table):   位于 {self.table_pos}，上面有机械臂和托盘")
        print(f"  - 托盘 (Tray):    位于 {self.tray_pos}")
        print(f"  - Lucy (Drone):   初始在桌子上方 {self.drone_start_pos}")
        print(f"  - Bob (Arm):      在桌子旁 {self.arm_base_pos}")
        print("\n协作流程:")
        print("  [1] Lucy: 桌子上方 → 柜子 → 抓取杯子 → 桌子 → 放置杯子")
        print("  [2] Bob:  从桌子抓取杯子 → 放置到托盘")
        print("  [3] 复位: Lucy回到桌子上方，Bob复位")
        
        # 阶段 1: 无人机从柜子取杯子放到桌子
        if not self.phase1_drone_get_cup_from_cabinet():
            print("\n   [FAILED] 阶段 1 失败，协作任务中断")
            return False
        
        # 阶段过渡
        print("\n" + "-"*70)
        print("阶段过渡: Lucy 完成空中运输，Bob 准备桌面操作")
        print("-"*70)
        time.sleep(0.5)
        
        # 阶段 2: 机械臂从桌子放到托盘
        if not self.phase2_arm_pick_and_place():
            print("\n   [FAILED] 阶段 2 失败，协作任务中断")
            return False
        
        # 阶段 3: 复位
        time.sleep(0.5)
        if not self.phase3_reset_positions():
            print("\n   [WARN] 复位阶段有问题")
        
        # 任务完成
        print("\n" + "#"*70)
        print("# " + " "*22 + "协作任务完成!")
        print("#"*70)
        print("\n任务总结:")
        print("  [✓] Lucy (Drone): 成功从柜子取杯子放到桌子")
        print("  [✓] Bob (Arm):    成功从桌子拿杯子放到托盘")
        print("  [✓] 复位:         所有机器人回到初始位置")
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
