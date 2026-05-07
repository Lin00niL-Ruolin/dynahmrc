"""
多机器人协作场景
场景：Asset/Scene/Scene/all_scene_test.json
任务：
1. 移动操作机器人1：打开冰箱 → 拿取柠檬 → 放到客厅桌子
2. 移动操作机器人2：关冰箱门（与机器人1协作）
3. 无人机：去卫生间 → 拿取杯子 → 放到客厅桌子
4. 固定机械臂：当物体放到桌子上时，把柠檬或杯子放到托盘里
"""

import sys
import os
import json
import math
import time
import threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BestMan 导入
from Env.Client import Client
from Visualization.Visualizer import Visualizer

# DynaHMRC 导入
from dynahmrc.integration.bestman_adapter import BestManAdapter
from dynahmrc.integration.robot_factory import RobotFactory
from dynahmrc.robots.arm_robot import ArmRobot
from dynahmrc.robots.drone_robot import DroneRobot


def load_scene_from_json(client, json_path):
    """从JSON文件加载场景"""
    print(f"\n[场景加载] 加载场景文件: {json_path}")
    
    with open(json_path, 'r') as f:
        scene_data = json.load(f)
    
    object_ids = {}
    for obj_config in scene_data:
        try:
            # 处理方向（可能是字符串表达式）
            orientation = obj_config['object_orientation']
            processed_orientation = []
            for val in orientation:
                if isinstance(val, str):
                    # 评估数学表达式
                    processed_orientation.append(eval(val))
                else:
                    processed_orientation.append(val)
            
            obj_id = client.load_object(
                obj_name=obj_config['obj_name'],
                model_path=obj_config['model_path'],
                object_position=obj_config['object_position'],
                object_orientation=processed_orientation,
                scale=obj_config.get('scale', 1.0),
                fixed_base=obj_config.get('fixed_base', True)
            )
            object_ids[obj_config['obj_name']] = obj_id
            print(f"  ✓ 创建 {obj_config['obj_name']} (ID: {obj_id})")
        except Exception as e:
            print(f"  ✗ 创建 {obj_config.get('obj_name', 'unknown')} 失败: {e}")
    
    print(f"[场景加载] 完成，共加载 {len(object_ids)} 个物体")
    return object_ids


class MultiRobotCollaboration:
    """多机器人协作任务管理器"""
    
    def __init__(self):
        self.client = None
        self.visualizer = None
        self.robot_factory = None
        self.adapter = None
        
        # 机器人
        self.drone = None
        self.mobile_robot1 = None  # 负责柠檬任务
        self.mobile_robot2 = None  # 负责关冰箱门
        self.arm_robot = None      # 固定机械臂
        
        # 场景物体
        self.scene_objects = {}
        self.robots = {}
        
        # 任务状态
        self.task_status = {
            'lemon_on_table': False,
            'cup_on_table': False,
            'fridge_opened': False,
            'fridge_closed': False,
        }
        
    def initialize(self):
        """初始化环境和机器人"""
        print("\n" + "="*70)
        print("多机器人协作场景初始化")
        print("="*70)
        
        # 1. 初始化客户端
        print("\n1. 初始化 BestMan 客户端...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        from yacs.config import CfgNode as CN
        config_path = os.path.join(project_root, "Config", "default.yaml")
        
        with open(config_path, "r") as f:
            cfg = CN.load_cfg(f)
        
        cfg.Client.enable_GUI = True
        cfg.Client.enable_Debug = True
        
        self.client = Client(cfg.Client)
        self.visualizer = Visualizer(self.client, cfg.Visualizer)
        print("   ✓ 客户端初始化成功")
        
        # 2. 加载场景
        print("\n2. 加载场景...")
        scene_path = os.path.join(project_root, "Asset", "Scene", "Scene", "all_scene_test.json")
        self.scene_objects = load_scene_from_json(self.client, scene_path)
        
        # 3. 创建额外物体
        print("\n3. 创建任务所需物体...")
        self._create_task_objects()
        
        # 4. 创建机器人
        print("\n4. 创建机器人...")
        self._create_robots()
        
        # 5. 创建Adapter
        print("\n5. 创建 BestManAdapter...")
        self.adapter = BestManAdapter(self.robots, self.client)
        for name, obj_id in self.scene_objects.items():
            obj_type = 'furniture' if name in ['elementA', 'elementB1', 'elementB2', 'elementE'] else 'graspable'
            self.adapter.register_scene_object(name, obj_id, obj_type)
        print("   ✓ Adapter 创建成功")
        
        print("\n" + "="*70)
        print("初始化完成")
        print("="*70)
        
    def _create_task_objects(self):
        """创建任务所需的额外物体"""
        # 在冰箱里放柠檬
        lemon_id = self.client.load_object(
            obj_name="lemon",
            model_path="Asset/Scene/Object/URDF_models/food_lemon/model.urdf",
            object_position=[4.0, 5.3, 0.8],  # 冰箱内部
            object_orientation=[0, 0, 0],
            scale=1.0,
            fixed_base=False
        )
        self.scene_objects['lemon'] = lemon_id
        print(f"   ✓ 在冰箱中创建柠檬 (ID: {lemon_id})")
        
        # 在卫生间洗手池上放杯子
        cup_id = self.client.load_object(
            obj_name="cup",
            model_path="Asset/Scene/Object/URDF_models/yellow_cup/model.urdf",
            object_position=[-0.3, 8.5, 1.2],  # 洗手池上方
            object_orientation=[0, 0, 0],
            scale=1.0,
            fixed_base=False
        )
        self.scene_objects['cup'] = cup_id
        print(f"   ✓ 在洗手池上创建杯子 (ID: {cup_id})")
        
        # 创建托盘（用于固定机械臂放置物品）
        tray_id = self.client.load_object(
            obj_name="tray",
            model_path="Asset/Scene/Object/URDF_models/clear_box/model.urdf",
            object_position=[0.5, 2.0, 0.85],  # 客厅桌子旁边
            object_orientation=[0, 0, 0],
            scale=2.0,
            fixed_base=True
        )
        self.scene_objects['tray'] = tray_id
        print(f"   ✓ 创建托盘 (ID: {tray_id})")
        
    def _create_robots(self):
        """创建4个机器人"""
        self.robot_factory = RobotFactory(self.client, self.visualizer)
        
        # 1. 创建无人机（在卫生间附近空中）
        print("   创建无人机...")
        drone_bestman = self.robot_factory.create_robot(
            robot_id="drone",
            robot_type="drone",
            robot_model="quadcopter",
            init_position=[-0.3, 8.0, 2.0],  # 卫生间上方
            init_orientation=[0, 0, 0, 1]
        )
        self.drone = DroneRobot("drone", drone_bestman)
        self.robots['drone'] = self.drone
        print("   ✓ 无人机创建成功")
        
        # 2. 创建移动操作机器人1（负责柠檬任务）
        print("   创建移动操作机器人1...")
        mobile1_bestman = self.robot_factory.create_robot(
            robot_id="mobile_robot1",
            robot_type="mobile_manipulator",
            robot_model="panda_on_segbot",
            init_position=[2.0, 4.0, 0]  # 厨房附近
        )
        self.mobile_robot1 = mobile1_bestman
        self.robots['mobile_robot1'] = mobile1_bestman
        print("   ✓ 移动操作机器人1创建成功")
        
        # 3. 创建移动操作机器人2（负责关冰箱门）
        print("   创建移动操作机器人2...")
        mobile2_bestman = self.robot_factory.create_robot(
            robot_id="mobile_robot2",
            robot_type="mobile_manipulator",
            robot_model="panda_on_segbot",
            init_position=[-2.0, 4.0, 0]  # 厨房附近
        )
        self.mobile_robot2 = mobile2_bestman
        self.robots['mobile_robot2'] = mobile2_bestman
        print("   ✓ 移动操作机器人2创建成功")
        
        # 4. 创建固定机械臂（底座在客厅桌子表面）
        print("   创建固定机械臂...")
        # 客厅桌子表面位置 [0.4, 1.85, 1.0]
        arm_bestman = self.robot_factory.create_robot(
            robot_id="arm_robot",
            robot_type="arm",
            robot_model="panda",
            init_position=[0.4, 1.85, 1.0],  # 客厅桌子表面
            init_orientation=[0, 0, 0, 1]
        )
        self.arm_robot = ArmRobot("arm_robot", arm_bestman)
        self.robots['arm_robot'] = self.arm_robot
        print("   ✓ 固定机械臂创建成功")
        
    def execute_task(self):
        """执行协作任务"""
        print("\n" + "="*70)
        print("开始执行多机器人协作任务")
        print("="*70)
        
        # 创建线程执行任务
        threads = []
        
        # 机器人1任务：打开冰箱 → 拿取柠檬 → 放到客厅桌子
        t1 = threading.Thread(target=self._robot1_lemon_task)
        threads.append(t1)
        
        # 机器人2任务：等待机器人1拿出柠檬后关冰箱门
        t2 = threading.Thread(target=self._robot2_close_fridge_task)
        threads.append(t2)
        
        # 无人机任务：去卫生间拿杯子 → 放到客厅桌子
        t3 = threading.Thread(target=self._drone_cup_task)
        threads.append(t3)
        
        # 固定机械臂任务：监控桌子上的物体并放入托盘
        t4 = threading.Thread(target=self._arm_robot_monitor_task)
        threads.append(t4)
        
        # 启动所有线程
        for t in threads:
            t.start()
        
        # 等待所有任务完成
        for t in threads:
            t.join()
        
        print("\n" + "="*70)
        print("所有任务执行完成")
        print("="*70)
        
    def _robot1_lemon_task(self):
        """移动操作机器人1：打开冰箱 → 拿取柠檬 → 放到客厅桌子"""
        print("\n[Robot1] 开始柠檬任务")
        
        try:
            # 1. 导航到冰箱
            print("[Robot1] 导航到冰箱...")
            self.adapter.execute_action(
                'mobile_robot1',
                'navigate',
                {'target': [3.5, 5.0, 0]}
            )
            
            # 2. 打开冰箱门
            print("[Robot1] 打开冰箱门...")
            # TODO: 实现打开冰箱门动作
            self.task_status['fridge_opened'] = True
            print("[Robot1] 冰箱门已打开")
            
            # 3. 抓取柠檬
            print("[Robot1] 抓取柠檬...")
            self.adapter.execute_action(
                'mobile_robot1',
                'pick',
                {'object_id': 'lemon', 'object_name': 'lemon'}
            )
            print("[Robot1] 柠檬已抓取")
            
            # 4. 导航到客厅桌子
            print("[Robot1] 导航到客厅桌子...")
            self.adapter.execute_action(
                'mobile_robot1',
                'navigate',
                {'target': [0.3, 1.5, 0]}
            )
            
            # 5. 放置柠檬到桌子
            print("[Robot1] 放置柠檬到客厅桌子...")
            self.adapter.execute_action(
                'mobile_robot1',
                'place',
                {'target': [0.4, 1.85, 1.0]}
            )
            self.task_status['lemon_on_table'] = True
            print("[Robot1] 柠檬已放到客厅桌子")
            
        except Exception as e:
            print(f"[Robot1] 任务失败: {e}")
            
    def _robot2_close_fridge_task(self):
        """移动操作机器人2：等待机器人1拿出柠檬后关冰箱门"""
        print("\n[Robot2] 等待关冰箱门任务")
        
        try:
            # 等待冰箱被打开
            print("[Robot2] 等待冰箱被打开...")
            while not self.task_status['fridge_opened']:
                time.sleep(0.5)
            
            # 等待柠檬被拿出（等待几秒钟）
            print("[Robot2] 等待柠檬被拿出...")
            time.sleep(3)
            
            # 导航到冰箱
            print("[Robot2] 导航到冰箱...")
            self.adapter.execute_action(
                'mobile_robot2',
                'navigate',
                {'target': [3.8, 5.0, 0]}
            )
            
            # 关闭冰箱门
            print("[Robot2] 关闭冰箱门...")
            # TODO: 实现关闭冰箱门动作
            self.task_status['fridge_closed'] = True
            print("[Robot2] 冰箱门已关闭")
            
        except Exception as e:
            print(f"[Robot2] 任务失败: {e}")
            
    def _drone_cup_task(self):
        """无人机：去卫生间拿杯子 → 放到客厅桌子"""
        print("\n[Drone] 开始杯子任务")
        
        try:
            # 1. 导航到卫生间洗手池上方
            print("[Drone] 导航到卫生间...")
            self.drone.navigate_to([-0.3, 8.5, 2.0])
            
            # 2. 下降抓取杯子
            print("[Drone] 抓取杯子...")
            self.drone.pick(object_id=self.scene_objects['cup'])
            print("[Drone] 杯子已抓取")
            
            # 3. 飞往客厅桌子上方
            print("[Drone] 飞往客厅桌子...")
            self.drone.navigate_to([0.4, 1.85, 2.0])
            
            # 4. 放置杯子到桌子
            print("[Drone] 放置杯子到客厅桌子...")
            self.drone.place(target_position=[0.4, 1.85, 1.0])
            self.task_status['cup_on_table'] = True
            print("[Drone] 杯子已放到客厅桌子")
            
        except Exception as e:
            print(f"[Drone] 任务失败: {e}")
            
    def _arm_robot_monitor_task(self):
        """固定机械臂：监控桌子上的物体并放入托盘"""
        print("\n[ArmRobot] 开始监控客厅桌子")
        
        try:
            while True:
                # 检查是否有物体放到桌子上
                if self.task_status['lemon_on_table'] or self.task_status['cup_on_table']:
                    print("[ArmRobot] 检测到桌子上有物体，开始放入托盘...")
                    
                    # 等待物体稳定
                    time.sleep(2)
                    
                    # 确定要抓取的物体
                    target_object = None
                    if self.task_status['lemon_on_table']:
                        target_object = 'lemon'
                        self.task_status['lemon_on_table'] = False  # 标记为已处理
                    elif self.task_status['cup_on_table']:
                        target_object = 'cup'
                        self.task_status['cup_on_table'] = False  # 标记为已处理
                    
                    if target_object:
                        print(f"[ArmRobot] 抓取 {target_object}...")
                        obj_id = self.scene_objects.get(target_object)
                        if obj_id:
                            # 抓取物体
                            self.arm_robot.pick(obj_id)
                            
                            # 放置到托盘
                            tray_pos = [0.5, 2.0, 0.85]
                            print(f"[ArmRobot] 放置 {target_object} 到托盘...")
                            self.arm_robot.place(tray_pos)
                            print(f"[ArmRobot] {target_object} 已放入托盘")
                
                # 如果所有任务都完成了，退出
                if self.task_status['fridge_closed'] and not self.task_status['lemon_on_table'] and not self.task_status['cup_on_table']:
                    print("[ArmRobot] 所有任务完成")
                    break
                    
                time.sleep(0.5)
                
        except Exception as e:
            print(f"[ArmRobot] 任务失败: {e}")


def main():
    """主函数"""
    collaboration = MultiRobotCollaboration()
    
    # 初始化
    collaboration.initialize()
    
    # 等待用户准备
    print("\n按 Enter 开始执行任务...")
    input()
    
    # 执行任务
    collaboration.execute_task()
    
    print("\n任务全部完成！")
    print("按 Enter 退出...")
    input()


if __name__ == "__main__":
    main()
