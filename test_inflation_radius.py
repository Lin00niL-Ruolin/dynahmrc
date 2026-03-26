"""
膨胀半径可视化测试
加载场景、机器人和物品，并用颜色标记所有膨胀半径
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BestMan 导入
from Env.Client import Client
from Visualization.Visualizer import Visualizer
from dynahmrc.integration.robot_factory import RobotFactory
from dynahmrc.utils.path_planning import PathPlanner
import pybullet as p


def load_config(config_path: str = "Config/default.yaml"):
    """加载配置文件"""
    from yacs.config import CfgNode as CN
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    with open(config_path, "r") as f:
        cfg = CN.load_cfg(f)
    
    return cfg


class InflationRadiusVisualizer:
    """膨胀半径可视化器"""
    
    def __init__(self):
        self.client = None
        self.visualizer = None
        self.robot_factory = None
        self.path_planner = None
        self.scene_objects = {}
        self.robots = {}
        
        # 颜色配置 - 不同膨胀半径用不同颜色
        self.colors = {
            'small': [0, 1, 0, 0.3],      # 绿色 - 小半径
            'medium': [1, 1, 0, 0.3],     # 黄色 - 中等半径
            'large': [1, 0.5, 0, 0.3],    # 橙色 - 大半径
            'xlarge': [1, 0, 0, 0.3],     # 红色 - 超大半径
            'robot': [0, 0, 1, 0.5],      # 蓝色 - 机器人本身
        }
        
        self.visual_shapes = []  # 存储可视化形状ID
        
    def setup_environment(self, gui=True):
        """设置仿真环境"""
        print("\n" + "="*60)
        print("1. 初始化 BestMan 仿真环境")
        print("="*60)
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "Config", "default.yaml")
        
        try:
            cfg = load_config(config_path)
            cfg.Client.enable_GUI = gui
            cfg.Client.enable_Debug = True
            
            self.client = Client(cfg.Client)
            print("   ✓ BestMan 客户端初始化成功")
            
            self.visualizer = Visualizer(self.client, cfg.Visualizer)
            print("   ✓ Visualizer 初始化成功")
            
            # 初始化路径规划器
            self.path_planner = PathPlanner(
                client_id=self.client
            )
            print("   ✓ 路径规划器初始化成功")
            
            # 设置机器人半径
            self.robot_radius = 0.3
            
        except Exception as e:
            print(f"   [ERROR] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def load_scene(self, scene_config):
        """加载场景配置"""
        print("\n" + "="*60)
        print("2. 加载场景和物体")
        print("="*60)
        
        try:
            # 检查是否有 JSON 场景文件路径
            scene_path = scene_config.get('scene_path')
            if scene_path:
                # 从 JSON 文件加载场景
                return self._load_scene_from_json(scene_path)
            
            # 否则从配置字典加载
            # 加载地面
            if scene_config.get('ground'):
                ground = scene_config['ground']
                ground_id = self.client.load_object(
                    obj_name="ground",
                    model_path=ground['model_path'],
                    object_position=ground['position'],
                    object_orientation=ground.get('orientation', [0, 0, 0, 1]),
                    scale=ground.get('scale', 1.0),
                    fixed_base=True
                )
                self.scene_objects['ground'] = {
                    'id': ground_id,
                    'type': 'ground',
                    'position': ground['position'],
                    'radius': 0  # 地面不需要膨胀
                }
                print(f"   ✓ 加载地面 (ID: {ground_id})")
            
            # 加载家具/障碍物
            for obj_name, obj_config in scene_config.get('furniture', {}).items():
                obj_id = self.client.load_object(
                    obj_name=obj_name,
                    model_path=obj_config['model_path'],
                    object_position=obj_config['position'],
                    object_orientation=obj_config.get('orientation', [0, 0, 0, 1]),
                    scale=obj_config.get('scale', 1.0),
                    fixed_base=obj_config.get('fixed_base', True)
                )
                
                # 获取物体半径（从配置或估算）
                obj_radius = obj_config.get('radius', 0.3)
                
                self.scene_objects[obj_name] = {
                    'id': obj_id,
                    'type': 'furniture',
                    'position': obj_config['position'],
                    'radius': obj_radius
                }
                print(f"   ✓ 加载家具 {obj_name} (ID: {obj_id}, 半径: {obj_radius}m)")
            
            # 加载可抓取物品
            for obj_name, obj_config in scene_config.get('objects', {}).items():
                obj_id = self.client.load_object(
                    obj_name=obj_name,
                    model_path=obj_config['model_path'],
                    object_position=obj_config['position'],
                    object_orientation=obj_config.get('orientation', [0, 0, 0, 1]),
                    scale=obj_config.get('scale', 1.0),
                    fixed_base=False
                )
                
                obj_radius = obj_config.get('radius', 0.1)
                
                self.scene_objects[obj_name] = {
                    'id': obj_id,
                    'type': 'object',
                    'position': obj_config['position'],
                    'radius': obj_radius
                }
                print(f"   ✓ 加载物品 {obj_name} (ID: {obj_id}, 半径: {obj_radius}m)")
            
            return True
            
        except Exception as e:
            print(f"   [ERROR] 加载场景失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_scene_from_json(self, scene_path):
        """从 JSON 文件加载场景"""
        import json
        
        try:
            # 获取项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            if not os.path.isabs(scene_path):
                scene_path = os.path.join(project_root, scene_path)
            
            print(f"   加载场景文件: {scene_path}")
            
            with open(scene_path, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)
            
            # 加载场景中的物体
            objects = scene_data.get('objects', [])
            print(f"   场景中有 {len(objects)} 个物体")
            
            for obj in objects:
                obj_name = obj.get('name', 'unknown')
                obj_type = obj.get('type', 'object')
                model_path = obj.get('model_path', '')
                position = obj.get('position', [0, 0, 0])
                orientation = obj.get('orientation', [0, 0, 0, 1])
                scale = obj.get('scale', 1.0)
                fixed = obj.get('fixed', True)
                
                # 估算物体半径（从 size 或 scale）
                size = obj.get('size', [0.1, 0.1, 0.1])
                if isinstance(size, list) and len(size) >= 2:
                    obj_radius = max(size[0], size[1]) / 2.0
                else:
                    obj_radius = 0.1 * scale
                
                # 加载物体
                if model_path:
                    # 处理相对路径
                    if not os.path.isabs(model_path):
                        model_path = os.path.join(project_root, model_path)
                    
                    try:
                        obj_id = self.client.load_object(
                            obj_name=obj_name,
                            model_path=model_path,
                            object_position=position,
                            object_orientation=orientation,
                            scale=scale,
                            fixed_base=fixed
                        )
                        
                        # 根据类型分类
                        if obj_type in ['furniture', 'obstacle']:
                            obj_category = 'furniture'
                        elif obj_type in ['graspable', 'object', 'item']:
                            obj_category = 'object'
                        else:
                            obj_category = 'object'
                        
                        self.scene_objects[obj_name] = {
                            'id': obj_id,
                            'type': obj_category,
                            'position': position,
                            'radius': obj_radius
                        }
                        
                        print(f"   ✓ 加载 {obj_category} {obj_name} (ID: {obj_id}, 半径: {obj_radius:.2f}m)")
                        
                    except Exception as e:
                        print(f"   [警告] 加载物体 {obj_name} 失败: {e}")
            
            print(f"   ✓ 从 JSON 加载场景完成，共 {len(self.scene_objects)} 个物体")
            return True
            
        except Exception as e:
            print(f"   [ERROR] 从 JSON 加载场景失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_robots(self, robot_configs):
        """加载机器人"""
        print("\n" + "="*60)
        print("3. 加载机器人")
        print("="*60)
        
        try:
            self.robot_factory = RobotFactory(self.client, self.visualizer)
            
            for config in robot_configs:
                robot_id = config['robot_id']
                robot_type = config['robot_type']
                robot_model = config['robot_model']
                init_position = config.get('init_position', [0, 0, 0])
                init_orientation = config.get('init_orientation', [0, 0, 0, 1])
                
                robot = self.robot_factory.create_robot(
                    robot_id=robot_id,
                    robot_type=robot_type,
                    robot_model=robot_model,
                    init_position=init_position,
                    init_orientation=init_orientation
                )
                
                # 获取机器人半径
                robot_radius = getattr(robot, 'robot_radius', 0.3)
                if hasattr(robot, 'path_planner') and robot.path_planner:
                    robot_radius = getattr(robot.path_planner, 'robot_radius', robot_radius)
                
                self.robots[robot_id] = {
                    'instance': robot,
                    'type': robot_type,
                    'position': init_position,
                    'radius': robot_radius
                }
                
                print(f"   ✓ 加载机器人 {robot_id} (类型: {robot_type}, 半径: {robot_radius}m)")
            
            return True
            
        except Exception as e:
            print(f"   [ERROR] 加载机器人失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_inflation_radius(self, obj_radius, robot_radius=None):
        """计算膨胀半径"""
        if robot_radius is None:
            robot_radius = getattr(self, 'robot_radius', 0.3)
        
        # 与 path_planning.py 中相同的计算方式
        total_radius = robot_radius + obj_radius * 0.5 + 0.05
        return total_radius
    
    def get_color_for_radius(self, radius):
        """根据半径大小返回颜色"""
        if radius < 0.3:
            return self.colors['small']
        elif radius < 0.5:
            return self.colors['medium']
        elif radius < 0.8:
            return self.colors['large']
        else:
            return self.colors['xlarge']
    
    def visualize_inflation_radii(self):
        """可视化所有膨胀半径"""
        print("\n" + "="*60)
        print("4. 可视化膨胀半径")
        print("="*60)
        
        try:
            # 清除之前的可视化
            self.clear_visualization()
            
            robot_radius = getattr(self, 'robot_radius', 0.3)
            print(f"\n   机器人半径: {robot_radius}m")
            print(f"   膨胀半径计算公式: total_radius = robot_radius + obj_radius * 0.5 + 0.05")
            print()
            
            # 可视化场景物体
            print("   场景物体膨胀半径:")
            for obj_name, obj_info in self.scene_objects.items():
                if obj_info['type'] == 'ground':
                    continue
                
                position = obj_info['position']
                obj_radius = obj_info['radius']
                inflation_radius = self.calculate_inflation_radius(obj_radius)
                color = self.get_color_for_radius(inflation_radius)
                
                # 创建膨胀区域可视化（半透明圆柱体）
                visual_id = self._create_cylinder_visual(
                    position=[position[0], position[1], 0.05],
                    radius=inflation_radius,
                    height=0.1,
                    color=color
                )
                
                self.visual_shapes.append(visual_id)
                
                print(f"      {obj_name}: 物体半径={obj_radius:.2f}m, 膨胀半径={inflation_radius:.2f}m")
            
            # 可视化机器人
            print("\n   机器人膨胀半径:")
            for robot_id, robot_info in self.robots.items():
                position = robot_info['position']
                robot_r = robot_info['radius']
                inflation_radius = self.calculate_inflation_radius(0, robot_r)  # 物体半径为0
                
                # 机器人用蓝色标记
                visual_id = self._create_cylinder_visual(
                    position=[position[0], position[1], 0.15],
                    radius=inflation_radius,
                    height=0.2,
                    color=self.colors['robot']
                )
                
                self.visual_shapes.append(visual_id)
                
                print(f"      {robot_id}: 机器人半径={robot_r:.2f}m, 膨胀半径={inflation_radius:.2f}m")
            
            # 添加图例
            self._add_legend()
            
            print("\n   ✓ 膨胀半径可视化完成")
            print(f"   共创建 {len(self.visual_shapes)} 个可视化形状")
            
            return True
            
        except Exception as e:
            print(f"   [ERROR] 可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_cylinder_visual(self, position, radius, height, color):
        """创建圆柱体可视化"""
        # 创建视觉形状
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=color
        )
        
        # 创建多体（仅视觉，无物理）
        multi_body_id = p.createMultiBody(
            baseMass=0,  # 无质量
            baseCollisionShapeIndex=-1,  # 无碰撞形状
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1]
        )
        
        return multi_body_id
    
    def _add_legend(self):
        """添加颜色图例"""
        legend_text = [
            "Inflation Radius Legend:",
            "Green: < 0.3m (Small)",
            "Yellow: 0.3-0.5m (Medium)",
            "Orange: 0.5-0.8m (Large)",
            "Red: > 0.8m (XLarge)",
            "Blue: Robot"
        ]
        
        print("\n   颜色图例:")
        for text in legend_text:
            print(f"      {text}")
    
    def clear_visualization(self):
        """清除所有可视化"""
        for shape_id in self.visual_shapes:
            try:
                p.removeBody(shape_id)
            except:
                pass
        self.visual_shapes.clear()
        print("   已清除之前的可视化")
    
    def update_obstacles_in_planner(self):
        """更新障碍物到路径规划器"""
        print("\n" + "="*60)
        print("5. 更新路径规划器障碍物")
        print("="*60)
        
        try:
            obstacle_positions = []
            obstacle_sizes = []
            
            for obj_name, obj_info in self.scene_objects.items():
                if obj_info['type'] != 'ground':
                    obstacle_positions.append(obj_info['position'])
                    obstacle_sizes.append(obj_info['radius'])
            
            self.path_planner.update_obstacles_from_scene(
                self.scene_objects,
                robot_type='mobile_manipulator'
            )
            
            print(f"   ✓ 已更新 {len(obstacle_positions)} 个障碍物到路径规划器")
            
            return True
            
        except Exception as e:
            print(f"   [ERROR] 更新障碍物失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_test(self, scene_config, robot_configs):
        """运行完整测试"""
        print("\n" + "="*60)
        print("膨胀半径可视化测试")
        print("="*60)
        
        # 设置环境
        if not self.setup_environment(gui=True):
            return False
        
        # 加载场景
        if not self.load_scene(scene_config):
            return False
        
        # 加载机器人
        if not self.load_robots(robot_configs):
            return False
        
        # 更新障碍物到规划器
        if not self.update_obstacles_in_planner():
            return False
        
        # 可视化膨胀半径
        if not self.visualize_inflation_radii():
            return False
        
        print("\n" + "="*60)
        print("测试完成！按 Ctrl+C 退出")
        print("="*60)
        
        # 保持运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n   退出测试")
        
        return True


def main():
    """主函数"""
    # 场景配置 - 使用 JSON 场景文件
    scene_config = {
        "config_path": "Config/default.yaml",
        "gui": True,
        "scene_path": "Asset/Scene/Scene/all_scene_test.json"  # JSON场景文件路径
    }

    # 机器人配置
    robot_configs = [
        {
            "robot_id": "alice",
            "robot_type": "arm",
            "robot_model": "panda",
            "init_position": [0.2, 1.6, 0.1],
            "init_orientation": [0, 0, 0, 1],
            "capabilities": ["manipulation", "perception"]
        },
        {
            "robot_id": "bob",
            "robot_type": "mobile_manipulator",
            "robot_model": "panda_on_segbot",
            "init_position": [1.2, 2.9, 0],
            "init_orientation": [0, 0, -1, 0],  # 朝向装配区
            "capabilities": ["navigation", "manipulation", "transport", "perception"]
        },
        {
            "robot_id": "david",
            "robot_type": "drone",
            "robot_model": "drone",
            "init_position": [-0.5, 0.5, 1.1],
            "init_orientation": [0, 0, 1, 0],  # 朝向另一侧
            "capabilities": ["manipulation", "transport", "perception"]
        }
    ]
    
    # 创建可视化器并运行测试
    visualizer = InflationRadiusVisualizer()
    visualizer.run_test(scene_config, robot_configs)


if __name__ == "__main__":
    main()
