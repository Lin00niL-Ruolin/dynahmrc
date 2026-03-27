"""
碰撞半径测试与可视化
加载场景、机器人和物品，检测并可视化碰撞
"""

import sys
import os
import time
import math

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


class CollisionRadiusVisualizer:
    """碰撞半径可视化器"""
    
    def __init__(self):
        self.client = None
        self.visualizer = None
        self.robot_factory = None
        self.path_planner = None
        self.scene_objects = {}
        self.robots = {}
        
        # 颜色配置
        self.colors = {
            'collision': [1, 0, 0, 0.5],      # 红色 - 碰撞区域
            'safe': [0, 1, 0, 0.3],           # 绿色 - 安全距离
            'warning': [1, 1, 0, 0.4],        # 黄色 - 警告距离
            'robot': [0, 0, 1, 0.5],          # 蓝色 - 机器人
            'obstacle': [0.5, 0.5, 0.5, 0.3], # 灰色 - 障碍物
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
                    'radius': 0
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
    
    def _eval_orientation(self, orientation):
        """处理朝向中的数学表达式"""
        import math
        
        if isinstance(orientation, list):
            result = []
            for val in orientation:
                if isinstance(val, str):
                    try:
                        # 安全地评估数学表达式
                        val = eval(val, {"__builtins__": {}}, {"math": math})
                    except:
                        val = 0.0
                result.append(float(val))
            return result
        return [0, 0, 0, 1]
    
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
            
            # 处理不同的 JSON 结构
            # 如果是字典，获取 objects 列表；如果是列表，直接使用
            if isinstance(scene_data, dict):
                objects = scene_data.get('objects', [])
            elif isinstance(scene_data, list):
                objects = scene_data
            else:
                print(f"   [警告] 未知的场景数据格式: {type(scene_data)}")
                objects = []
            
            print(f"   场景中有 {len(objects)} 个物体")
            
            for obj in objects:
                # 支持多种字段名格式
                obj_name = obj.get('obj_name') or obj.get('name', 'unknown')
                obj_type = obj.get('type', 'object')
                model_path = obj.get('model_path') or obj.get('urdf_path', '')
                
                # 位置字段可能是 position 或 object_position
                position = obj.get('object_position') or obj.get('position', [0, 0, 0])
                
                # 朝向字段可能是 orientation 或 object_orientation
                orientation = obj.get('object_orientation') or obj.get('orientation', [0, 0, 0, 1])
                
                # 处理朝向中的数学表达式（如 "math.pi"）
                orientation = self._eval_orientation(orientation)
                
                scale = obj.get('scale', 1.0)
                fixed = obj.get('fixed_base') or obj.get('fixed', True)
                
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
    
    def check_collisions(self):
        """检测碰撞并返回碰撞信息"""
        collisions = []
        
        for robot_id, robot_info in self.robots.items():
            robot_pos = robot_info['position']
            robot_radius = robot_info['radius']
            
            for obj_name, obj_info in self.scene_objects.items():
                if obj_info['type'] == 'ground':
                    continue
                
                obj_pos = obj_info['position']
                obj_radius = obj_info['radius']
                
                # 计算距离
                distance = math.sqrt(
                    (robot_pos[0] - obj_pos[0]) ** 2 +
                    (robot_pos[1] - obj_pos[1]) ** 2
                )
                
                # 计算碰撞半径之和
                collision_threshold = robot_radius + obj_radius
                
                # 判断是否碰撞
                is_colliding = distance < collision_threshold
                is_warning = distance < collision_threshold + 0.2  # 警告距离
                
                collisions.append({
                    'robot_id': robot_id,
                    'object_name': obj_name,
                    'distance': distance,
                    'robot_radius': robot_radius,
                    'object_radius': obj_radius,
                    'collision_threshold': collision_threshold,
                    'is_colliding': is_colliding,
                    'is_warning': is_warning and not is_colliding,
                    'robot_pos': robot_pos,
                    'object_pos': obj_pos
                })
        
        return collisions
    
    def visualize_collisions(self):
        """可视化碰撞检测"""
        print("\n" + "="*60)
        print("4. 碰撞检测与可视化")
        print("="*60)
        
        try:
            # 清除之前的可视化
            self.clear_visualization()
            
            # 检测碰撞
            collisions = self.check_collisions()
            
            print(f"\n   检测到的碰撞关系:")
            collision_count = 0
            warning_count = 0
            
            for coll in collisions:
                robot_id = coll['robot_id']
                obj_name = coll['object_name']
                distance = coll['distance']
                threshold = coll['collision_threshold']
                
                if coll['is_colliding']:
                    collision_count += 1
                    print(f"   🔴 碰撞: {robot_id} <-> {obj_name}")
                    print(f"      距离: {distance:.3f}m, 碰撞阈值: {threshold:.3f}m")
                    
                    # 可视化碰撞区域
                    self._visualize_collision_area(coll)
                    
                elif coll['is_warning']:
                    warning_count += 1
                    print(f"   🟡 警告: {robot_id} <-> {obj_name}")
                    print(f"      距离: {distance:.3f}m, 碰撞阈值: {threshold:.3f}m")
                    
                    # 可视化警告区域
                    self._visualize_warning_area(coll)
            
            if collision_count == 0 and warning_count == 0:
                print(f"   🟢 没有碰撞或警告，所有物体都在安全距离")
            
            # 可视化所有机器人和障碍物的碰撞半径
            print(f"\n   可视化碰撞半径:")
            self._visualize_all_radii()
            
            print(f"\n   ✓ 碰撞可视化完成")
            print(f"   碰撞数: {collision_count}, 警告数: {warning_count}")
            
            return True
            
        except Exception as e:
            print(f"   [ERROR] 碰撞可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _visualize_collision_area(self, coll_info):
        """可视化碰撞区域"""
        robot_pos = coll_info['robot_pos']
        obj_pos = coll_info['object_pos']
        
        # 计算碰撞区域的中点
        mid_x = (robot_pos[0] + obj_pos[0]) / 2
        mid_y = (robot_pos[1] + obj_pos[1]) / 2
        
        # 创建碰撞区域可视化（红色球体）
        visual_id = self._create_sphere_visual(
            position=[mid_x, mid_y, 0.2],
            radius=0.1,
            color=self.colors['collision']
        )
        self.visual_shapes.append(visual_id)
        
        # 绘制连线
        line_id = self._create_line_visual(
            start=[robot_pos[0], robot_pos[1], 0.1],
            end=[obj_pos[0], obj_pos[1], 0.1],
            color=[1, 0, 0]
        )
        self.visual_shapes.append(line_id)
    
    def _visualize_warning_area(self, coll_info):
        """可视化警告区域"""
        robot_pos = coll_info['robot_pos']
        obj_pos = coll_info['object_pos']
        
        # 计算中点
        mid_x = (robot_pos[0] + obj_pos[0]) / 2
        mid_y = (robot_pos[1] + obj_pos[1]) / 2
        
        # 创建警告区域可视化（黄色球体）
        visual_id = self._create_sphere_visual(
            position=[mid_x, mid_y, 0.15],
            radius=0.08,
            color=self.colors['warning']
        )
        self.visual_shapes.append(visual_id)
    
    def _visualize_all_radii(self):
        """可视化所有机器人和障碍物的实际碰撞形状"""
        # 可视化机器人
        for robot_id, robot_info in self.robots.items():
            position = robot_info['position']
            radius = robot_info['radius']
            
            # 机器人用半透明蓝色球体表示碰撞体积
            visual_id = self._create_sphere_visual(
                position=[position[0], position[1], position[2]],
                radius=radius,
                color=self.colors['robot']
            )
            self.visual_shapes.append(visual_id)
            
            print(f"      {robot_id}: 碰撞半径={radius:.2f}m")
        
        # 可视化障碍物 - 使用实际碰撞形状
        for obj_name, obj_info in self.scene_objects.items():
            if obj_info['type'] == 'ground':
                continue
            
            obj_id = obj_info['id']
            position = obj_info['position']
            
            # 获取物体的实际碰撞形状
            visual_id = self._create_object_collision_visual(obj_id, obj_name, self.colors['obstacle'])
            if visual_id is not None:
                self.visual_shapes.append(visual_id)
    
    def _create_object_collision_visual(self, obj_id, obj_name, color):
        """创建物体的实际碰撞形状可视化"""
        try:
            # 获取物体的碰撞数据
            collision_data = p.getCollisionShapeData(obj_id, -1, physicsClientId=self.client.client_id)
            
            if not collision_data:
                # 如果没有碰撞数据，使用默认球体
                pos, orn = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client.client_id)
                return self._create_sphere_visual([pos[0], pos[1], pos[2]], 0.1, color)
            
            # 为每个碰撞形状创建可视化
            visual_ids = []
            shape_types = []
            for collision in collision_data:
                geom_type = collision[2]
                geom_dimensions = collision[3]
                local_pos = collision[4]
                local_orn = collision[5]
                
                # 获取物体世界位置
                world_pos, world_orn = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client.client_id)
                
                # 根据几何类型创建可视化
                if geom_type == p.GEOM_BOX:
                    shape_name = "BOX"
                    # 盒子形状
                    half_extents = geom_dimensions
                    visual_shape_id = p.createVisualShape(
                        shapeType=p.GEOM_BOX,
                        halfExtents=half_extents,
                        rgbaColor=color
                    )
                elif geom_type == p.GEOM_SPHERE:
                    shape_name = "SPHERE"
                    # 球体形状
                    radius = geom_dimensions[0]
                    visual_shape_id = p.createVisualShape(
                        shapeType=p.GEOM_SPHERE,
                        radius=radius,
                        rgbaColor=color
                    )
                elif geom_type == p.GEOM_CYLINDER:
                    shape_name = "CYLINDER"
                    # 圆柱体形状
                    radius = geom_dimensions[0]
                    height = geom_dimensions[1]
                    visual_shape_id = p.createVisualShape(
                        shapeType=p.GEOM_CYLINDER,
                        radius=radius,
                        length=height,
                        rgbaColor=color
                    )
                elif geom_type == p.GEOM_CAPSULE:
                    shape_name = "CAPSULE"
                    # 胶囊体形状
                    radius = geom_dimensions[0]
                    height = geom_dimensions[1]
                    visual_shape_id = p.createVisualShape(
                        shapeType=p.GEOM_CAPSULE,
                        radius=radius,
                        length=height,
                        rgbaColor=color
                    )
                elif geom_type == p.GEOM_MESH:
                    shape_name = "MESH"
                    # 网格形状（使用默认球体表示）
                    visual_shape_id = p.createVisualShape(
                        shapeType=p.GEOM_SPHERE,
                        radius=0.1,
                        rgbaColor=color
                    )
                else:
                    shape_name = f"OTHER({geom_type})"
                    # 其他形状使用默认球体
                    visual_shape_id = p.createVisualShape(
                        shapeType=p.GEOM_SPHERE,
                        radius=0.1,
                        rgbaColor=color
                    )
                
                shape_types.append(shape_name)
                
                # 计算世界位置（考虑本地偏移）
                final_pos = [
                    world_pos[0] + local_pos[0],
                    world_pos[1] + local_pos[1],
                    world_pos[2] + local_pos[2]
                ]
                
                # 创建多体
                multi_body_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=final_pos,
                    baseOrientation=world_orn
                )
                visual_ids.append(multi_body_id)
            
            # 打印检测到的形状类型
            if shape_types:
                unique_shapes = list(set(shape_types))
                print(f"      {obj_name}: 形状={unique_shapes}, 共{len(shape_types)}个碰撞体")
            
            # 返回第一个可视化ID（如果有多个，只返回第一个作为代表）
            return visual_ids[0] if visual_ids else None
            
        except Exception as e:
            print(f"   [警告] 获取物体 {obj_name} 碰撞形状失败: {e}")
            # 失败时使用默认球体
            try:
                pos, orn = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client.client_id)
                return self._create_sphere_visual([pos[0], pos[1], pos[2]], 0.1, color)
            except:
                return None
    
    def _create_sphere_visual(self, position, radius, color):
        """创建球体可视化"""
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color
        )
        
        multi_body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1]
        )
        
        return multi_body_id
    
    def _create_line_visual(self, start, end, color, width=3):
        """创建连线可视化"""
        # 使用 debug line
        line_id = p.addUserDebugLine(
            lineFromXYZ=start,
            lineToXYZ=end,
            lineColorRGB=color,
            lineWidth=width
        )
        return line_id
    
    def clear_visualization(self):
        """清除所有可视化"""
        # 清除多体
        for shape_id in self.visual_shapes:
            try:
                if isinstance(shape_id, int) and shape_id >= 0:
                    p.removeBody(shape_id)
            except:
                pass
        
        # 清除 debug lines
        p.removeAllUserDebugItems()
        
        self.visual_shapes.clear()
        print("   已清除之前的可视化")
    
    def run_test(self, scene_config, robot_configs):
        """运行完整测试"""
        print("\n" + "="*60)
        print("碰撞半径可视化测试")
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
        
        # 可视化碰撞检测
        if not self.visualize_collisions():
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
    # 场景配置 - 使用内置场景
    scene_config = {
        'ground': {
            'model_path': 'Asset/Scene/Object/URDF_models/clear_box/model.urdf',
            'position': [0, 0, 0],
            'orientation': [0, 0, 0, 1],
            'scale': 5.0
        },
        'furniture': {
            'table': {
                'model_path': 'Asset/Scene/Object/URDF_models/furniture_table_rectangle_high/table.urdf',
                'position': [0, 0, 0],
                'orientation': [0, 0, 0, 1],
                'scale': 1.0,
                'fixed_base': True,
                'radius': 0.5  # 桌子半径
            }
        },
        'objects': {
            'box': {
                'model_path': 'Asset/Scene/Object/URDF_models/cracker_box/model.urdf',
                'position': [0.5, 0, 0.5],
                'orientation': [0, 0, 0, 1],
                'scale': 1.0,
                'radius': 0.15  # 箱子半径
            },
            'target_zone': {
                'model_path': 'Asset/Scene/Object/URDF_models/clear_box/model.urdf',
                'position': [2.0, 0, 0.1],
                'orientation': [0, 0, 0, 1],
                'scale': 1.0,
                'radius': 0.3  # 目标区域半径
            },
            'prep_station': {
                'model_path': 'Asset/Scene/Object/URDF_models/clear_box/model.urdf',
                'position': [0.0, 2.0, 0.1],
                'orientation': [0, 0, 0, 1],
                'scale': 1.0,
                'radius': 0.3  # 准备区域半径
            }
        }
    }

    # 机器人配置
    robot_configs = [
        {
            'robot_id': 'helper_mobile_manipulator',
            'robot_type': 'mobile_manipulator',
            'robot_model': 'panda_on_segbot',
            'init_position': [-1.0, 0, 0],
            'init_orientation': [0, 0, 0, 1]
        },
        {
            'robot_id': 'logistics_mobile_base',
            'robot_type': 'mobile_base',
            'robot_model': 'segbot',
            'init_position': [-1.0, 1.0, 0],
            'init_orientation': [0, 0, 0, 1]
        },
        {
            'robot_id': 'precision_arm_1',
            'robot_type': 'arm',
            'robot_model': 'panda',
            'init_position': [1.0, -1.0, 0],
            'init_orientation': [0, 0, 0, 1]
        },
        {
            'robot_id': 'precision_arm_2',
            'robot_type': 'arm',
            'robot_model': 'panda',
            'init_position': [2.0, -1.0, 0],
            'init_orientation': [0, 0, 0, 1]
        },
        {
            'robot_id': 'aerial_drone',
            'robot_type': 'drone',
            'robot_model': 'drone',
            'init_position': [0.0, 0.0, 1.5],  # 空中初始位置
            'init_orientation': [0, 0, 0, 1]
        }
    ]
    
    # 创建可视化器并运行测试
    visualizer = CollisionRadiusVisualizer()
    visualizer.run_test(scene_config, robot_configs)


if __name__ == "__main__":
    main()
