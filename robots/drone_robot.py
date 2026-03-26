"""
DroneRobot - 无人机机器人类
支持空中导航、抓取和放置操作
对应 asset/robot/ 中的 drone/UAV 类型
"""

from typing import Dict, List, Optional, Any, Tuple
import math
import numpy as np
import pybullet as p

from ..utils.path_planning import PathPlanner


class DroneRobot:
    """
    无人机机器人包装类
    
    能力:
        - navigation: 空中自主导航、路径规划
        - pick: 空中抓取（轻量物体）
        - place: 空中放置
        - perception: 空中视角感知
    
    对应论文中的 UAV (Unmanned Aerial Vehicle) 类型
    """
    
    def __init__(self, robot_id: str, bestman_instance: Any):
        """
        初始化无人机机器人
        
        Args:
            robot_id: 机器人唯一标识
            bestman_instance: BestMan 控制器实例
        """
        self.robot_id = robot_id
        self.robot_type = "drone"  # 机器人类型标识
        self.bestman = bestman_instance
        
        # 能力列表
        self.capabilities = ["navigation", "pick", "place", "perception"]
        
        # 基础属性
        self.position = [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, 0.0, 1.0]
        self.yaw = 0.0
        
        # 状态
        self.is_busy = False
        self.current_task = None
        self.error_status = None
        
        # 抓取状态
        self.is_holding_object = False
        self.held_object_id = None
        
        # 无人机特定属性
        self.flight_height = 1.5  # 默认飞行高度
        self.max_payload = 0.5    # 最大负载（kg）
        
        # 路径规划器
        self.path_planner = None
        if hasattr(bestman_instance, 'client'):
            try:
                self.path_planner = PathPlanner(
                    resolution=0.1,
                    robot_radius=0.2,  # 无人机半径较小
                    client_id=bestman_instance.client
                )
                print(f"[DroneRobot] 路径规划器初始化成功")
            except Exception as e:
                print(f"[DroneRobot] 路径规划器初始化失败: {e}")
        
        # 更新初始位姿
        self._update_pose()
        
        print(f"[DroneRobot] 初始化完成: {robot_id}")
    
    def _update_pose(self):
        """更新当前位姿"""
        try:
            if hasattr(self.bestman, 'get_base_pose'):
                pose = self.bestman.get_base_pose()
                if pose:
                    self.position = pose.get_position()
                    self.orientation = pose.get_orientation()
                    self.yaw = p.getEulerFromQuaternion(self.orientation)[2]
        except Exception as e:
            print(f"[DroneRobot] 更新位姿失败: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """获取机器人状态"""
        self._update_pose()
        return {
            "robot_id": self.robot_id,
            "robot_type": "drone",
            "position": self.position,
            "orientation": self.orientation,
            "is_busy": self.is_busy,
            "is_holding_object": self.is_holding_object,
            "held_object_id": self.held_object_id,
            "flight_height": self.flight_height
        }
    
    def navigate_to(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None,
        scene_objects: Optional[Dict[str, Dict]] = None
    ) -> Tuple[bool, str]:
        """
        空中导航到目标位置
        
        Args:
            target_position: 目标位置 [x, y, z]（z 为飞行高度）
            target_orientation: 目标朝向（四元数，可选）
            scene_objects: 场景物体信息，用于避障
        
        Returns:
            (是否成功, 消息)
        """
        try:
            self.is_busy = True
            self.current_task = f"navigate_to_{target_position}"
            
            print(f"[DroneRobot] 开始空中导航到 {target_position}")
            
            # 确保目标高度至少为飞行高度
            target_pos = target_position.copy()
            if len(target_pos) < 3 or target_pos[2] < self.flight_height:
                target_pos = [target_pos[0], target_pos[1], self.flight_height]
            
            # 更新障碍物信息
            if scene_objects and self.path_planner:
                print(f"[DroneRobot] 更新障碍物信息...")
                self.path_planner.update_obstacles_from_scene(scene_objects)
            
            # 规划路径
            start_pos = self.position.copy()
            
            if self.path_planner:
                global_path = self.path_planner.plan_global_path(
                    start_pos[:2], target_pos[:2],
                    scene_objects=scene_objects or {},
                    max_search_radius=5.0,
                    radius_step=0.5
                )
                
                if global_path is None:
                    print("[DroneRobot] 路径规划失败")
                    return False, "路径规划失败"
                
                print(f"[DroneRobot] 路径规划成功，路径点数: {len(global_path)}")
                
                # 沿路径飞行
                for waypoint in global_path:
                    # 设置飞行高度
                    waypoint_3d = [waypoint[0], waypoint[1], target_pos[2]]
                    self._fly_to(waypoint_3d)
            else:
                # 直接飞行到目标
                self._fly_to(target_pos)
            
            # 调整最终朝向
            if target_orientation:
                target_yaw = p.getEulerFromQuaternion(target_orientation)[2]
                self._rotate_to_yaw(target_yaw)
            
            self._update_pose()
            print(f"[DroneRobot] 导航完成，当前位置: {self.position}")
            return True, f"成功导航到 {target_pos}"
            
        except Exception as e:
            import traceback
            self.error_status = f"navigation_failed: {str(e)}\n{traceback.format_exc()}"
            print(f"[DroneRobot] 导航失败: {self.error_status}")
            return False, str(e)
        finally:
            self.is_busy = False
            self.current_task = None
    
    def _fly_to(self, target_pos: List[float]):
        """飞行到指定位置（3D）"""
        try:
            print(f"[DroneRobot] _fly_to: 从 {self.position} 飞到 {target_pos}")
            
            # 使用 BestMan 的移动功能
            if hasattr(self.bestman, 'move_base_to'):
                print(f"[DroneRobot] 使用 move_base_to")
                from Robotics_API.Pose import Pose
                target_pose = Pose(target_pos, self.orientation)
                result = self.bestman.move_base_to(target_pose)
                print(f"[DroneRobot] move_base_to 返回: {result}")
            elif hasattr(self.bestman, 'set_base_pose'):
                print(f"[DroneRobot] 使用 set_base_pose")
                self.bestman.set_base_pose(target_pos, self.orientation)
            else:
                # 备用：直接设置位置
                print(f"[DroneRobot] 使用直接设置位置")
                self._set_position_directly(target_pos)
            
            self._update_pose()
            print(f"[DroneRobot] 飞行后位置: {self.position}")
            
        except Exception as e:
            print(f"[DroneRobot] 飞行失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _set_position_directly(self, target_pos: List[float]):
        """直接设置位置（用于测试）"""
        try:
            if hasattr(self.bestman, 'client') and hasattr(self.bestman, 'robot_id'):
                p.resetBasePositionAndOrientation(
                    self.bestman.robot_id,
                    target_pos,
                    self.orientation,
                    physicsClientId=self.bestman.client
                )
        except Exception as e:
            print(f"[DroneRobot] 直接设置位置失败: {e}")
    
    def _rotate_to_yaw(self, target_yaw: float):
        """旋转到指定朝向"""
        try:
            if hasattr(self.bestman, 'rotate_base_to_yaw'):
                self.bestman.rotate_base_to_yaw(target_yaw)
            else:
                # 备用：直接设置朝向
                new_orientation = p.getQuaternionFromEuler([0, 0, target_yaw])
                self.orientation = list(new_orientation)
                self._set_position_directly(self.position)
            
            self._update_pose()
            
        except Exception as e:
            print(f"[DroneRobot] 旋转失败: {e}")
    
    def pick(self, object_id: str, object_position: List[float]) -> Tuple[bool, str]:
        """
        空中抓取物体
        
        Args:
            object_id: 物体ID
            object_position: 物体位置 [x, y, z]
        
        Returns:
            (是否成功, 消息)
        """
        try:
            self.is_busy = True
            self.current_task = f"pick_{object_id}"
            
            print(f"[DroneRobot] 开始抓取物体 {object_id} 在 {object_position}")
            
            # 检查是否已持有物体
            if self.is_holding_object:
                return False, "已经持有物体"
            
            # 导航到物体上方
            approach_pos = [object_position[0], object_position[1], object_position[2] + 0.3]
            success, msg = self.navigate_to(approach_pos)
            
            if not success:
                return False, f"无法接近物体: {msg}"
            
            # 执行抓取
            if hasattr(self.bestman, 'grasp'):
                grasp_success = self.bestman.grasp(object_id)
                if grasp_success:
                    self.is_holding_object = True
                    self.held_object_id = object_id
                    print(f"[DroneRobot] 抓取成功: {object_id}")
                    return True, f"成功抓取物体 {object_id}"
                else:
                    return False, "抓取失败"
            else:
                # 简化版：直接标记为已抓取
                self.is_holding_object = True
                self.held_object_id = object_id
                print(f"[DroneRobot] 抓取成功（简化）: {object_id}")
                return True, f"成功抓取物体 {object_id}"
            
        except Exception as e:
            import traceback
            self.error_status = f"pick_failed: {str(e)}\n{traceback.format_exc()}"
            print(f"[DroneRobot] 抓取失败: {self.error_status}")
            return False, str(e)
        finally:
            self.is_busy = False
            self.current_task = None
    
    def place(self, target_position: List[float]) -> Tuple[bool, str]:
        """
        放置物体
        
        Args:
            target_position: 目标位置 [x, y, z]
        
        Returns:
            (是否成功, 消息)
        """
        try:
            self.is_busy = True
            self.current_task = f"place_at_{target_position}"
            
            print(f"[DroneRobot] 开始放置物体到 {target_position}")
            
            # 检查是否持有物体
            if not self.is_holding_object:
                return False, "没有持有物体"
            
            # 导航到目标位置上方
            place_pos = [target_position[0], target_position[1], target_position[2] + 0.3]
            success, msg = self.navigate_to(place_pos)
            
            if not success:
                return False, f"无法到达放置位置: {msg}"
            
            # 执行放置
            if hasattr(self.bestman, 'release'):
                release_success = self.bestman.release()
                if release_success:
                    self.is_holding_object = False
                    self.held_object_id = None
                    print(f"[DroneRobot] 放置成功")
                    return True, "成功放置物体"
                else:
                    return False, "放置失败"
            else:
                # 简化版：直接标记为已放置
                self.is_holding_object = False
                self.held_object_id = None
                print(f"[DroneRobot] 放置成功（简化）")
                return True, "成功放置物体"
            
        except Exception as e:
            import traceback
            self.error_status = f"place_failed: {str(e)}\n{traceback.format_exc()}"
            print(f"[DroneRobot] 放置失败: {self.error_status}")
            return False, str(e)
        finally:
            self.is_busy = False
            self.current_task = None
    
    def communicate(self, message: str, to: Optional[str] = None) -> Tuple[bool, str]:
        """
        发送通信消息
        
        Args:
            message: 消息内容
            to: 接收者（None 表示广播）
        
        Returns:
            (是否成功, 消息)
        """
        try:
            print(f"[DroneRobot] 发送消息{'给 ' + to if to else '（广播）'}: {message}")
            return True, "消息已发送"
        except Exception as e:
            return False, str(e)
    
    def receive_message(self, from_robot: str, message: str) -> bool:
        """
        接收消息
        
        Args:
            from_robot: 发送者名称
            message: 消息内容
        
        Returns:
            是否成功接收
        """
        print(f"[DroneRobot] 收到来自 {from_robot} 的消息: {message}")
        return True
    
    def wait(self, duration: float = 1.0) -> Tuple[bool, str]:
        """
        等待
        
        Args:
            duration: 等待时间（秒）
        
        Returns:
            (是否成功, 消息)
        """
        try:
            print(f"[DroneRobot] 等待 {duration} 秒...")
            if hasattr(self.bestman, 'client'):
                for _ in range(int(duration * 240)):  # 假设 240Hz
                    p.stepSimulation(physicsClientId=self.bestman.client)
            else:
                import time
                time.sleep(duration)
            return True, f"等待了 {duration} 秒"
        except Exception as e:
            return False, str(e)
