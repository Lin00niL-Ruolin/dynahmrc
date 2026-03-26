"""
MobileBase - 移动基座机器人类
继承 BestMan 的 MobileBase 能力，支持导航和避障任务
对应 asset/robot/ 中的 mobile_base 类型（如 segbot）
"""

from typing import Dict, List, Optional, Any, Set, Tuple
import math
import numpy as np
import pybullet as p


class MobileBase:
    """
    移动基座机器人包装类
    
    能力:
        - navigation: 自主导航、路径规划、避障
        - transport: 运输轻量物体（无抓取能力）
    
    限制:
        - 无操作能力（无机械臂）
        - 只能执行运输类任务
    """
    
    def __init__(
        self,
        robot_id: str,
        bestman_instance: Any,
        capabilities: Optional[List[str]] = None
    ):
        """
        初始化移动基座机器人
        
        Args:
            robot_id: 机器人唯一标识
            bestman_instance: BestMan 移动基座实例
            capabilities: 能力列表，默认 ["navigation", "transport"]
        """
        self.robot_id = robot_id
        self.robot_type = "mobile_base"
        self.bestman = bestman_instance
        self.capabilities = capabilities or ["navigation", "transport"]
        
        # 状态信息
        self.position = [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, 0.0, 1.0]
        self.yaw = 0.0
        self.is_busy = False
        self.current_task = None
        self.error_status = None
        
        # 导航参数
        self.navigation_threshold = 0.05  # 到达目标阈值（米）
        
        # 路径规划器（用于避障导航）
        try:
            from ..utils.path_planning import PathPlanner
            self.path_planner = PathPlanner()
            print(f"[MobileBase] 路径规划器初始化成功")
        except Exception as e:
            print(f"[MobileBase] 路径规划器初始化失败: {e}")
            self.path_planner = None
        
        # 更新初始位置
        self._update_pose()
    
    def _update_pose(self):
        """从 BestMan 实例更新当前位姿"""
        if hasattr(self.bestman, 'sim_get_current_base_pose'):
            pose = self.bestman.sim_get_current_base_pose()
            self.position = pose.get_position()
            self.orientation = pose.get_orientation()
            # 计算 yaw 角
            euler = p.getEulerFromQuaternion(self.orientation)
            self.yaw = euler[2]
    
    def get_state(self) -> Dict[str, Any]:
        """获取机器人当前状态"""
        self._update_pose()
        return {
            "robot_id": self.robot_id,
            "robot_type": self.robot_type,
            "position": self.position,
            "orientation": self.orientation,
            "yaw": self.yaw,
            "is_busy": self.is_busy,
            "current_task": self.current_task,
            "error_status": self.error_status,
            "capabilities": self.capabilities
        }
    
    def navigate_to(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None,
        scene_objects: Optional[Dict[str, Dict]] = None
    ) -> Tuple[bool, str]:
        """
        导航到目标位置（支持避障）
        
        Args:
            target_position: 目标位置 [x, y, z]（z 通常忽略）
            target_orientation: 目标朝向（四元数，可选）
            scene_objects: 场景物体信息，用于避障
        
        Returns:
            (是否成功, 消息)
        """
        try:
            self.is_busy = True
            self.current_task = f"navigate_to_{target_position}"
            
            print(f"[MobileBase] 开始导航到 {target_position}")
            
            # 如果有路径规划器且提供了场景物体，使用 A* + 避障导航
            if self.path_planner and scene_objects:
                print(f"[MobileBase] 使用避障导航，场景物体数: {len(scene_objects)}")
                success = self._navigate_with_avoidance(target_position, target_orientation, scene_objects)
            else:
                # 使用简单导航
                print(f"[MobileBase] 使用简单导航")
                success = self._simple_navigate(target_position, target_orientation)
            
            self._update_pose()
            if success:
                print(f"[MobileBase] 导航完成，当前位置: {self.position}")
                return True, f"成功导航到 {target_position}"
            else:
                return False, f"导航到 {target_position} 失败"
            
        except Exception as e:
            import traceback
            self.error_status = f"navigation_failed: {str(e)}\n{traceback.format_exc()}"
            print(f"[MobileBase] 导航失败: {self.error_status}")
            return False, str(e)
        finally:
            self.is_busy = False
            self.current_task = None
    
    def _simple_navigate(self, target_position: List[float], target_orientation: Optional[List[float]] = None) -> bool:
        """简单导航（无避障）"""
        # 构建目标位姿
        from Robotics_API.Pose import Pose
        if target_orientation:
            target_pose = Pose(target_position, target_orientation)
        else:
            target_pose = Pose(target_position, self.orientation)
        
        # 使用 BestMan 的导航功能
        if hasattr(self.bestman, 'sim_move_base_to_waypoint'):
            self.bestman.sim_move_base_to_waypoint(
                target_pose,
                threshold=self.navigation_threshold
            )
        else:
            # 备用：简单导航
            self._simple_navigation(target_position)
        
        # 如果需要调整最终朝向
        if target_orientation:
            target_yaw = p.getEulerFromQuaternion(target_orientation)[2]
            self.rotate_to_yaw(target_yaw)
        
        return True
    
    def _navigate_with_avoidance(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]],
        scene_objects: Dict[str, Dict]
    ) -> bool:
        """使用 A* 路径规划和局部避障导航"""
        # 更新障碍物信息
        self.path_planner.update_obstacles_from_scene(scene_objects)
        
        # 规划全局路径（A*）
        start_pos = [self.position[0], self.position[1]]
        goal_pos = [target_position[0], target_position[1]]
        
        print(f"[MobileBase] 规划全局路径: {start_pos} -> {goal_pos}")
        global_path = self.path_planner.plan_global_path(
            start_pos, goal_pos,
            scene_objects=scene_objects,
            max_search_radius=5.0,
            radius_step=0.5
        )
        
        if global_path is None:
            print("[MobileBase] 全局路径规划失败，使用简单导航")
            return self._simple_navigate(target_position, target_orientation)
        
        print(f"[MobileBase] 全局路径规划成功，路径点数: {len(global_path)}")
        
        # 沿路径点导航
        for i, waypoint in enumerate(global_path):
            print(f"[MobileBase] 前往路径点 {i+1}/{len(global_path)}: {waypoint}")
            
            # 计算到路径点的距离
            dx = waypoint[0] - self.position[0]
            dy = waypoint[1] - self.position[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < self.navigation_threshold:
                continue
            
            # 计算目标朝向
            target_yaw = math.atan2(dy, dx)
            
            # 旋转到目标方向
            self.rotate_to_yaw(target_yaw)
            
            # 向前移动
            self.move_forward(distance)
            
            self._update_pose()
        
        # 调整最终朝向
        if target_orientation:
            target_yaw = p.getEulerFromQuaternion(target_orientation)[2]
            self.rotate_to_yaw(target_yaw)
        
        return True
    
    def _simple_navigation(self, target_pos: List[float]):
        """简单导航实现（备用）"""
        max_steps = 1000
        step = 0
        
        while step < max_steps:
            self._update_pose()
            
            # 计算距离和方向
            dx = target_pos[0] - self.position[0]
            dy = target_pos[1] - self.position[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < self.navigation_threshold:
                break
            
            # 计算目标朝向
            target_yaw = math.atan2(dy, dx)
            
            # 旋转到目标方向
            self.rotate_to_yaw(target_yaw)
            
            # 向前移动一小步
            step_size = min(0.05, distance)
            self.move_forward(step_size)
            
            step += 1
            self.bestman.client.run(1)
    
    def rotate_to_yaw(self, target_yaw: float, gradual: bool = True) -> bool:
        """
        旋转到目标朝向
        
        Args:
            target_yaw: 目标 yaw 角（弧度）
            gradual: 是否平滑旋转
        
        Returns:
            是否成功
        """
        try:
            if hasattr(self.bestman, 'sim_rotate_base_to_target_yaw'):
                self.bestman.sim_rotate_base_to_target_yaw(target_yaw, gradual=gradual)
            else:
                # 备用：直接设置朝向
                self._set_yaw(target_yaw)
            
            self._update_pose()
            return True
            
        except Exception as e:
            self.error_status = f"rotation_failed: {str(e)}"
            return False
    
    def _set_yaw(self, target_yaw: float):
        """直接设置朝向（备用）"""
        orientation = [0, 0, math.sin(target_yaw / 2.0), math.cos(target_yaw / 2.0)]
        p.resetBasePositionAndOrientation(
            self.bestman.base_id,
            self.position,
            orientation,
            physicsClientId=self.bestman.client_id
        )
        self.bestman.client.run(5)
    
    def move_forward(self, distance: float) -> bool:
        """
        向前移动指定距离
        
        Args:
            distance: 移动距离（米）
        
        Returns:
            是否成功
        """
        try:
            if hasattr(self.bestman, 'sim_move_base_forward'):
                self.bestman.sim_move_base_forward(distance)
            else:
                # 备用实现
                new_x = self.position[0] + distance * math.cos(self.yaw)
                new_y = self.position[1] + distance * math.sin(self.yaw)
                p.resetBasePositionAndOrientation(
                    self.bestman.base_id,
                    [new_x, new_y, self.position[2]],
                    self.orientation,
                    physicsClientId=self.bestman.client_id
                )
                self.bestman.client.run(10)
            
            self._update_pose()
            return True
            
        except Exception as e:
            self.error_status = f"move_forward_failed: {str(e)}"
            return False
    
    def move_backward(self, distance: float) -> bool:
        """
        向后移动指定距离
        
        Args:
            distance: 移动距离（米）
        
        Returns:
            是否成功
        """
        try:
            if hasattr(self.bestman, 'sim_move_base_backward'):
                self.bestman.sim_move_base_backward(distance)
            else:
                # 备用实现
                new_x = self.position[0] - distance * math.cos(self.yaw)
                new_y = self.position[1] - distance * math.sin(self.yaw)
                p.resetBasePositionAndOrientation(
                    self.bestman.base_id,
                    [new_x, new_y, self.position[2]],
                    self.orientation,
                    physicsClientId=self.bestman.client_id
                )
                self.bestman.client.run(10)
            
            self._update_pose()
            return True
            
        except Exception as e:
            self.error_status = f"move_backward_failed: {str(e)}"
            return False
    
    def follow_path(self, waypoints: List[List[float]]) -> bool:
        """
        沿路径点导航
        
        Args:
            waypoints: 路径点列表，每个点为 [x, y, z]
        
        Returns:
            是否成功
        """
        try:
            self.is_busy = True
            self.current_task = "follow_path"
            
            for i, waypoint in enumerate(waypoints):
                if not self.navigate_to(waypoint):
                    self.error_status = f"path_following_failed_at_waypoint_{i}"
                    return False
            
            return True
            
        except Exception as e:
            self.error_status = f"follow_path_failed: {str(e)}"
            return False
        finally:
            self.is_busy = False
            self.current_task = None
    
    def transport_object(
        self,
        object_id: int,
        source_position: List[float],
        target_position: List[float]
    ) -> bool:
        """
        运输物体（仅适用于可被基座运输的物体）
        
        Args:
            object_id: 物体 ID
            source_position: 源位置
            target_position: 目标位置
        
        Returns:
            是否成功
        """
        try:
            self.is_busy = True
            self.current_task = f"transport_object_{object_id}"
            
            # 1. 导航到源位置
            if not self.navigate_to(source_position):
                return False
            
            # 2. 装载物体（假设物体已经在基座上或可以被基座携带）
            # 这里简化处理，实际可能需要特定的装载机制
            
            # 3. 导航到目标位置
            if not self.navigate_to(target_position):
                return False
            
            # 4. 卸载物体
            
            return True
            
        except Exception as e:
            self.error_status = f"transport_failed: {str(e)}"
            return False
        finally:
            self.is_busy = False
            self.current_task = None
    
    def stop(self):
        """停止移动"""
        if hasattr(self.bestman, 'sim_stop_base'):
            self.bestman.sim_stop_base()
    
    def emergency_stop(self):
        """紧急停止"""
        self.is_busy = False
        self.current_task = None
        self.stop()
    
    def get_distance_to(self, position: List[float]) -> float:
        """计算到目标位置的距离"""
        self._update_pose()
        dx = position[0] - self.position[0]
        dy = position[1] - self.position[1]
        return math.sqrt(dx**2 + dy**2)
    
    def receive_message(self, sender: str, message: str):
        """
        接收来自其他机器人的消息
        
        Args:
            sender: 发送者名称
            message: 消息内容
        """
        print(f"[{self.robot_id}] 收到来自 {sender} 的消息: {message[:50]}...")
        # 可以在这里添加消息处理逻辑
        # 例如：存储消息、触发动作等
