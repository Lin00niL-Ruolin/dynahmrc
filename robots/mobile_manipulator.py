"""
MobileManipulator - 移动操作复合机器人类
继承 BestMan 的 MobileManipulation 能力，支持导航+操作
对应 asset/robot/ 中的 mobile_manipulator 类型
"""

from typing import Dict, List, Optional, Any, Tuple
import math
import numpy as np
import pybullet as p

from ..utils.path_planning import PathPlanner


class MobileManipulator:
    """
    移动操作复合机器人包装类
    
    能力:
        - navigation: 自主导航、路径规划
        - manipulation: 抓取、放置、操作物体
        - transport: 运输物体（抓取后携带）
        - perception: 通过相机感知环境
    
    这是功能最全面的机器人类型，可以独立执行完整的 pick-and-place 任务。
    """
    
    def __init__(
        self,
        robot_id: str,
        bestman_instance: Any,
        capabilities: Optional[List[str]] = None
    ):
        """
        初始化移动操作复合机器人
        
        Args:
            robot_id: 机器人唯一标识
            bestman_instance: BestMan 移动操作实例
            capabilities: 能力列表，默认 ["navigation", "manipulation", "transport", "perception"]
        """
        self.robot_id = robot_id
        self.robot_type = "mobile_manipulator"
        self.bestman = bestman_instance
        self.capabilities = capabilities or ["navigation", "manipulation", "transport", "perception"]
        
        # 状态信息
        self.position = [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, 0.0, 1.0]
        self.yaw = 0.0
        self.is_busy = False
        self.current_task = None
        self.error_status = None
        self.is_holding_object = False
        self.held_object_id = None
        
        # 导航参数
        self.navigation_threshold = 0.05
        self.manipulation_range = 0.8  # 操作范围（米）
        
        # 路径规划器（A* + DWA）
        self.path_planner = PathPlanner(client_id=self.bestman.client_id)
        
        # 更新初始位置
        self._update_pose()
    
    def _update_pose(self):
        """从 BestMan 实例更新当前位姿"""
        if hasattr(self.bestman, 'sim_get_current_base_pose'):
            pose = self.bestman.sim_get_current_base_pose()
            self.position = pose.get_position()
            self.orientation = pose.get_orientation()
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
            "capabilities": self.capabilities,
            "is_holding_object": self.is_holding_object,
            "held_object_id": self.held_object_id
        }
    
    def navigate_to(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None,
        scene_objects: Optional[Dict[str, Dict]] = None
    ) -> bool:
        """
        导航到目标位置（使用 A* + DWA 路径规划）
        
        Args:
            target_position: 目标位置 [x, y, z]
            target_orientation: 目标朝向（四元数，可选）
            scene_objects: 场景物体信息，用于避障
        
        Returns:
            是否成功
        """
        try:
            self.is_busy = True
            self.current_task = f"navigate_to_{target_position}"
            
            print(f"[MobileManipulator] 开始导航到 {target_position}")
            
            # 更新障碍物信息
            print(f"[MobileManipulator] 更新障碍物信息...")
            if scene_objects:
                self.path_planner.update_obstacles_from_scene(scene_objects)
            
            # 规划全局路径（A*）
            start_pos = [self.position[0], self.position[1]]
            goal_pos = [target_position[0], target_position[1]]
            print(f"[MobileManipulator] 规划路径: {start_pos} -> {goal_pos}")
            
            global_path = self.path_planner.plan_global_path(start_pos, goal_pos)
            print(f"[MobileManipulator] 路径规划结果: {global_path is not None}")
            
            if global_path is None:
                print("[MobileManipulator] 全局路径规划失败，使用简单导航")
                self._simple_navigation(target_position)
            else:
                print(f"[MobileManipulator] 全局路径规划成功，路径点数: {len(global_path)}")
                
                # 检查路径是否经过其他机器人
                if scene_objects:
                    for obj_name, obj_info in scene_objects.items():
                        if obj_info.get('type') == 'robot':
                            robot_pos = obj_info.get('position', [0, 0, 0])
                            min_dist_to_robot = float('inf')
                            closest_point = None
                            for i, path_point in enumerate(global_path):
                                dist = math.sqrt(
                                    (path_point[0] - robot_pos[0])**2 + 
                                    (path_point[1] - robot_pos[1])**2
                                )
                                if dist < min_dist_to_robot:
                                    min_dist_to_robot = dist
                                    closest_point = (i, path_point)
                            print(f"[MobileManipulator] 路径与机器人 {obj_name} 的最小距离: {min_dist_to_robot:.3f}m")
                            if min_dist_to_robot < 0.5:  # 小于安全距离
                                print(f"[MobileManipulator] ⚠️ 警告: 路径经过机器人 {obj_name} 附近! 最近点 {closest_point}")
                
                # 使用 DWA 沿路径导航
                self._dwa_navigation(global_path, scene_objects or {})
            
            # 调整最终朝向
            if target_orientation:
                target_yaw = p.getEulerFromQuaternion(target_orientation)[2]
                self.rotate_to_yaw(target_yaw)
            
            self._update_pose()
            print(f"[MobileManipulator] 导航完成，当前位置: {self.position}")
            return True
            
        except Exception as e:
            import traceback
            self.error_status = f"navigation_failed: {str(e)}\n{traceback.format_exc()}"
            print(f"[MobileManipulator] 导航失败: {self.error_status}")
            return False
        finally:
            self.is_busy = False
            self.current_task = None
    
    def _dwa_navigation(self, global_path: List[List[float]], scene_objects: Dict[str, Dict]):
        """使用 DWA 沿全局路径导航"""
        max_steps = 1000
        step = 0
        current_v = 0.0
        current_yaw_rate = 0.0
        
        print(f"[MobileManipulator] 开始 DWA 导航，全局路径 {len(global_path)} 点")
        print(f"[MobileManipulator] 起点: {self.position[:2]}, 终点: {global_path[-1]}")
        
        # 提取障碍物位置用于调试
        obstacle_positions = []
        for obj_name, obj_info in scene_objects.items():
            if obj_info.get('type') != 'graspable':
                pos = obj_info.get('position', [0, 0, 0])
                obstacle_positions.append([pos[0], pos[1]])
        print(f"[MobileManipulator] 障碍物数量: {len(obstacle_positions)}, 位置: {obstacle_positions}")
        
        while step < max_steps:
            self._update_pose()
            
            # 检查是否到达终点
            goal = global_path[-1]
            distance_to_goal = math.sqrt(
                (goal[0] - self.position[0]) ** 2 +
                (goal[1] - self.position[1]) ** 2
            )
            
            if distance_to_goal < self.navigation_threshold:
                print(f"[MobileManipulator] 到达目标，距离: {distance_to_goal:.3f}m")
                break
            
            # 检查是否碰撞障碍物
            collision = False
            for obs_pos in obstacle_positions:
                dist_to_obs = math.sqrt(
                    (obs_pos[0] - self.position[0]) ** 2 +
                    (obs_pos[1] - self.position[1]) ** 2
                )
                if dist_to_obs < 0.3:  # 机器人半径
                    print(f"[MobileManipulator] ⚠️ 警告: 距离障碍物 {dist_to_obs:.3f}m，位置 {obs_pos}")
                    collision = True
            
            # 使用 DWA 计算速度
            v, yaw_rate = self.path_planner.compute_velocity(
                [self.position[0], self.position[1]],
                self.yaw,
                current_v,
                current_yaw_rate,
                scene_objects
            )
            
            # 调试信息：显示每一步的速度
            if step % 10 == 0 or v > 0.001:
                print(f"[MobileManipulator] Step {step}: v={v:.4f}m/s, yaw_rate={yaw_rate:.4f}rad/s, pos=[{self.position[0]:.3f}, {self.position[1]:.3f}]")
            
            # 应用速度
            old_pos = self.position.copy()
            self._apply_velocity(v, yaw_rate, obstacle_positions)
            
            # 检查位置是否变化
            self._update_pose()
            pos_change = math.sqrt((self.position[0] - old_pos[0])**2 + (self.position[1] - old_pos[1])**2)
            if v > 0.001 and pos_change < 0.001:
                print(f"[MobileManipulator] ⚠️ 警告: 速度 {v:.4f} 但位置未变化 {pos_change:.6f}m")
            
            current_v = v
            current_yaw_rate = yaw_rate
            
            step += 1
            self.bestman.client.run(5)
            
            if step % 50 == 0:
                print(f"[MobileManipulator] 导航中... 步骤: {step}, 位置: [{self.position[0]:.3f}, {self.position[1]:.3f}], 速度: {v:.3f}m/s")
        
        if step >= max_steps:
            print(f"[MobileManipulator] 导航达到最大步数 {max_steps}")
        print(f"[MobileManipulator] DWA 导航结束，最终位置: [{self.position[0]:.3f}, {self.position[1]:.3f}]")
    
    def _apply_velocity(self, v: float, yaw_rate: float, obstacles: List[List[float]] = None):
        """应用速度指令，带障碍物检测，同时移动底座和机械臂"""
        # 计算新位置
        dt = 0.1
        new_yaw = self.yaw + yaw_rate * dt
        new_x = self.position[0] + v * math.cos(new_yaw) * dt
        new_y = self.position[1] + v * math.sin(new_yaw) * dt
        
        # 检查新位置是否会碰撞障碍物
        if obstacles:
            for obs_pos in obstacles:
                dist_to_obs = math.sqrt(
                    (obs_pos[0] - new_x) ** 2 +
                    (obs_pos[1] - new_y) ** 2
                )
                if dist_to_obs < 0.35:  # 机器人半径 + 安全距离
                    # 会碰撞，停止移动
                    print(f"[MobileManipulator] 🚫 碰撞检测: 新位置 [{new_x:.3f}, {new_y:.3f}] 距离障碍物 {dist_to_obs:.3f}m，停止移动")
                    return
        
        # 更新朝向
        orientation = [0, 0, math.sin(new_yaw / 2.0), math.cos(new_yaw / 2.0)]
        
        # 计算底座移动的位移和旋转
        dx = new_x - self.position[0]
        dy = new_y - self.position[1]
        dyaw = new_yaw - self.yaw
        
        # 设置新位置到底座
        p.resetBasePositionAndOrientation(
            self.bestman.base_id,
            [new_x, new_y, self.position[2]],
            orientation,
            physicsClientId=self.bestman.client_id
        )
        
        # 同时移动机械臂（如果有的话）
        if hasattr(self.bestman, 'arm_id') and self.bestman.arm_id is not None:
            # 获取机械臂当前位置和朝向
            arm_pos, arm_orn = p.getBasePositionAndOrientation(
                self.bestman.arm_id, physicsClientId=self.bestman.client_id
            )
            
            # 计算机械臂相对于底座的位置
            rel_x = arm_pos[0] - self.position[0]
            rel_y = arm_pos[1] - self.position[1]
            
            # 应用旋转（如果底座旋转了）
            if abs(dyaw) > 0.001:
                cos_yaw = math.cos(dyaw)
                sin_yaw = math.sin(dyaw)
                new_rel_x = rel_x * cos_yaw - rel_y * sin_yaw
                new_rel_y = rel_x * sin_yaw + rel_y * cos_yaw
                rel_x = new_rel_x
                rel_y = new_rel_y
                
                # 旋转机械臂朝向
                arm_euler = p.getEulerFromQuaternion(arm_orn)
                new_arm_yaw = arm_euler[2] + dyaw
                new_arm_orn = p.getQuaternionFromEuler([arm_euler[0], arm_euler[1], new_arm_yaw])
            else:
                new_arm_orn = arm_orn
            
            # 计算新的绝对位置
            new_arm_x = new_x + rel_x
            new_arm_y = new_y + rel_y
            new_arm_z = arm_pos[2]  # 保持高度不变
            
            # 更新机械臂位置和朝向
            p.resetBasePositionAndOrientation(
                self.bestman.arm_id,
                [new_arm_x, new_arm_y, new_arm_z],
                new_arm_orn,
                physicsClientId=self.bestman.client_id
            )
            
            # 更新夹爪位置（如果有的话）
            if hasattr(self.bestman, 'gripper_id') and self.bestman.gripper_id is not None:
                gripper_pos, gripper_orn = p.getBasePositionAndOrientation(
                    self.bestman.gripper_id, physicsClientId=self.bestman.client_id
                )
                
                # 计算夹爪相对于底座的位置
                rel_gx = gripper_pos[0] - self.position[0]
                rel_gy = gripper_pos[1] - self.position[1]
                
                # 应用旋转
                if abs(dyaw) > 0.001:
                    new_rel_gx = rel_gx * cos_yaw - rel_gy * sin_yaw
                    new_rel_gy = rel_gx * sin_yaw + rel_gy * cos_yaw
                    rel_gx = new_rel_gx
                    rel_gy = new_rel_gy
                    
                    # 旋转夹爪朝向
                    gripper_euler = p.getEulerFromQuaternion(gripper_orn)
                    new_gripper_yaw = gripper_euler[2] + dyaw
                    new_gripper_orn = p.getQuaternionFromEuler([gripper_euler[0], gripper_euler[1], new_gripper_yaw])
                else:
                    new_gripper_orn = gripper_orn
                
                # 计算新的绝对位置
                new_gripper_x = new_x + rel_gx
                new_gripper_y = new_y + rel_gy
                new_gripper_z = gripper_pos[2]
                
                p.resetBasePositionAndOrientation(
                    self.bestman.gripper_id,
                    [new_gripper_x, new_gripper_y, new_gripper_z],
                    new_gripper_orn,
                    physicsClientId=self.bestman.client_id
                )
    
    def _simple_navigation(self, target_pos: List[float]):
        """简单导航实现（备用）"""
        max_steps = 1000
        step = 0
        
        print(f"[MobileManipulator] 使用简单导航到 {target_pos}")
        
        while step < max_steps:
            self._update_pose()
            
            dx = target_pos[0] - self.position[0]
            dy = target_pos[1] - self.position[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < self.navigation_threshold:
                print(f"[MobileManipulator] 简单导航到达目标")
                break
            
            target_yaw = math.atan2(dy, dx)
            self.rotate_to_yaw(target_yaw)
            self.move_forward(min(0.05, distance))
            
            step += 1
            self.bestman.client.run(1)
    
    def rotate_to_yaw(self, target_yaw: float, gradual: bool = True) -> bool:
        """旋转到目标朝向"""
        try:
            if hasattr(self.bestman, 'sim_rotate_base_to_target_yaw'):
                self.bestman.sim_rotate_base_to_target_yaw(target_yaw, gradual=gradual)
            else:
                self._set_yaw(target_yaw)
            
            self._update_pose()
            return True
            
        except Exception as e:
            self.error_status = f"rotation_failed: {str(e)}"
            return False
    
    def _set_yaw(self, target_yaw: float):
        """直接设置朝向"""
        orientation = [0, 0, math.sin(target_yaw / 2.0), math.cos(target_yaw / 2.0)]
        p.resetBasePositionAndOrientation(
            self.bestman.base_id,
            self.position,
            orientation,
            physicsClientId=self.bestman.client_id
        )
        self.bestman.client.run(5)
    
    def move_forward(self, distance: float) -> bool:
        """向前移动"""
        try:
            if hasattr(self.bestman, 'sim_move_base_forward'):
                self.bestman.sim_move_base_forward(distance)
            else:
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
    
    def pick(self, object_id: int, grasp_pose: Optional[Dict] = None, scene_objects: Optional[Dict[str, Dict]] = None) -> bool:
        """
        抓取物体

        算法流程:
        1. 检查物体是否在操作范围内
        2. 如有必要，导航到合适位置
        3. 执行抓取动作
        """
        try:
            self.is_busy = True
            self.current_task = f"pick_object_{object_id}"
            
            # 更新当前位置
            self._update_pose()

            # 获取物体位置
            obj_pos, obj_orn = p.getBasePositionAndOrientation(
                object_id, physicsClientId=self.bestman.client_id
            )

            # 检查距离
            distance = math.sqrt(
                (obj_pos[0] - self.position[0])**2 +
                (obj_pos[1] - self.position[1])**2
            )

            # 如果物体太远，先导航到合适位置
            if distance > self.manipulation_range:
                # 计算合适的抓取位置（物体前方）
                approach_distance = self.manipulation_range * 0.8
                angle = math.atan2(
                    obj_pos[1] - self.position[1],
                    obj_pos[0] - self.position[0]
                )
                approach_pos = [
                    obj_pos[0] - approach_distance * math.cos(angle),
                    obj_pos[1] - approach_distance * math.sin(angle),
                    self.position[2]
                ]
                
                print(f"[MobileManipulator] pick: 物体距离 {distance:.3f}m，超出操作范围 {self.manipulation_range}m")
                print(f"[MobileManipulator] pick: 物体位置 {obj_pos[:2]}，机器人位置 {self.position[:2]}")
                print(f"[MobileManipulator] pick: 计算接近位置 {approach_pos[:2]}，角度 {math.degrees(angle):.1f}°")

                if not self.navigate_to(approach_pos, scene_objects=scene_objects):
                    self.error_status = f"pick_failed: 导航到抓取位置失败"
                    return False

                # 转向物体
                self.rotate_to_yaw(angle)
            
            # 执行抓取
            pre_grasp_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.1]
            grasp_pos = [obj_pos[0], obj_pos[1], obj_pos[2]]
            
            # 1. 移动到预抓取位置
            if not self._move_arm_to_position(pre_grasp_pos):
                self.error_status = f"pick_failed: 移动到预抓取位置失败"
                return False
            
            # 2. 打开夹爪
            self.open_gripper()
            
            # 3. 下降到抓取位置
            if not self._move_arm_to_position(grasp_pos):
                self.error_status = f"pick_failed: 下降到抓取位置失败"
                return False
            
            # 4. 关闭夹爪
            self.close_gripper()
            
            # 5. 创建约束
            self._create_grasp_constraint(object_id)
            
            # 6. 抬升
            if not self._move_arm_to_position(pre_grasp_pos):
                self.error_status = f"pick_failed: 抬升物体失败"
                return False
            
            self.is_holding_object = True
            self.held_object_id = object_id
            
            return True
            
        except Exception as e:
            import traceback
            self.error_status = f"pick_failed: {str(e)}\n{traceback.format_exc()}"
            return False
        finally:
            self.is_busy = False
            self.current_task = None
    
    def place(self, target_position: List[float], scene_objects: Optional[Dict[str, Dict]] = None) -> bool:
        """
        放置物体到目标位置

        算法流程:
        1. 检查目标位置是否在操作范围内
        2. 如有必要，导航到合适位置
        3. 执行放置动作
        """
        try:
            self.is_busy = True
            self.current_task = f"place_at_{target_position}"

            print(f"[MobileManipulator] 开始放置，目标位置: {target_position}")
            print(f"[MobileManipulator] 当前持有物体: {self.is_holding_object}, ID: {self.held_object_id}")
            print(f"[MobileManipulator] 约束ID: {getattr(self, 'constraint_id', None)}")

            # 检查是否持有物体
            if not self.is_holding_object:
                self.error_status = "place_failed: 没有持有物体"
                print(f"[MobileManipulator] 错误: 没有持有物体")
                return False

            # 检查距离
            distance = math.sqrt(
                (target_position[0] - self.position[0])**2 +
                (target_position[1] - self.position[1])**2
            )
            print(f"[MobileManipulator] 到目标距离: {distance}")

            # 如果目标太远，先导航
            if distance > self.manipulation_range:
                approach_distance = self.manipulation_range * 0.8
                angle = math.atan2(
                    target_position[1] - self.position[1],
                    target_position[0] - self.position[0]
                )
                approach_pos = [
                    target_position[0] - approach_distance * math.cos(angle),
                    target_position[1] - approach_distance * math.sin(angle),
                    self.position[2]
                ]

                print(f"[MobileManipulator] 导航到接近位置: {approach_pos}")
                if not self.navigate_to(approach_pos, scene_objects=scene_objects):
                    self.error_status = "place_failed: 导航到接近位置失败"
                    return False

                self.rotate_to_yaw(angle)
            
            # 执行放置
            pre_place_pos = [target_position[0], target_position[1], target_position[2] + 0.1]
            place_pos = target_position
            
            # 1. 移动到预放置位置
            print(f"[MobileManipulator] 移动到预放置位置: {pre_place_pos}")
            if not self._move_arm_to_position(pre_place_pos):
                self.error_status = "place_failed: 移动到预放置位置失败"
                return False
            
            # 2. 下降到放置位置
            print(f"[MobileManipulator] 下降到放置位置: {place_pos}")
            if not self._move_arm_to_position(place_pos):
                self.error_status = "place_failed: 下降到放置位置失败"
                return False
            
            # 3. 打开夹爪
            print(f"[MobileManipulator] 打开夹爪")
            self.open_gripper()
            
            # 4. 移除约束
            print(f"[MobileManipulator] 移除约束")
            self._remove_grasp_constraint()
            
            # 5. 抬升
            print(f"[MobileManipulator] 抬升手臂")
            if not self._move_arm_to_position(pre_place_pos):
                self.error_status = "place_failed: 抬升失败"
                return False
            
            self.is_holding_object = False
            self.held_object_id = None
            
            print(f"[MobileManipulator] 放置完成")
            return True
            
        except Exception as e:
            import traceback
            self.error_status = f"place_failed: {str(e)}\n{traceback.format_exc()}"
            print(f"[MobileManipulator] 放置异常: {self.error_status}")
            return False
        finally:
            self.is_busy = False
            self.current_task = None
    
    def transport(
        self,
        object_id: int,
        source_position: List[float],
        target_position: List[float]
    ) -> bool:
        """
        完整的运输任务：导航到源位置 -> 抓取 -> 导航到目标位置 -> 放置
        
        Args:
            object_id: 物体 ID
            source_position: 源位置
            target_position: 目标位置
        
        Returns:
            是否成功
        """
        try:
            self.is_busy = True
            self.current_task = f"transport_{object_id}_to_{target_position}"
            
            # 1. 抓取
            if not self.pick(object_id):
                return False
            
            # 2. 放置
            if not self.place(target_position):
                return False
            
            return True
            
        except Exception as e:
            self.error_status = f"transport_failed: {str(e)}"
            return False
        finally:
            self.is_busy = False
            self.current_task = None
    
    def _move_arm_to_position(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None
    ) -> bool:
        """移动机械臂到目标位置"""
        try:
            from Robotics_API.Pose import Pose
            target_pose = Pose(target_position, target_orientation or [0, 0, 0, 1])
            
            if hasattr(self.bestman, 'sim_move_arm_to_target_pose'):
                self.bestman.sim_move_arm_to_target_pose(target_pose)
                return True
            else:
                return self._simple_ik_move(target_position, target_orientation)
                
        except Exception as e:
            return False
    
    def _simple_ik_move(self, target_pos, target_orn):
        """简单的 IK 移动"""
        try:
            joint_positions = p.calculateInverseKinematics(
                self.bestman.arm_id,
                self.bestman.eef_id,
                target_pos,
                targetOrientation=target_orn,
                maxNumIterations=100,
                physicsClientId=self.bestman.client_id
            )
            
            for i, pos in enumerate(joint_positions[:self.bestman.arm_num_dofs]):
                p.setJointMotorControl2(
                    self.bestman.arm_id,
                    i,
                    p.POSITION_CONTROL,
                    pos,
                    force=100,
                    physicsClientId=self.bestman.client_id
                )
            
            self.bestman.client.run(50)
            return True
            
        except Exception as e:
            return False
    
    def open_gripper(self):
        """打开夹爪"""
        if hasattr(self.bestman, 'sim_open_gripper'):
            self.bestman.sim_open_gripper()
    
    def close_gripper(self):
        """关闭夹爪"""
        if hasattr(self.bestman, 'sim_close_gripper'):
            self.bestman.sim_close_gripper()
    
    def _create_grasp_constraint(self, object_id: int):
        """创建抓取约束"""
        # 直接使用 PyBullet 创建约束，避免 BestMan 方法中的 getLinkState 问题
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.bestman.arm_id,
            parentLinkIndex=self.bestman.eef_id,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,  # -1 表示 base
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.bestman.client_id
        )
    
    def _remove_grasp_constraint(self):
        """移除抓取约束"""
        # 优先使用我们创建的约束 ID
        if hasattr(self, 'constraint_id') and self.constraint_id is not None:
            try:
                p.removeConstraint(self.constraint_id, physicsClientId=self.bestman.client_id)
                print(f"[MobileManipulator] 移除约束 {self.constraint_id}")
            except Exception as e:
                print(f"[MobileManipulator] 移除约束失败: {e}")
            self.constraint_id = None
        # 备选：使用 BestMan 的方法
        elif hasattr(self.bestman, 'sim_remove_gripper_constraint'):
            self.bestman.sim_remove_gripper_constraint()
    
    def get_end_effector_pose(self) -> Tuple[List[float], List[float]]:
        """获取末端执行器位姿"""
        link_state = p.getLinkState(
            self.bestman.arm_id,
            self.bestman.eef_id,
            physicsClientId=self.bestman.client_id
        )
        return link_state[0], link_state[1]
    
    def stop(self):
        """停止移动"""
        if hasattr(self.bestman, 'sim_stop_base'):
            self.bestman.sim_stop_base()
    
    def emergency_stop(self):
        """紧急停止"""
        self.is_busy = False
        self.current_task = None
        self.stop()
        # 停止机械臂
        for i in range(self.bestman.arm_num_dofs):
            p.setJointMotorControl2(
                self.bestman.arm_id,
                i,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=100,
                physicsClientId=self.bestman.client_id
            )
