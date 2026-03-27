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
        
        # 路径规划器（A* + DWA）- 初始为None，可以通过set_path_planner设置外部实例
        self.path_planner = None
        self._client_id = self.bestman.client_id
        
        # 更新初始位置
        self._update_pose()
    
    def set_path_planner(self, path_planner):
        """
        设置外部路径规划器实例
        
        Args:
            path_planner: PathPlanner 实例，用于与提示词计算共享相同的障碍物地图
        """
        self.path_planner = path_planner
        print(f"[MobileManipulator] {self.robot_id} 使用外部路径规划器实例")
    
    def _get_path_planner(self):
        """
        获取路径规划器实例
        如果没有设置外部实例，则创建一个新的
        """
        if self.path_planner is None:
            print(f"[MobileManipulator] {self.robot_id} 创建新的路径规划器实例")
            self.path_planner = PathPlanner(client_id=self._client_id)
        return self.path_planner
    
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
    ) -> Tuple[bool, str]:
        """
        导航到目标位置（使用 A* + DWA 路径规划）
        
        Args:
            target_position: 目标位置 [x, y, z]
            target_orientation: 目标朝向（四元数，可选）
            scene_objects: 场景物体信息，用于避障
        
        Returns:
            (是否成功, 消息)
        """
        try:
            self.is_busy = True
            self.current_task = f"navigate_to_{target_position}"
            
            print(f"[MobileManipulator] 开始导航到 {target_position}")
            print(f"[MobileManipulator] 当前位置: {self.position}, 场景物体数: {len(scene_objects) if scene_objects else 0}")
            
            # 获取路径规划器实例
            planner = self._get_path_planner()
            
            # 更新障碍物信息
            if scene_objects:
                print(f"[MobileManipulator] 更新障碍物信息...")
                planner.update_obstacles_from_scene(scene_objects)
                print(f"[MobileManipulator] 障碍物更新完成")
            
            # 规划全局路径（A*），PathPlanner 会自动处理替代目标
            start_pos = [self.position[0], self.position[1]]
            goal_pos = [target_position[0], target_position[1]]
            
            print(f"[MobileManipulator] 规划全局路径: {start_pos} -> {goal_pos}")
            global_path = planner.plan_global_path(
                start_pos, goal_pos,
                scene_objects=scene_objects or {},
                max_search_radius=5.0,
                radius_step=0.5
            )
            
            if global_path is None:
                print("[MobileManipulator] 全局路径规划失败，无法找到可达路径")
                return False, f"目标 {goal_pos} 不可达，且无法找到替代目标"
            
            print(f"[MobileManipulator] 全局路径规划成功，路径点数: {len(global_path)}")
            
            # 可视化规划的路径
            self._visualize_path(global_path, color=[0, 1, 0], line_width=2)  # 绿色路径
            
            # 使用 DWA 沿路径导航
            print(f"[MobileManipulator] 开始 DWA 导航...")
            reached = self._dwa_navigation(global_path, scene_objects or {})
            print(f"[MobileManipulator] DWA 导航完成")
            
            # 可视化实际走过的路径
            if hasattr(self, '_actual_path') and self._actual_path:
                self._visualize_path(self._actual_path, color=[1, 0, 0], line_width=3)  # 红色实际路径
            
            if reached:
                # 调整最终朝向
                if target_orientation:
                    target_yaw = p.getEulerFromQuaternion(target_orientation)[2]
                    self.rotate_to_yaw(target_yaw)
                
                self._update_pose()
                final_distance = math.sqrt(
                    (goal_pos[0] - self.position[0]) ** 2 +
                    (goal_pos[1] - self.position[1]) ** 2
                )
                print(f"[MobileManipulator] 导航完成，当前位置: {self.position}, 距离原目标: {final_distance:.3f}m")
                
                if final_distance < 0.5:  # 距离原目标足够近
                    return True, f"成功导航到目标附近，距离原目标 {final_distance:.3f}m"
                else:
                    return True, f"导航到替代目标，距离原目标 {final_distance:.3f}m"
            else:
                return False, "DWA 导航未完成（可能卡住）"
            
        except Exception as e:
            import traceback
            self.error_status = f"navigation_failed: {str(e)}\n{traceback.format_exc()}"
            print(f"[MobileManipulator] 导航失败: {self.error_status}")
            return False, str(e)
        finally:
            self.is_busy = False
            self.current_task = None
    
    def _visualize_path(self, path: List[List[float]], color: List[float] = [0, 1, 0], line_width: int = 2):
        """可视化路径
        
        Args:
            path: 路径点列表 [[x, y], [x, y], ...]
            color: RGB 颜色
            line_width: 线宽
        """
        try:
            import pybullet as p
            
            if not path or len(path) < 2:
                return
            
            # 创建路径线条
            for i in range(len(path) - 1):
                p.addUserDebugLine(
                    lineFromXYZ=[path[i][0], path[i][1], 0.1],
                    lineToXYZ=[path[i+1][0], path[i+1][1], 0.1],
                    lineColorRGB=color,
                    lineWidth=line_width,
                    lifeTime=10.0,  # 10秒后消失
                    physicsClientId=self.bestman.client_id
                )
            
            # 在每个路径点添加小标记
            for i, point in enumerate(path):
                if i % max(1, len(path) // 10) == 0:  # 每隔几个点标记一次
                    p.addUserDebugText(
                        text=str(i),
                        textPosition=[point[0], point[1], 0.2],
                        textColorRGB=color,
                        textSize=1.0,
                        lifeTime=10.0,
                        physicsClientId=self.bestman.client_id
                    )
            
            print(f"[MobileManipulator] 路径可视化完成: {len(path)} 个点, 颜色: {color}")
            
        except Exception as e:
            print(f"[MobileManipulator] 路径可视化失败: {e}")
    
    def _dwa_navigation(self, global_path: List[List[float]], scene_objects: Dict[str, Dict]) -> bool:
        """
        使用 DWA 沿全局路径导航
        
        Returns:
            是否成功到达目标
        """
        max_steps = 1000
        step = 0
        current_v = 0.0
        current_yaw_rate = 0.0
        
        print(f"[MobileManipulator] 开始 DWA 导航，全局路径 {len(global_path)} 点")
        print(f"[MobileManipulator] 起点: {self.position[:2]}, 终点: {global_path[-1]}")
        
        # 提取障碍物位置用于调试（排除自己）
        obstacle_positions = []
        obstacle_details = []
        for obj_name, obj_info in scene_objects.items():
            obj_type = obj_info.get('type', 'unknown')
            # 排除可抓取物体和机器人自己
            if obj_type == 'graspable':
                continue
            # 排除自己（匹配 robot_id 或 robot_{robot_id}）
            if obj_name == self.robot_id or obj_name == f"robot_{self.robot_id}":
                print(f"[MobileManipulator] 排除自己 '{obj_name}' 不作为障碍物")
                continue
            pos = obj_info.get('position', [0, 0, 0])
            obstacle_positions.append([pos[0], pos[1]])
            obstacle_details.append(f"{obj_name}({obj_type}):[{pos[0]:.2f},{pos[1]:.2f}]")
        print(f"[MobileManipulator] 障碍物数量: {len(obstacle_positions)}")
        print(f"[MobileManipulator] 障碍物详情: {obstacle_details}")
        
        # 记录是否卡住
        stuck_counter = 0
        last_position = [self.position[0], self.position[1]]
        stuck_threshold = 100  # 100步内移动距离小于阈值则认为卡住
        
        # 记录实际走过的路径
        self._actual_path = [[self.position[0], self.position[1]]]
        
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
                return True
            
            # 检查是否卡住
            moved_distance = math.sqrt(
                (self.position[0] - last_position[0]) ** 2 +
                (self.position[1] - last_position[1]) ** 2
            )
            if moved_distance < 0.01:  # 移动距离小于1cm
                stuck_counter += 1
                if stuck_counter > stuck_threshold:
                    print(f"[MobileManipulator] 检测到卡住，已停止移动 {stuck_counter} 步")
                    return False
            else:
                stuck_counter = 0
                last_position = [self.position[0], self.position[1]]
            
            # 检查是否碰撞障碍物（排除自己）
            collision = False
            for obj_name, obj_info in scene_objects.items():
                if obj_info.get('type') == 'graspable':
                    continue
                # 排除自己（匹配 robot_id 或 robot_{robot_id}）
                if obj_name == self.robot_id or obj_name == f"robot_{self.robot_id}":
                    continue
                obs_pos = obj_info.get('position', [0, 0, 0])
                dist_to_obs = math.sqrt(
                    (obs_pos[0] - self.position[0]) ** 2 +
                    (obs_pos[1] - self.position[1]) ** 2
                )
                if dist_to_obs < 0.3:  # 机器人半径
                    print(f"[MobileManipulator] ⚠️ 警告: 距离障碍物 '{obj_name}' {dist_to_obs:.3f}m，位置 [{obs_pos[0]:.3f}, {obs_pos[1]:.3f}]")
                    collision = True
            
            # 使用 DWA 计算速度
            planner = self._get_path_planner()
            v, yaw_rate = planner.compute_velocity(
                [self.position[0], self.position[1]],
                self.yaw,
                current_v,
                current_yaw_rate,
                scene_objects
            )
            
            # 应用速度
            self._apply_velocity(v, yaw_rate, scene_objects)
            
            current_v = v
            current_yaw_rate = yaw_rate
            
            step += 1
            self.bestman.client.run(5)
            
            # 记录实际走过的路径点
            if step % 5 == 0:  # 每5步记录一次
                self._actual_path.append([self.position[0], self.position[1]])
            
            if step % 50 == 0:
                print(f"[MobileManipulator] 导航中... 步骤: {step}, 位置: [{self.position[0]:.3f}, {self.position[1]:.3f}], 速度: {v:.3f}m/s")
        
        if step >= max_steps:
            print(f"[MobileManipulator] 导航达到最大步数 {max_steps}")
        print(f"[MobileManipulator] DWA 导航结束，最终位置: [{self.position[0]:.3f}, {self.position[1]:.3f}]")
        return False
    
    def _apply_velocity(self, v: float, yaw_rate: float, scene_objects: Dict[str, Dict] = None):
        """应用速度指令，带障碍物检测，同时移动底座和机械臂"""
        # 计算新位置
        dt = 0.1
        new_yaw = self.yaw + yaw_rate * dt
        new_x = self.position[0] + v * math.cos(new_yaw) * dt
        new_y = self.position[1] + v * math.sin(new_yaw) * dt
        
        # 检查新位置是否会碰撞障碍物（排除自己）
        if scene_objects:
            for obj_name, obj_info in scene_objects.items():
                if obj_info.get('type') == 'graspable':
                    continue
                # 排除自己（匹配 robot_id 或 robot_{robot_id}）
                if obj_name == self.robot_id or obj_name == f"robot_{self.robot_id}":
                    continue
                obs_pos = obj_info.get('position', [0, 0, 0])
                dist_to_obs = math.sqrt(
                    (obs_pos[0] - new_x) ** 2 +
                    (obs_pos[1] - new_y) ** 2
                )
                if dist_to_obs < 0.35:  # 机器人半径 + 安全距离
                    # 会碰撞，停止移动
                    print(f"[MobileManipulator] 🚫 碰撞检测: 新位置 [{new_x:.3f}, {new_y:.3f}] 距离障碍物 '{obj_name}' {dist_to_obs:.3f}m，位置 [{obs_pos[0]:.3f}, {obs_pos[1]:.3f}]，停止移动")
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
        2. 如有必要，导航到合适位置（排除目标物体作为障碍物）
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

                # 准备导航用的场景物体列表，排除目标物体（不要把它当障碍物）
                navigation_scene_objects = {}
                if scene_objects:
                    for obj_name, obj_info in scene_objects.items():
                        # 排除目标物体（通过 object_id 或物体名称匹配）
                        if obj_info.get('id') != object_id:
                            navigation_scene_objects[obj_name] = obj_info
                        else:
                            print(f"[MobileManipulator] pick: 排除目标物体 '{obj_name}' 不作为障碍物")
                
                if not self.navigate_to(approach_pos, scene_objects=navigation_scene_objects):
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
            print(f"[MobileManipulator] 到目标距离: {distance:.3f}m, 操作范围: {self.manipulation_range}m")
            print(f"[MobileManipulator] 当前机器人位置: {self.position}, 目标位置: {target_position}")

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

                # 准备导航用的场景物体列表，排除手中持有的物体（不要把它当障碍物）
                navigation_scene_objects = {}
                if scene_objects:
                    for obj_name, obj_info in scene_objects.items():
                        # 排除手中持有的物体
                        if obj_info.get('id') != self.held_object_id:
                            navigation_scene_objects[obj_name] = obj_info
                        else:
                            print(f"[MobileManipulator] place: 排除手中物体 '{obj_name}' 不作为障碍物")

                print(f"[MobileManipulator] 导航到接近位置: {approach_pos}")
                if not self.navigate_to(approach_pos, scene_objects=navigation_scene_objects):
                    self.error_status = "place_failed: 导航到接近位置失败"
                    return False

                self.rotate_to_yaw(angle)
            
            # 执行放置
            pre_place_pos = [target_position[0], target_position[1], target_position[2] + 0.1]
            place_pos = target_position
            
            # 检查机械臂是否能到达目标位置（通过尝试 IK）
            print(f"[MobileManipulator] 检查机械臂是否能到达目标...")
            try:
                test_joints = p.calculateInverseKinematics(
                    self.bestman.arm_id,
                    self.bestman.eef_id,
                    place_pos,
                    maxNumIterations=100,
                    physicsClientId=self.bestman.client_id
                )
                print(f"[MobileManipulator] IK 求解成功，可以尝试放置")
            except Exception as e:
                print(f"[MobileManipulator] 警告: IK 求解失败，目标可能不可达: {e}")
                print(f"[MobileManipulator] 尝试调整机器人位置...")
                # 这里可以添加调整机器人位置的逻辑
            
            # 1. 移动到预放置位置
            print(f"[MobileManipulator] === 开始移动到预放置位置: {pre_place_pos} ===")
            result = self._move_arm_to_position(pre_place_pos)
            print(f"[MobileManipulator] 移动到预放置位置结果: {result}")
            if not result:
                self.error_status = "place_failed: 移动到预放置位置失败"
                return False
            
            # 2. 下降到放置位置
            print(f"[MobileManipulator] === 开始下降到放置位置: {place_pos} ===")
            result = self._move_arm_to_position(place_pos)
            print(f"[MobileManipulator] 下降到放置位置结果: {result}")
            if not result:
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
        target_orientation: Optional[List[float]] = None,
        max_steps: int = 100,
        tolerance: float = 0.05
    ) -> bool:
        """移动机械臂到目标位置，并等待到达"""
        print(f"[MobileManipulator] _move_arm_to_position 被调用，目标: {target_position}")
        try:
            from Robotics_API.Pose import Pose
            target_pose = Pose(target_position, target_orientation or [0, 0, 0, 1])
            
            has_sim_move = hasattr(self.bestman, 'sim_move_arm_to_target_pose')
            print(f"[MobileManipulator] bestman 有 sim_move_arm_to_target_pose: {has_sim_move}")
            
            if has_sim_move:
                print(f"[MobileManipulator] 调用 sim_move_arm_to_target_pose 目标: {target_position}")
                self.bestman.sim_move_arm_to_target_pose(target_pose)
                
                # 等待机械臂到达目标位置
                print(f"[MobileManipulator] 等待机械臂到达目标...")
                for step in range(max_steps):
                    # 获取当前末端执行器位置
                    if hasattr(self.bestman, 'eef_id') and hasattr(self.bestman, 'arm_id'):
                        eef_state = p.getLinkState(
                            self.bestman.arm_id, 
                            self.bestman.eef_id,
                            physicsClientId=self.bestman.client_id
                        )
                        current_pos = eef_state[0]
                        
                        # 计算距离
                        distance = math.sqrt(
                            (current_pos[0] - target_position[0])**2 +
                            (current_pos[1] - target_position[1])**2 +
                            (current_pos[2] - target_position[2])**2
                        )
                        
                        if step % 20 == 0:  # 每20步打印一次
                            print(f"[MobileManipulator] 步骤 {step}: 当前位置 {current_pos}, 目标 {target_position}, 距离 {distance:.4f}m")
                        
                        if distance < tolerance:
                            print(f"[MobileManipulator] 机械臂到达目标位置: {target_position}, 误差: {distance:.4f}m")
                            return True
                    else:
                        print(f"[MobileManipulator] 警告: bestman 没有 eef_id 或 arm_id 属性")
                        return True  # 无法检查位置，直接返回成功
                    
                    # 运行一步仿真
                    self.bestman.client.run(1)
                
                # 超时未到达
                print(f"[MobileManipulator] 警告: 机械臂未能在 {max_steps} 步内到达目标位置")
                # 获取最终位置
                if hasattr(self.bestman, 'eef_id') and hasattr(self.bestman, 'arm_id'):
                    eef_state = p.getLinkState(
                        self.bestman.arm_id, 
                        self.bestman.eef_id,
                        physicsClientId=self.bestman.client_id
                    )
                    final_pos = eef_state[0]
                    final_distance = math.sqrt(
                        (final_pos[0] - target_position[0])**2 +
                        (final_pos[1] - target_position[1])**2 +
                        (final_pos[2] - target_position[2])**2
                    )
                    print(f"[MobileManipulator] 最终位置: {final_pos}, 距离目标: {final_distance:.4f}m")
                return False
            else:
                return self._simple_ik_move(target_position, target_orientation)
                
        except Exception as e:
            print(f"[MobileManipulator] _move_arm_to_position 错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _simple_ik_move(self, target_pos, target_orn, max_steps=100, tolerance=0.05):
        """简单的 IK 移动，带位置检查"""
        try:
            print(f"[MobileManipulator] _simple_ik_move 开始，目标: {target_pos}")
            
            joint_positions = p.calculateInverseKinematics(
                self.bestman.arm_id,
                self.bestman.eef_id,
                target_pos,
                targetOrientation=target_orn,
                maxNumIterations=100,
                physicsClientId=self.bestman.client_id
            )
            
            print(f"[MobileManipulator] IK 求解完成，关节数: {len(joint_positions)}")
            
            for i, pos in enumerate(joint_positions[:self.bestman.arm_num_dofs]):
                p.setJointMotorControl2(
                    self.bestman.arm_id,
                    i,
                    p.POSITION_CONTROL,
                    pos,
                    force=100,
                    physicsClientId=self.bestman.client_id
                )
            
            # 等待并检查是否到达目标
            print(f"[MobileManipulator] 等待机械臂移动...")
            for step in range(max_steps):
                self.bestman.client.run(1)
                
                # 获取当前末端位置
                eef_state = p.getLinkState(
                    self.bestman.arm_id,
                    self.bestman.eef_id,
                    physicsClientId=self.bestman.client_id
                )
                current_pos = eef_state[0]
                
                distance = math.sqrt(
                    (current_pos[0] - target_pos[0])**2 +
                    (current_pos[1] - target_pos[1])**2 +
                    (current_pos[2] - target_pos[2])**2
                )
                
                if step % 20 == 0:
                    print(f"[MobileManipulator] IK 步骤 {step}: 当前 {current_pos}, 距离 {distance:.4f}m")
                
                if distance < tolerance:
                    print(f"[MobileManipulator] IK 移动成功，到达目标，误差: {distance:.4f}m")
                    return True
            
            # 超时
            print(f"[MobileManipulator] IK 移动超时，未能到达目标")
            return False
            
        except Exception as e:
            print(f"[MobileManipulator] _simple_ik_move 错误: {e}")
            import traceback
            traceback.print_exc()
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
