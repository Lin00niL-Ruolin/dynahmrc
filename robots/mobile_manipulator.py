"""
MobileManipulator - 移动操作复合机器人类
继承 BestMan 的 MobileManipulation 能力，支持导航+操作
对应 asset/robot/ 中的 mobile_manipulator 类型
"""

from typing import Dict, List, Optional, Any, Tuple
import math
import numpy as np
import pybullet as p


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
        target_orientation: Optional[List[float]] = None
    ) -> bool:
        """
        导航到目标位置
        
        Args:
            target_position: 目标位置 [x, y, z]
            target_orientation: 目标朝向（四元数，可选）
        
        Returns:
            是否成功
        """
        try:
            self.is_busy = True
            self.current_task = f"navigate_to_{target_position}"
            
            # 使用简单导航实现（更稳定）
            self._simple_navigation(target_position)
            
            if target_orientation:
                target_yaw = p.getEulerFromQuaternion(target_orientation)[2]
                self.rotate_to_yaw(target_yaw)
            
            self._update_pose()
            return True
            
        except Exception as e:
            import traceback
            self.error_status = f"navigation_failed: {str(e)}\n{traceback.format_exc()}"
            return False
        finally:
            self.is_busy = False
            self.current_task = None
    
    def _simple_navigation(self, target_pos: List[float]):
        """简单导航实现（备用）"""
        max_steps = 1000
        step = 0
        
        while step < max_steps:
            self._update_pose()
            
            dx = target_pos[0] - self.position[0]
            dy = target_pos[1] - self.position[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < self.navigation_threshold:
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
    
    def pick(self, object_id: int, grasp_pose: Optional[Dict] = None) -> bool:
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
                
                if not self.navigate_to(approach_pos):
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
    
    def place(self, target_position: List[float]) -> bool:
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
                if not self.navigate_to(approach_pos):
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
