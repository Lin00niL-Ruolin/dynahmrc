"""
ArmRobot - 固定机械臂机器人类
继承 BestMan 的 Manipulation 能力，仅支持操作任务
对应 asset/robot/ 中的 arm 类型（如 panda、xarm6）
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pybullet as p


class ArmRobot:
    """
    固定机械臂机器人包装类
    
    能力:
        - manipulation: 抓取、放置、操作物体
        - perception: 通过相机感知环境
    
    限制:
        - 无移动能力（固定基座）
        - 工作空间受限（臂展范围内）
    """
    
    def __init__(
        self,
        robot_id: str,
        bestman_instance: Any,
        capabilities: Optional[List[str]] = None
    ):
        """
        初始化固定机械臂机器人
        
        Args:
            robot_id: 机器人唯一标识
            bestman_instance: BestMan 机器人实例（如 Bestman_sim_panda_with_gripper）
            capabilities: 能力列表，默认 ["manipulation", "perception"]
        """
        self.robot_id = robot_id
        self.robot_type = "arm"
        self.bestman = bestman_instance
        self.capabilities = capabilities or ["manipulation", "perception"]
        
        # 状态信息
        self.position = [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, 0.0, 1.0]
        self.is_busy = False
        self.current_task = None
        self.error_status = None
        
        # 更新初始位置
        self._update_pose()
    
    def _update_pose(self):
        """从 BestMan 实例更新当前位姿"""
        if hasattr(self.bestman, 'sim_get_current_base_pose'):
            pose = self.bestman.sim_get_current_base_pose()
            self.position = pose.get_position()
            self.orientation = pose.get_orientation()
    
    def get_state(self) -> Dict[str, Any]:
        """获取机器人当前状态"""
        self._update_pose()
        return {
            "robot_id": self.robot_id,
            "robot_type": self.robot_type,
            "position": self.position,
            "orientation": self.orientation,
            "is_busy": self.is_busy,
            "current_task": self.current_task,
            "error_status": self.error_status,
            "capabilities": self.capabilities
        }
    
    def pick(self, object_id: int, grasp_pose: Optional[Dict] = None) -> bool:
        """
        抓取物体
        
        Args:
            object_id: PyBullet 物体 ID
            grasp_pose: 抓取位姿（可选，自动计算）
        
        Returns:
            是否成功
        """
        try:
            self.is_busy = True
            self.current_task = f"pick_object_{object_id}"
            
            # 获取物体位置
            obj_pos, obj_orn = p.getBasePositionAndOrientation(
                object_id, physicsClientId=self.bestman.client_id
            )
            
            # 计算预抓取位置（物体上方）
            pre_grasp_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.1]
            grasp_pos = [obj_pos[0], obj_pos[1], obj_pos[2]]
            
            # 执行抓取流程
            # 1. 移动到预抓取位置
            if not self.move_to_position(pre_grasp_pos):
                return False
            
            # 2. 打开夹爪
            self.open_gripper()
            
            # 3. 下降到抓取位置
            if not self.move_to_position(grasp_pos):
                return False
            
            # 4. 关闭夹爪（抓取）
            self.close_gripper()
            
            # 5. 创建约束（模拟抓取）
            self._create_grasp_constraint(object_id)
            
            # 6. 抬升
            if not self.move_to_position(pre_grasp_pos):
                return False
            
            return True
            
        except Exception as e:
            self.error_status = f"pick_failed: {str(e)}"
            return False
        finally:
            self.is_busy = False
            self.current_task = None
    
    def place(self, target_position: List[float]) -> bool:
        """
        放置物体到目标位置
        
        Args:
            target_position: 目标位置 [x, y, z]
        
        Returns:
            是否成功
        """
        try:
            self.is_busy = True
            self.current_task = f"place_at_{target_position}"
            
            # 计算放置路径
            pre_place_pos = [target_position[0], target_position[1], target_position[2] + 0.1]
            place_pos = target_position
            
            # 1. 移动到预放置位置
            if not self.move_to_position(pre_place_pos):
                return False
            
            # 2. 下降到放置位置
            if not self.move_to_position(place_pos):
                return False
            
            # 3. 打开夹爪（释放）
            self.open_gripper()
            
            # 4. 移除约束
            self._remove_grasp_constraint()
            
            # 5. 抬升
            if not self.move_to_position(pre_place_pos):
                return False
            
            return True
            
        except Exception as e:
            self.error_status = f"place_failed: {str(e)}"
            return False
        finally:
            self.is_busy = False
            self.current_task = None
    
    def move_to_position(
        self,
        target_position: List[float],
        target_orientation: Optional[List[float]] = None
    ) -> bool:
        """
        移动末端执行器到目标位置
        
        Args:
            target_position: 目标位置 [x, y, z]
            target_orientation: 目标姿态（四元数）[x, y, z, w]
        
        Returns:
            是否成功
        """
        try:
            # 使用 BestMan 的 IK 和轨迹执行
            if hasattr(self.bestman, 'sim_move_arm_to_target_pose'):
                # 构建目标位姿
                from Robotics_API.Pose import Pose
                target_pose = Pose(target_position, target_orientation or [0, 0, 0, 1])
                
                # 执行运动
                self.bestman.sim_move_arm_to_target_pose(target_pose)
                return True
            else:
                # 备用：使用简单的关节控制
                return self._simple_ik_move(target_position, target_orientation)
                
        except Exception as e:
            self.error_status = f"move_failed: {str(e)}"
            return False
    
    def _simple_ik_move(self, target_pos, target_orn):
        """简单的 IK 移动（备用）"""
        try:
            # 使用 PyBullet 的 IK
            joint_positions = p.calculateInverseKinematics(
                self.bestman.arm_id,
                self.bestman.eef_id,
                target_pos,
                targetOrientation=target_orn,
                maxNumIterations=100,
                physicsClientId=self.bestman.client_id
            )
            
            # 设置关节位置
            for i, pos in enumerate(joint_positions[:self.bestman.arm_num_dofs]):
                p.setJointMotorControl2(
                    self.bestman.arm_id,
                    i,
                    p.POSITION_CONTROL,
                    pos,
                    force=100,
                    physicsClientId=self.bestman.client_id
                )
            
            # 等待执行
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
                print(f"[ArmRobot] 移除约束 {self.constraint_id}")
            except Exception as e:
                print(f"[ArmRobot] 移除约束失败: {e}")
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
    
    def emergency_stop(self):
        """紧急停止"""
        self.is_busy = False
        self.current_task = None
        # 停止关节运动
        for i in range(self.bestman.arm_num_dofs):
            p.setJointMotorControl2(
                self.bestman.arm_id,
                i,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=100,
                physicsClientId=self.bestman.client_id
            )
