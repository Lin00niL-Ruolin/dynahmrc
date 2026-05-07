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
        
        # 固定机械臂特性
        self.is_fixed_base = True  # 标记为固定基座，不可移动
        self.can_navigate = False  # 无导航能力
        
        # 状态信息
        self.position = [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, 0.0, 1.0]
        self.is_busy = False
        self.current_task = None
        self.error_status = None
        
        # 更新初始位置
        self._update_pose()
        
        # 同步机械臂位置与基座高度（修复基座抬高后手臂不跟随的问题）
        self._sync_arm_position_with_base()
        
        print(f"[ArmRobot] 固定机械臂 {robot_id} 初始化完成 (不可移动)")
    
    def _sync_arm_position_with_base(self):
        """
        同步机械臂位置与基座位置
        修复：当基座位置改变时，机械臂也应相应移动
        """
        try:
            # 获取基座当前位置
            base_pose = self.bestman.sim_get_current_base_pose()
            base_pos = base_pose.get_position()
            base_orn = base_pose.get_orientation()
            
            # 获取机械臂当前位置
            arm_id = self.bestman.arm_id
            arm_pos, arm_orn = p.getBasePositionAndOrientation(
                arm_id, physicsClientId=self.bestman.client_id
            )
            
            # 计算机械臂应该在的位置（基座上方0.1米）
            target_arm_pos = [base_pos[0], base_pos[1], base_pos[2] + 0.1]
            
            # 检查位置是否需要同步（X、Y、Z任一坐标差异超过0.01）
            pos_diff = [
                abs(arm_pos[0] - target_arm_pos[0]),
                abs(arm_pos[1] - target_arm_pos[1]),
                abs(arm_pos[2] - target_arm_pos[2])
            ]
            
            if any(diff > 0.01 for diff in pos_diff):
                p.resetBasePositionAndOrientation(
                    arm_id,
                    target_arm_pos,
                    base_orn,  # 使用基座朝向
                    physicsClientId=self.bestman.client_id
                )
                print(f"[ArmRobot] 机械臂位置已同步:")
                print(f"         从: [{arm_pos[0]:.2f}, {arm_pos[1]:.2f}, {arm_pos[2]:.2f}]")
                print(f"         到: [{target_arm_pos[0]:.2f}, {target_arm_pos[1]:.2f}, {target_arm_pos[2]:.2f}]")
                
                # 运行几步仿真让位置生效
                for _ in range(10):
                    p.stepSimulation(physicsClientId=self.bestman.client_id)
                
        except Exception as e:
            print(f"[ArmRobot] 同步机械臂位置失败: {e}")
    
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
            "is_fixed_base": self.is_fixed_base,  # 固定基座标记
            "can_navigate": self.can_navigate,    # 导航能力标记
            "position": self.position,
            "orientation": self.orientation,
            "is_busy": self.is_busy,
            "current_task": self.current_task,
            "error_status": self.error_status,
            "capabilities": self.capabilities
        }
    
    def pick(self, object_id: int, grasp_pose: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        抓取物体
        
        Args:
            object_id: PyBullet 物体 ID
            grasp_pose: 抓取位姿（可选，自动计算）
        
        Returns:
            (是否成功, 消息)
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
                return False, "无法移动到预抓取位置"
            
            # 2. 打开夹爪
            self.open_gripper()
            
            # 3. 下降到抓取位置（使用更多步数确保平滑）
            if not self.move_to_position(grasp_pos, steps=30):
                return False, "无法下降到抓取位置"
            
            # 3.5 确保机械臂已稳定到达抓取位置
            self._wait_for_arm_stable(grasp_pos)
            
            # 4. 关闭夹爪（抓取）
            self.close_gripper()
            
            # 5. 创建约束（模拟抓取）
            self._create_grasp_constraint(object_id)
            
            # 6. 抬升
            if not self.move_to_position(pre_grasp_pos):
                return False, "无法抬升"
            
            return True, "抓取成功"
            
        except Exception as e:
            self.error_status = f"pick_failed: {str(e)}"
            return False, str(e)
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
            
            # 2. 下降到放置位置（使用更多步数确保平滑到达）
            if not self.move_to_position(place_pos, steps=30):
                return False
            
            # 2.5 确保机械臂已稳定到达目标位置
            self._wait_for_arm_stable(place_pos)
            
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
        target_orientation: Optional[List[float]] = None,
        steps: int = 20
    ) -> bool:
        """
        移动末端执行器到目标位置
        
        Args:
            target_position: 目标位置 [x, y, z]
            target_orientation: 目标姿态（四元数）[x, y, z, w]
            steps: 插值步数，越大运动越平滑
        
        Returns:
            是否成功
        """
        try:
            # 使用 BestMan 的 sim_move_eef_to_goal_pose 进行平滑移动
            if hasattr(self.bestman, 'sim_move_eef_to_goal_pose'):
                from Robotics_API.Pose import Pose
                target_pose = Pose(target_position, target_orientation or [0, 0, 0, 1])
                
                # 执行平滑运动，使用更多步数
                self.bestman.sim_move_eef_to_goal_pose(target_pose, steps=steps)
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
            
            # 等待执行 - 增加步数并检查是否到达目标
            max_wait_steps = 200  # 最大等待步数
            for _ in range(max_wait_steps):
                self.bestman.client.run(1)
                # 检查是否接近目标
                current_pos, _ = self.get_end_effector_pose()
                dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
                if dist < 0.02:  # 2cm误差范围内认为到达
                    break
            
            return True
            
        except Exception as e:
            return False
    
    def open_gripper(self):
        """打开夹爪"""
        if hasattr(self.bestman, 'sim_open_gripper'):
            self.bestman.sim_open_gripper()
            # 等待夹爪完全打开
            self.bestman.client.run(50)
    
    def close_gripper(self):
        """关闭夹爪"""
        if hasattr(self.bestman, 'sim_close_gripper'):
            self.bestman.sim_close_gripper()
            # 等待夹爪完全关闭
            self.bestman.client.run(50)
    
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
    
    def _wait_for_arm_stable(self, target_pos: List[float], threshold: float = 0.01, max_steps: int = 100):
        """
        等待机械臂稳定到达目标位置
        
        Args:
            target_pos: 目标位置
            threshold: 位置误差阈值（默认1cm）
            max_steps: 最大等待步数
        """
        for _ in range(max_steps):
            self.bestman.client.run(1)
            current_pos, _ = self.get_end_effector_pose()
            dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
            if dist < threshold:
                print(f"[ArmRobot] 机械臂已稳定到达目标位置，误差: {dist:.4f}m")
                return True
        print(f"[ArmRobot] 警告: 机械臂未完全稳定，当前误差可能较大")
        return False
    
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
