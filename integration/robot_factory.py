"""
RobotFactory - 机器人动态工厂
根据 LLM 协调器的任务分配决策，动态创建对应类型的 BestMan 机器人
"""

from typing import Dict, List, Optional, Any, Type
import os


class RobotFactory:
    """
    机器人工厂类
    
    职责:
    - 根据机器人类型动态实例化 BestMan 控制器
    - 维护 robot_id → BestMan 实例的映射表
    - 管理机器人配置和初始化参数
    """
    
    # 默认 URDF 路径配置
    DEFAULT_URDF_PATHS = {
        "arm": {
            "panda": "Asset/Robot/mobile_manipulator/arm/franka/urdf/panda.urdf",
            "xarm6": "Asset/Robot/mobile_manipulator/arm/ufactory/urdf/xarm6.urdf",
        },
        "mobile_base": {
            "segbot": "Asset/Robot/mobile_manipulator/base/segbot/urdf/segbot.urdf",
        },
        "mobile_manipulator": {
            # 移动操作复合机器人通常需要组合基座和手臂
            "panda_on_segbot": {
                "base": "Asset/Robot/mobile_manipulator/base/segbot/urdf/segbot.urdf",
                "arm": "Asset/Robot/mobile_manipulator/arm/franka/urdf/panda.urdf",
            }
        }
    }
    
    def __init__(self, client: Any, visualizer: Any, config_loader: Optional[Any] = None):
        """
        初始化机器人工厂
        
        Args:
            client: PyBullet 客户端实例
            visualizer: 可视化器实例
            config_loader: 配置加载器（可选）
        """
        self.client = client
        self.visualizer = visualizer
        self.config_loader = config_loader
        
        # 机器人注册表
        self.robot_registry: Dict[str, Any] = {}
        
        # 机器人配置缓存
        self.robot_configs: Dict[str, Dict] = {}
    
    def create_robot(
        self,
        robot_id: str,
        robot_type: str,
        robot_model: str,
        init_position: List[float] = None,
        init_orientation: List[float] = None,
        custom_config: Optional[Dict] = None
    ) -> Any:
        """
        创建机器人实例
        
        Args:
            robot_id: 机器人唯一标识
            robot_type: 机器人类型（arm, mobile_base, mobile_manipulator）
            robot_model: 机器人型号（panda, xarm6, segbot 等）
            init_position: 初始位置 [x, y, z]
            init_orientation: 初始朝向（四元数）[x, y, z, w]
            custom_config: 自定义配置（可选）
        
        Returns:
            DynaHMRC 机器人包装实例
        """
        # 检查是否已存在
        if robot_id in self.robot_registry:
            raise ValueError(f"机器人 {robot_id} 已存在")
        
        # 获取配置
        config = custom_config or self._load_config(robot_type, robot_model)
        
        # 设置初始位姿
        if init_position:
            config['base_init_pose'] = init_position + (init_orientation or [0, 0, 0, 1])
        
        # 根据类型创建对应的 BestMan 实例
        if robot_type == "arm":
            bestman_instance = self._create_arm_robot(robot_model, config)
            from ..robots.arm_robot import ArmRobot
            robot = ArmRobot(robot_id, bestman_instance)
            
        elif robot_type == "mobile_base":
            bestman_instance = self._create_mobile_base(robot_model, config)
            from ..robots.mobile_base import MobileBase
            robot = MobileBase(robot_id, bestman_instance)
            
        elif robot_type == "mobile_manipulator":
            bestman_instance = self._create_mobile_manipulator(robot_model, config)
            from ..robots.mobile_manipulator import MobileManipulator
            robot = MobileManipulator(robot_id, bestman_instance)
            
        else:
            raise ValueError(f"不支持的机器人类型: {robot_type}")
        
        # 注册到映射表
        self.robot_registry[robot_id] = robot
        self.robot_configs[robot_id] = {
            "type": robot_type,
            "model": robot_model,
            "config": config
        }
        
        print(f"[RobotFactory] 创建机器人成功: {robot_id} ({robot_type}/{robot_model})")
        return robot
    
    def _create_arm_robot(self, model: str, config: Dict) -> Any:
        """创建固定机械臂 BestMan 实例"""
        # 获取项目根目录
        root_dir = self._get_project_root()
        
        if model == "panda":
            from Robotics_API.Bestman_sim_panda_with_gripper import Bestman_sim_panda_with_gripper
            
            # 构建配置对象
            cfg = self._build_config(config, {
                'arm_urdf_path': os.path.join(root_dir, "Asset/Robot/mobile_manipulator/arm/franka/urdf/panda.urdf"),
                'arm_num_dofs': 7,
                'eef_id': 7,
                'tcp_link': 7,
                'arm_place_height': 1.0,
                'arm_reset_jointValues': [0.0, -0.5, 0.0, -2.5, 0.0, 2.0, 0.0],
            })
            
            return Bestman_sim_panda_with_gripper(self.client, self.visualizer, cfg)
            
        elif model == "xarm6":
            # 可以添加 xarm6 支持
            raise NotImplementedError("xarm6 支持待实现")
        else:
            raise ValueError(f"不支持的机械臂型号: {model}")
    
    def _create_mobile_base(self, model: str, config: Dict) -> Any:
        """创建移动基座 BestMan 实例"""
        root_dir = self._get_project_root()
        
        if model == "segbot":
            # 移动基座通常使用 Bestman_sim 基类或专门的移动基座类
            from Robotics_API.Bestman_sim import Bestman_sim
            
            cfg = self._build_config(config, {
                'base_urdf_path': os.path.join(root_dir, "Asset/Robot/mobile_manipulator/base/segbot/urdf/segbot.urdf"),
                'arm_num_dofs': 0,  # 无机械臂
            })
            
            # 创建简化版 BestMan（仅基座）
            return MobileBaseAdapter(self.client, self.visualizer, cfg)
        else:
            raise ValueError(f"不支持的移动基座型号: {model}")
    
    def _create_mobile_manipulator(self, model: str, config: Dict) -> Any:
        """创建移动操作复合 BestMan 实例"""
        root_dir = self._get_project_root()
        
        if model == "panda_on_segbot":
            from Robotics_API.Bestman_sim_panda_with_gripper import Bestman_sim_panda_with_gripper
            
            cfg = self._build_config(config, {
                'base_urdf_path': os.path.join(root_dir, "Asset/Robot/mobile_manipulator/base/segbot/urdf/segbot.urdf"),
                'arm_urdf_path': os.path.join(root_dir, "Asset/Robot/mobile_manipulator/arm/franka/urdf/panda.urdf"),
                'arm_num_dofs': 7,
                'eef_id': 7,
                'tcp_link': 7,
                'arm_place_height': 1.0,
                'arm_reset_jointValues': [0.0, -0.5, 0.0, -2.5, 0.0, 2.0, 0.0],
            })
            
            return Bestman_sim_panda_with_gripper(self.client, self.visualizer, cfg)
        else:
            raise ValueError(f"不支持的移动操作型号: {model}")
    
    def _build_config(self, custom_config: Dict, defaults: Dict) -> Any:
        """构建配置对象"""
        # 合并默认配置和自定义配置
        merged = {**defaults, **custom_config}
        
        # 创建简单的配置对象
        class Config:
            pass
        
        cfg = Config()
        cfg.Robot = type('Robot', (), merged)()
        cfg.Controller = type('Controller', (), {
            'Kp': 1.0,
            'Ki': 0.1,
            'Kd': 0.05,
            'target_distance': 0.0
        })()
        cfg.Camera = type('Camera', (), {'enabled': False})()
        
        return cfg
    
    def _load_config(self, robot_type: str, model: str) -> Dict:
        """加载机器人配置"""
        # 简化实现，返回空配置
        return {}
    
    def _get_project_root(self) -> str:
        """获取项目根目录"""
        # 从当前文件路径推断
        current_file = os.path.abspath(__file__)
        # dynahmrc/integration/robot_factory.py -> 根目录
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        return root_dir
    
    def get_robot(self, robot_id: str) -> Optional[Any]:
        """获取机器人实例"""
        return self.robot_registry.get(robot_id)
    
    def remove_robot(self, robot_id: str) -> bool:
        """移除机器人"""
        if robot_id in self.robot_registry:
            del self.robot_registry[robot_id]
            del self.robot_configs[robot_id]
            return True
        return False
    
    def get_all_robots(self) -> Dict[str, Any]:
        """获取所有机器人"""
        return self.robot_registry.copy()
    
    def get_robot_info(self, robot_id: str) -> Optional[Dict]:
        """获取机器人信息"""
        if robot_id not in self.robot_registry:
            return None
        
        robot = self.robot_registry[robot_id]
        config = self.robot_configs[robot_id]
        
        return {
            "robot_id": robot_id,
            "robot_type": config["type"],
            "robot_model": config["model"],
            "capabilities": robot.capabilities,
            "state": robot.get_state()
        }


class MobileBaseAdapter:
    """
    移动基座适配器
    为纯移动基座提供统一的 BestMan 接口
    """
    
    def __init__(self, client, visualizer, cfg):
        self.client = client
        self.visualizer = visualizer
        self.client_id = client.get_client_id()
        self.robot_cfg = cfg.Robot
        
        # 初始化基座
        from Robotics_API.Pose import Pose
        self.base_init_pose = Pose(
            self.robot_cfg.base_init_pose[:3],
            self.robot_cfg.base_init_pose[3:]
        )
        
        self.base_id = self.client.load_object(
            obj_name="mobile_base",
            model_path=self.robot_cfg.base_urdf_path,
            object_position=self.base_init_pose.get_position(),
            object_orientation=self.base_init_pose.get_orientation(),
            fixed_base=False
        )
        
        # 初始化控制器
        from Controller import PIDController
        self.distance_controller = PIDController(
            Kp=1.0, Ki=0.1, Kd=0.05, setpoint=0.0
        )
        
        self.current_base_yaw = self.base_init_pose.get_orientation("euler")[2]
        self.arm_num_dofs = 0  # 无机械臂
    
    def sim_get_current_base_pose(self):
        """获取当前基座位姿"""
        from Robotics_API.Pose import Pose
        pos, orn = self.client.get_object_pose(self.base_id)
        return Pose(pos, orn)
    
    def sim_stop_base(self):
        """停止基座"""
        import pybullet as p
        p.resetBaseVelocity(
            self.base_id, [0, 0, 0], [0, 0, 0],
            physicsClientId=self.client_id
        )
    
    def sim_rotate_base_to_target_yaw(self, target_yaw, gradual=True, **kwargs):
        """旋转到目标朝向"""
        import math
        import pybullet as p
        
        def angle_to_quaternion(yaw):
            return [0, 0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
        
        orientation = angle_to_quaternion(target_yaw)
        position = self.sim_get_current_base_pose().get_position()
        
        p.resetBasePositionAndOrientation(
            self.base_id, position, orientation,
            physicsClientId=self.client_id
        )
        self.current_base_yaw = target_yaw
        self.client.run(5)
    
    def sim_move_base_to_waypoint(self, waypoint, threshold=0.01):
        """导航到路径点"""
        import math
        import pybullet as p
        
        while True:
            pose = self.sim_get_current_base_pose()
            x, y = pose.get_position()[:2]
            target = waypoint
            
            distance = math.sqrt((target.y - y)**2 + (target.x - x)**2)
            if distance < threshold:
                break
            
            yaw = math.atan2(target.y - y, target.x - x)
            self.sim_rotate_base_to_target_yaw(yaw)
            
            # 向前移动
            step_size = min(0.02, distance)
            new_x = x + step_size * math.cos(yaw)
            new_y = y + step_size * math.sin(yaw)
            
            p.resetBasePositionAndOrientation(
                self.base_id,
                [new_x, new_y, pose.get_position()[2]],
                [0, 0, math.sin(yaw/2), math.cos(yaw/2)],
                physicsClientId=self.client_id
            )
            self.client.run(1)
    
    def sim_move_base_forward(self, distance, **kwargs):
        """向前移动"""
        import math
        import pybullet as p
        
        pose = self.sim_get_current_base_pose()
        pos = pose.get_position()
        
        new_x = pos[0] + distance * math.cos(self.current_base_yaw)
        new_y = pos[1] + distance * math.sin(self.current_base_yaw)
        
        p.resetBasePositionAndOrientation(
            self.base_id,
            [new_x, new_y, pos[2]],
            pose.get_orientation(),
            physicsClientId=self.client_id
        )
        self.client.run(10)
    
    def sim_move_base_backward(self, distance, **kwargs):
        """向后移动"""
        self.sim_move_base_forward(-distance)
