"""
BestManAdapter - BestMan API 适配层
桥接 dyna_hmrc_web 的 LLM 逻辑与 BestMan 仿真平台

核心功能:
1. 将 LLM 输出的 high-level 动作转换为 BestMan 的具体 API 调用
2. 将 BestMan 的执行结果转换为 LLM 可理解的反馈
3. 处理三类机器人的异构动作映射
"""

from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import json
import traceback


class ActionType(Enum):
    """动作类型枚举"""
    NAVIGATE = "navigate"           # 导航到位置
    PICK = "pick"                   # 抓取物体
    PLACE = "place"                 # 放置物体
    TRANSPORT = "transport"         # 运输物体
    ROTATE = "rotate"               # 旋转
    MOVE_FORWARD = "move_forward"   # 向前移动
    MOVE_BACKWARD = "move_backward" # 向后移动
    WAIT = "wait"                   # 等待
    STOP = "stop"                   # 停止
    EMERGENCY_STOP = "emergency_stop"  # 紧急停止


@dataclass
class ExecutionFeedback:
    """执行反馈数据类"""
    success: bool
    action_type: str
    robot_id: str
    message: str
    execution_time: float = 0.0
    error_code: Optional[str] = None
    error_details: Optional[str] = None
    state_changes: Dict[str, Any] = field(default_factory=dict)
    sensor_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于 LLM 消费）"""
        return {
            "success": self.success,
            "action": self.action_type,
            "robot": self.robot_id,
            "message": self.message,
            "execution_time": round(self.execution_time, 2),
            "error": {
                "code": self.error_code,
                "details": self.error_details
            } if not self.success else None,
            "state_changes": self.state_changes,
            "sensor_data": self.sensor_data
        }
    
    def to_llm_string(self) -> str:
        """转换为 LLM 可读的字符串格式"""
        status = "✓ 成功" if self.success else "✗ 失败"
        result = f"[{status}] {self.robot_id} 执行 {self.action_type}: {self.message}"
        if not self.success and self.error_details:
            result += f"\n  错误详情: {self.error_details}"
        return result


class BestManAdapter:
    """
    BestMan API 适配器
    
    职责:
    - 解析 LLM 输出的动作指令
    - 调用对应机器人的 BestMan API
    - 捕获异常并转换为标准化反馈
    - 维护机器人状态映射
    """
    
    def __init__(self, robot_registry: Dict[str, Any]):
        """
        初始化适配器
        
        Args:
            robot_registry: 机器人注册表 {robot_id: robot_instance}
        """
        self.robot_registry = robot_registry
        self.action_handlers: Dict[ActionType, Callable] = {
            ActionType.NAVIGATE: self._handle_navigate,
            ActionType.PICK: self._handle_pick,
            ActionType.PLACE: self._handle_place,
            ActionType.TRANSPORT: self._handle_transport,
            ActionType.ROTATE: self._handle_rotate,
            ActionType.MOVE_FORWARD: self._handle_move_forward,
            ActionType.MOVE_BACKWARD: self._handle_move_backward,
            ActionType.WAIT: self._handle_wait,
            ActionType.STOP: self._handle_stop,
            ActionType.EMERGENCY_STOP: self._handle_emergency_stop,
        }
    
    def execute_action(
        self,
        robot_id: str,
        action: str,
        params: Dict[str, Any]
    ) -> ExecutionFeedback:
        """
        执行动作（主入口）
        
        Args:
            robot_id: 机器人 ID
            action: 动作类型字符串
            params: 动作参数
        
        Returns:
            ExecutionFeedback 执行反馈
        """
        import time
        start_time = time.time()
        
        # 检查机器人是否存在
        if robot_id not in self.robot_registry:
            return ExecutionFeedback(
                success=False,
                action_type=action,
                robot_id=robot_id,
                message=f"机器人 {robot_id} 未找到",
                error_code="ROBOT_NOT_FOUND",
                error_details=f"可用机器人: {list(self.robot_registry.keys())}"
            )
        
        robot = self.robot_registry[robot_id]
        
        # 解析动作类型
        try:
            action_type = ActionType(action.lower())
        except ValueError:
            return ExecutionFeedback(
                success=False,
                action_type=action,
                robot_id=robot_id,
                message=f"未知动作类型: {action}",
                error_code="UNKNOWN_ACTION",
                error_details=f"支持的动作: {[a.value for a in ActionType]}"
            )
        
        # 检查机器人能力是否匹配
        if not self._check_capability(robot, action_type):
            return ExecutionFeedback(
                success=False,
                action_type=action,
                robot_id=robot_id,
                message=f"机器人 {robot_id} 无法执行 {action}",
                error_code="CAPABILITY_MISMATCH",
                error_details=f"机器人类型: {robot.robot_type}, 能力: {robot.capabilities}"
            )
        
        # 执行动作
        try:
            handler = self.action_handlers[action_type]
            success, message, state_changes = handler(robot, params)
            
            execution_time = time.time() - start_time
            
            # 获取传感器数据
            sensor_data = self._get_sensor_data(robot)
            
            # 提取错误信息（如果有）
            error_code = None
            error_details = None
            if not success:
                error_code = "ACTION_FAILED"
                error_details = state_changes.get("error", "Unknown error")
            
            return ExecutionFeedback(
                success=success,
                action_type=action,
                robot_id=robot_id,
                message=message,
                execution_time=execution_time,
                state_changes=state_changes,
                sensor_data=sensor_data,
                error_code=error_code,
                error_details=error_details
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_trace = traceback.format_exc()
            
            return ExecutionFeedback(
                success=False,
                action_type=action,
                robot_id=robot_id,
                message=f"执行异常: {str(e)}",
                execution_time=execution_time,
                error_code="EXECUTION_EXCEPTION",
                error_details=error_trace
            )
    
    def _check_capability(self, robot: Any, action_type: ActionType) -> bool:
        """检查机器人是否有能力执行动作"""
        capability_map = {
            ActionType.NAVIGATE: ["navigation"],
            ActionType.PICK: ["manipulation"],
            ActionType.PLACE: ["manipulation"],
            ActionType.TRANSPORT: ["transport", "navigation"],
            ActionType.ROTATE: ["navigation"],
            ActionType.MOVE_FORWARD: ["navigation"],
            ActionType.MOVE_BACKWARD: ["navigation"],
            ActionType.WAIT: [],  # 所有机器人都可等待
            ActionType.STOP: [],  # 所有机器人都可停止
            ActionType.EMERGENCY_STOP: [],  # 所有机器人都可紧急停止
        }
        
        required = capability_map.get(action_type, [])
        if not required:
            return True
        
        robot_caps = getattr(robot, 'capabilities', [])
        return any(cap in robot_caps for cap in required)
    
    def _handle_navigate(self, robot: Any, params: Dict) -> tuple:
        """处理导航动作"""
        target = params.get("target")
        if not target:
            return False, "缺少目标位置参数", {}
        
        # 支持多种位置格式
        if isinstance(target, dict):
            position = [target.get("x", 0), target.get("y", 0), target.get("z", 0)]
            orientation = target.get("orientation")
        elif isinstance(target, list):
            position = target[:3]
            orientation = target[3:] if len(target) > 3 else None
        else:
            return False, f"不支持的目标格式: {type(target)}", {}
        
        try:
            success = robot.navigate_to(position, orientation)
            
            state = robot.get_state()
            
            if not success:
                error_info = getattr(robot, 'error_status', 'Unknown error')
                return False, f"导航到 {position}", {
                    "position": state.get("position"),
                    "orientation": state.get("orientation"),
                    "error": error_info
                }
            
            return True, f"导航到 {position}", {
                "position": state.get("position"),
                "orientation": state.get("orientation")
            }
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[ERROR] 导航失败: {error_msg}")
            return False, f"导航异常: {str(e)}", {"error": error_msg}
    
    def _handle_pick(self, robot: Any, params: Dict) -> tuple:
        """处理抓取动作"""
        object_id = params.get("object_id")
        object_name = params.get("object_name", str(object_id))
        
        if object_id is None:
            return False, "缺少物体 ID 参数", {}
        
        # 转换 object_id 为整数（PyBullet 物体 ID）
        if isinstance(object_id, str):
            # 尝试从场景对象映射中查找
            object_id = self._resolve_object_id(object_id)
        
        success = robot.pick(object_id)
        
        state = robot.get_state()
        return success, f"抓取物体 {object_name}", {
            "is_holding_object": state.get("is_holding_object", False),
            "held_object_id": state.get("held_object_id")
        }
    
    def _handle_place(self, robot: Any, params: Dict) -> tuple:
        """处理放置动作"""
        target = params.get("target")
        
        if isinstance(target, dict):
            position = [target.get("x", 0), target.get("y", 0), target.get("z", 0)]
        elif isinstance(target, list):
            position = target
        else:
            return False, f"不支持的目标格式: {type(target)}", {}
        
        success = robot.place(position)
        
        state = robot.get_state()
        return success, f"放置到 {position}", {
            "is_holding_object": state.get("is_holding_object", False),
            "held_object_id": state.get("held_object_id")
        }
    
    def _handle_transport(self, robot: Any, params: Dict) -> tuple:
        """处理运输动作"""
        object_id = params.get("object_id")
        source = params.get("source")
        target = params.get("target")
        
        if object_id is None:
            return False, "缺少物体 ID 参数", {}
        
        if isinstance(target, dict):
            target_pos = [target.get("x", 0), target.get("y", 0), target.get("z", 0)]
        elif isinstance(target, list):
            target_pos = target
        else:
            return False, f"不支持的目标格式: {type(target)}", {}
        
        # 转换 source 格式
        if isinstance(source, dict):
            source_pos = [source.get("x", 0), source.get("y", 0), source.get("z", 0)]
        elif isinstance(source, list):
            source_pos = source
        else:
            source_pos = None  # 让机器人自动获取
        
        success = robot.transport(object_id, source_pos, target_pos)
        
        state = robot.get_state()
        return success, f"运输物体到 {target_pos}", {
            "is_holding_object": state.get("is_holding_object", False)
        }
    
    def _handle_rotate(self, robot: Any, params: Dict) -> tuple:
        """处理旋转动作"""
        angle = params.get("angle", 0)
        
        # 支持度数或弧度
        if params.get("unit", "radians") == "degrees":
            import math
            angle = math.radians(angle)
        
        success = robot.rotate_to_yaw(angle)
        
        state = robot.get_state()
        return success, f"旋转到 {angle} 弧度", {
            "yaw": state.get("yaw"),
            "orientation": state.get("orientation")
        }
    
    def _handle_move_forward(self, robot: Any, params: Dict) -> tuple:
        """处理向前移动"""
        distance = params.get("distance", 0.1)
        success = robot.move_forward(distance)
        
        state = robot.get_state()
        return success, f"向前移动 {distance} 米", {
            "position": state.get("position")
        }
    
    def _handle_move_backward(self, robot: Any, params: Dict) -> tuple:
        """处理向后移动"""
        distance = params.get("distance", 0.1)
        success = robot.move_backward(distance)
        
        state = robot.get_state()
        return success, f"向后移动 {distance} 米", {
            "position": state.get("position")
        }
    
    def _handle_wait(self, robot: Any, params: Dict) -> tuple:
        """处理等待"""
        duration = params.get("duration", 1.0)
        import time
        time.sleep(duration)
        
        return True, f"等待 {duration} 秒", {}
    
    def _handle_stop(self, robot: Any, params: Dict) -> tuple:
        """处理停止"""
        robot.stop()
        return True, "停止运动", {}
    
    def _handle_emergency_stop(self, robot: Any, params: Dict) -> tuple:
        """处理紧急停止"""
        robot.emergency_stop()
        return True, "紧急停止", {}
    
    def _get_sensor_data(self, robot: Any) -> Dict[str, Any]:
        """获取机器人传感器数据"""
        state = robot.get_state()
        
        sensor_data = {
            "position": state.get("position"),
            "orientation": state.get("orientation"),
            "is_busy": state.get("is_busy"),
        }
        
        # 根据机器人类型添加特定数据
        if hasattr(robot, 'is_holding_object'):
            sensor_data["is_holding_object"] = state.get("is_holding_object")
        
        return sensor_data
    
    def _resolve_object_id(self, object_name: str) -> int:
        """解析物体名称到 PyBullet ID"""
        # 这里应该实现物体名称到 ID 的映射
        # 简化实现：假设名称就是 ID 的字符串形式
        try:
            return int(object_name)
        except ValueError:
            return -1
    
    def get_robot_states(self) -> Dict[str, Dict]:
        """获取所有机器人状态"""
        return {
            rid: robot.get_state()
            for rid, robot in self.robot_registry.items()
        }
    
    def get_robot_capabilities(self) -> Dict[str, List[str]]:
        """获取所有机器人能力"""
        return {
            rid: robot.capabilities
            for rid, robot in self.robot_registry.items()
        }
    
    def get_scene_graph(self) -> Dict[str, Dict]:
        """
        获取场景图（Scene Graph）
        包含场景中所有物体的位置和状态信息
        
        Returns:
            Dict: {object_name: {position, orientation, type, ...}}
        """
        scene_graph = {}
        
        # Try to get scene objects from the first robot's environment
        # In BestMan, scene objects are typically managed by the simulation environment
        try:
            # Get the first robot to access the simulation
            first_robot = next(iter(self.robot_registry.values()), None)
            if first_robot and hasattr(first_robot, 'env'):
                env = first_robot.env
                
                # Try to get objects from environment
                if hasattr(env, 'get_objects'):
                    objects = env.get_objects()
                    for obj_id, obj_info in objects.items():
                        scene_graph[obj_info.get('name', f'object_{obj_id}')] = {
                            'id': obj_id,
                            'position': obj_info.get('position', [0, 0, 0]),
                            'orientation': obj_info.get('orientation', [0, 0, 0, 1]),
                            'type': obj_info.get('type', 'unknown'),
                            'is_grasped': obj_info.get('is_grasped', False)
                        }
                elif hasattr(env, 'scene_objects'):
                    # Alternative: directly access scene_objects
                    for obj_name, obj_data in env.scene_objects.items():
                        scene_graph[obj_name] = {
                            'position': obj_data.get('position', [0, 0, 0]),
                            'orientation': obj_data.get('orientation', [0, 0, 0, 1]),
                            'type': obj_data.get('type', 'unknown'),
                            'model_path': obj_data.get('model_path', '')
                        }
        except Exception as e:
            print(f"[BestManAdapter] Failed to get scene graph: {e}")
        
        return scene_graph
