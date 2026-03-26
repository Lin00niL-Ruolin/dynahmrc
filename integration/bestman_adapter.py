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
    COMMUNICATE = "communicate"     # 机器人间通信


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
    
    def __init__(self, robot_registry: Dict[str, Any], client: Any = None):
        """
        初始化适配器
        
        Args:
            robot_registry: 机器人注册表 {robot_id: robot_instance}
            client: BestMan Client 实例（可选），用于获取场景信息
        """
        self.robot_registry = robot_registry
        self.client = client
        self.scene_objects: Dict[str, Dict] = {}  # 存储场景物体信息
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
            ActionType.COMMUNICATE: self._handle_communicate,
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
        
        print(f"[BestManAdapter.execute_action] 开始执行: robot_id={robot_id}, action={action}, params={params}")
        
        # 检查机器人是否存在
        if robot_id not in self.robot_registry:
            print(f"[BestManAdapter.execute_action] 错误: 机器人 {robot_id} 未找到")
            return ExecutionFeedback(
                success=False,
                action_type=action,
                robot_id=robot_id,
                message=f"机器人 {robot_id} 未找到",
                error_code="ROBOT_NOT_FOUND",
                error_details=f"可用机器人: {list(self.robot_registry.keys())}"
            )
        
        robot = self.robot_registry[robot_id]
        print(f"[BestManAdapter.execute_action] 找到机器人: {robot_id}, 类型={robot.robot_type}")
        
        # 解析动作类型
        try:
            action_type = ActionType(action.lower())
            print(f"[BestManAdapter.execute_action] 动作类型解析成功: {action_type}")
        except ValueError:
            print(f"[BestManAdapter.execute_action] 错误: 未知动作类型 {action}")
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
            print(f"[BestManAdapter.execute_action] 错误: 能力不匹配")
            return ExecutionFeedback(
                success=False,
                action_type=action,
                robot_id=robot_id,
                message=f"机器人 {robot_id} 无法执行 {action}",
                error_code="CAPABILITY_MISMATCH",
                error_details=f"机器人类型: {robot.robot_type}, 能力: {robot.capabilities}"
            )
        
        # 执行动作
        print(f"[BestManAdapter.execute_action] 调用 handler: {action_type}")
        try:
            handler = self.action_handlers[action_type]
            # 对于需要 robot_id 的 handler，传递 robot_id
            if action_type in [ActionType.NAVIGATE, ActionType.PICK, ActionType.COMMUNICATE]:
                success, message, state_changes = handler(robot, params, robot_id)
            else:
                success, message, state_changes = handler(robot, params)
            print(f"[BestManAdapter.execute_action] handler 返回: success={success}, message={message}")
            
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
            ActionType.COMMUNICATE: [],  # 所有机器人都可通信
        }
        
        required = capability_map.get(action_type, [])
        if not required:
            return True
        
        robot_caps = getattr(robot, 'capabilities', [])
        return any(cap in robot_caps for cap in required)
    
    def _handle_navigate(self, robot: Any, params: Dict, robot_id: str = None) -> tuple:
        """处理导航动作"""
        print(f"[_handle_navigate] 开始: robot_id={robot_id}, params={params}")
        
        position = None
        orientation = None
        
        # 支持多种参数格式
        # 格式1: params = {"target": [x, y, z]} 或 {"target": {"x": x, "y": y, "z": z}}
        if "target" in params:
            target = params["target"]
            if isinstance(target, dict):
                position = [target.get("x", 0), target.get("y", 0), target.get("z", 0)]
                orientation = target.get("orientation")
            elif isinstance(target, list):
                position = target[:3]
                orientation = target[3:] if len(target) > 3 else None
            elif isinstance(target, str):
                # target 是字符串（如物体名称），需要解析为位置
                print(f"[_handle_navigate] target 是字符串: {target}，尝试解析为位置...")
                position = self._resolve_target_position(target)
                if position is None:
                    return False, f"无法解析目标位置: {target}", {}
        
        # 格式2: params = {"x": x, "y": y, "z": z}
        elif "x" in params and "y" in params:
            position = [params.get("x", 0), params.get("y", 0), params.get("z", 0)]
            orientation = params.get("orientation")
        
        if position is None:
            print(f"[_handle_navigate] 错误: 无法解析目标位置参数")
            return False, "缺少或无法解析目标位置参数", {}
        
        print(f"[_handle_navigate] 目标位置: {position}, 朝向: {orientation}")
        
        try:
            # 根据机器人类型决定调用方式
            robot_type = getattr(robot, 'robot_type', None)
            
            if robot_type == 'arm':
                # ArmRobot 不支持导航
                print(f"[_handle_navigate] 错误: ArmRobot 不支持导航")
                return False, "ArmRobot 不支持导航", {}
            
            # 获取场景物体信息用于避障（排除当前机器人自身）
            scene_objects = self.get_scene_graph(exclude_robot_id=robot_id)
            print(f"[_handle_navigate] 获取场景物体: {len(scene_objects)} 个")
            
            # 调用导航方法，所有支持避障的机器人都传递 scene_objects
            print(f"[_handle_navigate] 调用 robot.navigate_to...")
            success = robot.navigate_to(position, orientation, scene_objects)
            print(f"[_handle_navigate] navigate_to 返回: success={success}")
            
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
    
    def _handle_pick(self, robot: Any, params: Dict, robot_id: str = None) -> tuple:
        """处理抓取动作"""
        print(f"[_handle_pick] 开始: robot_id={robot_id}, params={params}")
        
        object_id = params.get("object_id")
        object_name = params.get("object_name", str(object_id))
        
        if object_id is None:
            print(f"[_handle_pick] 错误: 缺少物体 ID 参数")
            return False, "缺少物体 ID 参数", {}
        
        # 转换 object_id 为整数（PyBullet 物体 ID）
        if isinstance(object_id, str):
            # 尝试从场景对象映射中查找
            object_id = self._resolve_object_id(object_id)
        
        try:
            # 根据机器人类型决定调用方式
            robot_type = getattr(robot, 'robot_type', None)
            
            if robot_type == 'arm':
                # ArmRobot 不支持 scene_objects 参数
                success = robot.pick(object_id)
            elif robot_type in ['mobile_manipulator', 'mobile_base']:
                # MobileManipulator 支持 scene_objects
                scene_objects = self.get_scene_graph(exclude_robot_id=robot_id)
                success = robot.pick(object_id, scene_objects=scene_objects)
            else:
                # 默认尝试带 scene_objects
                scene_objects = self.get_scene_graph(exclude_robot_id=robot_id)
                success = robot.pick(object_id, scene_objects=scene_objects)
            
            state = robot.get_state()
            
            if not success:
                error_info = getattr(robot, 'error_status', 'Unknown error')
                return False, f"抓取物体 {object_name}", {
                    "is_holding_object": state.get("is_holding_object", False),
                    "held_object_id": state.get("held_object_id"),
                    "error": error_info
                }
            
            return True, f"抓取物体 {object_name}", {
                "is_holding_object": state.get("is_holding_object", False),
                "held_object_id": state.get("held_object_id")
            }
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[ERROR] 抓取失败: {error_msg}")
            return False, f"抓取异常: {str(e)}", {"error": error_msg}
    
    def _handle_place(self, robot: Any, params: Dict) -> tuple:
        """处理放置动作"""
        # 支持多种参数格式
        # 格式1: params = {"target": [x, y, z]} 或 {"target": {"x": x, "y": y, "z": z}}
        # 格式2: params = {"location": [x, y, z]} 或 {"location": {"x": x, "y": y, "z": z}}
        target = params.get("target") or params.get("location")
        
        if target is None:
            return False, "缺少目标位置参数 (target 或 location)", {}
        
        if isinstance(target, dict):
            position = [target.get("x", 0), target.get("y", 0), target.get("z", 0)]
        elif isinstance(target, list):
            position = target
        else:
            return False, f"不支持的目标格式: {type(target)}", {}
        
        try:
            # 根据机器人类型决定调用方式
            robot_type = getattr(robot, 'robot_type', None)
            
            if robot_type == 'arm':
                # ArmRobot 不支持 scene_objects 参数
                success = robot.place(position)
            elif robot_type in ['mobile_manipulator', 'mobile_base']:
                # MobileManipulator 支持 scene_objects
                scene_objects = self.get_scene_graph()
                success = robot.place(position, scene_objects=scene_objects)
            else:
                # 默认尝试带 scene_objects
                scene_objects = self.get_scene_graph()
                success = robot.place(position, scene_objects=scene_objects)
            
            state = robot.get_state()
            
            if not success:
                error_info = getattr(robot, 'error_status', 'Unknown error')
                return False, f"放置到 {position}", {
                    "is_holding_object": state.get("is_holding_object", False),
                    "held_object_id": state.get("held_object_id"),
                    "error": error_info
                }
            
            return True, f"放置到 {position}", {
                "is_holding_object": state.get("is_holding_object", False),
                "held_object_id": state.get("held_object_id")
            }
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[ERROR] 放置失败: {error_msg}")
            return False, f"放置异常: {str(e)}", {"error": error_msg}
    
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
    
    def _handle_communicate(self, robot: Any, params: Dict, robot_id: str = None) -> tuple:
        """处理机器人间通信"""
        print(f"[_handle_communicate] 开始: robot_id={robot_id}, params={params}")
        
        to_robot = params.get("to")
        message = params.get("message", "")
        broadcast = params.get("broadcast", False)
        
        if not message:
            print(f"[_handle_communicate] 错误: 消息为空")
            return False, "消息不能为空", {}
        
        # 获取发送者名称
        sender_name = robot_id or "unknown"
        
        if broadcast:
            # 广播给所有机器人
            print(f"[_handle_communicate] 广播消息: {sender_name} -> all: {message[:50]}...")
            for rid, other_robot in self.robot_registry.items():
                if rid != robot_id and hasattr(other_robot, 'receive_message'):
                    try:
                        other_robot.receive_message(sender_name, message)
                    except Exception as e:
                        print(f"[_handle_communicate] 发送给 {rid} 失败: {e}")
            return True, f"广播消息: {message[:50]}...", {"broadcast": True}
        
        elif to_robot:
            # 发送给特定机器人
            print(f"[_handle_communicate] 发送消息: {sender_name} -> {to_robot}: {message[:50]}...")
            if to_robot in self.robot_registry:
                target_robot = self.robot_registry[to_robot]
                if hasattr(target_robot, 'receive_message'):
                    try:
                        target_robot.receive_message(sender_name, message)
                        return True, f"发送消息给 {to_robot}: {message[:50]}...", {"to": to_robot}
                    except Exception as e:
                        print(f"[_handle_communicate] 发送失败: {e}")
                        return False, f"发送消息失败: {e}", {"error": str(e)}
                else:
                    return False, f"机器人 {to_robot} 不支持接收消息", {}
            else:
                return False, f"机器人 {to_robot} 未找到", {}
        
        else:
            print(f"[_handle_communicate] 错误: 未指定接收者")
            return False, "必须指定接收者 (to) 或设置为广播 (broadcast)", {}
    
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
        """
        解析物体名称到 PyBullet ID
        
        Args:
            object_name: 物体名称（如 "box"）或 ID 字符串（如 "1"）
        
        Returns:
            PyBullet 物体 ID，如果找不到返回 -1
        """
        # 1. 首先尝试从 scene_objects 中查找
        if object_name in self.scene_objects:
            return self.scene_objects[object_name].get('id', -1)
        
        # 2. 尝试将名称作为整数 ID 解析
        try:
            return int(object_name)
        except ValueError:
            pass
        
        # 3. 尝试模糊匹配（忽略大小写）
        for name, info in self.scene_objects.items():
            if name.lower() == object_name.lower():
                return info.get('id', -1)
        
        print(f"[BestManAdapter] 警告: 找不到物体 '{object_name}'")
        return -1
    
    def _resolve_target_position(self, target_name: str) -> Optional[List[float]]:
        """
        解析目标名称到位置坐标
        
        Args:
            target_name: 目标名称（如 "prep_station", "box" 等）
        
        Returns:
            位置坐标 [x, y, z]，如果找不到返回 None
        """
        # 1. 从场景物体中查找
        if target_name in self.scene_objects:
            obj_info = self.scene_objects[target_name]
            pos = obj_info.get('position', [0, 0, 0])
            print(f"[_resolve_target_position] 从场景物体找到 '{target_name}': {pos}")
            return pos
        
        # 2. 尝试模糊匹配（忽略大小写）
        for name, info in self.scene_objects.items():
            if name.lower() == target_name.lower():
                pos = info.get('position', [0, 0, 0])
                print(f"[_resolve_target_position] 模糊匹配找到 '{target_name}' -> '{name}': {pos}")
                return pos
        
        # 3. 从机器人注册表中查找（其他机器人的位置）
        for rid, robot in self.robot_registry.items():
            if rid == target_name or rid.lower() == target_name.lower():
                state = robot.get_state()
                pos = state.get('position', [0, 0, 0])
                print(f"[_resolve_target_position] 从机器人找到 '{target_name}': {pos}")
                return pos
        
        print(f"[_resolve_target_position] 警告: 无法解析目标 '{target_name}'")
        return None
    
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
    
    def register_scene_object(self, obj_name: str, obj_id: int, obj_type: str = "object"):
        """
        注册场景物体
        
        Args:
            obj_name: 物体名称
            obj_id: PyBullet 物体 ID
            obj_type: 物体类型
        """
        self.scene_objects[obj_name] = {
            'id': obj_id,
            'type': obj_type,
            'name': obj_name
        }
        print(f"[BestManAdapter] 注册场景物体: {obj_name} (ID: {obj_id})")
    
    def get_scene_graph(self, exclude_robot_id: str = None) -> Dict[str, Dict]:
        """
        获取场景图（Scene Graph）
        包含场景中所有物体和其他机器人的位置和状态信息
        
        Args:
            exclude_robot_id: 要排除的机器人ID（通常是当前正在规划的机器人）
        
        Returns:
            Dict: {object_name: {position, orientation, type, size, ...}}
        """
        scene_graph = {}
        
        # 从注册的物体中获取信息
        try:
            import pybullet as p
            
            for obj_name, obj_info in self.scene_objects.items():
                obj_id = obj_info.get('id')
                if obj_id is not None:
                    try:
                        # 从 PyBullet 获取物体当前位置和朝向
                        pos, orn = p.getBasePositionAndOrientation(obj_id)
                        
                        # 尝试获取物体尺寸（AABB）
                        try:
                            aabb_min, aabb_max = p.getAABB(obj_id)
                            size = [
                                aabb_max[0] - aabb_min[0],
                                aabb_max[1] - aabb_min[1],
                                aabb_max[2] - aabb_min[2]
                            ]
                        except:
                            size = [0.1, 0.1, 0.1]  # 默认尺寸
                        
                        scene_graph[obj_name] = {
                            'id': obj_id,
                            'position': list(pos),
                            'orientation': list(orn),
                            'type': obj_info.get('type', 'unknown'),
                            'size': size,
                            'is_grasped': False  # TODO: 检查是否被抓取
                        }
                    except Exception as e:
                        print(f"[BestManAdapter] 获取物体 {obj_name} 状态失败: {e}")
            
            # 添加其他机器人作为动态障碍物
            for rid, robot in self.robot_registry.items():
                if rid == exclude_robot_id:
                    continue  # 排除当前机器人自身
                
                try:
                    state = robot.get_state()
                    pos = state.get('position', [0, 0, 0])
                    orn = state.get('orientation', [0, 0, 0, 1])
                    
                    # 机器人尺寸（近似为圆柱体）
                    robot_size = [0.6, 0.6, 1.0]  # 默认机器人尺寸
                    
                    scene_graph[f"robot_{rid}"] = {
                        'id': -1,  # 机器人没有PyBullet ID
                        'position': pos,
                        'orientation': orn,
                        'type': 'robot',
                        'size': robot_size,
                        'robot_id': rid,
                        'is_grasped': False
                    }
                    print(f"[BestManAdapter] 添加机器人障碍物: {rid} 位置 {pos[:2]}")
                except Exception as e:
                    print(f"[BestManAdapter] 获取机器人 {rid} 状态失败: {e}")
                    
        except Exception as e:
            print(f"[BestManAdapter] Failed to get scene graph: {e}")
        
        return scene_graph
