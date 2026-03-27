"""
Robot Agent for DynaHMRC
Implements the four-stage collaboration framework
Based on paper Section IV: System Architecture
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from .memory import MemoryModule


class CollaborationPhase(Enum):
    """Four-stage collaboration process"""
    SELF_DESCRIPTION = "self_description"
    TASK_ALLOCATION = "task_allocation"
    LEADER_ELECTION = "leader_election"
    EXECUTION = "execution"
    REFLECTION = "reflection"


class RobotAgent:
    """
    Robot Agent - Decentralized independent LLM Agent
    Each robot has its own Memory, Observation, and Planning modules
    
    Attributes:
        name: Robot identifier
        robot_type: Type of robot (MobileManipulation, Manipulator, Mobile, Drone)
        capabilities: List of capabilities
        memory: MemoryModule instance
        is_leader: Whether this robot is the leader
        current_phase: Current collaboration phase
    """
    
    # Action sets for different robot types (Table II in paper)
    # 支持多种命名方式以兼容不同的 LLM 输出
    ACTION_SETS = {
        # 移动操作复合机器人 (Mobile Manipulator / Ma-Mo)
        'MobileManipulation': ['navigate', 'open', 'pick', 'place', 'move', 'communicate', 'wait'],
        'MobileManipulator': ['navigate', 'open', 'pick', 'place', 'move', 'communicate', 'wait'],
        'mobile_manipulation': ['navigate', 'open', 'pick', 'place', 'move', 'communicate', 'wait'],
        'mobile_manipulator': ['navigate', 'open', 'pick', 'place', 'move', 'communicate', 'wait'],
        'ma-mo': ['navigate', 'open', 'pick', 'place', 'move', 'communicate', 'wait'],
        'ma_mo': ['navigate', 'open', 'pick', 'place', 'move', 'communicate', 'wait'],
        'mamo': ['navigate', 'open', 'pick', 'place', 'move', 'communicate', 'wait'],
        
        # 固定机械臂 (Manipulator / Arm / Ma)
        'Manipulator': ['pick', 'place', 'communicate', 'wait'],  # No navigate
        'manipulator': ['pick', 'place', 'communicate', 'wait'],
        'Arm': ['pick', 'place', 'communicate', 'wait'],
        'arm': ['pick', 'place', 'communicate', 'wait'],
        'ma': ['pick', 'place', 'communicate', 'wait'],
        'fixed_arm': ['pick', 'place', 'communicate', 'wait'],
        'fixed manipulator': ['pick', 'place', 'communicate', 'wait'],
        
        # 移动基座 (Mobile / Mobile Base / Mo)
        'Mobile': ['navigate', 'communicate', 'wait'],  # No pick/place
        'mobile': ['navigate', 'communicate', 'wait'],
        'MobileBase': ['navigate', 'communicate', 'wait'],
        'Mobile_Base': ['navigate', 'communicate', 'wait'],
        'mobile_base': ['navigate', 'communicate', 'wait'],
        'mobile base': ['navigate', 'communicate', 'wait'],
        'mo': ['navigate', 'communicate', 'wait'],
        'agv': ['navigate', 'communicate', 'wait'],
        'AGV': ['navigate', 'communicate', 'wait'],
        
        # 无人机 (Drone / UAV)
        'Drone': ['navigate', 'pick', 'place', 'communicate', 'wait'],
        'drone': ['navigate', 'pick', 'place', 'communicate', 'wait'],
        'UAV': ['navigate', 'pick', 'place', 'communicate', 'wait'],
        'uav': ['navigate', 'pick', 'place', 'communicate', 'wait'],
        'aerial': ['navigate', 'pick', 'place', 'communicate', 'wait'],
        'Aerial': ['navigate', 'pick', 'place', 'communicate', 'wait'],
        'quadrotor': ['navigate', 'pick', 'place', 'communicate', 'wait'],
        'Quadrotor': ['navigate', 'pick', 'place', 'communicate', 'wait'],
    }
    
    # 机器人类型映射表 - 用于标准化 LLM 返回的类型名称
    ROBOT_TYPE_MAPPING = {
        # 移动操作复合机器人
        'mobile_manipulation': 'MobileManipulation',
        'mobile_manipulator': 'MobileManipulation',
        'ma-mo': 'MobileManipulation',
        'ma_mo': 'MobileManipulation',
        'mamo': 'MobileManipulation',
        'mobile manipulation': 'MobileManipulation',
        
        # 固定机械臂
        'manipulator': 'Manipulator',
        'arm': 'Manipulator',
        'ma': 'Manipulator',
        'fixed_arm': 'Manipulator',
        'fixed arm': 'Manipulator',
        'fixed manipulator': 'Manipulator',
        
        # 移动基座
        'mobile': 'Mobile',
        'mobile_base': 'Mobile',
        'mobile base': 'Mobile',
        'mo': 'Mobile',
        'agv': 'Mobile',
        
        # 无人机
        'drone': 'Drone',
        'uav': 'Drone',
        'aerial': 'Drone',
        'quadrotor': 'Drone',
    }
    
    def __init__(
        self,
        name: str,
        robot_type: str,  # MobileManipulation, Manipulator, Mobile, Drone
        capabilities: List[str],
        llm_client: Any,
        avatar: str = "🤖",
        max_history: int = 10
    ):
        """
        Initialize Robot Agent
        
        Args:
            name: Robot identifier
            robot_type: Type of robot
            capabilities: List of capabilities
            llm_client: LLM client for generating responses
            avatar: Emoji avatar for display
            max_history: Maximum history length
        """
        self.name = name
        self.robot_type = robot_type
        self.capabilities = capabilities
        self.llm_client = llm_client
        self.avatar = avatar
        
        # 标准化机器人类型名称
        self.normalized_robot_type = self._normalize_robot_type(robot_type)
        
        # Action set based on robot type (使用标准化后的类型)
        self.available_actions = self._get_available_actions(robot_type)
        
        print(f"[RobotAgent] {name} initialized as {robot_type} (normalized: {self.normalized_robot_type})")
        print(f"[RobotAgent] {name} available actions: {self.available_actions}")
        
        # Collaboration state
        self.is_leader = False
        self.leader_name = None
        self.teammates = {}  # {name: description}
        self.current_phase = CollaborationPhase.SELF_DESCRIPTION
        
        # Execution state
        self.step_count = 0
        self.current_action = None
        
        # Modules
        self.memory = MemoryModule(max_history=max_history)
        
        # Path planner for reachability calculation
        self.path_planner = None
        
        # Communication callback
        self.send_message_callback = None
    
    def set_path_planner(self, path_planner):
        """设置路径规划器"""
        self.path_planner = path_planner
        print(f"[RobotAgent] {self.name} path planner set")
    
    def _normalize_robot_type(self, robot_type: str) -> str:
        """
        标准化机器人类型名称
        将 LLM 可能返回的各种变体映射到标准类型
        """
        if not robot_type:
            return 'MobileManipulation'  # 默认类型
        
        # 直接匹配
        if robot_type in self.ACTION_SETS:
            # 返回映射后的标准名称，如果没有映射则返回原值
            return self.ROBOT_TYPE_MAPPING.get(robot_type.lower(), robot_type)
        
        # 尝试小写匹配
        robot_type_lower = robot_type.lower()
        if robot_type_lower in self.ROBOT_TYPE_MAPPING:
            return self.ROBOT_TYPE_MAPPING[robot_type_lower]
        
        # 尝试模糊匹配
        for key, standard_type in self.ROBOT_TYPE_MAPPING.items():
            if key in robot_type_lower or robot_type_lower in key:
                return standard_type
        
        # 如果包含特定关键词，推断类型
        if any(kw in robot_type_lower for kw in ['drone', 'uav', 'aerial', 'quadrotor', '飞行', '无人机']):
            return 'Drone'
        elif any(kw in robot_type_lower for kw in ['mobile_manipulator', 'mobile manipulation', 'ma-mo', 'mamo']):
            return 'MobileManipulation'
        elif any(kw in robot_type_lower for kw in ['mobile_base', 'mobile base', 'agv', '移动基地']):
            return 'Mobile'
        elif any(kw in robot_type_lower for kw in ['manipulator', 'arm', 'ma', '机械臂']):
            return 'Manipulator'
        
        # 无法识别，返回原值
        print(f"[RobotAgent] Warning: Unknown robot type '{robot_type}', using as-is")
        return robot_type
    
    def _get_available_actions(self, robot_type: str) -> List[str]:
        """
        获取机器人可用的动作集合
        支持多种命名方式
        """
        # 首先尝试直接匹配
        if robot_type in self.ACTION_SETS:
            return self.ACTION_SETS[robot_type]
        
        # 尝试标准化后的类型
        normalized = self._normalize_robot_type(robot_type)
        if normalized in self.ACTION_SETS:
            return self.ACTION_SETS[normalized]
        
        # 尝试小写匹配
        robot_type_lower = robot_type.lower()
        for key, actions in self.ACTION_SETS.items():
            if key.lower() == robot_type_lower:
                return actions
        
        # 默认返回最基本的动作
        print(f"[RobotAgent] Warning: No action set found for type '{robot_type}', using default")
        return ['communicate', 'wait']
        
    def set_message_callback(self, callback):
        """Set callback for sending messages"""
        self.send_message_callback = callback
    
    # ========== Stage 1: Self-Description ==========
    def self_describe(self, task: str) -> Tuple[str, str]:
        """
        Stage 1: Self-Description
        Generate self-introduction including capabilities and understanding of task
        
        Args:
            task: Task description
            
        Returns:
            Tuple of (thought, description)
        """
        prompt = self._build_self_description_prompt(task)
        max_retries = 3
        
        for attempt in range(max_retries):
            response = self.llm_client.generate(prompt, temperature=1.0)
            
            print(f"[DEBUG] Self-Description Response (attempt {attempt+1}): {response}")
            
            # Parse response
            thought, description = self._parse_self_description_response(response)
            
            # Check if response is valid
            if thought and description:
                self.memory.store_self_description(description)
                self.current_phase = CollaborationPhase.TASK_ALLOCATION
                return thought, description
            elif not thought and description:
                self.memory.store_self_description(description)
                self.current_phase = CollaborationPhase.TASK_ALLOCATION
                return "", description
            else:
                print(f"[DEBUG] Invalid response, retrying...")
        
        # Fallback if all retries fail
        fallback_desc = f"I am {self.name}, a {self.normalized_robot_type} robot with {', '.join(self.capabilities)} capabilities."
        self.memory.store_self_description(fallback_desc)
        return "", fallback_desc
    
    def _build_self_description_prompt(self, task: str) -> str:
        """Build prompt for Self-Description stage"""
        # 获取职责说明
        responsibilities = self._get_robot_responsibilities()
        
        return f"""You are {self.name}, a {self.normalized_robot_type} robot.

=== YOUR RESPONSIBILITIES ===
{responsibilities}

Your capabilities: {', '.join(self.capabilities)}
Your available actions: {', '.join(self.available_actions)}

Task: {task}

## Instructions
Introduce yourself to your teammates. Explain:
1. Who you are and your robot type
2. Your specific capabilities AND limitations (be clear about what you CANNOT do)
3. How you can contribute to this task based on your actual capabilities
4. Your confidence level in handling this task

IMPORTANT: Be honest about your limitations:
- If you are a Fixed Arm, state clearly that you cannot move/navigate
- If you are a Mobile Base, state clearly that you cannot pick/place objects
- If you are a Drone, mention your payload limitations
- If you are a Mobile Manipulator, highlight your versatility

## Output Format
Thought: <Your internal reasoning>
Description: <Your introduction to teammates (2-3 sentences)>"""
    
    def _parse_self_description_response(self, response: str) -> Tuple[str, str]:
        """Parse Self-Description response"""
        thought = ""
        description = ""
        
        response = response.strip()
        
        # Extract Thought
        if "Thought:" in response:
            thought_part = response.split("Thought:")[1]
            if "Description:" in thought_part:
                thought = thought_part.split("Description:")[0].strip()
            else:
                thought = thought_part.strip()
        
        # Extract Description
        if "Description:" in response:
            description_part = response.split("Description:")[1]
            description = description_part.strip()
        
        return thought, description
    
    # ========== Stage 2: Task Allocation + Leadership Bidding ==========
    def propose_allocation(self, task: str, teammates_descriptions: Dict[str, str]) -> Tuple[Dict, str, str]:
        """
        Stage 2: Task Allocation and Leadership Bidding
        Propose task division and campaign speech
        
        Args:
            task: Task description
            teammates_descriptions: Dict of {robot_name: description}
            
        Returns:
            Tuple of (plan, thought, campaign_speech)
        """
        self.teammates = teammates_descriptions
        
        prompt = self._build_allocation_prompt(task, teammates_descriptions)
        max_retries = 3
        
        for attempt in range(max_retries):
            response = self.llm_client.generate(prompt, temperature=1.0)
            
            print(f"[DEBUG] Task Allocation Response (attempt {attempt+1}): {response}")
            
            plan, thought, campaign_speech = self._parse_allocation_response(response)
            
            if plan and campaign_speech:
                self.memory.store_task_plan(plan)
                self.current_phase = CollaborationPhase.LEADER_ELECTION
                return plan, thought, campaign_speech
        
        # Fallback
        fallback_plan = {
            "task_decomposition": [{"id": "task_1", "description": task, "assigned_to": self.name}],
            "coordination_points": []
        }
        fallback_speech = f"I propose to handle this task. I am {self.name} with {', '.join(self.capabilities)}."
        return fallback_plan, "", fallback_speech
    
    def _build_allocation_prompt(self, task: str, teammates_descriptions: Dict[str, str]) -> str:
        """Build prompt for Task Allocation stage"""
        teammates_str = "\n".join([f"- {name}: {desc}" for name, desc in teammates_descriptions.items()])
        
        # 获取职责说明
        responsibilities = self._get_robot_responsibilities()
        
        return f"""You are {self.name}, a {self.normalized_robot_type} robot.

=== YOUR RESPONSIBILITIES ===
{responsibilities}

Your capabilities: {', '.join(self.capabilities)}
Your available actions: {', '.join(self.available_actions)}

Task: {task}

Teammates:
{teammates_str}

## Instructions
Propose a task allocation plan and give a campaign speech for leadership:

1. Analyze the task and divide it into subtasks
2. Assign each subtask to the most suitable robot (including yourself)
   - Consider each robot's capabilities and limitations
   - Fixed Arm robots CANNOT navigate - don't assign navigation tasks to them
   - Mobile Base robots CANNOT pick/place - only assign transport tasks
   - Drones are good for aerial tasks but have limited payload
   - Mobile Manipulators can handle most tasks independently
3. Identify coordination points where robots need to synchronize
4. Give a persuasive campaign speech explaining why you should be the leader

## Output Format
```json
{{
  "task_decomposition": [
    {{"id": "subtask_1", "description": "...", "assigned_to": "robot_name", "estimated_steps": 5}}
  ],
  "coordination_points": ["point_1", "point_2"]
}}
```

Thought: <Your internal reasoning>
Campaign Speech: <Your leadership campaign speech (2-3 sentences)>"""
    
    def _parse_allocation_response(self, response: str) -> Tuple[Dict, str, str]:
        """Parse Task Allocation response"""
        plan = {}
        thought = ""
        campaign_speech = ""
        
        response = response.strip()
        
        # Extract JSON plan
        if "```json" in response:
            try:
                json_str = response.split("```json")[1].split("```")[0].strip()
                plan = json.loads(json_str)
            except:
                pass
        elif "```" in response:
            try:
                json_str = response.split("```")[1].split("```")[0].strip()
                plan = json.loads(json_str)
            except:
                pass
        
        # Extract Thought
        if "Thought:" in response:
            thought_part = response.split("Thought:")[1]
            if "Campaign Speech:" in thought_part:
                thought = thought_part.split("Campaign Speech:")[0].strip()
            else:
                thought = thought_part.strip()
        
        # Extract Campaign Speech
        if "Campaign Speech:" in response:
            speech_part = response.split("Campaign Speech:")[1]
            campaign_speech = speech_part.strip()
        
        return plan, thought, campaign_speech
    
    # ========== Stage 3: Leader Election ==========
    def vote_leader(self, proposals: Dict[str, Tuple[Dict, str]]) -> str:
        """
        Stage 3: Leader Election
        Vote for the most capable leader based on proposals
        
        Args:
            proposals: Dict of {robot_name: (plan, campaign_speech)}
            
        Returns:
            Name of the elected leader
        """
        prompt = self._build_election_prompt(proposals)
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.llm_client.generate(prompt, temperature=1.0)
                
                print(f"[DEBUG] Leader Election Response (attempt {attempt+1}): {response}")
                
                leader_name = self._parse_leader_response(response)
                
                if leader_name and leader_name in proposals:
                    self.leader_name = leader_name
                    self.is_leader = (leader_name == self.name)
                    self.current_phase = CollaborationPhase.EXECUTION
                    return leader_name
                    
            except Exception as e:
                print(f"[ERROR] Leader Election failed (attempt {attempt+1}): {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback: vote for self
        print(f"[WARN] Using fallback: voting for self ({self.name})")
        self.leader_name = self.name
        self.is_leader = True
        self.current_phase = CollaborationPhase.EXECUTION
        return self.name
    
    def _build_election_prompt(self, proposals: Dict[str, Tuple[Dict, str]]) -> str:
        """Build prompt for Leader Election stage"""
        proposals_str = ""
        for name, (plan, speech) in proposals.items():
            proposals_str += f"\n=== {name} ===\n"
            proposals_str += f"Campaign Speech: {speech}\n"
            proposals_str += f"Plan: {json.dumps(plan, indent=2)}\n"
        
        robot_list = ", ".join(proposals.keys())
        
        return f"""You are {self.name}, a {self.normalized_robot_type} robot.

You need to vote for a leader among: {robot_list}

Proposals from candidates:
{proposals_str}

## Instructions
Analyze each candidate's proposal and campaign speech. Vote for the robot who:
1. Has the most comprehensive and feasible plan
2. Demonstrates strong leadership qualities
3. Can effectively coordinate the team

## Output Format
Thought: <Your reasoning for the vote>
Vote: <Name of the robot you vote for>"""
    
    def _parse_leader_response(self, response: str) -> str:
        """Parse Leader Election response"""
        response = response.strip()
        
        if "Vote:" in response:
            vote_part = response.split("Vote:")[1].strip()
            # Extract just the name (first word)
            leader_name = vote_part.split()[0].strip()
            return leader_name
        
        return ""
    
    # ========== Stage 4: Closed-Loop Execution ==========
    def execute_step(self, observation: Dict, leader_plan: Dict) -> Dict:
        """
        Stage 4: Closed-Loop Execution
        Execute one atomic action based on observation and plan
        
        Args:
            observation: Current observation (scene graph, robot states, etc.)
            leader_plan: The leader's plan
            
        Returns:
            Action dictionary
        """
        prompt = self._build_execution_prompt(observation, leader_plan)
        
        response = self.llm_client.generate(prompt, temperature=1.0)
        
        print(f"[DEBUG] {self.name} Execution Response: {response}")
        
        action = self._parse_action_response(response)
        print(f"[DEBUG] {self.name} Parsed action: {action}")
        
        # Validate action against capabilities
        if not self._validate_action(action):
            print(f"[DEBUG] {self.name} Action validation failed: {action.get('action')} not in {self.available_actions}")
            action = self._fallback_action()
        else:
            print(f"[DEBUG] {self.name} Action validation passed")
        
        self.current_action = action
        self.step_count += 1
        
        return action
    
    def _get_robot_responsibilities(self) -> str:
        """获取机器人类型的详细职责说明"""
        responsibilities = {
            'MobileManipulation': """You are a MOBILE MANIPULATOR robot (移动操作复合机器人).
Your responsibilities:
- CAN navigate to any location in the environment
- CAN pick up objects using your manipulator arm
- CAN place objects at specified locations
- CAN open/close containers and doors
- CAN transport objects from one place to another
- CANNOT fly or operate at heights above ground level
You are the most versatile robot type and can handle most tasks independently.""",
            
            'Manipulator': """You are a FIXED ARM robot (固定机械臂).
Your responsibilities:
- CANNOT navigate or move your base - you are fixed in place
- CAN pick up objects that are within your arm's reach
- CAN place objects at nearby locations
- CAN manipulate objects in your workspace
- MUST rely on other robots (Mobile or Mobile Manipulator) to bring objects to you or take objects away
- Your workspace is limited to your arm's reach radius
You specialize in precise manipulation but cannot move.""",
            
            'Mobile': """You are a MOBILE BASE robot (移动基座).
Your responsibilities:
- CAN navigate to any location in the environment
- CANNOT pick up or manipulate objects - you have no arm
- CAN transport objects only if they are already loaded onto you by other robots
- CAN communicate and coordinate with other robots
- Your primary role is transportation and logistics support
You are a transport specialist but cannot manipulate objects directly.""",
            
            'Drone': """You are a DRONE/UAV robot (无人机).
Your responsibilities:
- CAN navigate in 3D space including flying over obstacles
- CAN pick up LIGHTWEIGHT objects (max 0.5kg payload)
- CAN place objects from aerial positions
- CAN provide aerial reconnaissance and perception
- CAN access elevated or hard-to-reach areas
- CANNOT pick up heavy objects
- CANNOT open doors or containers requiring force
You specialize in aerial operations and accessing elevated areas."""
        }
        return responsibilities.get(self.normalized_robot_type, "Unknown robot type")
    
    def _calculate_reachable_positions(self, scene_graph: Dict, robot_states: Dict, 
                                        path_planner=None) -> Dict[str, List[float]]:
        """
        计算机器人可以到达的位置，使用A*算法验证路径可达性
        
        Args:
            scene_graph: 场景图
            robot_states: 机器人状态
            path_planner: 路径规划器（可选），用于A*验证
        
        Returns:
            Dict[str, List[float]]: 可到达位置字典 {位置名称: [x, y, z]}
        """
        reachable_positions = {}
        
        # 获取当前机器人位置
        my_position = robot_states.get(self.name, {}).get('position', [0, 0, 0])
        
        # 更新障碍物地图（如果有path_planner且支持）
        if path_planner and hasattr(path_planner, 'update_obstacles_from_scene'):
            try:
                path_planner.update_obstacles_from_scene(scene_graph)
                print(f"[_calculate_reachable_positions] {self.name} 障碍物地图已更新")
            except Exception as e:
                print(f"[_calculate_reachable_positions] 更新障碍物地图失败: {e}")
        
        # 过滤场景图中的物体（排除机器人和地面）
        scene_items = {k: v for k, v in scene_graph.items() 
                      if not k.startswith('robot_') and k != 'ground'}
        total_items = len(scene_items)
        
        # 根据机器人类型计算候选位置，并使用A*验证
        if self.normalized_robot_type == 'Manipulator':
            # 固定机械臂：只能在当前位置操作
            arm_reach = 0.8
            print(f"\n[{self.name}] 计算可达位置 (机械臂模式): 0/{total_items} [{' ' * 20}] 0%", end='', flush=True)
            for idx, (name, info) in enumerate(scene_items.items()):
                pos = info.get('position', [0, 0, 0])
                distance = ((pos[0] - my_position[0])**2 + 
                           (pos[1] - my_position[1])**2 + 
                           (pos[2] - my_position[2])**2) ** 0.5
                if distance <= arm_reach:
                    reachable_positions[f"{name}_approach"] = pos
                
                # 更新进度条
                progress = (idx + 1) / total_items
                bar_length = 20
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                percent = int(progress * 100)
                print(f"\r[{self.name}] 计算可达位置 (机械臂模式): {idx+1}/{total_items} [{bar}] {percent}%", end='', flush=True)
            print()  # 换行
        
        elif self.normalized_robot_type == 'Mobile':
            # 移动基座：使用A*验证地面路径
            print(f"\n[{self.name}] 计算可达位置 (移动基座模式): 0/{total_items} [{' ' * 20}] 0%", end='', flush=True)
            for idx, (name, info) in enumerate(scene_items.items()):
                pos = info.get('position', [0, 0, 0])
                target_pos = [pos[0], pos[1], 0.0]
                
                # 使用A*验证路径
                if path_planner:
                    # 使用plan_global_path而不是plan，保持一致性
                    if hasattr(path_planner, 'plan_global_path'):
                        path = path_planner.plan_global_path(
                            my_position[:2], 
                            target_pos[:2],
                            scene_objects=scene_graph,
                            max_search_radius=5.0,
                            radius_step=0.5
                        )
                    else:
                        path = path_planner.plan(
                            my_position[:2], 
                            target_pos[:2],
                            robot_id=self.name,
                            use_cache=True
                        )
                    if path is not None:
                        reachable_positions[f"{name}_approach"] = target_pos
                else:
                    # 无路径规划器时，直接添加
                    reachable_positions[f"{name}_approach"] = target_pos
                
                # 更新进度条
                progress = (idx + 1) / total_items
                bar_length = 20
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                percent = int(progress * 100)
                status = "✓" if f"{name}_approach" in reachable_positions else "✗"
                print(f"\r[{self.name}] 计算可达位置 (移动基座模式): {idx+1}/{total_items} [{bar}] {percent}% {status}", end='', flush=True)
            print()  # 换行
        
        elif self.normalized_robot_type == 'Drone':
            # 无人机：A*验证空中路径（无人机可以飞直线）
            print(f"\n[{self.name}] 计算可达位置 (无人机模式): 0/{total_items} [{' ' * 20}] 0%", end='', flush=True)
            for idx, (name, info) in enumerate(scene_items.items()):
                pos = info.get('position', [0, 0, 0])
                
                # 上方位置
                above_pos = [pos[0], pos[1], pos[2] + 1.5]
                # 低空位置
                nearby_pos = [pos[0], pos[1], pos[2] + 0.3]
                
                # 无人机路径相对简单，主要检查距离
                distance_above = ((above_pos[0] - my_position[0])**2 + 
                                 (above_pos[1] - my_position[1])**2 + 
                                 (above_pos[2] - my_position[2])**2) ** 0.5
                
                if distance_above < 10.0:  # 最大飞行距离
                    reachable_positions[f"{name}_above"] = above_pos
                    reachable_positions[f"{name}_nearby"] = nearby_pos
                
                # 更新进度条
                progress = (idx + 1) / total_items
                bar_length = 20
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                percent = int(progress * 100)
                print(f"\r[{self.name}] 计算可达位置 (无人机模式): {idx+1}/{total_items} [{bar}] {percent}%", end='', flush=True)
            print()  # 换行
        
        else:  # MobileManipulation
            # 移动操作机器人：使用A*验证路径
            print(f"\n[{self.name}] 计算可达位置 (移动操作模式): 0/{total_items} [{' ' * 20}] 0%", end='', flush=True)
            for idx, (name, info) in enumerate(scene_items.items()):
                pos = info.get('position', [0, 0, 0])
                
                # 物体前方位置
                approach_pos = [pos[0] + 0.3, pos[1], 0.0]
                
                # 使用A*验证路径
                if path_planner:
                    # 使用plan_global_path而不是plan，保持一致性
                    if hasattr(path_planner, 'plan_global_path'):
                        path = path_planner.plan_global_path(
                            my_position[:2],
                            approach_pos[:2],
                            scene_objects=scene_graph,
                            max_search_radius=5.0,
                            radius_step=0.5
                        )
                    else:
                        path = path_planner.plan(
                            my_position[:2],
                            approach_pos[:2],
                            robot_id=self.name,
                            use_cache=True
                        )
                    if path is not None:
                        reachable_positions[f"{name}_approach"] = approach_pos
                        reachable_positions[f"{name}_position"] = pos
                else:
                    reachable_positions[f"{name}_approach"] = approach_pos
                    reachable_positions[f"{name}_position"] = pos
                
                # 更新进度条
                progress = (idx + 1) / total_items
                bar_length = 20
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                percent = int(progress * 100)
                status = "✓" if f"{name}_approach" in reachable_positions else "✗"
                print(f"\r[{self.name}] 计算可达位置 (移动操作模式): {idx+1}/{total_items} [{bar}] {percent}% {status}", end='', flush=True)
            print()  # 换行
        
        print(f"[{self.name}] 共找到 {len(reachable_positions)} 个可达位置")
        return reachable_positions
    
    def _build_execution_prompt(self, observation: Dict, leader_plan: Dict) -> str:
        """Build prompt for Execution stage"""
        history_str = self.memory.format_history_for_prompt(k=5)
        messages_str = self.memory.format_messages_for_prompt()
        
        scene_graph = observation.get('scene_graph', {})
        scene_str = "Scene Graph:\n"
        for name, info in scene_graph.items():
            pos = info.get('position', [0, 0, 0])
            scene_str += f"  - {name}: at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})\n"
        
        # 获取机器人状态信息
        robot_states = observation.get('robot_states', {})
        teammates_str = ""
        if robot_states:
            teammates_str = "\nTeammates:\n"
            for robot_name, state in robot_states.items():
                if robot_name != self.name:
                    pos = state.get('position', [0, 0, 0])
                    teammates_str += f"  - {robot_name}: at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})\n"
        
        # Leader 信息
        leader_info = ""
        if self.leader_name:
            leader_info = f"\nLeader: {self.leader_name}"
            if self.is_leader:
                leader_info += " (You are the leader)"
        
        # 获取职责说明
        responsibilities = self._get_robot_responsibilities()
        
        # 计算可到达位置（使用A*验证）
        reachable_positions = self._calculate_reachable_positions(
            scene_graph, robot_states, self.path_planner
        )
        
        if reachable_positions:
            reachable_str = "\n".join([f"  - {name}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]" 
                                       for name, pos in reachable_positions.items()])
        else:
            reachable_str = "  (No reachable positions calculated - planner not available or paths blocked)"
        
        return f"""You are {self.name}, a {self.normalized_robot_type} robot.

=== YOUR RESPONSIBILITIES ===
{responsibilities}

=== YOUR CAPABILITIES ===
Your available actions: {', '.join(self.available_actions)}
Your capabilities list: {', '.join(self.capabilities)}{leader_info}

=== CURRENT SITUATION ===
{scene_str}{teammates_str}

=== REACHABLE POSITIONS FOR NAVIGATION ===
When you need to navigate, use these pre-calculated reachable positions as targets:
{reachable_str}

Note: These positions are ONLY for navigation actions. For pick/place/manipulate actions, use object names directly.

Leader's Plan: {json.dumps(leader_plan, indent=2)}

=== HISTORY ===
{history_str}

{messages_str}

=== ACTION SELECTION GUIDE ===
Based on the current observation and leader's plan, choose your next action.

STEP-BY-STEP DECISION PROCESS:
1. Analyze your current task: What needs to be done?
2. Check your current state: Where are you? What are you holding?
3. Choose the appropriate action type:

   IF you need to move to a location:
   → Use "navigate" action with one of the pre-calculated reachable positions
   
   IF you are at the object location and need to grab it:
   → Use "pick" action with the object name
   
   IF you are holding an object and need to put it down:
   → Use "place" action with the target location/object
   
   IF you need to manipulate something (open/close):
   → Use "manipulate" action
   
   IF you need help from teammates:
   → Use "communicate" action

CRITICAL RULES:
1. You can ONLY use these action types: {', '.join(self.available_actions)}
2. DO NOT use any other action types - this will cause execution failure
3. Respect your robot type limitations:
   - If you are a Fixed Arm: NEVER try to navigate - ask others to bring objects to you
   - If you are a Mobile Base: NEVER try to pick/place - only transport
   - If you are a Drone: Only handle lightweight objects, avoid heavy lifting
   - If you are a Mobile Manipulator: You can navigate, pick, place, and manipulate

4. BALANCED ACTION SELECTION:
   - Navigation is just ONE step in completing a task
   - After navigating to an object, you MUST pick it up
   - After picking up, you MUST navigate to destination and place it
   - Do NOT get stuck in infinite navigation loops

5. If the task requires capabilities you don't have:
   - Use "communicate" action to request help from appropriate teammates
   - Do not attempt actions outside your capabilities

6. Consider:
   - Your assigned subtasks
   - Current state of the environment  
   - Recent action history and feedback
   - Messages from teammates

=== OUTPUT FORMAT ===
```json
{{
  "action": "<action_type>",
  "params": {{<action-specific parameters>}},
  "reasoning": "<why you chose this action>"
}}
```

Valid action types: {', '.join(self.available_actions)}"""
    
    def _parse_action_response(self, response: str) -> Dict:
        """Parse Execution response"""
        action = {"action": "wait", "params": {}, "reasoning": "Default wait action"}
        
        response = response.strip()
        
        # Extract JSON
        if "```json" in response:
            try:
                json_str = response.split("```json")[1].split("```")[0].strip()
                action = json.loads(json_str)
            except:
                pass
        elif "```" in response:
            try:
                json_str = response.split("```")[1].split("```")[0].strip()
                action = json.loads(json_str)
            except:
                pass
        elif response.startswith("{"):
            try:
                action = json.loads(response)
            except:
                pass
        
        return action
    
    def _validate_action(self, action: Dict) -> bool:
        """Validate action against robot capabilities"""
        action_type = action.get('action', '')
        return action_type in self.available_actions
    
    def _fallback_action(self) -> Dict:
        """Return fallback action"""
        return {
            "action": "wait",
            "params": {"duration": 1},
            "reasoning": "Fallback wait action due to invalid action"
        }
    
    def store_action_result(self, action: Dict, feedback: Dict):
        """Store action and its feedback in memory"""
        self.memory.store_action(action, feedback)
    
    def receive_message(self, from_robot: str, content: str):
        """Receive message from another robot"""
        self.memory.store_received_message(from_robot, content)
    
    def send_message(self, to_robot: str, content: str):
        """Send message to another robot"""
        if self.send_message_callback:
            self.send_message_callback(self.name, to_robot, content)
    
    # ========== Stage 5: Reflection ==========
    def reflect(self, task: str, team_history: Dict) -> Tuple[str, str]:
        """
        Stage 5: Reflection
        Analyze past experiences and generate future plan improvements
        
        Args:
            task: Task description
            team_history: Aggregated team history including actions and feedback
            
        Returns:
            Tuple of (reflection_summary, future_plan_adjustments)
        """
        prompt = self._build_reflection_prompt(task, team_history)
        
        response = self.llm_client.generate(prompt, temperature=1.0)
        
        print(f"[DEBUG] Reflection Response: {response}")
        
        summary, adjustments = self._parse_reflection_response(response)
        
        self.current_phase = CollaborationPhase.REFLECTION
        
        return summary, adjustments
    
    def _build_reflection_prompt(self, task: str, team_history: Dict) -> str:
        """Build prompt for Reflection stage"""
        history_str = self.memory.format_history_for_prompt(k=10)
        messages_str = self.memory.format_messages_for_prompt()
        
        team_progress = team_history.get('total_steps', 0)
        robot_states = team_history.get('robot_states', {})
        
        team_status = "\n".join([
            f"  - {name}: {info.get('actions', 0)} actions taken"
            for name, info in robot_states.items()
        ])
        
        return f"""You are {self.name}, a {self.normalized_robot_type} robot.
Your capabilities: {', '.join(self.capabilities)}

Task: {task}

Team Progress:
- Total steps taken: {team_progress}
- Team member status:
{team_status}

Your History:
{history_str}

Messages:
{messages_str}

## Instructions
Reflect on the task execution so far:
1. Compare current task state with the target objectives
2. Analyze what has been accomplished and what remains
3. Identify successful strategies and lessons learned
4. Identify any failures or inefficiencies
5. Propose adjustments to future task allocation and coordination

## Output Format
Reflection Summary: <Your analysis of progress, successes, and issues>
Future Plan Adjustments: <Specific recommendations for improving the plan>"""
    
    def _parse_reflection_response(self, response: str) -> Tuple[str, str]:
        """Parse Reflection response"""
        summary = ""
        adjustments = ""
        
        response = response.strip()
        
        # Extract Reflection Summary
        if "Reflection Summary:" in response:
            summary_part = response.split("Reflection Summary:")[1]
            if "Future Plan Adjustments:" in summary_part:
                summary = summary_part.split("Future Plan Adjustments:")[0].strip()
            else:
                summary = summary_part.strip()
        
        # Extract Future Plan Adjustments
        if "Future Plan Adjustments:" in response:
            adjustments_part = response.split("Future Plan Adjustments:")[1]
            adjustments = adjustments_part.strip()
        
        return summary, adjustments
    
    def update_leader_plan(self, reflections: Dict[str, Tuple[str, str]], current_plan: Dict) -> Dict:
        """
        Leader integrates team reflections and updates the plan
        
        Args:
            reflections: Dict of {robot_name: (summary, adjustments)}
            current_plan: Current task plan
            
        Returns:
            Updated task plan
        """
        if not self.is_leader:
            return current_plan
        
        prompt = self._build_plan_update_prompt(reflections, current_plan)
        
        response = self.llm_client.generate(prompt, temperature=1.0)
        
        print(f"[DEBUG] Plan Update Response: {response}")
        
        updated_plan = self._parse_plan_update_response(response, current_plan)
        
        return updated_plan
    
    def _build_plan_update_prompt(self, reflections: Dict[str, Tuple[str, str]], current_plan: Dict) -> str:
        """Build prompt for leader plan update"""
        reflections_str = ""
        for name, (summary, adjustments) in reflections.items():
            reflections_str += f"\n=== {name} ===\n"
            reflections_str += f"Summary: {summary}\n"
            reflections_str += f"Adjustments: {adjustments}\n"
        
        return f"""You are {self.name}, the leader of the team.
Your role: Integrate team reflections and update the task plan.

Current Plan:
{json.dumps(current_plan, indent=2)}

Team Reflections:
{reflections_str}

## Instructions
As the leader, analyze all team reflections and:
1. Identify common issues or patterns
2. Determine necessary adjustments to the task plan
3. Reallocate tasks if needed
4. Update coordination points
5. Maintain the overall goal while improving efficiency

## Output Format
```json
{{
  "task_decomposition": [
    {{"id": "subtask_1", "description": "...", "assigned_to": "robot_name", "estimated_steps": 5}}
  ],
  "coordination_points": ["point_1", "point_2"],
  "adjustment_reasoning": "<why these changes were made>"
}}
```"""
    
    def _parse_plan_update_response(self, response: str, fallback_plan: Dict) -> Dict:
        """Parse Plan Update response"""
        response = response.strip()
        
        # Extract JSON
        if "```json" in response:
            try:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            except:
                pass
        elif "```" in response:
            try:
                json_str = response.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            except:
                pass
        elif response.startswith("{"):
            try:
                return json.loads(response)
            except:
                pass
        
        return fallback_plan
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the robot agent"""
        return {
            'name': self.name,
            'robot_type': self.robot_type,
            'is_leader': self.is_leader,
            'leader_name': self.leader_name,
            'current_phase': self.current_phase.value,
            'step_count': self.step_count,
            'current_action': self.current_action,
            'capabilities': self.capabilities
        }
