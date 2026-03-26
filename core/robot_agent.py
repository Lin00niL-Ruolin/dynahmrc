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
    ACTION_SETS = {
        'MobileManipulation': ['navigate', 'open', 'pick', 'place', 'move', 'communicate', 'wait'],
        'Manipulator': ['pick', 'place', 'communicate', 'wait'],  # No navigate
        'Mobile': ['navigate', 'communicate', 'wait'],  # No pick/place
        'Drone': ['navigate', 'pick', 'place', 'communicate', 'wait']
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
        
        # Action set based on robot type
        self.available_actions = self.ACTION_SETS.get(robot_type, ['communicate', 'wait'])
        
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
        
        # Communication callback
        self.send_message_callback = None
        
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
        fallback_desc = f"I am {self.name}, a {self.robot_type} robot with {', '.join(self.capabilities)} capabilities."
        self.memory.store_self_description(fallback_desc)
        return "", fallback_desc
    
    def _build_self_description_prompt(self, task: str) -> str:
        """Build prompt for Self-Description stage"""
        return f"""You are {self.name}, a {self.robot_type} robot with capabilities: {', '.join(self.capabilities)}.

Task: {task}

## Instructions
Introduce yourself to your teammates. Explain:
1. Who you are and your robot type
2. Your specific capabilities
3. How you can contribute to this task
4. Your confidence level in handling this task

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
        
        return f"""You are {self.name}, a {self.robot_type} robot with capabilities: {', '.join(self.capabilities)}.

Task: {task}

Teammates:
{teammates_str}

## Instructions
Propose a task allocation plan and give a campaign speech for leadership:
1. Analyze the task and divide it into subtasks
2. Assign each subtask to the most suitable robot (including yourself)
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
        
        return f"""You are {self.name}, a {self.robot_type} robot.

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
    
    def _build_execution_prompt(self, observation: Dict, leader_plan: Dict) -> str:
        """Build prompt for Execution stage"""
        history_str = self.memory.format_history_for_prompt(k=5)
        messages_str = self.memory.format_messages_for_prompt()
        
        scene_graph = observation.get('scene_graph', {})
        scene_str = "Scene Graph:\n"
        for name, info in scene_graph.items():
            pos = info.get('position', [0, 0, 0])
            scene_str += f"  - {name}: at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})\n"
        
        return f"""You are {self.name}, a {self.robot_type} robot.
Your capabilities: {', '.join(self.capabilities)}
Your available actions: {', '.join(self.available_actions)}

{scene_str}

Leader's Plan: {json.dumps(leader_plan, indent=2)}

{history_str}

{messages_str}

## Instructions
Based on the current observation and leader's plan, choose your next action.
Consider:
1. Your assigned subtasks
2. Current state of the environment
3. Recent action history and feedback
4. Messages from teammates

## Output Format
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
        
        return f"""You are {self.name}, a {self.robot_type} robot.
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
