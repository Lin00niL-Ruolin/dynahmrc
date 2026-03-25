"""
Four-Stage Collaboration Framework for DynaHMRC
Implements the complete four-stage collaboration process:
1. Self-Description
2. Task Allocation
3. Leader Election
4. Closed-Loop Execution

Based on paper Section IV: System Architecture
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field

from .robot_agent import RobotAgent, CollaborationPhase


class CollaborationPhase(Enum):
    """Four-stage collaboration process"""
    IDLE = "idle"
    SELF_DESCRIPTION = "self_description"
    TASK_ALLOCATION = "task_allocation"
    LEADER_ELECTION = "leader_election"
    EXECUTION = "execution"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CollaborationResult:
    """Result of the four-stage collaboration"""
    success: bool
    message: str
    leader_name: Optional[str] = None
    task_plan: Optional[Dict] = None
    execution_steps: int = 0
    duration: float = 0.0
    robot_assignments: Dict[str, List[str]] = field(default_factory=dict)


class CollaborationManager:
    """
    Manages the collaboration state and synchronization
    """
    
    def __init__(self):
        self.current_phase = CollaborationPhase.IDLE
        self.phase_handlers: Dict[CollaborationPhase, Callable] = {}
        self.execution_history: List[Dict] = []
        
    def register_phase_handler(self, phase: CollaborationPhase, handler: Callable):
        """Register a handler for a specific phase"""
        self.phase_handlers[phase] = handler
    
    def transition_to(self, phase: CollaborationPhase, context: Optional[Dict] = None):
        """Transition to a new phase"""
        old_phase = self.current_phase
        self.current_phase = phase
        
        # Record history
        self.execution_history.append({
            'timestamp': time.time(),
            'from': old_phase.value,
            'to': phase.value,
            'context': context or {}
        })
        
        # Call handler if registered
        if phase in self.phase_handlers:
            self.phase_handlers[phase](context)
        
        print(f"[CollaborationManager] Phase transition: {old_phase.value} -> {phase.value}")
    
    def get_current_phase(self) -> CollaborationPhase:
        """Get current collaboration phase"""
        return self.current_phase
    
    def is_phase(self, phase: CollaborationPhase) -> bool:
        """Check if currently in a specific phase"""
        return self.current_phase == phase


class FourStageCollaboration:
    """
    Four-Stage Collaboration Framework
    Orchestrates the complete collaboration process among multiple robots
    
    Stage 1: Self-Description - Each robot introduces itself
    Stage 2: Task Allocation - Propose task division and leadership bidding
    Stage 3: Leader Election - Vote for the leader
    Stage 4: Closed-Loop Execution - Execute task with feedback loop
    """
    
    def __init__(
        self,
        robots: List[RobotAgent],
        max_execution_steps: int = 100,
        enable_communication: bool = True
    ):
        """
        Initialize four-stage collaboration
        
        Args:
            robots: List of robot agents
            max_execution_steps: Maximum execution steps
            enable_communication: Whether to enable inter-robot communication
        """
        self.robots: Dict[str, RobotAgent] = {r.name: r for r in robots}
        self.max_execution_steps = max_execution_steps
        self.enable_communication = enable_communication
        
        # Collaboration state
        self.manager = CollaborationManager()
        self.leader_name: Optional[str] = None
        self.task_plan: Optional[Dict] = None
        
        # Results
        self.execution_history: List[Dict] = []
        self.start_time: Optional[float] = None
        
        # Setup communication
        if enable_communication:
            self._setup_communication()
    
    def _setup_communication(self):
        """Setup communication callbacks between robots"""
        def message_callback(from_robot: str, to_robot: str, content: str):
            if to_robot in self.robots:
                self.robots[to_robot].receive_message(from_robot, content)
        
        for robot in self.robots.values():
            robot.set_message_callback(message_callback)
    
    def run_collaboration(self, task: str) -> CollaborationResult:
        """
        Run the complete four-stage collaboration
        
        Args:
            task: Task description
            
        Returns:
            CollaborationResult with success status and details
        """
        self.start_time = time.time()
        self.manager.transition_to(CollaborationPhase.SELF_DESCRIPTION)
        
        try:
            # Stage 1: Self-Description
            descriptions = self._run_self_description(task)
            print(f"\n[FourStageCollaboration] Stage 1 Complete - Descriptions: {descriptions}")
            
            # Stage 2: Task Allocation
            proposals = self._run_task_allocation(task, descriptions)
            print(f"\n[FourStageCollaboration] Stage 2 Complete - Proposals received from {list(proposals.keys())}")
            
            # Stage 3: Leader Election
            self.leader_name = self._run_leader_election(proposals)
            print(f"\n[FourStageCollaboration] Stage 3 Complete - Leader: {self.leader_name}")
            
            # Get the leader's plan
            if self.leader_name in proposals:
                self.task_plan = proposals[self.leader_name][0]
            
            # Stage 4: Execution
            execution_success = self._run_execution(task)
            print(f"\n[FourStageCollaboration] Stage 4 Complete - Success: {execution_success}")
            
            # Calculate results
            duration = time.time() - self.start_time
            
            # Get robot assignments
            robot_assignments = self._extract_robot_assignments()
            
            self.manager.transition_to(
                CollaborationPhase.COMPLETED if execution_success else CollaborationPhase.FAILED
            )
            
            return CollaborationResult(
                success=execution_success,
                message="Task completed successfully" if execution_success else "Task execution failed",
                leader_name=self.leader_name,
                task_plan=self.task_plan,
                execution_steps=len(self.execution_history),
                duration=duration,
                robot_assignments=robot_assignments
            )
            
        except Exception as e:
            duration = time.time() - self.start_time if self.start_time else 0
            self.manager.transition_to(CollaborationPhase.FAILED, {'error': str(e)})
            
            return CollaborationResult(
                success=False,
                message=f"Collaboration failed: {str(e)}",
                duration=duration
            )
    
    def _run_self_description(self, task: str) -> Dict[str, str]:
        """
        Stage 1: Self-Description
        Each robot generates its self-introduction
        
        Returns:
            Dict of {robot_name: description}
        """
        print("\n" + "="*60)
        print("Stage 1: Self-Description")
        print("="*60)
        
        descriptions = {}
        
        for name, robot in self.robots.items():
            print(f"\n[Self-Description] Robot {name} is introducing itself...")
            thought, description = robot.self_describe(task)
            descriptions[name] = description
            print(f"[Self-Description] {name}: {description}")
        
        self.manager.transition_to(CollaborationPhase.TASK_ALLOCATION)
        return descriptions
    
    def _run_task_allocation(
        self,
        task: str,
        descriptions: Dict[str, str]
    ) -> Dict[str, Tuple[Dict, str]]:
        """
        Stage 2: Task Allocation
        Each robot proposes a task allocation plan and leadership bid
        
        Args:
            task: Task description
            descriptions: Dict of robot descriptions
            
        Returns:
            Dict of {robot_name: (plan, campaign_speech)}
        """
        print("\n" + "="*60)
        print("Stage 2: Task Allocation & Leadership Bidding")
        print("="*60)
        
        proposals = {}
        
        for name, robot in self.robots.items():
            # Exclude self from teammates
            teammates = {k: v for k, v in descriptions.items() if k != name}
            
            print(f"\n[TaskAllocation] Robot {name} is proposing allocation...")
            plan, thought, speech = robot.propose_allocation(task, teammates)
            proposals[name] = (plan, speech)
            
            print(f"[TaskAllocation] {name}'s campaign speech: {speech}")
            print(f"[TaskAllocation] {name}'s plan: {json.dumps(plan, indent=2)}")
        
        self.manager.transition_to(CollaborationPhase.LEADER_ELECTION)
        return proposals
    
    def _run_leader_election(
        self,
        proposals: Dict[str, Tuple[Dict, str]]
    ) -> str:
        """
        Stage 3: Leader Election
        Each robot votes for the leader
        
        Args:
            proposals: Dict of robot proposals
            
        Returns:
            Name of the elected leader
        """
        print("\n" + "="*60)
        print("Stage 3: Leader Election")
        print("="*60)
        
        votes = {name: 0 for name in self.robots.keys()}
        
        for name, robot in self.robots.items():
            print(f"\n[LeaderElection] Robot {name} is voting...")
            voted_for = robot.vote_leader(proposals)
            votes[voted_for] = votes.get(voted_for, 0) + 1
            print(f"[LeaderElection] {name} voted for {voted_for}")
        
        # Count votes and determine winner
        leader = max(votes.keys(), key=lambda k: votes[k])
        print(f"\n[LeaderElection] Final votes: {votes}")
        print(f"[LeaderElection] Elected leader: {leader} with {votes[leader]} votes")
        
        self.manager.transition_to(CollaborationPhase.EXECUTION)
        return leader
    
    def _run_execution(self, task: str) -> bool:
        """
        Stage 4: Closed-Loop Execution
        Execute the task with continuous feedback
        
        Args:
            task: Task description
            
        Returns:
            True if execution successful, False otherwise
        """
        print("\n" + "="*60)
        print("Stage 4: Closed-Loop Execution")
        print("="*60)
        
        if not self.leader_name or not self.task_plan:
            print("[Execution] Error: No leader or task plan")
            return False
        
        step = 0
        
        while step < self.max_execution_steps:
            print(f"\n[Execution] Step {step + 1}/{self.max_execution_steps}")
            
            # Get current observation (would come from simulation in real implementation)
            observation = self._get_observation()
            
            # Each robot executes one step
            all_robots_done = True
            
            for name, robot in self.robots.items():
                # Skip if robot has completed its tasks
                if self._is_robot_done(robot):
                    continue
                
                all_robots_done = False
                
                # Get action from robot
                action = robot.execute_step(observation, self.task_plan)
                print(f"[Execution] {name} action: {action}")
                
                # Execute action and get feedback (would use BestMan APIs in real implementation)
                feedback = self._execute_action(name, action)
                
                # Store result in robot's memory
                robot.store_action_result(action, feedback)
                
                # Record in history
                self.execution_history.append({
                    'step': step,
                    'robot': name,
                    'action': action,
                    'feedback': feedback,
                    'timestamp': time.time()
                })
                
                # Check if task is complete
                if self._is_task_complete():
                    print("[Execution] Task completed!")
                    return True
            
            if all_robots_done:
                print("[Execution] All robots have completed their tasks")
                return True
            
            step += 1
            time.sleep(0.1)  # Small delay between steps
        
        print(f"[Execution] Reached maximum steps ({self.max_execution_steps})")
        return False
    
    def _get_observation(self) -> Dict:
        """Get current observation from the environment"""
        # This would integrate with BestMan in real implementation
        return {
            'scene_graph': {},
            'robot_states': {name: robot.get_status() for name, robot in self.robots.items()},
            'timestamp': time.time()
        }
    
    def _execute_action(self, robot_name: str, action: Dict) -> Dict:
        """
        Execute an action and return feedback
        This is a placeholder - would integrate with BestMan in real implementation
        """
        action_type = action.get('action', 'wait')
        
        # Simulate action execution
        return {
            'success': True,
            'message': f"Executed {action_type}",
            'state_change': {}
        }
    
    def _is_robot_done(self, robot: RobotAgent) -> bool:
        """Check if a robot has completed its tasks"""
        # This would check if robot has completed all assigned subtasks
        return False
    
    def _is_task_complete(self) -> bool:
        """Check if the overall task is complete"""
        # This would check task completion conditions
        return False
    
    def _extract_robot_assignments(self) -> Dict[str, List[str]]:
        """Extract which tasks are assigned to each robot"""
        assignments = {name: [] for name in self.robots.keys()}
        
        if self.task_plan and 'task_decomposition' in self.task_plan:
            for subtask in self.task_plan['task_decomposition']:
                assigned_to = subtask.get('assigned_to')
                task_id = subtask.get('id', 'unknown')
                if assigned_to in assignments:
                    assignments[assigned_to].append(task_id)
        
        return assignments
    
    def get_execution_history(self) -> List[Dict]:
        """Get the execution history"""
        return self.execution_history
    
    def get_collaboration_status(self) -> Dict[str, Any]:
        """Get current collaboration status"""
        return {
            'phase': self.manager.get_current_phase().value,
            'leader': self.leader_name,
            'robots': {name: robot.get_status() for name, robot in self.robots.items()},
            'execution_steps': len(self.execution_history)
        }
