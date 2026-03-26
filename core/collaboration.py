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
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from .robot_agent import RobotAgent, CollaborationPhase

# Visualization disabled
VISUALIZATION_AVAILABLE = False
SpeechBubbleVisualizer = None


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
        enable_communication: bool = True,
        enable_visualization: bool = True,
        max_workers: int = 4,
        reflection_interval: int = 10,
        enable_reflection: bool = True
    ):
        """
        Initialize four-stage collaboration
        
        Args:
            robots: List of robot agents
            max_execution_steps: Maximum execution steps
            enable_communication: Whether to enable inter-robot communication
            enable_visualization: Whether to enable speech bubble visualization
            max_workers: Maximum number of worker threads for parallel processing
            reflection_interval: Steps between reflection stages (∆t in paper)
            enable_reflection: Whether to enable periodic reflection
        """
        self.robots: Dict[str, RobotAgent] = {r.name: r for r in robots}
        self.max_execution_steps = max_execution_steps
        self.enable_communication = enable_communication
        self.enable_visualization = enable_visualization and VISUALIZATION_AVAILABLE
        self.max_workers = max_workers
        self.reflection_interval = reflection_interval
        self.enable_reflection = enable_reflection
        
        # Collaboration state
        self.manager = CollaborationManager()
        self.leader_name: Optional[str] = None
        self.task_plan: Optional[Dict] = None
        
        # Results
        self.execution_history: List[Dict] = []
        self.start_time: Optional[float] = None
        self.reflection_history: List[Dict] = []
        
        # Thread safety
        self._visualizer_lock = Lock()
        self._history_lock = Lock()
        
        # Adapter for executing actions
        self.adapter = None
        
        # Visualization
        self.visualizer: Optional[SpeechBubbleVisualizer] = None
        if self.enable_visualization:
            try:
                self.visualizer = SpeechBubbleVisualizer()
                self.visualizer.start()
                print("[FourStageCollaboration] Speech bubble visualization enabled")
            except Exception as e:
                print(f"[FourStageCollaboration] Failed to start visualization: {e}")
                self.visualizer = None
        
        # Setup communication
        if enable_communication:
            self._setup_communication()
    
    def _setup_communication(self):
        """Setup communication callbacks between robots with visualization"""
        def message_callback(from_robot: str, to_robot: str, content: str):
            # Show speech bubble for communication
            if self.visualizer:
                self.visualizer.show_communication(from_robot, to_robot, content)
            
            if to_robot in self.robots:
                self.robots[to_robot].receive_message(from_robot, content)
        
        for robot in self.robots.values():
            robot.set_message_callback(message_callback)
    
    def set_robot_position(self, robot_name: str, position: Tuple[float, float, float]):
        """Set robot position for visualization"""
        if self.visualizer:
            self.visualizer.update_robot_position(robot_name, position)
    
    def update_leader_visualization(self):
        """Update leader indicator in visualization"""
        if self.visualizer and self.leader_name:
            self.visualizer.set_leader(self.leader_name)
    
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
            
            result = CollaborationResult(
                success=execution_success,
                message="Task completed successfully" if execution_success else "Task execution failed",
                leader_name=self.leader_name,
                task_plan=self.task_plan,
                execution_steps=len(self.execution_history),
                duration=duration,
                robot_assignments=robot_assignments
            )
            
            # Show completion message
            if self.visualizer:
                if execution_success:
                    self.visualizer.show_speech(
                        self.leader_name or "System",
                        "🎉 Task completed successfully!",
                        duration=5.0
                    )
                else:
                    self.visualizer.show_speech(
                        self.leader_name or "System",
                        "❌ Task execution failed",
                        duration=5.0
                    )
            
            return result
            
        except Exception as e:
            duration = time.time() - self.start_time if self.start_time else 0
            self.manager.transition_to(CollaborationPhase.FAILED, {'error': str(e)})
            
            # Show error message
            if self.visualizer:
                self.visualizer.show_speech(
                    "System",
                    f"💥 Error: {str(e)[:50]}",
                    duration=5.0
                )
            
            return CollaborationResult(
                success=False,
                message=f"Collaboration failed: {str(e)}",
                duration=duration
            )
    
    def _run_self_description(self, task: str) -> Dict[str, str]:
        """
        Stage 1: Self-Description
        Each robot generates its self-introduction in parallel
        
        Returns:
            Dict of {robot_name: description}
        """
        print("\n" + "="*60)
        print("Stage 1: Self-Description (Parallel)")
        print("="*60)
        
        descriptions = {}
        
        def describe_robot(name_robot_pair: Tuple[str, RobotAgent]) -> Tuple[str, str]:
            """Worker function for parallel self-description"""
            name, robot = name_robot_pair
            print(f"\n[Self-Description] Robot {name} is introducing itself...")
            thought, description = robot.self_describe(task)
            print(f"[Self-Description] {name}: {description}")
            
            # Show speech bubble (thread-safe)
            if self.visualizer:
                with self._visualizer_lock:
                    self.visualizer.show_self_description(name, description)
            
            return name, description
        
        # Execute self-descriptions in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(describe_robot, (name, robot)): name 
                for name, robot in self.robots.items()
            }
            
            for future in as_completed(futures):
                name, description = future.result()
                descriptions[name] = description
        
        self.manager.transition_to(CollaborationPhase.TASK_ALLOCATION)
        return descriptions
    
    def _run_task_allocation(
        self,
        task: str,
        descriptions: Dict[str, str]
    ) -> Dict[str, Tuple[Dict, str]]:
        """
        Stage 2: Task Allocation
        Each robot proposes a task allocation plan and leadership bid in parallel
        
        Args:
            task: Task description
            descriptions: Dict of robot descriptions
            
        Returns:
            Dict of {robot_name: (plan, campaign_speech)}
        """
        print("\n" + "="*60)
        print("Stage 2: Task Allocation & Leadership Bidding (Parallel)")
        print("="*60)
        
        proposals = {}
        
        def allocate_task(name_robot_pair: Tuple[str, RobotAgent]) -> Tuple[str, Dict, str]:
            """Worker function for parallel task allocation"""
            name, robot = name_robot_pair
            # Exclude self from teammates
            teammates = {k: v for k, v in descriptions.items() if k != name}
            
            print(f"\n[TaskAllocation] Robot {name} is proposing allocation...")
            plan, thought, speech = robot.propose_allocation(task, teammates)
            
            print(f"[TaskAllocation] {name}'s campaign speech: {speech}")
            print(f"[TaskAllocation] {name}'s plan: {json.dumps(plan, indent=2)}")
            
            # Show speech bubble for campaign speech (thread-safe)
            if self.visualizer and speech:
                with self._visualizer_lock:
                    self.visualizer.show_campaign_speech(name, speech)
            
            return name, plan, speech
        
        # Execute task allocations in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(allocate_task, (name, robot)): name 
                for name, robot in self.robots.items()
            }
            
            for future in as_completed(futures):
                name, plan, speech = future.result()
                proposals[name] = (plan, speech)
        
        self.manager.transition_to(CollaborationPhase.LEADER_ELECTION)
        return proposals
    
    def _run_leader_election(
        self,
        proposals: Dict[str, Tuple[Dict, str]]
    ) -> str:
        """
        Stage 3: Leader Election
        Each robot votes for the leader in parallel
        
        Args:
            proposals: Dict of robot proposals
            
        Returns:
            Name of the elected leader
        """
        print("\n" + "="*60)
        print("Stage 3: Leader Election (Parallel)")
        print("="*60)
        
        votes = {name: 0 for name in self.robots.keys()}
        votes_lock = Lock()
        
        def cast_vote(name_robot_pair: Tuple[str, RobotAgent]) -> Tuple[str, str]:
            """Worker function for parallel voting"""
            name, robot = name_robot_pair
            print(f"\n[LeaderElection] Robot {name} is voting...")
            voted_for = robot.vote_leader(proposals)
            print(f"[LeaderElection] {name} voted for {voted_for}")
            
            # Show speech bubble for voting (thread-safe)
            if self.visualizer:
                with self._visualizer_lock:
                    self.visualizer.show_vote(name, voted_for)
            
            return name, voted_for
        
        # Execute voting in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(cast_vote, (name, robot)): name 
                for name, robot in self.robots.items()
            }
            
            for future in as_completed(futures):
                name, voted_for = future.result()
                with votes_lock:
                    votes[voted_for] = votes.get(voted_for, 0) + 1
        
        # Count votes and determine winner
        leader = max(votes.keys(), key=lambda k: votes[k])
        print(f"\n[LeaderElection] Final votes: {votes}")
        
        # Update leader in visualizer
        self.leader_name = leader
        self.update_leader_visualization()
        print(f"[LeaderElection] Elected leader: {leader} with {votes[leader]} votes")
        
        self.manager.transition_to(CollaborationPhase.EXECUTION)
        return leader
    
    def _run_execution(self, task: str) -> bool:
        """
        Stage 4: Closed-Loop Execution with Periodic Reflection
        Execute the task with continuous feedback using parallel processing
        
        Args:
            task: Task description
            
        Returns:
            True if execution successful, False otherwise
        """
        print("\n" + "="*60)
        print("Stage 4: Closed-Loop Execution with Reflection (Parallel)")
        print("="*60)
        
        if not self.leader_name or not self.task_plan:
            print("[Execution] Error: No leader or task plan")
            return False
        
        step = 0
        
        print(f"[Execution] 开始执行阶段，最大步数: {self.max_execution_steps}")
        print(f"[Execution] 机器人数量: {len(self.robots)}, Leader: {self.leader_name}")
        print(f"[Execution] 任务计划: {self.task_plan}")
        
        while step < self.max_execution_steps:
            print(f"\n[Execution] ===== Step {step + 1}/{self.max_execution_steps} =====")
            
            # Stage 5: Reflection (triggered at fixed intervals)
            if self.enable_reflection and step > 0 and step % self.reflection_interval == 0:
                self._run_reflection(task)
            
            # Get current observation (would come from simulation in real implementation)
            observation = self._get_observation()
            print(f"[Execution] 获取观察: {len(observation.get('scene_graph', {}))} 个场景物体")
            
            # Execute robot steps in parallel
            active_robots = [
                (name, robot) for name, robot in self.robots.items() 
                if not self._is_robot_done(robot)
            ]
            
            if not active_robots:
                print("[Execution] All robots have completed their tasks")
                return True
            
            def execute_robot_step(name_robot_pair: Tuple[str, RobotAgent]) -> Optional[Dict]:
                """Worker function for parallel execution"""
                name, robot = name_robot_pair
                
                # Get action from robot
                action = robot.execute_step(observation, self.task_plan)
                print(f"[Execution] {name} action: {action}")
                
                # Show speech bubble for action (thread-safe)
                if self.visualizer:
                    action_str = action.get('action', 'wait')
                    reasoning = action.get('reasoning', '')
                    with self._visualizer_lock:
                        self.visualizer.show_action(name, action_str, reasoning)
                
                # Execute action and get feedback
                print(f"[Execution] 调用 _execute_action: robot={name}, action={action}")
                feedback = self._execute_action(name, action)
                print(f"[Execution] _execute_action 返回: {feedback}")
                
                # Store result in robot's memory
                robot.store_action_result(action, feedback)
                
                return {
                    'step': step,
                    'robot': name,
                    'action': action,
                    'feedback': feedback,
                    'timestamp': time.time()
                }
            
            # Execute all robot steps in parallel
            step_results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(execute_robot_step, (name, robot)): name 
                    for name, robot in active_robots
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        step_results.append(result)
            
            # Record all results in history (thread-safe)
            with self._history_lock:
                self.execution_history.extend(step_results)
            
            # Check if task is complete
            if self._is_task_complete():
                print("[Execution] Task completed!")
                return True
            
            step += 1
            time.sleep(0.1)  # Small delay between steps
        
        print(f"[Execution] Reached maximum steps ({self.max_execution_steps})")
        return False
    
    def _run_reflection(self, task: str):
        """
        Stage 5: Reflection and Group Discussion
        Periodic team reflection to improve task planning
        """
        print("\n" + "="*60)
        print("Stage 5: Reflection & Group Discussion (Parallel)")
        print("="*60)
        
        # Aggregate team history
        team_history = self._aggregate_team_history()
        
        # Each robot reflects in parallel
        reflections = {}
        
        def reflect_robot(name_robot_pair: Tuple[str, RobotAgent]) -> Tuple[str, str, str]:
            """Worker function for parallel reflection"""
            name, robot = name_robot_pair
            print(f"\n[Reflection] Robot {name} is reflecting...")
            
            summary, adjustments = robot.reflect(task, team_history)
            
            print(f"[Reflection] {name}'s summary: {summary[:100]}...")
            print(f"[Reflection] {name}'s adjustments: {adjustments[:100]}...")
            
            return name, summary, adjustments
        
        # Execute reflections in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(reflect_robot, (name, robot)): name 
                for name, robot in self.robots.items()
            }
            
            for future in as_completed(futures):
                name, summary, adjustments = future.result()
                reflections[name] = (summary, adjustments)
        
        # Leader integrates reflections and updates plan
        if self.leader_name and self.leader_name in self.robots:
            print(f"\n[Reflection] Leader {self.leader_name} is updating plan...")
            leader = self.robots[self.leader_name]
            
            updated_plan = leader.update_leader_plan(reflections, self.task_plan)
            
            if updated_plan and updated_plan != self.task_plan:
                print(f"[Reflection] Plan updated successfully")
                self.task_plan = updated_plan
                
                # Show speech bubble for plan update
                if self.visualizer:
                    with self._visualizer_lock:
                        self.visualizer.show_speech(
                            self.leader_name,
                            "📋 Plan updated based on team reflection!",
                            duration=3.0
                        )
            else:
                print(f"[Reflection] No plan changes needed")
        
        # Record reflection
        self.reflection_history.append({
            'step': len(self.execution_history),
            'reflections': reflections,
            'updated_plan': self.task_plan,
            'timestamp': time.time()
        })
        
        self.manager.transition_to(CollaborationPhase.EXECUTION)
    
    def _aggregate_team_history(self) -> Dict:
        """Aggregate history from all team members for reflection"""
        max_steps = max([robot.step_count for robot in self.robots.values()])
        
        robot_states = {}
        for name, robot in self.robots.items():
            robot_states[name] = {
                'actions': robot.step_count,
                'current_task': robot.current_action,
                'is_leader': robot.is_leader
            }
        
        return {
            'total_steps': max_steps,
            'robot_states': robot_states,
            'execution_history_length': len(self.execution_history)
        }
    
    def set_adapter(self, adapter):
        """Set the BestManAdapter for executing real robot actions"""
        self.adapter = adapter
        print(f"[FourStageCollaboration] Adapter set: {adapter}")
    
    def _get_observation(self) -> Dict:
        """Get current observation from the environment"""
        # Get scene graph from adapter if available
        scene_graph = {}
        if self.adapter and hasattr(self.adapter, 'get_scene_graph'):
            try:
                scene_graph = self.adapter.get_scene_graph()
            except Exception as e:
                print(f"[WARN] Failed to get scene graph: {e}")
        
        # Get robot states from adapter if available
        robot_states = {}
        if self.adapter and hasattr(self.adapter, 'get_robot_states'):
            try:
                robot_states = self.adapter.get_robot_states()
            except Exception as e:
                print(f"[WARN] Failed to get robot states: {e}")
        
        # Fallback to robot's own status if adapter not available
        if not robot_states:
            robot_states = {name: robot.get_status() for name, robot in self.robots.items()}
        
        return {
            'scene_graph': scene_graph,
            'robot_states': robot_states,
            'timestamp': time.time()
        }
    
    def _execute_action(self, robot_name: str, action: Dict) -> Dict:
        """
        Execute an action and return feedback
        Integrates with BestManAdapter to execute real robot actions
        """
        action_type = action.get('action', 'wait')
        params = action.get('params', {})
        
        print(f"[_execute_action] 开始执行: robot={robot_name}, action_type={action_type}, params={params}")
        
        # Use BestManAdapter to execute the action
        if hasattr(self, 'adapter') and self.adapter:
            from dynahmrc.integration.bestman_adapter import ExecutionFeedback
            print(f"[_execute_action] 调用 adapter.execute_action...")
            feedback = self.adapter.execute_action(robot_name, action_type, params)
            print(f"[_execute_action] adapter 返回: success={feedback.success}, message={feedback.message}")
            return {
                'success': feedback.success,
                'message': feedback.message,
                'state_change': feedback.state_changes,
                'execution_time': feedback.execution_time,
                'sensor_data': feedback.sensor_data
            }
        else:
            # Fallback: simulate action execution
            print(f"[WARN] No adapter available, simulating action: {action_type}")
            return {
                'success': True,
                'message': f"Simulated {action_type}",
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
