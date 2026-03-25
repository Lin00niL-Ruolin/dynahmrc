"""
Base Task Class - Common interface for all tasks
Supports dynamic variations: CTO, IRZ, ANC, REC
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from enum import Enum
import random


class TaskVariation(Enum):
    """Task variation types from DynaHMRC paper"""
    STATIC = "static"
    CTO = "cto"  # Change Task Objective
    IRZ = "irz"  # Inaccessible Region Zone
    ANC = "anc"  # Add New Collaborator
    REC = "rec"  # Remove Existing Collaborator


class BaseTask(ABC):
    """
    Base class for collaborative tasks
    Supports dynamic variations for evaluating adaptability
    """
    
    def __init__(self, task_id: str, config: Dict):
        self.task_id = task_id
        self.config = config
        self.max_steps = config.get('max_steps', 100)
        self.reflection_interval = config.get('reflection_interval', 10)
        
        # Task state
        self.current_step = 0
        self.completed = False
        self.success_rate = 0.0
        
        # Dynamic variations
        self.variation_type = config.get('variation', TaskVariation.STATIC)
        self.variation_step = config.get('variation_step', 30)
        self.variation_applied = False
        self.variations_history = []
        self.pending_changes = []
        
        # Variation-specific data
        self.inaccessible_objects = []  # For IRZ
        self.new_collaborator = None    # For ANC
        self.removed_collaborator = None  # For REC
    
    @abstractmethod
    def get_goal_description(self) -> str:
        """Return task goal description for LLM"""
        pass
    
    @abstractmethod
    def check_completion(self, robot_states: Dict) -> bool:
        """Check if task is completed"""
        pass
    
    @abstractmethod
    def get_reward(self, robot_states: Dict) -> float:
        """Calculate task reward"""
        pass
    
    @abstractmethod
    def get_partial_success(self) -> float:
        """Calculate partial success rate (0-1)"""
        pass
    
    def check_and_apply_variation(self, robot_team: List[str] = None) -> Optional[Dict]:
        """
        Check if variation should be applied and apply it
        
        Args:
            robot_team: Current list of robot IDs (for ANC/REC)
        
        Returns:
            Variation details if applied, None otherwise
        """
        if self.variation_applied or self.variation_type == TaskVariation.STATIC:
            return None
        
        if self.current_step < self.variation_step:
            return None
        
        # Apply variation based on type
        if self.variation_type == TaskVariation.CTO:
            return self._apply_cto()
        elif self.variation_type == TaskVariation.IRZ:
            return self._apply_irz()
        elif self.variation_type == TaskVariation.ANC:
            return self._apply_anc(robot_team or [])
        elif self.variation_type == TaskVariation.REC:
            return self._apply_rec(robot_team or [])
        
        return None
    
    def _apply_cto(self) -> Dict:
        """
        Change Task Objective variation
        Modifies task goal during execution
        """
        variation_data = {
            'type': 'CTO',
            'step': self.current_step,
            'description': ''
        }
        
        # To be overridden by specific tasks
        self.variation_applied = True
        self.variations_history.append(variation_data)
        
        return variation_data
    
    def _apply_irz(self) -> Dict:
        """
        Inaccessible Region Zone variation
        Makes certain objects/regions inaccessible
        """
        # Get all objects in the task
        all_objects = self._get_all_objects()
        
        if len(all_objects) >= 2:
            # Randomly select 1-2 objects to make inaccessible
            num_inaccessible = min(random.randint(1, 2), len(all_objects) - 1)
            self.inaccessible_objects = random.sample(all_objects, num_inaccessible)
        
        variation_data = {
            'type': 'IRZ',
            'step': self.current_step,
            'inaccessible_objects': self.inaccessible_objects,
            'description': f'Objects {self.inaccessible_objects} are now inaccessible'
        }
        
        self.variation_applied = True
        self.variations_history.append(variation_data)
        
        return variation_data
    
    def _apply_anc(self, current_robots: List[str]) -> Dict:
        """
        Add New Collaborator variation
        Adds a new robot to the team mid-task
        """
        # Define possible new robots by type
        available_robots = {
            'mobile_manipulator': ['Eve', 'Frank', 'Hannah'],
            'manipulator': ['Ivan', 'Julia'],
            'mobile': ['Kevin', 'Liam'],
            'drone': ['Megan', 'Nathan']
        }
        
        # Select a robot type and name
        robot_type = random.choice(list(available_robots.keys()))
        candidates = [r for r in available_robots[robot_type] if r not in current_robots]
        
        if candidates:
            self.new_collaborator = {
                'robot_id': random.choice(candidates),
                'robot_type': robot_type
            }
        
        variation_data = {
            'type': 'ANC',
            'step': self.current_step,
            'new_robot': self.new_collaborator,
            'description': f'New robot {self.new_collaborator["robot_id"]} ({robot_type}) joined the team'
        }
        
        self.variation_applied = True
        self.variations_history.append(variation_data)
        
        return variation_data
    
    def _apply_rec(self, current_robots: List[str]) -> Dict:
        """
        Remove Existing Collaborator variation
        Removes a robot from the team mid-task
        """
        if len(current_robots) > 1:
            self.removed_collaborator = random.choice(current_robots)
        
        variation_data = {
            'type': 'REC',
            'step': self.current_step,
            'removed_robot': self.removed_collaborator,
            'description': f'Robot {self.removed_collaborator} left the team'
        }
        
        self.variation_applied = True
        self.variations_history.append(variation_data)
        
        return variation_data
    
    def _get_all_objects(self) -> List[str]:
        """Get all objects involved in the task (to be overridden)"""
        return []
    
    def is_object_accessible(self, obj_name: str) -> bool:
        """Check if an object is accessible (for IRZ variation)"""
        return obj_name not in self.inaccessible_objects
    
    def inject_variation(self, variation_type: str, params: Dict):
        """
        Manually inject dynamic task variation
        Types: CTO, IRZ, ANC, REC
        """
        self.pending_changes.append({
            'type': variation_type,
            'params': params,
            'step': self.current_step
        })
    
    def apply_pending_changes(self) -> List[Dict]:
        """Apply and return pending dynamic changes"""
        changes = self.pending_changes.copy()
        self.pending_changes = []
        self.variations_history.extend(changes)
        return changes
    
    def get_task_status(self) -> Dict:
        """Get current task status for feedback"""
        return {
            'task_id': self.task_id,
            'step': self.current_step,
            'max_steps': self.max_steps,
            'completed': self.completed,
            'variation_type': self.variation_type.value if isinstance(self.variation_type, TaskVariation) else self.variation_type,
            'variation_applied': self.variation_applied,
            'variations_history': self.variations_history,
            'inaccessible_objects': self.inaccessible_objects if self.inaccessible_objects else None
        }
    
    def step(self):
        """Increment step counter"""
        self.current_step += 1
        return self.current_step < self.max_steps
