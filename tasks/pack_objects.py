"""
Pack Objects Task - Fundamental picking and placing
Supports dynamic variations: CTO, IRZ, ANC, REC
"""

from typing import Dict, List
from .base_task import BaseTask, TaskVariation


class PackObjectsTask(BaseTask):
    """
    Pack objects into a designated tray
    Evaluates basic manipulation capabilities
    """
    
    def __init__(self, task_id: str, config: Dict):
        super().__init__(task_id, config)
        self.target_objects = config.get('target_objects', [])
        self.original_target_objects = self.target_objects.copy()
        self.target_container = config.get('target_container', 'tray')
        self.placed_objects = []
        
    def get_goal_description(self) -> str:
        """Return task goal"""
        objects_str = ', '.join(self.target_objects)
        return f"Pack the following objects into the {self.target_container}: {objects_str}"
    
    def check_completion(self, robot_states: Dict) -> bool:
        """Check if all objects are in tray"""
        # Check container contents from scene graph
        container_contents = robot_states.get('container_contents', {}).get(self.target_container, [])
        self.placed_objects = container_contents
        
        # Check if all target objects are placed
        return all(obj in container_contents for obj in self.target_objects)
    
    def get_partial_success(self) -> float:
        """Calculate partial success rate"""
        if not self.target_objects:
            return 0.0
        placed = sum(1 for obj in self.target_objects if obj in self.placed_objects)
        return placed / len(self.target_objects)
    
    def get_reward(self, robot_states: Dict) -> float:
        """Calculate reward based on progress"""
        completion = self.get_partial_success()
        step_penalty = self.current_step * 0.01
        return completion - step_penalty
    
    def get_task_status(self) -> Dict:
        """Get detailed task status"""
        status = super().get_task_status()
        status.update({
            'target_objects': self.target_objects,
            'placed_objects': self.placed_objects,
            'target_container': self.target_container,
            'partial_success': self.get_partial_success()
        })
        return status
    
    def _get_all_objects(self) -> List[str]:
        """Get all objects involved in the task"""
        return self.original_target_objects.copy()
    
    def _apply_cto(self) -> Dict:
        """
        Change Task Objective variation for Pack Objects
        Modifies target objects during execution
        """
        import random
        
        original = self.target_objects.copy()
        
        # CTO strategies:
        # 1. Remove some objects from target
        # 2. Add new objects to target
        # 3. Change container
        
        strategy = random.choice(['remove', 'add', 'change_container'])
        
        if strategy == 'remove' and len(self.target_objects) > 1:
            # Remove 1-2 objects from target
            num_remove = min(random.randint(1, 2), len(self.target_objects) - 1)
            removed = random.sample(self.target_objects, num_remove)
            self.target_objects = [obj for obj in self.target_objects if obj not in removed]
            description = f"Target objects changed: removed {removed}"
            
        elif strategy == 'add':
            # Add new objects (from original list that were not in target)
            available = [obj for obj in self.original_target_objects if obj not in self.target_objects]
            if available:
                num_add = min(random.randint(1, 2), len(available))
                added = random.sample(available, num_add)
                self.target_objects.extend(added)
                description = f"Target objects changed: added {added}"
            else:
                description = "No change possible (all objects already targeted)"
        
        elif strategy == 'change_container':
            # Change target container
            alternative_containers = ['box', 'bin', 'shelf', 'cart']
            new_container = random.choice([c for c in alternative_containers if c != self.target_container])
            old_container = self.target_container
            self.target_container = new_container
            description = f"Target container changed from {old_container} to {new_container}"
        
        else:
            description = "No change applied"
        
        variation_data = {
            'type': 'CTO',
            'step': self.current_step,
            'original_objects': original,
            'new_objects': self.target_objects.copy(),
            'target_container': self.target_container,
            'strategy': strategy,
            'description': description
        }
        
        self.variation_applied = True
        self.variations_history.append(variation_data)
        
        return variation_data
