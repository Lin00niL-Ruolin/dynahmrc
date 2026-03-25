"""
Sort Solids Task - Color-based matching
Supports dynamic variations: CTO, IRZ, ANC, REC
"""

from typing import Dict, List
from .base_task import BaseTask, TaskVariation


class SortSolidsTask(BaseTask):
    """
    Sort colored solids onto matching colored panels
    Requires color recognition and precise placement
    """
    
    def __init__(self, task_id: str, config: Dict):
        super().__init__(task_id, config)
        self.color_pairs = config.get('color_pairs', [])  # [('red', 'red_panel'), ...]
        self.original_color_pairs = self.color_pairs.copy()
        self.sorted_objects = {}
        
    def get_goal_description(self) -> str:
        """Return task goal"""
        pairs_str = '; '.join([f"{obj} -> {panel}" for obj, panel in self.color_pairs])
        return f"Sort colored solids onto matching colored panels: {pairs_str}"
    
    def check_completion(self, robot_states: Dict) -> bool:
        """Check if all objects are on correct panels"""
        panel_states = robot_states.get('panel_contents', {})
        
        for obj_color, panel in self.color_pairs:
            # Extract color from object name (e.g., "red_cube" -> "red")
            obj_name = f"{obj_color}_solid"
            target_panel = f"{panel}"
            
            # Check if object is on correct panel
            panel_contents = panel_states.get(target_panel, [])
            if obj_name not in panel_contents:
                return False
        
        return True
    
    def get_partial_success(self) -> float:
        """Calculate partial success rate"""
        if not self.color_pairs:
            return 0.0
        
        # Get panel states
        # This is a simplified version - actual implementation would check scene state
        correct = 0
        for obj_color, panel in self.color_pairs:
            obj_name = f"{obj_color}_solid"
            # Check if correctly placed (would need actual state)
            # For now, use sorted_objects tracking
            if self.sorted_objects.get(obj_color, False):
                correct += 1
        
        return correct / len(self.color_pairs)
    
    def get_reward(self, robot_states: Dict) -> float:
        """Calculate reward"""
        return self.get_partial_success() - self.current_step * 0.01
    
    def get_task_status(self) -> Dict:
        """Get detailed task status"""
        status = super().get_task_status()
        status.update({
            'color_pairs': self.color_pairs,
            'sorted_count': sum(self.sorted_objects.values()),
            'partial_success': self.get_partial_success()
        })
        return status
    
    def _get_all_objects(self) -> List[str]:
        """Get all objects involved in the task"""
        return [f"{color}_solid" for color, _ in self.original_color_pairs]
    
    def _apply_cto(self) -> Dict:
        """
        Change Task Objective variation for Sort Solids
        Modifies color-object to panel mappings
        """
        import random
        
        original_pairs = self.color_pairs.copy()
        
        # CTO strategies:
        # 1. Swap panel assignments
        # 2. Add new color pair
        # 3. Remove a color pair
        # 4. Change to different matching rule (e.g., shape instead of color)
        
        strategy = random.choice(['swap', 'add', 'remove', 'shuffle'])
        
        if strategy == 'swap' and len(self.color_pairs) >= 2:
            # Swap two panel assignments
            idx1, idx2 = random.sample(range(len(self.color_pairs)), 2)
            obj1, panel1 = self.color_pairs[idx1]
            obj2, panel2 = self.color_pairs[idx2]
            
            # Swap panels
            self.color_pairs[idx1] = (obj1, panel2)
            self.color_pairs[idx2] = (obj2, panel1)
            
            description = f"Swapped panels: {obj1}->{panel2}, {obj2}->{panel1}"
            
        elif strategy == 'add':
            # Add a new color pair
            all_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
            available = [c for c in all_colors if c not in [p[0] for p in self.color_pairs]]
            
            if available:
                new_color = random.choice(available)
                # Match with same color panel
                new_panel = f"{new_color}_panel"
                self.color_pairs.append((new_color, new_panel))
                description = f"Added new color pair: {new_color} -> {new_panel}"
            else:
                description = "No new colors available to add"
        
        elif strategy == 'remove':
            # Remove a color pair
            if len(self.color_pairs) > 2:
                removed = random.choice(self.color_pairs)
                self.color_pairs.remove(removed)
                description = f"Removed color pair: {removed[0]} -> {removed[1]}"
            else:
                description = "Cannot remove (minimum pairs reached)"
        
        elif strategy == 'shuffle':
            # Shuffle all panel assignments
            objects = [p[0] for p in self.color_pairs]
            panels = [p[1] for p in self.color_pairs]
            
            # Shuffle panels while keeping objects
            random.shuffle(panels)
            self.color_pairs = list(zip(objects, panels))
            
            description = "All panel assignments shuffled"
        
        else:
            description = "No change applied"
        
        variation_data = {
            'type': 'CTO',
            'step': self.current_step,
            'original_pairs': original_pairs,
            'new_pairs': self.color_pairs.copy(),
            'strategy': strategy,
            'description': description
        }
        
        self.variation_applied = True
        self.variations_history.append(variation_data)
        
        return variation_data
