"""
Make Sandwich Task - Sequential stacking
Supports dynamic variations: CTO, IRZ, ANC, REC
"""

from typing import Dict, List
from .base_task import BaseTask, TaskVariation


class MakeSandwichTask(BaseTask):
    """
    Assemble sandwich by stacking ingredients in specific order
    Tests sequential manipulation and ordering constraints
    """
    
    def __init__(self, task_id: str, config: Dict):
        super().__init__(task_id, config)
        self.ingredient_order = config.get('ingredient_order', [])  # ['bread', 'lettuce', 'tomato', 'bread']
        self.original_ingredient_order = self.ingredient_order.copy()
        self.target_location = config.get('target_location', 'cutting_board')
        self.current_stack = []
        
    def get_goal_description(self) -> str:
        """Return task goal"""
        order_str = ' -> '.join(self.ingredient_order)
        return f"Make a sandwich by stacking ingredients in order on {self.target_location}: {order_str}"
    
    def check_completion(self, robot_states: Dict) -> bool:
        """Check if sandwich is correctly assembled"""
        # Get current stack state from target location
        location_contents = robot_states.get('location_contents', {}).get(self.target_location, [])
        self.current_stack = location_contents
        
        # Check if stack matches required order
        return self.current_stack == self.ingredient_order
    
    def get_partial_success(self) -> float:
        """Calculate partial success based on correct prefix"""
        if not self.ingredient_order:
            return 0.0
        
        correct_prefix = 0
        for i, ingredient in enumerate(self.ingredient_order):
            if i < len(self.current_stack) and self.current_stack[i] == ingredient:
                correct_prefix += 1
            else:
                break
        
        return correct_prefix / len(self.ingredient_order)
    
    def get_reward(self, robot_states: Dict) -> float:
        """Calculate reward"""
        # Higher reward for correct order, penalty for wrong order
        if self.current_stack != self.ingredient_order[:len(self.current_stack)]:
            return -0.1  # Wrong order penalty
        
        return self.get_partial_success() - self.current_step * 0.01
    
    def get_task_status(self) -> Dict:
        """Get detailed task status"""
        status = super().get_task_status()
        status.update({
            'required_order': self.ingredient_order,
            'current_stack': self.current_stack,
            'partial_success': self.get_partial_success(),
            'target_location': self.target_location
        })
        return status
    
    def _get_all_objects(self) -> List[str]:
        """Get all objects involved in the task"""
        return list(set(self.original_ingredient_order))
    
    def _apply_cto(self) -> Dict:
        """
        Change Task Objective variation for Make Sandwich
        Modifies ingredient order or target location
        """
        import random
        
        original_order = self.ingredient_order.copy()
        
        # CTO strategies:
        # 1. Reverse order
        # 2. Swap two ingredients
        # 3. Add/remove ingredients
        # 4. Change target location
        
        strategy = random.choice(['reverse', 'swap', 'modify', 'change_location'])
        
        if strategy == 'reverse' and len(self.ingredient_order) > 2:
            # Reverse the order
            self.ingredient_order = self.ingredient_order[::-1]
            description = f"Ingredient order reversed"
            
        elif strategy == 'swap' and len(self.ingredient_order) >= 2:
            # Swap two random ingredients
            idx1, idx2 = random.sample(range(len(self.ingredient_order)), 2)
            self.ingredient_order[idx1], self.ingredient_order[idx2] = \
                self.ingredient_order[idx2], self.ingredient_order[idx1]
            description = f"Swapped positions {idx1} and {idx2}"
            
        elif strategy == 'modify':
            # Add or remove an ingredient
            all_ingredients = ['bread', 'lettuce', 'tomato', 'cheese', 'ham', 'onion', 'pickle']
            action = random.choice(['add', 'remove'])
            
            if action == 'add':
                available = [i for i in all_ingredients if i not in self.ingredient_order]
                if available:
                    new_ingredient = random.choice(available)
                    # Insert at random position (not first or last for bread)
                    if new_ingredient == 'bread':
                        self.ingredient_order.append(new_ingredient)
                    else:
                        pos = random.randint(1, max(1, len(self.ingredient_order) - 1))
                        self.ingredient_order.insert(pos, new_ingredient)
                    description = f"Added {new_ingredient} to the recipe"
                else:
                    description = "No new ingredients available to add"
            else:  # remove
                if len(self.ingredient_order) > 2:
                    # Don't remove all bread
                    removable = [i for i in self.ingredient_order if i != 'bread' or self.ingredient_order.count('bread') > 2]
                    if removable:
                        to_remove = random.choice(removable)
                        self.ingredient_order.remove(to_remove)
                        description = f"Removed {to_remove} from the recipe"
                    else:
                        description = "No ingredients can be removed"
                else:
                    description = "Cannot remove (minimum ingredients reached)"
        
        elif strategy == 'change_location':
            # Change target location
            alternative_locations = ['plate', 'tray', 'counter', 'table']
            new_location = random.choice([l for l in alternative_locations if l != self.target_location])
            old_location = self.target_location
            self.target_location = new_location
            description = f"Target location changed from {old_location} to {new_location}"
        
        else:
            description = "No change applied"
        
        # Check if rework is needed (stack doesn't match new order)
        requires_rework = len(self.current_stack) > 0 and self.current_stack != self.ingredient_order[:len(self.current_stack)]
        
        variation_data = {
            'type': 'CTO',
            'step': self.current_step,
            'original_order': original_order,
            'new_order': self.ingredient_order.copy(),
            'target_location': self.target_location,
            'strategy': strategy,
            'requires_rework': requires_rework,
            'description': description
        }
        
        self.variation_applied = True
        self.variations_history.append(variation_data)
        
        return variation_data
