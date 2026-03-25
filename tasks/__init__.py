"""
Tasks module for DynaHMRC
"""

from typing import Dict

from .base_task import BaseTask, TaskVariation
from .pack_objects import PackObjectsTask
from .make_sandwich import MakeSandwichTask
from .sort_solids import SortSolidsTask

__all__ = [
    'BaseTask',
    'TaskVariation',
    'PackObjectsTask',
    'MakeSandwichTask',
    'SortSolidsTask'
]

# Task factory
def create_task(task_type: str, task_id: str, config: Dict) -> BaseTask:
    """Factory function to create task instances"""
    task_map = {
        'pack_objects': PackObjectsTask,
        'make_sandwich': MakeSandwichTask,
        'sort_solids': SortSolidsTask
    }
    
    task_class = task_map.get(task_type)
    if task_class is None:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return task_class(task_id, config)
