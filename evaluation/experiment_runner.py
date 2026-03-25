"""
Experiment Runner for DynaHMRC
Runs batch experiments for different task types and variations
"""

import os
import sys
import json
import time
import random
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import MetricsCollector, TaskVariation, MetricsFormatter


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    task_type: str  # pack_objects, make_sandwich, sort_solids
    variation: TaskVariation
    robot_team_config: str  # e.g., "Ma-MoMa-UAV"
    max_steps: int = 100
    num_trials: int = 8  # Paper uses 8 trials per configuration
    enable_gui: bool = False
    
    def get_name(self) -> str:
        """Get experiment name"""
        return f"{self.task_type}_{self.variation.value}_{self.robot_team_config}"


class TaskVariationInjector:
    """Injects dynamic variations during task execution"""
    
    def __init__(self, variation: TaskVariation, inject_step: int = 30):
        self.variation = variation
        self.inject_step = inject_step
        self.injected = False
        self.variation_data = {}
    
    def should_inject(self, current_step: int) -> bool:
        """Check if variation should be injected at current step"""
        if self.injected:
            return False
        if self.variation == TaskVariation.STATIC:
            return False
        return current_step >= self.inject_step
    
    def inject_cto(self, task) -> Dict:
        """
        Change Task Objective variation
        Modifies task goal during execution
        """
        if hasattr(task, 'target_objects'):
            # For pack_objects: change target objects
            original = task.target_objects.copy()
            # Add or remove objects
            if len(original) > 1:
                task.target_objects = original[:-1]  # Remove last object
            self.variation_data = {
                'type': 'CTO',
                'original': original,
                'new': task.target_objects.copy(),
                'step': task.current_step
            }
        elif hasattr(task, 'ingredient_order'):
            # For make_sandwich: change ingredient order
            original = task.ingredient_order.copy()
            if len(original) > 2:
                # Swap order
                task.ingredient_order = original[::-1]
            self.variation_data = {
                'type': 'CTO',
                'original': original,
                'new': task.ingredient_order.copy(),
                'step': task.current_step
            }
        
        self.injected = True
        return self.variation_data
    
    def inject_irz(self, scene_objects: List[str]) -> Dict:
        """
        Inaccessible Region Zone variation
        Makes certain objects/regions inaccessible
        """
        # Mark some objects as inaccessible
        inaccessible = random.sample(scene_objects, min(2, len(scene_objects)))
        self.variation_data = {
            'type': 'IRZ',
            'inaccessible_objects': inaccessible,
            'description': f'Objects {inaccessible} are now inaccessible'
        }
        self.injected = True
        return self.variation_data
    
    def inject_anc(self, current_robots: List[str]) -> Dict:
        """
        Add New Collaborator variation
        Adds a new robot to the team mid-task
        """
        # Define possible new robots
        available_robots = ['Charlie', 'Eve', 'Frank', 'Grace']
        new_robot = random.choice([r for r in available_robots if r not in current_robots])
        
        self.variation_data = {
            'type': 'ANC',
            'new_robot': new_robot,
            'robot_type': random.choice(['mobile_manipulator', 'manipulator', 'mobile', 'drone']),
            'step': self.inject_step
        }
        self.injected = True
        return self.variation_data
    
    def inject_rec(self, current_robots: List[str]) -> Dict:
        """
        Remove Existing Collaborator variation
        Removes a robot from the team mid-task
        """
        if len(current_robots) > 1:
            removed_robot = random.choice(current_robots)
            self.variation_data = {
                'type': 'REC',
                'removed_robot': removed_robot,
                'step': self.inject_step
            }
            self.injected = True
            return self.variation_data
        return {}


class ExperimentRunner:
    """Runs batch experiments and collects metrics"""
    
    def __init__(self, output_dir: str = "experiments/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_collector = MetricsCollector()
        self.experiment_results: Dict[str, List[Dict]] = {}
        
        # Robot team configurations from paper
        self.team_configs = {
            "Ma-MoMa-UAV": [
                {"robot_id": "Alice", "robot_type": "mobile_manipulator"},
                {"robot_id": "Bob", "robot_type": "manipulator"},
                {"robot_id": "Charlie", "robot_type": "drone"},
            ],
            "Ma-MoMa": [
                {"robot_id": "Alice", "robot_type": "mobile_manipulator"},
                {"robot_id": "Bob", "robot_type": "manipulator"},
            ],
            "Ma-UAV": [
                {"robot_id": "Alice", "robot_type": "mobile_manipulator"},
                {"robot_id": "Charlie", "robot_type": "drone"},
            ],
            "Ma-MoMa-Mo": [
                {"robot_id": "Alice", "robot_type": "mobile_manipulator"},
                {"robot_id": "Bob", "robot_type": "manipulator"},
                {"robot_id": "David", "robot_type": "mobile"},
            ],
            "Ma-MoMa-Mo-UAV": [
                {"robot_id": "Alice", "robot_type": "mobile_manipulator"},
                {"robot_id": "Bob", "robot_type": "manipulator"},
                {"robot_id": "David", "robot_type": "mobile"},
                {"robot_id": "Charlie", "robot_type": "drone"},
            ],
            "Ma-Mo-UAV": [
                {"robot_id": "Alice", "robot_type": "mobile_manipulator"},
                {"robot_id": "David", "robot_type": "mobile"},
                {"robot_id": "Charlie", "robot_type": "drone"},
            ],
        }
    
    def run_experiment(self, config: ExperimentConfig, 
                       task_executor: Callable) -> Dict[str, Any]:
        """
        Run a single experiment configuration
        
        Args:
            config: Experiment configuration
            task_executor: Function that executes the task and returns results
        
        Returns:
            Summary metrics for this experiment
        """
        print(f"\n{'='*60}")
        print(f"Running Experiment: {config.get_name()}")
        print(f"Task: {config.task_type}, Variation: {config.variation.value}")
        print(f"Team: {config.robot_team_config}, Trials: {config.num_trials}")
        print(f"{'='*60}\n")
        
        trial_results = []
        
        for trial in range(config.num_trials):
            print(f"  Trial {trial + 1}/{config.num_trials}...", end=" ")
            
            try:
                # Run trial
                result = self._run_trial(config, task_executor, trial)
                trial_results.append(result)
                print(f"Success={result.get('success', False)}")
                
            except Exception as e:
                print(f"FAILED: {e}")
                traceback.print_exc()
                trial_results.append({
                    'trial': trial,
                    'success': False,
                    'error': str(e)
                })
        
        # Calculate aggregate metrics
        summary = self._calculate_summary(trial_results)
        
        # Store results
        self.experiment_results[config.get_name()] = trial_results
        
        print(f"\n  Summary for {config.get_name()}:")
        print(f"    SUCC: {summary.get('SUCC', 0):.4f}")
        print(f"    PS: {summary.get('PS', 0):.4f}")
        print(f"    TS: {summary.get('TS', 0):.2f}")
        print(f"    CC: {summary.get('CC', 0):.2f}")
        
        return summary
    
    def _run_trial(self, config: ExperimentConfig, 
                   task_executor: Callable, trial_num: int) -> Dict:
        """Run a single trial"""
        # Get robot team
        robot_team = self.team_configs.get(config.robot_team_config, [])
        robot_ids = [r['robot_id'] for r in robot_team]
        robot_types = {r['robot_id']: r['robot_type'] for r in robot_team}
        
        # Start metrics collection
        task_id = f"{config.get_name()}_trial_{trial_num}"
        self.metrics_collector.start_task(
            task_id=task_id,
            task_type=config.task_type,
            variation=config.variation,
            robot_team=robot_ids,
            robot_types=robot_types
        )
        
        # Create variation injector
        injector = TaskVariationInjector(config.variation)
        
        # Execute task
        result = task_executor(
            task_type=config.task_type,
            robot_team=robot_team,
            variation_injector=injector,
            max_steps=config.max_steps,
            enable_gui=config.enable_gui
        )
        
        # End metrics collection
        self.metrics_collector.end_task(
            success=result.get('success', False),
            partial_success=result.get('partial_success', 0.0)
        )
        
        return {
            'trial': trial_num,
            'success': result.get('success', False),
            'partial_success': result.get('partial_success', 0.0),
            'steps': result.get('steps', 0),
            'communications': result.get('communications', 0),
        }
    
    def _calculate_summary(self, trial_results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics from trial results"""
        if not trial_results:
            return {}
        
        successful = sum(1 for r in trial_results if r.get('success', False))
        total = len(trial_results)
        
        return {
            'SUCC': successful / total if total > 0 else 0,
            'PS': sum(r.get('partial_success', 0) for r in trial_results) / total,
            'TS': sum(r.get('steps', 0) for r in trial_results) / total,
            'CC': sum(r.get('communications', 0) for r in trial_results) / total,
            'num_trials': total,
            'num_success': successful
        }
    
    def run_all_experiments(self, task_executor: Callable,
                           task_types: List[str] = None,
                           variations: List[TaskVariation] = None,
                           team_configs: List[str] = None):
        """
        Run all experiment configurations
        
        Args:
            task_executor: Function to execute tasks
            task_types: List of task types to run (default: all)
            variations: List of variations to run (default: all)
            team_configs: List of team configs to run (default: all)
        """
        task_types = task_types or ['pack_objects', 'make_sandwich', 'sort_solids']
        variations = variations or list(TaskVariation)
        team_configs = team_configs or list(self.team_configs.keys())
        
        all_summaries = {}
        
        for task_type in task_types:
            for variation in variations:
                for team_config in team_configs:
                    config = ExperimentConfig(
                        task_type=task_type,
                        variation=variation,
                        robot_team_config=team_config,
                        num_trials=8  # Paper standard
                    )
                    
                    summary = self.run_experiment(config, task_executor)
                    all_summaries[config.get_name()] = summary
        
        # Export results
        self._export_results(all_summaries)
        
        return all_summaries
    
    def _export_results(self, summaries: Dict[str, Dict[str, Any]]):
        """Export experiment results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Export metrics
        metrics_file = self.output_dir / f"metrics_{timestamp}.json"
        self.metrics_collector.export_to_json(str(metrics_file))
        
        # Export CSV
        csv_file = self.output_dir / f"results_{timestamp}.csv"
        self.metrics_collector.export_to_csv(str(csv_file))
        
        # Export formatted table
        table_file = self.output_dir / f"table_{timestamp}.txt"
        with open(table_file, 'w') as f:
            f.write("DynaHMRC Experiment Results\n")
            f.write("=" * 100 + "\n\n")
            f.write(MetricsFormatter.format_table(summaries))
        
        # Export LaTeX table
        latex_file = self.output_dir / f"latex_table_{timestamp}.txt"
        with open(latex_file, 'w') as f:
            f.write(MetricsFormatter.format_latex_table(summaries))
        
        print(f"\n{'='*60}")
        print("Results exported to:")
        print(f"  JSON: {metrics_file}")
        print(f"  CSV: {csv_file}")
        print(f"  Table: {table_file}")
        print(f"  LaTeX: {latex_file}")
        print(f"{'='*60}\n")


def create_mock_task_executor():
    """Create a mock task executor for testing"""
    def mock_executor(task_type: str, robot_team: List[Dict],
                     variation_injector: TaskVariationInjector,
                     max_steps: int = 100, enable_gui: bool = False) -> Dict:
        """Mock task execution for testing"""
        # Simulate task execution
        steps = random.randint(10, max_steps)
        success = random.random() > 0.3
        partial_success = random.random() if not success else 1.0
        
        return {
            'success': success,
            'partial_success': partial_success,
            'steps': steps,
            'communications': random.randint(5, 50),
        }
    
    return mock_executor


if __name__ == "__main__":
    # Test the experiment runner
    runner = ExperimentRunner()
    
    # Run a small test
    test_config = ExperimentConfig(
        task_type="pack_objects",
        variation=TaskVariation.STATIC,
        robot_team_config="Ma-MoMa-UAV",
        num_trials=2  # Small number for testing
    )
    
    mock_executor = create_mock_task_executor()
    result = runner.run_experiment(test_config, mock_executor)
    
    print("\nTest completed!")
    print(f"Result: {result}")
