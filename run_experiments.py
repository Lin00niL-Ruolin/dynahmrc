"""
Batch Experiment Runner for DynaHMRC Paper Reproduction
Runs all experiments from the paper and generates result tables
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynahmrc.evaluation.experiment_runner import (
    ExperimentRunner, ExperimentConfig, TaskVariationInjector
)
from dynahmrc.evaluation.metrics import TaskVariation, MetricsFormatter
from dynahmrc.system import DynaHMRCSystem


def create_llm_config(provider: str = "mock", api_key: str = None, model: str = None) -> Dict:
    """Create LLM configuration"""
    configs = {
        "mock": {
            "provider": "mock",
            "api_key": "",
            "model": "mock"
        },
        "kimi": {
            "provider": "kimi",
            "api_key": api_key or os.getenv("MOONSHOT_API_KEY", ""),
            "model": model or "kimi-k2.5"
        },
        "openai": {
            "provider": "openai",
            "api_key": api_key or os.getenv("OPENAI_API_KEY", ""),
            "model": model or "gpt-4o"
        }
    }
    return configs.get(provider, configs["mock"])


def create_scene_config(task_type: str) -> Dict:
    """Create scene configuration for a task type"""
    base_config = {
        "config_path": "Config/default.yaml",
        "gui": False
    }
    
    # Task-specific scene configurations
    if task_type == "pack_objects":
        base_config["objects"] = [
            {"name": "apple", "model_path": "Asset/Scene/Object/URDF_models/apple.urdf", 
             "position": [1.0, 1.0, 0.5]},
            {"name": "banana", "model_path": "Asset/Scene/Object/URDF_models/banana.urdf",
             "position": [1.2, 1.0, 0.5]},
            {"name": "orange", "model_path": "Asset/Scene/Object/URDF_models/orange.urdf",
             "position": [0.8, 1.2, 0.5]},
            {"name": "tray", "model_path": "Asset/Scene/Object/URDF_models/tray.urdf",
             "position": [2.0, 2.0, 0.5]},
        ]
    elif task_type == "make_sandwich":
        base_config["objects"] = [
            {"name": "bread_1", "model_path": "Asset/Scene/Object/URDF_models/bread.urdf",
             "position": [1.0, 1.0, 0.5]},
            {"name": "bread_2", "model_path": "Asset/Scene/Object/URDF_models/bread.urdf",
             "position": [1.2, 1.0, 0.5]},
            {"name": "lettuce", "model_path": "Asset/Scene/Object/URDF_models/lettuce.urdf",
             "position": [1.0, 1.2, 0.5]},
            {"name": "tomato", "model_path": "Asset/Scene/Object/URDF_models/tomato.urdf",
             "position": [1.2, 1.2, 0.5]},
            {"name": "cutting_board", "model_path": "Asset/Scene/Object/URDF_models/board.urdf",
             "position": [2.0, 2.0, 0.5]},
        ]
    elif task_type == "sort_solids":
        base_config["objects"] = [
            {"name": "red_solid", "model_path": "Asset/Scene/Object/URDF_models/red_cube.urdf",
             "position": [1.0, 1.0, 0.5]},
            {"name": "blue_solid", "model_path": "Asset/Scene/Object/URDF_models/blue_cube.urdf",
             "position": [1.2, 1.0, 0.5]},
            {"name": "green_solid", "model_path": "Asset/Scene/Object/URDF_models/green_cube.urdf",
             "position": [1.0, 1.2, 0.5]},
            {"name": "red_panel", "model_path": "Asset/Scene/Object/URDF_models/red_panel.urdf",
             "position": [2.0, 2.0, 0.1]},
            {"name": "blue_panel", "model_path": "Asset/Scene/Object/URDF_models/blue_panel.urdf",
             "position": [2.2, 2.0, 0.1]},
            {"name": "green_panel", "model_path": "Asset/Scene/Object/URDF_models/green_panel.urdf",
             "position": [2.0, 2.2, 0.1]},
        ]
    
    return base_config


def create_robot_configs(team_config: str) -> List[Dict]:
    """Create robot configurations for a team setup"""
    team_configs = {
        "Ma-MoMa-UAV": [
            {"robot_id": "Alice", "robot_type": "mobile_manipulator", 
             "robot_model": "panda_on_segbot", "init_position": [0, 0, 0]},
            {"robot_id": "Bob", "robot_type": "manipulator",
             "robot_model": "panda", "init_position": [0.5, 0, 0]},
            {"robot_id": "Charlie", "robot_type": "drone",
             "robot_model": "drone", "init_position": [0, 0.5, 1.0]},
        ],
        "Ma-MoMa": [
            {"robot_id": "Alice", "robot_type": "mobile_manipulator",
             "robot_model": "panda_on_segbot", "init_position": [0, 0, 0]},
            {"robot_id": "Bob", "robot_type": "manipulator",
             "robot_model": "panda", "init_position": [0.5, 0, 0]},
        ],
        "Ma-UAV": [
            {"robot_id": "Alice", "robot_type": "mobile_manipulator",
             "robot_model": "panda_on_segbot", "init_position": [0, 0, 0]},
            {"robot_id": "Charlie", "robot_type": "drone",
             "robot_model": "drone", "init_position": [0, 0.5, 1.0]},
        ],
        "Ma-MoMa-Mo": [
            {"robot_id": "Alice", "robot_type": "mobile_manipulator",
             "robot_model": "panda_on_segbot", "init_position": [0, 0, 0]},
            {"robot_id": "Bob", "robot_type": "manipulator",
             "robot_model": "panda", "init_position": [0.5, 0, 0]},
            {"robot_id": "David", "robot_type": "mobile",
             "robot_model": "segbot", "init_position": [0, 0.5, 0]},
        ],
        "Ma-MoMa-Mo-UAV": [
            {"robot_id": "Alice", "robot_type": "mobile_manipulator",
             "robot_model": "panda_on_segbot", "init_position": [0, 0, 0]},
            {"robot_id": "Bob", "robot_type": "manipulator",
             "robot_model": "panda", "init_position": [0.5, 0, 0]},
            {"robot_id": "David", "robot_type": "mobile",
             "robot_model": "segbot", "init_position": [0, 0.5, 0]},
            {"robot_id": "Charlie", "robot_type": "drone",
             "robot_model": "drone", "init_position": [0, 0.5, 1.0]},
        ],
        "Ma-Mo-UAV": [
            {"robot_id": "Alice", "robot_type": "mobile_manipulator",
             "robot_model": "panda_on_segbot", "init_position": [0, 0, 0]},
            {"robot_id": "David", "robot_type": "mobile",
             "robot_model": "segbot", "init_position": [0, 0.5, 0]},
            {"robot_id": "Charlie", "robot_type": "drone",
             "robot_model": "drone", "init_position": [0, 0.5, 1.0]},
        ],
    }
    
    return team_configs.get(team_config, team_configs["Ma-MoMa-UAV"])


def execute_task_with_system(task_type: str, robot_team: List[Dict],
                            variation_injector: TaskVariationInjector,
                            max_steps: int = 100, enable_gui: bool = False,
                            llm_config: Dict = None) -> Dict:
    """
    Execute a task using DynaHMRCSystem
    
    This is the actual task executor that uses the full system
    """
    # Create configurations
    scene_config = create_scene_config(task_type)
    scene_config["gui"] = enable_gui
    
    robot_configs = create_robot_configs(
        "-".join(set(r["robot_type"].replace("_", "").title() for r in robot_team))
    )
    
    llm_config = llm_config or create_llm_config("mock")
    
    try:
        # Initialize system
        system = DynaHMRCSystem(scene_config, robot_configs, llm_config, 
                               enable_visualization=enable_gui)
        
        if not system.initialize():
            return {
                'success': False,
                'partial_success': 0.0,
                'steps': 0,
                'communications': 0,
                'error': 'System initialization failed'
            }
        
        # Create task description based on type
        task_descriptions = {
            'pack_objects': 'Pack all objects into the tray',
            'make_sandwich': 'Make a sandwich by stacking ingredients in order',
            'sort_solids': 'Sort colored solids onto matching colored panels'
        }
        task_description = task_descriptions.get(task_type, 'Complete the task')
        
        # Execute task
        result = system.execute_task(task_description, max_steps=max_steps)
        
        return {
            'success': result.get('success', False),
            'partial_success': result.get('partial_success', 0.0),
            'steps': result.get('steps', 0),
            'communications': result.get('communications', 0),
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'partial_success': 0.0,
            'steps': 0,
            'communications': 0,
            'error': str(e)
        }


def run_paper_experiments(llm_provider: str = "mock", 
                         num_trials: int = 8,
                         task_types: List[str] = None,
                         variations: List[str] = None,
                         team_configs: List[str] = None,
                         output_dir: str = "experiments/results"):
    """
    Run all experiments from the DynaHMRC paper
    
    Args:
        llm_provider: LLM provider (mock, kimi, openai)
        num_trials: Number of trials per configuration (paper uses 8)
        task_types: List of task types to run
        variations: List of variations to run
        team_configs: List of team configurations to run
        output_dir: Directory to save results
    """
    print("="*80)
    print("DynaHMRC Paper Experiment Reproduction")
    print("="*80)
    
    # Default configurations from paper
    task_types = task_types or ['pack_objects', 'make_sandwich', 'sort_solids']
    variations = variations or ['static', 'cto', 'irz', 'anc', 'rec']
    team_configs = team_configs or [
        'Ma-MoMa-UAV', 'Ma-MoMa', 'Ma-UAV', 
        'Ma-MoMa-Mo', 'Ma-MoMa-Mo-UAV', 'Ma-Mo-UAV'
    ]
    
    # Create LLM config
    llm_config = create_llm_config(llm_provider)
    
    # Create experiment runner
    runner = ExperimentRunner(output_dir=output_dir)
    
    all_results = {}
    
    # Run all combinations
    for task_type in task_types:
        for variation_str in variations:
            for team_config in team_configs:
                variation = TaskVariation(variation_str)
                
                config = ExperimentConfig(
                    task_type=task_type,
                    variation=variation,
                    robot_team_config=team_config,
                    num_trials=num_trials,
                    enable_gui=False
                )
                
                # Create task executor with LLM config
                def task_executor(tt, rt, vi, ms, eg):
                    return execute_task_with_system(
                        tt, rt, vi, ms, eg, llm_config
                    )
                
                # Run experiment
                summary = runner.run_experiment(config, task_executor)
                all_results[config.get_name()] = summary
    
    # Export all results
    runner._export_results(all_results)
    
    # Print summary table
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(MetricsFormatter.format_table(all_results))
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run DynaHMRC paper experiments'
    )
    parser.add_argument(
        '--llm', 
        type=str, 
        default='mock',
        choices=['mock', 'kimi', 'openai'],
        help='LLM provider to use'
    )
    parser.add_argument(
        '--trials', 
        type=int, 
        default=8,
        help='Number of trials per configuration (default: 8)'
    )
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default=['pack_objects', 'make_sandwich', 'sort_solids'],
        help='Task types to run'
    )
    parser.add_argument(
        '--variations',
        type=str,
        nargs='+',
        default=['static', 'cto', 'irz', 'anc', 'rec'],
        help='Variations to run'
    )
    parser.add_argument(
        '--teams',
        type=str,
        nargs='+',
        default=['Ma-MoMa-UAV'],
        help='Team configurations to run'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='experiments/results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (1 trial, 1 task, static only)'
    )
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.trials = 1
        args.tasks = ['pack_objects']
        args.variations = ['static']
        args.teams = ['Ma-MoMa-UAV']
        print("[Quick Mode] Running minimal test...")
    
    # Run experiments
    results = run_paper_experiments(
        llm_provider=args.llm,
        num_trials=args.trials,
        task_types=args.tasks,
        variations=args.variations,
        team_configs=args.teams,
        output_dir=args.output
    )
    
    print("\n" + "="*80)
    print("Experiments completed!")
    print(f"Results saved to: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
