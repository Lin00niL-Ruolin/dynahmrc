"""
Evaluation Metrics Module
Implements the metrics used in DynaHMRC paper:
- SUCC: Success Rate
- PS: Partial Success Rate
- TS: Task Steps
- AS: Action Steps (per robot)
- CC: Communication Count
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import time


class TaskVariation(Enum):
    """Task variation types from paper"""
    STATIC = "static"
    CTO = "cto"  # Change Task Objective
    IRZ = "irz"  # Inaccessible Region Zone
    ANC = "anc"  # Add New Collaborator
    REC = "rec"  # Remove Existing Collaborator


@dataclass
class RobotMetrics:
    """Metrics for a single robot"""
    robot_id: str
    robot_type: str
    action_count: int = 0
    navigation_count: int = 0
    manipulation_count: int = 0
    communication_count: int = 0
    idle_time: float = 0.0


@dataclass
class TaskMetrics:
    """Metrics for a single task execution"""
    # Task identification
    task_id: str
    task_type: str  # pack_objects, make_sandwich, sort_solids
    variation: TaskVariation = TaskVariation.STATIC
    
    # Team composition
    robot_team: List[str] = field(default_factory=list)
    robot_types: Dict[str, str] = field(default_factory=dict)
    
    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Steps
    total_steps: int = 0
    robot_steps: Dict[str, int] = field(default_factory=dict)
    
    # Success metrics
    success: bool = False
    partial_success_rate: float = 0.0
    
    # Communication
    total_communications: int = 0
    communications_per_robot: Dict[str, int] = field(default_factory=dict)
    
    # Replanning
    replan_count: int = 0
    
    # Detailed history
    action_history: List[Dict] = field(default_factory=list)
    communication_history: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'variation': self.variation.value,
            'robot_team': self.robot_team,
            'robot_types': self.robot_types,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.get_duration(),
            'total_steps': self.total_steps,
            'robot_steps': self.robot_steps,
            'success': self.success,
            'partial_success_rate': self.partial_success_rate,
            'total_communications': self.total_communications,
            'communications_per_robot': self.communications_per_robot,
            'replan_count': self.replan_count,
        }
    
    def get_duration(self) -> float:
        """Get task duration in seconds"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def finalize(self, success: bool, partial_success: float):
        """Finalize metrics at task completion"""
        self.end_time = time.time()
        self.success = success
        self.partial_success_rate = partial_success
    
    def record_action(self, robot_id: str, action_type: str, action_details: Dict):
        """Record a robot action"""
        self.total_steps += 1
        self.robot_steps[robot_id] = self.robot_steps.get(robot_id, 0) + 1
        
        self.action_history.append({
            'step': self.total_steps,
            'robot_id': robot_id,
            'action_type': action_type,
            'details': action_details,
            'timestamp': time.time()
        })
    
    def record_communication(self, from_robot: str, to_robot: Optional[str], 
                            message_type: str, content: str):
        """Record a communication event"""
        self.total_communications += 1
        self.communications_per_robot[from_robot] = \
            self.communications_per_robot.get(from_robot, 0) + 1
        
        self.communication_history.append({
            'from': from_robot,
            'to': to_robot,
            'type': message_type,
            'content': content[:100],  # Truncate for storage
            'timestamp': time.time()
        })


class MetricsCollector:
    """Collects and aggregates metrics across multiple task runs"""
    
    def __init__(self):
        self.task_metrics: List[TaskMetrics] = []
        self.current_task: Optional[TaskMetrics] = None
    
    def start_task(self, task_id: str, task_type: str, 
                   variation: TaskVariation,
                   robot_team: List[str],
                   robot_types: Dict[str, str]) -> TaskMetrics:
        """Start recording metrics for a new task"""
        self.current_task = TaskMetrics(
            task_id=task_id,
            task_type=task_type,
            variation=variation,
            robot_team=robot_team,
            robot_types=robot_types
        )
        return self.current_task
    
    def end_task(self, success: bool, partial_success: float):
        """End current task recording"""
        if self.current_task:
            self.current_task.finalize(success, partial_success)
            self.task_metrics.append(self.current_task)
            self.current_task = None
    
    def record_action(self, robot_id: str, action_type: str, 
                     action_details: Dict = None):
        """Record an action in current task"""
        if self.current_task:
            self.current_task.record_action(
                robot_id, action_type, action_details or {}
            )
    
    def record_communication(self, from_robot: str, to_robot: Optional[str],
                           message_type: str, content: str):
        """Record a communication in current task"""
        if self.current_task:
            self.current_task.record_communication(
                from_robot, to_robot, message_type, content
            )
    
    def record_replan(self):
        """Record a replanning event"""
        if self.current_task:
            self.current_task.replan_count += 1
    
    def get_summary(self, task_type: Optional[str] = None,
                   variation: Optional[TaskVariation] = None) -> Dict[str, Any]:
        """Get summary statistics"""
        # Filter metrics
        filtered = self.task_metrics
        if task_type:
            filtered = [m for m in filtered if m.task_type == task_type]
        if variation:
            filtered = [m for m in filtered if m.variation == variation]
        
        if not filtered:
            return {}
        
        # Calculate aggregates
        total_tasks = len(filtered)
        successful_tasks = sum(1 for m in filtered if m.success)
        
        return {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'SUCC': successful_tasks / total_tasks if total_tasks > 0 else 0,
            'PS': sum(m.partial_success_rate for m in filtered) / total_tasks,
            'TS': sum(m.total_steps for m in filtered) / total_tasks,
            'AS': self._calculate_avg_action_steps(filtered),
            'CC': sum(m.total_communications for m in filtered) / total_tasks,
            'avg_replan_count': sum(m.replan_count for m in filtered) / total_tasks,
            'avg_duration': sum(m.get_duration() for m in filtered) / total_tasks,
        }
    
    def _calculate_avg_action_steps(self, metrics: List[TaskMetrics]) -> Dict[str, float]:
        """Calculate average action steps per robot type"""
        robot_steps_by_type: Dict[str, List[int]] = {}
        
        for m in metrics:
            for robot_id, steps in m.robot_steps.items():
                robot_type = m.robot_types.get(robot_id, 'unknown')
                if robot_type not in robot_steps_by_type:
                    robot_steps_by_type[robot_type] = []
                robot_steps_by_type[robot_type].append(steps)
        
        return {
            robot_type: sum(steps) / len(steps)
            for robot_type, steps in robot_steps_by_type.items()
        }
    
    def export_to_json(self, filepath: str):
        """Export all metrics to JSON file"""
        data = {
            'tasks': [m.to_dict() for m in self.task_metrics],
            'summary': self.get_summary()
        }
        
        # Add per-task-type summaries
        task_types = set(m.task_type for m in self.task_metrics)
        data['by_task_type'] = {
            tt: self.get_summary(task_type=tt)
            for tt in task_types
        }
        
        # Add per-variation summaries
        variations = set(m.variation for m in self.task_metrics)
        data['by_variation'] = {
            v.value: self.get_summary(variation=v)
            for v in variations
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[MetricsCollector] Metrics exported to {filepath}")
    
    def export_to_csv(self, filepath: str):
        """Export metrics to CSV format (for easy analysis)"""
        import csv
        
        if not self.task_metrics:
            print("[MetricsCollector] No metrics to export")
            return
        
        fieldnames = [
            'task_id', 'task_type', 'variation', 'robot_team',
            'success', 'partial_success_rate', 'total_steps',
            'total_communications', 'replan_count', 'duration'
        ]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for m in self.task_metrics:
                row = m.to_dict()
                row['robot_team'] = ','.join(row['robot_team'])
                row['duration'] = m.get_duration()
                writer.writerow({k: row.get(k, '') for k in fieldnames})
        
        print(f"[MetricsCollector] Metrics exported to {filepath}")


class MetricsFormatter:
    """Format metrics for display and reporting"""
    
    @staticmethod
    def format_table(results: Dict[str, Dict[str, Any]]) -> str:
        """Format results as a table string"""
        lines = []
        lines.append("=" * 100)
        lines.append(f"{'Method':<25} {'SUCC':>8} {'PS':>8} {'TS':>8} {'AS':>8} {'CC':>8}")
        lines.append("-" * 100)
        
        for method, metrics in results.items():
            succ = metrics.get('SUCC', 0)
            ps = metrics.get('PS', 0)
            ts = metrics.get('TS', 0)
            
            # AS might be a dict for different robot types
            as_val = metrics.get('AS', 0)
            if isinstance(as_val, dict):
                as_str = f"{sum(as_val.values()) / len(as_val):.2f}" if as_val else "N/A"
            else:
                as_str = f"{as_val:.2f}"
            
            cc = metrics.get('CC', 0)
            
            lines.append(
                f"{method:<25} {succ:>8.4f} {ps:>8.4f} "
                f"{ts:>8.2f} {as_str:>8} {cc:>8.2f}"
            )
        
        lines.append("=" * 100)
        return '\n'.join(lines)
    
    @staticmethod
    def format_latex_table(results: Dict[str, Dict[str, Any]]) -> str:
        """Format results as LaTeX table"""
        lines = []
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lccccc}")
        lines.append("\\toprule")
        lines.append(r"Method & SUCC $\uparrow$ & PS $\uparrow$ & TS $\downarrow$ & AS $\downarrow$ & CC $\downarrow$ \\")
        lines.append("\\midrule")
        
        for method, metrics in results.items():
            succ = metrics.get('SUCC', 0)
            ps = metrics.get('PS', 0)
            ts = metrics.get('TS', 0)
            
            as_val = metrics.get('AS', 0)
            if isinstance(as_val, dict):
                as_str = f"{sum(as_val.values()) / len(as_val):.2f}" if as_val else "N/A"
            else:
                as_str = f"{as_val:.2f}"
            
            cc = metrics.get('CC', 0)
            
            lines.append(
                f"{method} & {succ:.4f} & {ps:.4f} & "
                f"{ts:.2f} & {as_str} & {cc:.2f} \\\\"
            )
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        return '\n'.join(lines)
