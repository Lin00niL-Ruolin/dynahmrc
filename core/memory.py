"""
Memory Module for DynaHMRC
Manages historical context with FIFO queues
Based on paper Section IV-B: Memory Module
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from collections import deque


class MemoryModule:
    """
    Memory Module - Manages historical context
    Three independent FIFO queues (length limited to K)
    
    Attributes:
        feedback_history: Recent environment feedback
        received_message_history: Recent messages from teammates
        action_history: Recent actions taken
        long_feedback_history: Extended history for reflection
        long_action_history: Extended history for reflection
    """
    
    def __init__(self, max_history: int = 10, reflection_history: int = 50):
        """
        Initialize memory module
        
        Args:
            max_history: K - Length of short-term memory queues
            reflection_history: K̄ - Length of long-term memory queues (K̄ > K)
        """
        self.max_history = max_history
        self.reflection_history = reflection_history
        
        # Three independent queues for short-term memory
        self.feedback_history: deque = deque(maxlen=max_history)
        self.received_message_history: deque = deque(maxlen=max_history)
        self.action_history: deque = deque(maxlen=max_history)
        
        # Long history for Reflection
        self.long_feedback_history: deque = deque(maxlen=reflection_history)
        self.long_action_history: deque = deque(maxlen=reflection_history)
        
        # Persistent storage
        self.self_description = ""
        self.task_plan = {}
        
    def store_feedback(self, feedback: Dict):
        """Store environment feedback"""
        entry = {
            'step': len(self.action_history),
            'timestamp': time.time(),
            **feedback
        }
        self.feedback_history.append(entry)
        self.long_feedback_history.append(entry)
    
    def store_action(self, action: Dict, feedback: Dict):
        """Store executed action and its feedback"""
        entry = {
            'step': len(self.action_history),
            'timestamp': time.time(),
            'action': action,
            'feedback': feedback
        }
        self.action_history.append(entry)
        self.long_action_history.append(entry)
        self.store_feedback(feedback)
    
    def store_received_message(self, from_robot: str, content: str):
        """Store message received from another robot"""
        self.received_message_history.append({
            'from': from_robot,
            'content': content,
            'step': len(self.action_history),
            'timestamp': time.time()
        })
    
    def store_self_description(self, description: str):
        """Store self-introduction"""
        self.self_description = description
    
    def store_task_plan(self, plan: Dict):
        """Store task plan"""
        self.task_plan = plan
    
    def get_recent_history(self, k: int = None) -> List[Dict]:
        """Get recent k steps of history"""
        if k is None:
            k = self.max_history
        
        recent = []
        for i in range(min(k, len(self.action_history))):
            idx = len(self.action_history) - 1 - i
            recent.append({
                'action': list(self.action_history)[idx],
                'feedback': list(self.feedback_history)[idx] if idx < len(self.feedback_history) else None
            })
        return list(reversed(recent))
    
    def get_long_history(self) -> Tuple[List[Dict], List[Dict]]:
        """Get long history for Reflection"""
        return list(self.long_action_history), list(self.long_feedback_history)
    
    def get_received_messages(self) -> List[Dict]:
        """Get received message history"""
        return list(self.received_message_history)
    
    def format_history_for_prompt(self, k: int = 5) -> str:
        """Format history as LLM prompt text"""
        recent = self.get_recent_history(k)
        lines = ["Action and Feedback History:"]
        
        for item in recent:
            action = item['action']['action']
            feedback = item['feedback']
            step = item['action']['step']
            lines.append(f"  Step {step}: {action}")
            if feedback:
                status = "✓" if feedback.get('success') else "✗"
                msg = feedback.get('message', 'No feedback')
                lines.append(f"    -> {status} {msg}")
        
        return '\n'.join(lines)
    
    def format_messages_for_prompt(self) -> str:
        """Format received messages as prompt text"""
        messages = self.get_received_messages()
        if not messages:
            return "No messages received."
        
        lines = ["Received Messages:"]
        for msg in messages:
            lines.append(f"  From {msg['from']}: {msg['content']}")
        return '\n'.join(lines)
    
    def clear(self):
        """Clear all memory"""
        self.feedback_history.clear()
        self.received_message_history.clear()
        self.action_history.clear()
        self.long_feedback_history.clear()
        self.long_action_history.clear()
        self.self_description = ""
        self.task_plan = {}
