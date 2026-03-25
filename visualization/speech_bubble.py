"""
Speech Bubble Visualizer for DynaHMRC
Displays robot communication messages as speech bubbles above robots in PyBullet
"""

import pybullet as p
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from threading import Lock
import threading


@dataclass
class BubbleConfig:
    """Configuration for speech bubble appearance"""
    bubble_height: float = 0.6  # Height above robot base
    text_size: float = 1.0
    text_color: Tuple[float, float, float] = (1, 1, 1)  # White text
    leader_color: Tuple[float, float, float] = (1, 0.84, 0)  # Gold for leader
    normal_color: Tuple[float, float, float] = (0.3, 0.6, 1.0)  # Blue for normal
    max_width: int = 25  # Max characters per line
    display_duration: float = 5.0  # Seconds to display bubble


class SpeechBubble:
    """Represents a single speech bubble using PyBullet debug text"""
    
    def __init__(
        self,
        robot_name: str,
        message: str,
        position: Tuple[float, float, float],
        config: BubbleConfig,
        is_leader: bool = False
    ):
        self.robot_name = robot_name
        self.message = message
        self.position = position
        self.config = config
        self.is_leader = is_leader
        self.created_time = time.time()
        self.item_ids: List[int] = []  # PyBullet debug item IDs
        self._create_bubble()
    
    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        """Wrap text into multiple lines"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_len = len(word)
            if current_length + word_len + 1 <= max_width:
                current_line.append(word)
                current_length += word_len + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_len
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text]
    
    def _create_bubble(self):
        """Create the visual bubble in PyBullet"""
        # Remove emoji and special characters that PyBullet may not support
        clean_message = self._clean_text(self.message)
        lines = self._wrap_text(clean_message, self.config.max_width)
        
        x, y, z = self.position
        
        # Text color based on role
        text_color = self.config.leader_color if self.is_leader else self.config.normal_color
        
        # Build multi-line text
        display_text = f"[{self.robot_name}]:\n"
        display_text += "\n".join(lines)
        
        # Add leader indicator
        if self.is_leader:
            display_text = "[LEADER] " + display_text
        
        # Position above robot
        text_pos = [x, y, z + self.config.bubble_height]
        
        try:
            # Create debug text
            text_id = p.addUserDebugText(
                text=display_text,
                textPosition=text_pos,
                textColorRGB=text_color,
                textSize=self.config.text_size,
                physicsClientId=0
            )
            self.item_ids.append(text_id)
            
            # Draw a simple line from robot to text
            line_id = p.addUserDebugLine(
                lineFromXYZ=[x, y, z + 0.2],
                lineToXYZ=[x, y, z + self.config.bubble_height - 0.1],
                lineColorRGB=text_color,
                lineWidth=2.0,
                physicsClientId=0
            )
            self.item_ids.append(line_id)
            
        except Exception as e:
            print(f"[SpeechBubble] Error creating bubble: {e}")
    
    def _clean_text(self, text: str) -> str:
        """Remove emoji and special characters"""
        # Remove common emoji
        emoji_ranges = [
            (0x1F600, 0x1F64F),  # Emoticons
            (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
            (0x1F680, 0x1F6FF),  # Transport and Map
            (0x2600, 0x26FF),    # Misc symbols
            (0x2700, 0x27BF),    # Dingbats
        ]
        
        result = []
        for char in text:
            code = ord(char)
            is_emoji = any(start <= code <= end for start, end in emoji_ranges)
            if not is_emoji:
                result.append(char)
        
        return ''.join(result)
    
    def update(self, current_time: float) -> bool:
        """
        Update bubble state, return False if should be removed
        """
        elapsed = current_time - self.created_time
        return elapsed <= self.config.display_duration
    
    def remove(self):
        """Remove all visual elements"""
        for item_id in self.item_ids:
            try:
                p.removeUserDebugItem(item_id, physicsClientId=0)
            except Exception as e:
                pass
        self.item_ids.clear()


class SpeechBubbleVisualizer:
    """
    Manages speech bubbles for all robots in the simulation
    """
    
    def __init__(self, client_id: int = 0):
        self.client_id = client_id
        self.bubbles: Dict[str, SpeechBubble] = {}
        self.config = BubbleConfig()
        self.lock = Lock()
        self.running = False
        self.update_thread: Optional[threading.Thread] = None
        self.leader_name: Optional[str] = None
        
        # Robot positions (would be updated from simulation)
        self.robot_positions: Dict[str, Tuple[float, float, float]] = {}
        
        print("[SpeechBubbleVisualizer] Initialized")
    
    def start(self):
        """Start the visualizer update loop"""
        if self.running:
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        print("[SpeechBubbleVisualizer] Started update loop")
    
    def stop(self):
        """Stop the visualizer and clear all bubbles"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        
        with self.lock:
            for bubble in list(self.bubbles.values()):
                bubble.remove()
            self.bubbles.clear()
        
        print("[SpeechBubbleVisualizer] Stopped")
    
    def _update_loop(self):
        """Background thread to update bubbles"""
        while self.running:
            try:
                current_time = time.time()
                
                with self.lock:
                    # Remove expired bubbles
                    expired = []
                    for robot_name, bubble in list(self.bubbles.items()):
                        if not bubble.update(current_time):
                            bubble.remove()
                            expired.append(robot_name)
                    
                    for robot_name in expired:
                        if robot_name in self.bubbles:
                            del self.bubbles[robot_name]
                
                time.sleep(0.2)  # Update at 5Hz
            except Exception as e:
                print(f"[SpeechBubbleVisualizer] Update loop error: {e}")
                time.sleep(0.5)
    
    def set_leader(self, leader_name: str):
        """Set the current leader robot"""
        self.leader_name = leader_name
        print(f"[SpeechBubbleVisualizer] Leader set to: {leader_name}")
    
    def update_robot_position(
        self,
        robot_name: str,
        position: Tuple[float, float, float]
    ):
        """Update robot position for bubble placement"""
        self.robot_positions[robot_name] = position
    
    def show_speech(
        self,
        robot_name: str,
        message: str,
        duration: Optional[float] = None
    ):
        """
        Show a speech bubble for a robot
        
        Args:
            robot_name: Name of the robot speaking
            message: Message to display
            duration: Optional custom duration (seconds)
        """
        if robot_name not in self.robot_positions:
            print(f"[SpeechBubbleVisualizer] Warning: No position for robot {robot_name}")
            # Use default position
            self.robot_positions[robot_name] = (0, 0, 0.5)
        
        with self.lock:
            try:
                # Remove existing bubble for this robot
                if robot_name in self.bubbles:
                    self.bubbles[robot_name].remove()
                    del self.bubbles[robot_name]
                
                # Create config with custom duration if specified
                config = self.config
                if duration:
                    from dataclasses import replace
                    config = replace(self.config, display_duration=duration)
                
                # Create new bubble
                position = self.robot_positions[robot_name]
                is_leader = (robot_name == self.leader_name)
                
                bubble = SpeechBubble(
                    robot_name=robot_name,
                    message=message,
                    position=position,
                    config=config,
                    is_leader=is_leader
                )
                
                self.bubbles[robot_name] = bubble
                
                print(f"[SpeechBubbleVisualizer] {robot_name}: {message[:50]}...")
            except Exception as e:
                print(f"[SpeechBubbleVisualizer] Error showing speech: {e}")
    
    def show_self_description(self, robot_name: str, description: str):
        """Show self-description speech bubble"""
        prefix = "[Self-Desc] "
        self.show_speech(robot_name, prefix + description, duration=8.0)
    
    def show_campaign_speech(self, robot_name: str, speech: str):
        """Show leadership campaign speech bubble"""
        prefix = "[Campaign] "
        self.show_speech(robot_name, prefix + speech, duration=10.0)
    
    def show_vote(self, robot_name: str, voted_for: str):
        """Show voting action"""
        msg = f"[Vote] I vote for {voted_for} as leader!"
        self.show_speech(robot_name, msg, duration=5.0)
    
    def show_action(self, robot_name: str, action: str, reasoning: str = ""):
        """Show action execution"""
        msg = f"[Action] {action}"
        if reasoning:
            msg += f" | {reasoning}"
        self.show_speech(robot_name, msg, duration=4.0)
    
    def show_communication(self, from_robot: str, to_robot: str, content: str):
        """Show communication message"""
        msg = f"[To {to_robot}] {content}"
        self.show_speech(from_robot, msg, duration=6.0)
    
    def show_reflection(self, robot_name: str, summary: str):
        """Show reflection summary"""
        msg = f"[Reflection] {summary}"
        self.show_speech(robot_name, msg, duration=7.0)
    
    def show_leader_update(self, leader_name: str, update: str):
        """Show leader plan update"""
        msg = f"[Leader Update] {update}"
        self.show_speech(leader_name, msg, duration=8.0)
    
    def clear_all(self):
        """Clear all speech bubbles"""
        with self.lock:
            for bubble in list(self.bubbles.values()):
                bubble.remove()
            self.bubbles.clear()


# Convenience function for integration
def create_speech_bubble_visualizer(client_id: int = 0) -> SpeechBubbleVisualizer:
    """Factory function to create and start a speech bubble visualizer"""
    visualizer = SpeechBubbleVisualizer(client_id)
    visualizer.start()
    return visualizer
