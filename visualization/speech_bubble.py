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
    bubble_height: float = 0.8  # Height above robot base
    text_size: float = 1.2
    text_color: Tuple[float, float, float] = (0, 0, 0)  # Black text
    bg_color: Tuple[float, float, float] = (1, 1, 1)  # White background
    border_color: Tuple[float, float, float] = (0.3, 0.3, 0.3)  # Gray border
    max_width: int = 30  # Max characters per line
    display_duration: float = 5.0  # Seconds to display bubble
    fade_duration: float = 1.0  # Seconds to fade out


class SpeechBubble:
    """Represents a single speech bubble"""
    
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
            if current_length + len(word) + 1 <= max_width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text]
    
    def _create_bubble(self):
        """Create the visual bubble in PyBullet"""
        lines = self._wrap_text(self.message, self.config.max_width)
        
        # Calculate bubble dimensions
        line_height = 0.15
        bubble_width = max(len(line) * 0.03, 0.5)
        bubble_height = len(lines) * line_height + 0.2
        
        x, y, z = self.position
        
        # Leader crown icon
        if self.is_leader:
            crown_text = "👑"
            crown_id = p.addUserDebugText(
                text=crown_text,
                textPosition=[x, y, z + self.config.bubble_height + bubble_height + 0.1],
                textColorRGB=[1, 0.84, 0],  # Gold color
                textSize=self.config.text_size * 1.5,
                physicsClientId=0
            )
            self.item_ids.append(crown_id)
        
        # Draw bubble background (rectangle using lines)
        bg_z = z + self.config.bubble_height
        half_width = bubble_width / 2
        half_height = bubble_height / 2
        
        corners = [
            [x - half_width, y, bg_z - half_height],  # Bottom left
            [x + half_width, y, bg_z - half_height],  # Bottom right
            [x + half_width, y, bg_z + half_height],  # Top right
            [x - half_width, y, bg_z + half_height],  # Top left
        ]
        
        # Draw border lines
        border_lines = [
            (0, 1), (1, 2), (2, 3), (3, 0)  # Rectangle
        ]
        
        for start_idx, end_idx in border_lines:
            line_id = p.addUserDebugLine(
                lineFromXYZ=corners[start_idx],
                lineToXYZ=corners[end_idx],
                lineColorRGB=self.config.border_color,
                lineWidth=2.0,
                physicsClientId=0
            )
            self.item_ids.append(line_id)
        
        # Draw pointer line from robot to bubble
        pointer_id = p.addUserDebugLine(
            lineFromXYZ=[x, y, z + 0.3],
            lineToXYZ=[x, y, bg_z - half_height],
            lineColorRGB=self.config.border_color,
            lineWidth=2.0,
            physicsClientId=0
        )
        self.item_ids.append(pointer_id)
        
        # Draw text lines
        text_z = bg_z + half_height - 0.1
        for i, line in enumerate(lines):
            line_pos = [x, y, text_z - i * line_height]
            text_id = p.addUserDebugText(
                text=line,
                textPosition=line_pos,
                textColorRGB=self.config.text_color,
                textSize=self.config.text_size,
                physicsClientId=0
            )
            self.item_ids.append(text_id)
        
        # Draw robot name label
        name_pos = [x, y, bg_z - half_height - 0.1]
        name_color = (1, 0.84, 0) if self.is_leader else (0.2, 0.4, 0.8)
        name_id = p.addUserDebugText(
            text=f"[{self.robot_name}]",
            textPosition=name_pos,
            textColorRGB=name_color,
            textSize=self.config.text_size * 0.8,
            physicsClientId=0
        )
        self.item_ids.append(name_id)
    
    def update(self, current_time: float) -> bool:
        """
        Update bubble state, return False if should be removed
        """
        elapsed = current_time - self.created_time
        
        if elapsed > self.config.display_duration + self.config.fade_duration:
            return False
        
        # Could add fade effect here by changing alpha (not directly supported in PyBullet)
        
        return True
    
    def remove(self):
        """Remove all visual elements"""
        for item_id in self.item_ids:
            try:
                p.removeUserDebugItem(item_id, physicsClientId=0)
            except:
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
    
    def start(self):
        """Start the visualizer update loop"""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        print("[SpeechBubbleVisualizer] Started")
    
    def stop(self):
        """Stop the visualizer and clear all bubbles"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        
        with self.lock:
            for bubble in self.bubbles.values():
                bubble.remove()
            self.bubbles.clear()
        
        print("[SpeechBubbleVisualizer] Stopped")
    
    def _update_loop(self):
        """Background thread to update bubbles"""
        while self.running:
            current_time = time.time()
            
            with self.lock:
                # Remove expired bubbles
                expired = []
                for robot_name, bubble in self.bubbles.items():
                    if not bubble.update(current_time):
                        bubble.remove()
                        expired.append(robot_name)
                
                for robot_name in expired:
                    del self.bubbles[robot_name]
            
            time.sleep(0.1)  # Update at 10Hz
    
    def set_leader(self, leader_name: str):
        """Set the current leader robot"""
        self.leader_name = leader_name
    
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
            return
        
        with self.lock:
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
    
    def show_self_description(self, robot_name: str, description: str):
        """Show self-description speech bubble"""
        self.show_speech(
            robot_name,
            f"📢 {description}",
            duration=8.0
        )
    
    def show_campaign_speech(self, robot_name: str, speech: str):
        """Show leadership campaign speech bubble"""
        self.show_speech(
            robot_name,
            f"🎯 {speech}",
            duration=10.0
        )
    
    def show_vote(self, robot_name: str, voted_for: str):
        """Show voting action"""
        self.show_speech(
            robot_name,
            f"🗳️ I vote for {voted_for} as leader!",
            duration=5.0
        )
    
    def show_action(self, robot_name: str, action: str, reasoning: str = ""):
        """Show action execution"""
        msg = f"⚡ {action}"
        if reasoning:
            msg += f"\n💭 {reasoning}"
        self.show_speech(robot_name, msg, duration=4.0)
    
    def show_communication(self, from_robot: str, to_robot: str, content: str):
        """Show communication message"""
        self.show_speech(
            from_robot,
            f"📨 To {to_robot}: {content}",
            duration=6.0
        )
    
    def show_reflection(self, robot_name: str, summary: str):
        """Show reflection summary"""
        self.show_speech(
            robot_name,
            f"🤔 Reflection: {summary}",
            duration=7.0
        )
    
    def show_leader_update(self, leader_name: str, update: str):
        """Show leader plan update"""
        self.show_speech(
            leader_name,
            f"👑 Leader Update: {update}",
            duration=8.0
        )
    
    def clear_all(self):
        """Clear all speech bubbles"""
        with self.lock:
            for bubble in self.bubbles.values():
                bubble.remove()
            self.bubbles.clear()


# Convenience function for integration
def create_speech_bubble_visualizer(client_id: int = 0) -> SpeechBubbleVisualizer:
    """Factory function to create and start a speech bubble visualizer"""
    visualizer = SpeechBubbleVisualizer(client_id)
    visualizer.start()
    return visualizer
