"""
Bridge: DynaHMRC TypeScript Engine → BestMan 3D Service

Connects to the DynaHMRC engine via WebSocket and translates actions
to BestMan /act API calls, plus syncs robot positions.
"""

import json
import time
import logging
import re
import threading
from typing import Optional, Dict, Any

import requests
import websocket

logging.basicConfig(
    level=logging.INFO,
    format="[Bridge] %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bridge")


class DynaHMRCBridge:
    """Bridges DynaHMRC actions to BestMan 3D simulation."""

    def __init__(
        self,
        run_id: str,
        ts_ws_url: str = "ws://localhost:3001",
        bestman_url: str = "http://localhost:5001",
    ):
        self.run_id = run_id
        self.ws_url = f"{ts_ws_url}/ws/{run_id}"
        self.bestman_url = bestman_url
        self.ws: Optional[websocket.WebSocketApp] = None
        self.running = False
        self.bridge_thread: Optional[threading.Thread] = None
        self.last_state: Optional[Dict] = None

        # Track robot positions in 3D space
        self.robot_positions: Dict[str, list] = {}
        self.robot_grippers: Dict[str, Optional[str]] = {}

    def _parse_action_from_dialogue(self, content: str) -> Optional[Dict]:
        """Parse an action from dialogue content like 'Action: navigate({"target":"fridge"})\n...'."""
        m = re.match(r"Action:\s*(\w+)\((.*)\)", content)
        if not m:
            return None

        action_type = m.group(1)
        params_str = m.group(2)

        # Try parsing params as JSON (the engine emits JSON params)
        try:
            params = json.loads(params_str)
        except (json.JSONDecodeError, ValueError):
            # Fall back to simple string parsing
            params = {}
            if action_type in ("navigate", "open", "pick"):
                params = {"target": params_str.strip()}
            elif action_type == "place":
                parts = [p.strip() for p in params_str.split(",")]
                if len(parts) >= 2:
                    params = {"object": parts[0], "target": parts[1]}
                else:
                    params = {"object": params_str.strip(), "target": ""}
            elif action_type == "move":
                parts = [p.strip() for p in params_str.split(",")]
                if len(parts) >= 2:
                    params = {"dx": float(parts[0]), "dy": float(parts[1])}
            elif action_type == "communicate":
                parts = [p.strip() for p in params_str.rsplit(",", 1)]
                if len(parts) >= 2:
                    params = {"content": parts[0].strip('"'), "recipient": parts[1]}
                else:
                    params = {"content": params_str.strip(), "recipient": "*"}

        return {"action_type": action_type, "params": params}

    def _translate_to_bestman(self, robot_name: str, action_info: Dict) -> Dict:
        """Translate a DynaHMRC action to a BestMan /act request."""
        action_type = action_info["action_type"]
        params = action_info["params"]

        # Map robot names
        robot_id = robot_name.lower()

        if action_type == "navigate":
            target = params.get("target", "")
            # Look up target position from state or use default
            pos = self._get_target_position(target)
            return {
                "robot_id": robot_id,
                "action": "navigate",
                "params": {"target": pos},
            }

        elif action_type == "pick":
            obj_name = params.get("target", "")
            self.robot_grippers[robot_name] = obj_name
            return {
                "robot_id": robot_id,
                "action": "pick",
                "params": {"object": obj_name},
            }

        elif action_type == "place":
            obj_name = params.get("object", "")
            target = params.get("target", "")
            self.robot_grippers[robot_name] = None
            return {
                "robot_id": robot_id,
                "action": "place",
                "params": {"object": obj_name, "target": target},
            }

        elif action_type == "open":
            container = params.get("target", "")
            return {
                "robot_id": robot_id,
                "action": "open",
                "params": {"container": container},
            }

        elif action_type == "wait":
            return {
                "robot_id": robot_id,
                "action": "wait",
                "params": {},
            }

        elif action_type == "communicate":
            log.info(f"[COMM] {robot_name}: {params.get('content', '')}")
            return {
                "robot_id": robot_id,
                "action": "wait",
                "params": {},
            }

        else:
            log.warning(f"Unknown action type: {action_type}")
            return {
                "robot_id": robot_id,
                "action": "wait",
                "params": {},
            }

    def _get_target_position(self, target: str) -> list:
        """Get 3D position for a target furniture/object."""
        # Known positions from scene1 (x, y) → 3D (x, y, z)
        known_positions = {
            "fridge": [9.7, 0.5, 0.5],
            "counter_elementa": [7.7, 0.5, 0.5],
            "counter_elementb": [6.2, 0.5, 0.5],
            "dishwasher": [8.9, 0.5, 0.5],
            "microwave": [8.4, 0.3, 0.8],
            "table_dining": [3, 2, 0.5],
            "chair_bottom": [3, 1, 0.3],
            "chair_top": [3, 3, 0.3],
            "bookshelf_1": [0.5, 0.5, 0.5],
            "bookshelf_2": [0.5, 1.5, 0.5],
            "bookshelf_3": [0.5, 2.5, 0.5],
            "table_bob": [8.5, 5.5, 0.5],
            "table_extra": [8.5, 4, 0.5],
            "chair_bob_1": [8.5, 3, 0.3],
            "chair_bob_2": [7.5, 5, 0.3],
            "cutting_board": [8.5, 5.8, 0.83],
            "toilet": [1.5, 7, 0.3],
            "bathtub": [1, 7, 0.3],
            "bread_bottom": [8.5, 5.8, 0.86],
            "lettuce": [9.7, 0.5, 0.5],
            "tomato": [7.7, 0.5, 0.5],
            "cheese": [3, 2, 0.5],
            "bread_top": [8.55, 5.82, 0.86],
            "tray": [8.5, 5.8, 0.83],
        }
        key = target.lower().strip()
        if key in known_positions:
            return known_positions[key]

        log.warning(f"Unknown target '{target}', using center")
        return [5, 4, 0.3]

    def _on_message(self, ws, message: str):
        """Handle WebSocket messages from DynaHMRC engine."""
        try:
            msg = json.loads(message)
            msg_type = msg.get("type", "")
            data = msg.get("data", {})

            if msg_type == "dialogue":
                self._handle_dialogue(data)
            elif msg_type == "state":
                self._handle_state(data)
            elif msg_type == "control":
                log.info(f"Control: {data}")
            elif msg_type == "error":
                log.error(f"Engine error: {data}")

        except json.JSONDecodeError:
            log.warning(f"Invalid JSON: {message[:100]}")
        except Exception as e:
            log.error(f"Handler error: {e}")

    def _handle_dialogue(self, data: Dict):
        """Handle a dialogue event - parse actions from execution content."""
        robot_name = data.get("robotName", "?")
        content = data.get("content", "")

        # Only parse execution-stage dialogues
        stage = data.get("stage", "")
        if stage != "execution_reflection":
            return

        action_info = self._parse_action_from_dialogue(content)
        if action_info:
            log.info(
                f"[{robot_name}] Action: {action_info['action_type']}"
                f"({json.dumps(action_info['params'])})"
            )
            self._send_to_bestman(robot_name, action_info)

    def _handle_state(self, data: Dict):
        """Handle state update - sync robot positions."""
        self.last_state = data
        robots = data.get("robots", {})

        for name, info in robots.items():
            x = info.get("posX", 0)
            y = info.get("posY", 0)
            # Map 2D sim position to 3D (z=0 for ground)
            self.robot_positions[name] = [x, y, 0]
            self.robot_grippers[name] = info.get("graspingObject", None)

    def _send_to_bestman(self, robot_name: str, action_info: Dict):
        """Send translated action to BestMan /act endpoint."""
        req = self._translate_to_bestman(robot_name, action_info)

        try:
            resp = requests.post(
                f"{self.bestman_url}/act",
                json=req,
                timeout=10,
            )
            result = resp.json()
            if result.get("success"):
                log.info(f"  ✓ BestMan: {robot_name} {action_info['action_type']}")
            else:
                log.warning(
                    f"  ✗ BestMan failed: {result.get('message', '?')}"
                )
        except requests.exceptions.RequestException as e:
            log.warning(f"  ⚠ BestMan unreachable: {e}")

    def _on_error(self, ws, error):
        log.error(f"WS Error: {error}")

    def _on_close(self, ws, close_status, close_msg):
        log.info("WS Connection closed")
        self.running = False

    def _on_open(self, ws):
        log.info(f"Connected to DynaHMRC engine at {self.ws_url}")
        # Send start command
        ws.send(json.dumps({"command": "start"}))

    def start(self):
        """Start the bridge in a background thread."""
        self.running = True
        self.bridge_thread = threading.Thread(target=self._run, daemon=True)
        self.bridge_thread.start()
        log.info(f"Bridge started for run {self.run_id}")

    def _run(self):
        """Run the WebSocket connection (blocking)."""
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        # Enable ping/pong for keepalive
        self.ws.run_forever(ping_interval=30, ping_timeout=10)

    def stop(self):
        """Stop the bridge."""
        self.running = False
        if self.ws:
            self.ws.close()
        log.info("Bridge stopped")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
        bridge = DynaHMRCBridge(run_id)
        bridge.start()
        try:
            while bridge.running:
                time.sleep(1)
        except KeyboardInterrupt:
            bridge.stop()
    else:
        print("Usage: python bridge.py <run_id>")
