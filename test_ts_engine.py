"""
Quick test: Verify the TS engine handles make_sandwich on updated scene1.
"""
import json
import time
import sys
import urllib.request
import urllib.error
import threading

import websocket

TS_URL = "http://localhost:3001"
results = []
event_count = 0
max_events = 50

def create_run():
    data = json.dumps({
        "taskType": "make_sandwich",
        "layout": "scene1",
        "maxSteps": 15,
    }).encode()
    req = urllib.request.Request(
        f"{TS_URL}/api/run",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=10)
    return json.loads(resp.read())["runId"]

def on_msg(ws, message):
    global event_count, results
    msg = json.loads(message)
    event_count += 1
    msg_type = msg.get("type", "")
    data = msg.get("data", {})

    if msg_type == "dialogue":
        robot = data.get("robotName", "?")
        content = data.get("content", "")
        stage = data.get("stage", "")
        results.append(f"[{stage[:15]}] {robot}: {content[:80]}...")
    elif msg_type == "state":
        stage = data.get("stage", "")
        completed = data.get("taskCompleted", False)
        progress = data.get("taskProgress", "")
        step = data.get("step", 0)
        if step > 0:
            print(f"  Step {step}: {progress} | Stage: {stage} | Done: {completed}")
    elif msg_type == "control":
        print(f"  Control: {data}")

    if event_count >= max_events:
        ws.close()

def on_open(ws):
    print("WebSocket connected, starting run...")
    ws.send(json.dumps({"command": "start"}))

def on_error(ws, err):
    print(f"WS Error: {err}", file=sys.stderr)

def on_close(ws, code, msg):
    print(f"WS closed: {code} {msg}")

def main():
    run_id = create_run()
    print(f"Run created: {run_id}")

    ws_url = f"ws://localhost:3001/ws/{run_id}"
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_msg,
        on_error=on_error,
        on_close=on_close,
    )

    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()

    time.sleep(60)  # Let it run for 60s
    ws.close()
    t.join(timeout=5)

    print(f"\n=== Results ({event_count} events) ===")
    for r in results[:20]:
        print(r)
    if len(results) > 20:
        print(f"... and {len(results)-20} more")

    # Check if scene1 objects are correct
    state_resp = urllib.request.urlopen(f"{TS_URL}/api/run/{run_id}", timeout=5)
    state = json.loads(state_resp.read())
    print(f"\nFinal state: {json.dumps(state, indent=2)}")

if __name__ == "__main__":
    main()
