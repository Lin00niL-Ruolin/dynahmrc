"""
Run DynaHMRC Make Sandwich task → BestMan 3D via bridge.

1. Create a run on the TS engine
2. Start the bridge
3. Wait for completion
"""

import json
import time
import logging
import threading
import sys
from typing import Optional

import requests

from bridge import DynaHMRCBridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_sandwich")


TS_URL = "http://localhost:3001"
BM_URL = "http://localhost:5001"


def create_run():
    """Create a make_sandwich run on scene1."""
    resp = requests.post(
        f"{TS_URL}/api/run",
        json={
            "taskType": "make_sandwich",
            "layout": "scene1",
            "robots": [
                {"name": "Alice"},
                {"name": "Bob"},
                {"name": "David"},
                {"name": "Lucy"},
            ],
            "maxSteps": 50,
        },
        timeout=10,
    )
    resp.raise_for_status()
    result = resp.json()
    log.info(f"Run created: {result}")
    return result["runId"]


def init_bestman():
    """Initialize BestMan 3D scene."""
    resp = requests.post(
        f"{BM_URL}/init",
        json={
            "scene": "scene1",
            "gui": True,
            "config_path": "Config/default.yaml",
        },
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()
    log.info(f"BestMan initialized: {result.get('message', '')}")
    return result


def wait_for_completion(run_id: str, timeout: int = 300):
    """Wait for the run to complete by polling the API."""
    start = time.time()
    last_stage = ""
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{TS_URL}/api/run/{run_id}", timeout=5)
            status = resp.json()
            stage = status.get("stage", "")
            step = status.get("step", 0)
            running = status.get("running", True)
            completed = status.get("completed", False)

            if stage != last_stage:
                log.info(
                    f"Stage: {stage} | Step: {step} | "
                    f"Running: {running} | Complete: {completed}"
                )
                last_stage = stage

            if completed or (not running and stage == "completed"):
                log.info(f"Task completed in {step} steps!")
                return True

            if not running and stage != "completed":
                log.warning(f"Task stopped unexpectedly at step {step}")
                return False

        except Exception as e:
            log.warning(f"Poll error: {e}")

        time.sleep(2)

    log.warning(f"Timeout after {timeout}s")
    return False


def main():
    log.info("=" * 60)
    log.info("   DynaHMRC → BestMan 3D: Make Sandwich")
    log.info("=" * 60)

    # 1. Initialize BestMan
    log.info("\n[1/4] Initializing BestMan 3D scene...")
    try:
        init_bestman()
    except Exception as e:
        log.warning(f"BestMan init failed (may already be initialized): {e}")

    # 2. Create DynaHMRC run
    log.info("\n[2/4] Creating DynaHMRC run...")
    run_id = create_run()
    log.info(f"Run ID: {run_id}")

    # 3. Start bridge
    log.info("\n[3/4] Starting bridge...")
    bridge = DynaHMRCBridge(run_id)
    bridge.start()
    time.sleep(1)  # Let the bridge connect and start the run

    # 4. Wait for completion
    log.info("\n[4/4] Waiting for task completion...")
    success = wait_for_completion(run_id)

    # Stop bridge
    log.info("Stopping bridge...")
    bridge.stop()

    if success:
        log.info("\n✅ Make Sandwich task completed successfully!")
    else:
        log.warning("\n⚠️ Task did not complete as expected")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
