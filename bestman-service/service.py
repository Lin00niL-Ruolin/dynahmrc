"""
BestMan 微服务
通过 HTTP/WebSocket 提供 BestMan 3D 仿真控制接口
被 Node.js 后端的 bestman-bridge.ts 调用
"""

import os
import sys
import json
import math
import yaml
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pybullet as p

# 设置 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(os.path.dirname(script_dir))  # workspace/
bestman_dir = os.path.join(workspace_dir, 'BestMan')
sys.path.insert(0, bestman_dir)

# 导入 BestMan
from Env.Client import Client

# 导入场景设置
sys.path.insert(0, script_dir)
from scenes.scene1_setup import setup_scene1
from scenes.scene2_setup import setup_scene2
from scenes.scene3_setup import setup_scene3


# ============ 数据模型 ============

class ActRequest(BaseModel):
    robot_id: str
    action: str          # navigate / pick / place / grasp / release / wait / stop
    params: Dict[str, Any] = {}


class InitRequest(BaseModel):
    scene: str = "scene1"
    gui: bool = True
    config_path: str = "Config/default.yaml"


class StateResponse(BaseModel):
    robots: Dict[str, Any] = {}
    objects: Dict[str, Any] = {}
    task_completed: bool = False
    step: int = 0


# ============ 全局状态 ============

app = FastAPI(title="BestMan DynaHMRC Service", version="1.0.0")

# CORS（允许 Node.js 前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 配置加载 ============

def load_yaml_config(config_path: str) -> SimpleNamespace:
    """加载 YAML 配置文件并转换为对象（支持属性访问）"""
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        return dict_to_namespace(cfg_dict)
    else:
        bm_path = os.path.join(bestman_dir, config_path)
        if os.path.exists(bm_path):
            with open(bm_path, 'r') as f:
                cfg_dict = yaml.safe_load(f)
            return dict_to_namespace(cfg_dict)
    raise FileNotFoundError(f"Config not found: {config_path}")


# ============ 服务状态 ============

class ServiceState:
    client: Optional[Client] = None
    robots: Dict[str, Any] = {}
    robot_positions: Dict[str, List[float]] = {}
    scene_objects: Dict[str, int] = {}
    step_count: int = 0
    is_initialized: bool = False


state = ServiceState()


# ============ API 端点 ============

@app.get("/")
def root():
    return {"status": "ok", "service": "BestMan DynaHMRC", "initialized": state.is_initialized}


@app.get("/status")
def get_status():
    """获取服务状态"""
    return {
        "initialized": state.is_initialized,
        "robots": len(state.robots),
        "step": state.step_count,
        "objects": len(state.scene_objects),
    }


@app.post("/init")
def initialize(req: InitRequest):
    """初始化 BestMan 并加载场景"""
    if state.is_initialized:
        return {"message": "Already initialized, call /reset to restart"}
    
    try:
        print(f"\n[BestMan Service] 初始化中... 场景={req.scene}")
        
        # 加载配置（开发模式直接用 Config/default.yaml）
        config_path = os.path.join(bestman_dir, req.config_path)
        cfg = load_yaml_config(config_path)
        
        # 覆盖 GUI 设置
        cfg.Client.enable_GUI = req.gui
        
        # 创建 Client（传入 cfg.Client 子对象）
        client = Client(cfg.Client)
        state.client = client
        state.is_initialized = True
        
        print("[BestMan Service] Client 创建成功")
        
        # 加载场景
        if req.scene == "scene1":
            scene_json = os.path.join(script_dir, "scenes", "scene1.json")
            setup_scene1(client, scene_json)
        elif req.scene == "scene2":
            scene_json = os.path.join(script_dir, "scenes", "scene2.json")
            setup_scene2(client, scene_json)
        elif req.scene == "scene3":
            setup_scene3(client)
        else:
            raise ValueError(f"Unknown scene: {req.scene}")
        
        # ✅ 机器人已由 scene_setup 创建
        # 此处跳过重复加载，只从场景中反向注册机器人 ID
        print("\n[BestMan Service] 扫描场景中的机器人...")
        try:
            state.robots['bob'] = client.bob_arm
            state.robots['alice'] = client.alice_base
            state.robots['david'] = client.david
            state.robots['lucy'] = client.drone_body
            print(f"[机器人] ✅ Bob / Alice / David / Lucy 已注册 (由场景脚本创建)")
        except Exception as e:
            print(f"[机器人] ⚠️ 注册机器人: {e}")
        
        client.run(30)
        
        return {
            "message": f"Scene '{req.scene}' initialized successfully",
            "gui": req.gui,
            "robots": list(state.robots.keys()),
        }
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[BestMan Service] 初始化失败: {e}\n{error_details}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/act")
def execute_action(req: ActRequest):
    """执行机器人动作"""
    if not state.is_initialized:
        raise HTTPException(status_code=400, detail="Not initialized. Call /init first.")
    
    action = req.action
    robot_id = req.robot_id
    params = req.params
    
    print(f"[动作] {robot_id} → {action} {params}")
    
    try:
        # 模拟执行，返回成功（后续替换为真实动作控制）
        state.step_count += 1
        
        result = {
            "success": True,
            "robot": robot_id,
            "action": action,
            "message": f"{robot_id} executed {action}",
            "step": state.step_count,
        }
        
        # 更新机器人位置（模拟）
        if robot_id not in state.robot_positions:
            state.robot_positions[robot_id] = [0.0, 0.0, 0.0]
        
        if action == "navigate" and "target" in params:
            state.robot_positions[robot_id] = params["target"]
        
        # 步进仿真
        if state.client:
            state.client.run(5)
        
        return result
    
    except Exception as e:
        import traceback
        print(f"[动作错误] {traceback.format_exc()}")
        return {
            "success": False,
            "robot": robot_id,
            "action": action,
            "message": str(e),
        }


@app.get("/state")
def get_full_state():
    """获取完整状态"""
    if not state.is_initialized:
        return StateResponse()
    
    return StateResponse(
        robots=state.robot_positions,
        objects={name: obj_id for name, obj_id in state.scene_objects.items()},
        step=state.step_count,
    )


@app.post("/step")
def step_simulation(n_steps: int = 10):
    """步进仿真"""
    if state.client:
        state.client.run(n_steps)
        state.step_count += n_steps
    return {"stepped": n_steps, "total": state.step_count}


@app.post("/reset")
def reset_scene():
    """重置场景"""
    if state.client:
        state.client.disconnect()
    
    state.client = None
    state.robots = {}
    state.robot_positions = {}
    state.scene_objects = {}
    state.step_count = 0
    state.is_initialized = False
    
    return {"message": "Reset complete"}


@app.on_event("shutdown")
def shutdown():
    """服务关闭时断开 BestMan 连接"""
    if state.client:
        state.client.disconnect()
        print("[BestMan Service] 已断开连接")


# ============ 启动入口 ============

if __name__ == "__main__":
    print("="*60)
    print("  BestMan DynaHMRC 微服务")
    print("  端口: 5001")
    print("="*60)
    
    uvicorn.run(
        "service:app",
        host="0.0.0.0",
        port=5001,
        reload=False,
        log_level="info",
    )
