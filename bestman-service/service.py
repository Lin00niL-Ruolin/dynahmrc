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
    robots: Dict[str, Any] = {}                 # robot_name → body_id
    robot_positions: Dict[str, List[float]] = {} # robot_name → [x, y, z]
    scene_objects: Dict[str, int] = {}           # object_name → body_id
    gripper_constraints: Dict[str, int] = {}     # robot_name → constraint_id
    step_count: int = 0
    is_initialized: bool = False

    # 已知可抓取的物品名称后缀（用于过滤场景物体）
    PICKABLE_NAMES = {
        'apple', 'blue_bowl', 'bowl', 'fork_0', 'fork', 'book_0', 'book',
        'soap', 'cup', 'lemon', 'bread_0', 'bacon_0', 'bread_bottom',
        'bread_top', 'ham', 'lettuce', 'tomato', 'cheese',
        'red_cube', 'blue_sphere', 'green_cylinder', 'tray',
        'phone', 'toy_duck', 'egg',
    }


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
        
        # 扫描场景中的机器人
        print("\n[BestMan Service] 扫描场景中的机器人...")
        robot_attrs = ['bob_arm', 'alice_base', 'alice_arm', 'david', 'drone_body', 'bob']
        for attr in robot_attrs:
            val = getattr(client, attr, None)
            if val is not None:
                state.robots[attr] = val
                print(f"  ✓ {attr} = {val}")
        if state.robots:
            print(f"[机器人] ✅ {', '.join(state.robots.keys())} 已注册")
        else:
            print("[机器人] ⚠️ 未找到任何机器人")
        
        # 扫描场景中的所有可抓取物体
        print("\n[BestMan Service] 扫描场景中的可抓取物体...")
        loaded = 0
        for attr_name in dir(client):
            if attr_name in ServiceState.PICKABLE_NAMES or attr_name.startswith('_') or attr_name == 'client':
                continue
            val = getattr(client, attr_name, None)
            if isinstance(val, int) and val > 0:
                if val not in state.robots.values():
                    state.scene_objects[attr_name] = val
                    print(f"  ✓ {attr_name} = {val}")
                    loaded += 1
        # 额外检查已知可抓取名称
        for name in ServiceState.PICKABLE_NAMES:
            val = getattr(client, name, None)
            if val is not None and isinstance(val, int) and val not in state.robots.values():
                state.scene_objects[name] = val
                loaded += 1
        print(f"[场景] ✅ 已注册 {loaded} 个可抓取物体")
        
        # 步进仿真让物体稳定
        print("\n[BestMan Service] 步进仿真...")
        for _ in range(10):
            client.run(10)
        
        return {
            "message": f"Scene '{req.scene}' initialized successfully",
            "gui": req.gui,
            "robots": list(state.robots.keys()),
            "objects": list(state.scene_objects.keys()),
        }
        
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
    """执行机器人动作 — 真实 PyBullet 控制"""
    if not state.is_initialized:
        raise HTTPException(status_code=400, detail="Not initialized. Call /init first.")
    
    action = req.action
    robot_id = req.robot_id
    params = req.params
    
    print(f"[动作] {robot_id} → {action} {params}")
    
    try:
        state.step_count += 1
        result = {"success": True, "robot": robot_id, "action": action, "step": state.step_count}
        
        if action == "pick":
            target = params.get("target", params.get("object", ""))
            # 查找物体
            body_id = state.scene_objects.get(target)
            if body_id is None:
                # 模糊匹配
                for name, bid in state.scene_objects.items():
                    if target.lower() in name.lower():
                        body_id = bid
                        break
            if body_id is None:
                return {"success": False, "message": f"未知物体: {target}", "action": action}
            
            # 查找机器人
            robot_body = state.robots.get(robot_id)
            if robot_body is None:
                # 尝试 robot_id + '_arm' 或 robot_id + '_base'
                for suffix in ['_arm', '_base', '']:
                    robot_body = state.robots.get(robot_id + suffix)
                    if robot_body:
                        break
            if robot_body is None:
                return {"success": False, "message": f"未知机器人: {robot_id}", "action": action}
            
            # 已抓起对象先释放
            if robot_id in state.gripper_constraints:
                p.removeConstraint(state.gripper_constraints[robot_id])
            
            # 获取物体当前位置，计算相对于机器人的偏移
            obj_pos, obj_orn = p.getBasePositionAndOrientation(body_id)
            rob_pos, rob_orn = p.getBasePositionAndOrientation(robot_body)
            offset = [obj_pos[i] - rob_pos[i] for i in range(3)]
            
            # 创建固定约束：物体→机器人
            constraint_id = p.createConstraint(
                parentBodyUniqueId=robot_body, parentLinkIndex=-1,
                childBodyUniqueId=body_id, childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=offset,
                childFramePosition=[0, 0, 0],
            )
            state.gripper_constraints[robot_id] = constraint_id
            result["message"] = f"{robot_id} picked {target}"
            print(f"  ✓ 抓起 {target} (body={body_id}) → 约束#{constraint_id}")
        
        elif action == "place":
            target = params.get("target", "")
            # 释放约束
            if robot_id in state.gripper_constraints:
                p.removeConstraint(state.gripper_constraints[robot_id])
                del state.gripper_constraints[robot_id]
            # 获取要放置的物体
            obj_name = params.get("object", "")
            body_id = state.scene_objects.get(obj_name)
            if body_id:
                # 在目标位置创建临时支撑让物体不掉下去
                target_pos = state.scene_objects.get(target)
                if target_pos is None and target:
                    # 已知位置映射
                    known_pos = {
                        'tray': [5, 5, 0.83],
                        'cutting_board': [8.5, 5.5, 0.86],
                        'table_bob': [8.5, 5.5, 0.86],
                    }
                    if target in known_pos:
                        pos = known_pos[target]
                        p.resetBasePositionAndOrientation(body_id, pos, [0, 0, 0, 1])
            
            result["message"] = f"{robot_id} placed object"
            print(f"  ✓ 放置物体")
        
        elif action == "navigate":
            target = params.get("target", "")
            robot_body = state.robots.get(robot_id)
            if robot_body is None:
                for suffix in ['_arm', '_base', '']:
                    robot_body = state.robots.get(robot_id + suffix)
                    if robot_body:
                        break
            if target and robot_body:
                # 已知家具位置
                known_pos = {
                    'fridge': [9.4, 0.5, 0], 'table_dining': [3, 2, 0],
                    'table_bob': [8.5, 5.5, 0], 'cutting_board': [8.5, 5.5, 0.86],
                    'counter_elementa': [7.4, 0.5, 0], 'counter_elementb': [5.9, 0.5, 0],
                }
                tgt = target.lower()
                if tgt in known_pos:
                    pos = known_pos[tgt]
                    p.resetBasePositionAndOrientation(robot_body, pos, p.getQuaternionFromEuler([0, 0, 0]))
                    print(f"  ✓ 导航到 {target} @ {pos}")
            
            if robot_id not in state.robot_positions:
                state.robot_positions[robot_id] = [0.0, 0.0, 0.0]
            result["message"] = f"{robot_id} navigated to {target}"
        
        elif action == "wait":
            result["message"] = f"{robot_id} is waiting"
        
        elif action == "communicate":
            content = params.get("content", "")
            result["message"] = f"{robot_id} said: {content}"
        
        else:
            result["message"] = f"{robot_id} executed {action} (simulated)"
        
        # 步进仿真
        if state.client:
            state.client.run(5)
        
        return result
    
    except Exception as e:
        import traceback
        print(f"[动作错误] {traceback.format_exc()}")
        return {"success": False, "message": str(e)}


@app.get("/state")
def get_full_state():
    """获取完整状态 (已存在的实现，维持不变)"""
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
