"""
BestMan 微服务
通过 HTTP/WebSocket 提供 BestMan 3D 仿真控制接口
被 Node.js 后端的 bestman-bridge.ts 调用
"""

import os
import sys
import json
import math
import struct
import zlib
import base64
import io
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

# CORS(允许 Node.js 前端跨域访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 配置加载 ============

def load_yaml_config(config_path: str) -> SimpleNamespace:
    """加载 YAML 配置文件并转换为对象(支持属性访问)"""
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
    robot_aliases: Dict[str, str] = {}           # WebUI name → scene body name

    # 已知可抓取的物品名称后缀(用于过滤场景物体)
    PICKABLE_NAMES = {
        'apple', 'blue_bowl', 'bowl', 'fork_0', 'fork', 'book_0', 'book',
        'soap', 'cup', 'lemon', 'bacon', 'bread_0', 'bread_1', 'bacon_0',
        'lettuce', 'tomato', 'cheese',
        'small_cube_red', 'cube_red', 'cube_green', 'cube_blue', 'cube_yellow', 'cube_purple', 'cube_orange', 'tray',
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

        # 加载配置(开发模式直接用 Config/default.yaml)
        config_path = os.path.join(bestman_dir, req.config_path)
        cfg = load_yaml_config(config_path)

        # 检测 DISPLAY 是否可用（避免 GUI 连接卡死）
        has_display = os.environ.get('DISPLAY') and os.environ.get('XAUTHORITY')
        if req.gui and has_display:
            cfg.Client.enable_GUI = True
            try:
                client = Client(cfg.Client)
                print(f"[BestMan] GUI mode started (DISPLAY={os.environ.get('DISPLAY')})")
            except Exception as gui_err:
                print(f"[BestMan] GUI failed ({gui_err}), falling back to DIRECT...")
                cfg.Client.enable_GUI = False
                client = Client(cfg.Client)
        else:
            print(f"[BestMan] DIRECT mode (DISPLAY={os.environ.get('DISPLAY')}, req.gui={req.gui})")
            cfg.Client.enable_GUI = False
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

        # 扫描场景中的机器人(支持多个场景的不同命名)
        print("\n[BestMan Service] 扫描场景中的机器人...")
        robot_attrs = [
            'bob_arm', 'bob',           # Bob (场景1 / 其他)
            'new_robot_base',             # Alice 移动底座 (场景1)
            'new_robot_arm',              # Alice 手臂 (场景1)
            'alice_base', 'alice_arm',    # Alice (场景2/3)
            'david',                      # David (场景1)
            'david_base',                 # David (其他场景)
            'drone_body', 'drone',        # Lucy (场景1 / 其他)
            'mobile_manipulator_1',       # 通用命名
            'arm_1', 'mobile_base_1',
        ]
        for attr in robot_attrs:
            val = getattr(client, attr, None)
            if val is not None and attr not in state.robots:
                state.robots[attr] = val
                print(f"  ✓ {attr} = {val}")
        if state.robots:
            print(f"[机器人] ✅ {', '.join(state.robots.keys())} 已注册")
            # 名称别名映射
            state.robot_aliases = {
                'Alice': 'alice_base',
                'Bob': 'bob_arm',
                'David': 'david',
                'Lucy': 'drone_body',
            }
        else:
            print("[机器人] ⚠️ 未找到任何机器人")

        # 扫描场景中的所有可抓取物体
        print("\n[BestMan Service] 扫描场景中的可抓取物体...")
        loaded = 0
        # 从场景 JSON 和 setup 脚本中注册所有物体
        for attr_name in dir(client):
            if attr_name.startswith('_') or attr_name == 'client':
                continue
            val = getattr(client, attr_name, None)
            if isinstance(val, int) and val > 0:
                if val not in state.robots.values():
                    if attr_name not in state.scene_objects:
                        state.scene_objects[attr_name] = val
                        print(f"  ✓ {attr_name} = {val}")
                        loaded += 1
        # 直接通过 PyBullet 枚举所有物体（比扫描 client 属性更可靠）
        try:
            for body_id in range(p.getNumBodies()):
                if body_id in state.robots.values() or body_id in state.scene_objects.values():
                    continue
                try:
                    info = p.getBodyInfo(body_id)
                    name = info[1].decode('utf-8') if isinstance(info[1], bytes) else str(info[1])
                    # 清理名字：去掉路径、特殊字符
                    name = name.split('/')[-1].split('.')[0].split('_gen')[0].split('_convex')[0]
                    if name not in state.scene_objects and body_id > 0:
                        state.scene_objects[name] = body_id
                        print(f"  ✓ {name} = {body_id}")
                        loaded += 1
                except:
                    pass
        except Exception as e:
            print(f"  [PyBullet scan error] {e}")
        # 检查已知可抓取物体
        for name in ServiceState.PICKABLE_NAMES:
            val = getattr(client, name, None)
            if val is not None and isinstance(val, int) and val not in state.robots.values():
                if name not in state.scene_objects:
                    state.scene_objects[name] = val
                    loaded += 1
        print(f"[场景] ✅ 已注册 {loaded} 个可抓取物体")

        # 步进仿真让物体稳定
        print("\n[BestMan Service] 步进仿真...")
        for _ in range(10):
            client.run(10)

        # 设置 PyBullet GUI 相机视角（对准工作区域）
        if cfg.Client.enable_GUI:
            try:
                p.resetDebugVisualizerCamera(
                    cameraDistance=10,
                    cameraYaw=45,
                    cameraPitch=-30,
                    cameraTargetPosition=[4, 4, 0]
                )
                print("[BestMan] Camera position set")
            except Exception as cam_err:
                print(f"[BestMan] Camera error: {cam_err}")

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
    """执行机器人动作 - 真实 PyBullet 控制"""
    if not state.is_initialized:
        raise HTTPException(status_code=400, detail="Not initialized. Call /init first.")

    action = req.action
    robot_id = req.robot_id
    params = req.params

    print(f"[动作] {robot_id} → {action} {params}")

    try:
        state.step_count += 1
        result = {"success": True, "robot": robot_id, "action": action, "step": state.step_count}

        # 机器人解析函数（支持别名和场景1的命名）
        def _resolve_robot(rid: str):
            # 1. 直接匹配
            body = state.robots.get(rid)
            if body is not None:
                return rid, body
            # 2. 通过别名映射
            alias = state.robot_aliases.get(rid)
            if alias:
                body = state.robots.get(alias)
                if body is not None:
                    return alias, body
            # 3. 场景1的 new_robot 系列
            if rid == 'Alice':
                for k in ['new_robot_base', 'new_robot']:
                    body = state.robots.get(k)
                    if body: return k, body
            # 4. 后缀试探
            for suffix in ['_arm', '_base', '']:
                body = state.robots.get(rid + suffix)
                if body: return rid + suffix, body
            return None, None

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

            resolved_name, robot_body = _resolve_robot(robot_id)
            if robot_body is None:
                return {"success": False, "message": f"未知机器人: {robot_id}，可用: {list(state.robots.keys())}", "action": action}
            
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
            state.gripper_constraints[resolved_name] = constraint_id
            result["message"] = f"{robot_id} picked {target}"
            print(f"  ✓ 抓起 {target} (body={body_id}) → 约束#{constraint_id}")

        elif action == "place":
            target = params.get("target", "")
            # 释放约束（尝试所有可能的 key）
            for rk in [robot_id] + list(state.robot_aliases.values()):
                if rk in state.gripper_constraints:
                    p.removeConstraint(state.gripper_constraints[rk])
                    del state.gripper_constraints[rk]
                    print(f"  ✓ 释放约束: {rk}")
                    break
            # 获取要放置的物体
            obj_name = params.get("object", "")
            body_id = state.scene_objects.get(obj_name)
            if body_id:
                # 在目标位置创建临时支撑让物体不掉下去
                target_pos = state.scene_objects.get(target)
                if target_pos is None and target:
                    # 完整的位置映射（与导航共用）
                    known_pos = {
                        'tray': [2, 4, 0.87],
                        'cutting_board': [8.5, 5.5, 0.86],
                        'table_new_2': [8.5, 5.5, 0.86],
                        'table_new_1': [8.5, 4, 0.86],
                        'table2': [3, 5, 0.86],
                        'table1': [1, 4, 0.86],
                        'source_table_1': [2, 4, 0.86],
                        'bobs_table': [8.5, 5.5, 0.86],
                        'bob_table': [8.5, 5.5, 0.86],
                        'packing_table': [8, 3, 0.86],
                    }
                    tgt = target.lower()
                    pos = known_pos.get(tgt)
                    if pos is None:
                        # 模糊匹配
                        for key, val in known_pos.items():
                            if tgt in key or key in tgt:
                                pos = val
                                break
                    if pos is None and ('bob' in tgt or 'source_table' in tgt):
                        pos = [2, 4, 0.86]  # Bob's table = source_table_1
                    if pos:
                        p.resetBasePositionAndOrientation(body_id, pos, [0, 0, 0, 1])
            
            result["message"] = f"{robot_id} placed object"
            print(f"  ✓ 放置物体")
        
        elif action == "navigate":
            target = params.get("target", "")
            resolved_name, robot_body = _resolve_robot(robot_id)
            if target and robot_body:
                # 完整家具位置字典（覆盖全部三个场景，与2D仿真一致）
                known_pos = {
                    # === SCENE1 (make_sandwich) ===
                    'fridge': [9.4, 0.5, 0],
                    'elementa': [7.4, 0.5, 0],
                    'elementb1': [5.9, 0.5, 0],
                    'elementc': [8.6, 0.5, 0],
                    'microwave': [8.1, 0.3, 0],
                    'table': [3, 2, 0],
                    'chair_bottom': [3, 1, 0],
                    'chair_top': [3, 3, 0],
                    'table_new_1': [8.5, 4, 0.86],
                    'table_new_2': [8.5, 5.5, 0.86],
                    'chair_3': [8.5, 3, 0],
                    'chair_4': [7.5, 5, 0],
                    'bookshelf_1': [0.5, 0.5, 0.96],
                    'bookshelf_2': [0.5, 1.5, 0.96],
                    'bookshelf_3': [0.5, 2.5, 0.96],
                    'cutting_board': [8.5, 5.5, 0.86],
                    'toilet': [7.0, 8, 0],
                    'bathtub': [1.0, 7, 0],
                    # === SCENE2 (kitchen / sort_solids) ===
                    'elementb1_kitchen': [1, 0.5, 0],
                    'elementa_kitchen': [2.5, 0.5, 0],
                    'microwave_kitchen': [3.2, 0.3, 0],
                    'elementc_kitchen': [3.7, 0.5, 0],
                    'fridge_kitchen': [4.5, 0.5, 0],
                    'table1': [1, 4, 0],
                    'table2': [3, 5, 0],
                    'bookcase': [1, 6.5, 0],
                    'sofa': [7.5, 9, 0],
                    'rug': [6, 2, 0],
                    'shelf_table': [9.5, 7.5, 0],
                    # === SCENE3 (living_room / pack_objects) ===
                    'kitchen_cabinet': [1.2, 0.5, 0],
                    'kitchen_counter': [3.1, 0.5, 0],
                    'microwave_lr': [3.8, 0.3, 0],
                    'dishwasher': [4.6, 0.7, 0],
                    'fridge_lr': [5.5, 0.5, 0],
                    'cabinet_2': [7.3, 0.6, 0],
                    'sofa_lr': [8.6, 0.8, 0],
                    'packing_table': [8, 3, 0],
                    'source_table_1': [2, 4, 0],
                    'source_table_2': [4, 4, 0],
                    'chair': [2, 3, 0],
                    'bookcase_lr': [7.5, 5.5, 0],
                    'bathtub_lr': [1.5, 9.4, 0],
                    'sink_base': [4.7, 9.5, 0],
                    'sink': [4.7, 9.6, 0.8],
                    'tray': [2, 4, 0.87],
                    'wall_shelf': [5.7, 6.0, 0],
                    'rug_lr': [8, 2.4, 0],
                    # === 旧名称/别名（兼容性）===
                    'counter_elementa': [7.4, 0.5, 0],
                    'counter_elementb': [5.9, 0.5, 0],
                    'table_dining': [3, 2, 0],
                    'chair_bob_1': [8.5, 3, 0],
                    'chair_bob_2': [7.5, 5, 0],
                    'bobs_table': [8.5, 5.5, 0.86],
                    'bob_table': [8.5, 5.5, 0.86],
                }
                tgt = target.lower()
                # 1. 精确匹配
                pos = known_pos.get(tgt)
                # 2. 模糊匹配
                if pos is None:
                    for key, val in known_pos.items():
                        if tgt in key or key in tgt:
                            pos = val
                            break
                # 3. 场景3的Bob's table = source_table_1
                if pos is None and ('bob' in tgt or 'table_new_2' in tgt or 'table_2' in tgt):
                    pos = known_pos.get('source_table_1')
                    if pos is None:
                        pos = known_pos.get('table_new_2') or known_pos.get('table2') or known_pos.get('bobs_table')
                if pos:
                    p.resetBasePositionAndOrientation(robot_body, pos, p.getQuaternionFromEuler([0, 0, 0]))
                    # 如果机器人携带着物体，也移动物体
                    for rk, cid in list(state.gripper_constraints.items()):
                        if rk == robot_id or rk == resolved_name:
                            pass  # 物体通过约束跟随机器人
                    print(f"  ✓ 导航到 {target} @ {pos}")
                else:
                    print(f"  ⚠️ 未知位置: {target}, 跳过")
            
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
    """获取完整状态 (已存在的实现,维持不变)"""
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


@app.get("/render")
def render_scene():
    """捕获 3D 场景截图，返回 base64 PNG"""
    if not state.is_initialized or not state.client:
        return {"image": None}
    try:
        # PyBullet 相机参数
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[12, 6, 6],
            cameraTargetPosition=[4, 4, 0],
            cameraUpVector=[0, 0, 1],
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=30
        )
        width, height, rgbPixels, _, _ = p.getCameraImage(
            width=480, height=360,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        # rgbPixels 是 (height, width, 4) uint8 RGBA，转为 PNG
        raw_data = bytes(rgbPixels)  # flattens to RGBA bytes
        # 纯 Python PNG 编码器 (无需 PIL)
        def _encode_png(w, h, rgba_bytes):
            def write_chunk(chunk_type, data):
                chunk = chunk_type + data
                return struct.pack('>I', len(data)) + chunk + struct.pack('>I', zlib.crc32(chunk) & 0xFFFFFFFF)
            
            buf = io.BytesIO()
            buf.write(b'\x89PNG\r\n\x1a\n')
            # IHDR
            ihdr = struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)  # 8bit RGBA
            buf.write(write_chunk(b'IHDR', ihdr))
            # IDAT - raw image data with filter byte per row
            raw = b''
            for y in range(h):
                raw += b'\x00'  # filter: None
                row_start = y * w * 4
                raw += rgba_bytes[row_start:row_start + w * 4]
            buf.write(write_chunk(b'IDAT', zlib.compress(raw)))
            buf.write(write_chunk(b'IEND', b''))
            return buf.getvalue()
        
        png_data = _encode_png(width, height, raw_data)
        b64 = base64.b64encode(png_data).decode()
        return {"image": f"data:image/png;base64,{b64}"}
    except Exception as e:
        print(f"[BestMan] Render error: {e}")
        return {"image": None}


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
