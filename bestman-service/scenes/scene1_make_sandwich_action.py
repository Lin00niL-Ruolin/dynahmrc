#!/usr/bin/env python3
"""
scene1_make_sandwich_action.py
Make Sandwich 任务 - 机器人协作动作脚本（硬编码）
在 BestMan PyBullet 场景1中播放机器人协作过程

使用方法: 
  cd /home/developer/.openclaw/workspace/dynahmrc/bestman-service
  python3 scenes/scene1_make_sandwich_action.py
"""

import os
import sys
import json
import math
import time

# 路径设置
script_dir = os.path.dirname(os.path.abspath(__file__))
# scene1_make_sandwich_action.py 在 dynahmrc/bestman-service/scenes/
# BestMan 在 workspace/BestMan/
workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
bestman_dir = os.path.join(workspace_dir, 'BestMan')
sys.path.insert(0, bestman_dir)
sys.path.insert(0, os.path.dirname(script_dir))

import pybullet as p
from Env.Client import Client
from scenes.scene1_setup import setup_scene1
from scenes.path_planner import AStarPathPlanner, navigate_along_path


def resolve_asset_path(relative_path):
    if relative_path.startswith("Asset/"):
        return os.path.join(bestman_dir, relative_path)
    return os.path.join(bestman_dir, "Asset", relative_path)


def load_yaml_config(config_path):
    import yaml
    from types import SimpleNamespace
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d
    full_path = config_path
    if not os.path.exists(full_path):
        full_path = os.path.join(bestman_dir, config_path)
    with open(full_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return dict_to_namespace(cfg_dict)


# ========== 初始化场景 ==========
print("=" * 60)
print("  Make Sandwich - 动作回放脚本")
print("=" * 60)

# 加载配置
cfg = load_yaml_config('Config/default.yaml')
cfg.Client.enable_GUI = True

print("[1/3] 连接 PyBullet (GUI模式)...")
client = Client(cfg.Client)

print("[2/3] 加载场景1...")
scene_json = os.path.join(script_dir, "scene1.json")
setup_scene1(client, scene_json)

print("[3/3] 初始化完成，开始播放动作...")

# 步进仿真让场景稳定
for _ in range(30):
    p.stepSimulation()

print("\n初始化 A* 路径规划器...")
planner = AStarPathPlanner(scene='scene1', grid_size=0.3)
print("A* 路径规划器就绪 (网格分辨率 0.3m)")

print("\n观察 PyBullet 窗口！机器人即将开始移动...")
time.sleep(3)

# 获取机器人 body_id
robot_bodies = {}
for attr in ['bob_arm', 'bob', 'new_robot_base', 'new_robot_arm', 'david', 'drone_body']:
    val = getattr(client, attr, None)
    if isinstance(val, int) and val > 0:
        robot_bodies[attr] = val
        print(f"  机器人: {attr} = {val}")

# 名称映射
ROBOT_NAME_MAP = {
    'Alice': 'new_robot_base',
    'Bob': 'bob_arm',
    'David': 'david',
    'Lucy': 'drone_body',
}
ROBOT_PAIRS = {'new_robot_base': 'new_robot_arm'}

# 获取物品 body_id
scene_objects = {}
for attr in dir(client):
    val = getattr(client, attr, None)
    if isinstance(val, int) and val > 0 and val not in robot_bodies.values():
        # 排除墙壁等
        if not attr.startswith('wall_') and attr != 'wood_floor' and attr != 'enable_cache':
            scene_objects[attr] = val

# 物品名映射（脚本名 → scene 中的名）
ITEM_NAME_MAP = {
    'bacon': 'bacon_0',
}

# 地面位置（Alice/David/Lucy 导航用，Z=0）
GROUND_POS = {
    'fridge': [9.4, 0.5, 0],
    'elementa': [7.4, 0.5, 0],
    'elementb1': [5.9, 0.5, 0],
    'elementc': [8.6, 0.5, 0],
    'microwave': [8.1, 0.3, 0],
    'table': [3, 2, 0],
    'table_new_1': [8.5, 4, 0],        # Z=0，地面！不是桌面
    'table_new_2': [8.5, 5.5, 0],      # Z=0，地面！
    'cutting_board': [8.5, 5.5, 0],    # 导航到附近地面
}

# 桌面位置（Bob 和物品放置用，Z=0.86）
TABLE_POS = {
    'table_new_1': [8.5, 4, 0.86],
    'table_new_2': [8.5, 5.5, 0.86],
    'cutting_board': [8.5, 5.5, 0.86],
}


def get_pos(name):
    t = name.lower() if isinstance(name, str) else ''
    return GROUND_POS.get(t)


def navigate(robot_name, target_name):
    """导航：用 A* 路径规划平滑移动"""
    key = ROBOT_NAME_MAP.get(robot_name)
    if not key:
        print(f"  ⚠️ 未知机器人名 {robot_name}")
        return False
    body = robot_bodies.get(key)
    if body is None:
        print(f"  ⚠️ 找不到 {robot_name} 的 body (key={key})")
        return False
    pos = get_pos(target_name)
    if pos is None:
        print(f"  ⚠️ 未知目标 {target_name}")
        return False
    
    # 当前位置
    old_pos = p.getBasePositionAndOrientation(body)[0]
    
    # A* 寻路
    path = planner.plan(old_pos[0], old_pos[1], pos[0], pos[1])
    if path is None:
        # A* 失败，直接瞬移（兜底）
        p.resetBasePositionAndOrientation(body, pos, p.getQuaternionFromEuler([0, 0, 0]))
        print(f"  ⚠️ A* 无路径，直接瞬移")
    else:
        # 沿路径平滑移动
        paired_key = ROBOT_PAIRS.get(key)
        paired_body = robot_bodies.get(paired_key) if paired_key else None
        navigate_along_path(p, body, path, paired_body, steps_per_point=8)
    
    # 步进仿真
    for _ in range(10):
        p.stepSimulation()
    
    # 确认到达
    final_pos = p.getBasePositionAndOrientation(body)[0]
    print(f"  ✓ {robot_name}: ({old_pos[0]:.1f},{old_pos[1]:.1f}) → ({final_pos[0]:.1f},{final_pos[1]:.1f})")
    return True


def pick(robot_name, object_name):
    """拾取物体"""
    real_name = ITEM_NAME_MAP.get(object_name, object_name)
    obj_body = scene_objects.get(real_name)
    if obj_body is None:
        print(f"  ⚠️ 找不到物体 {object_name} (搜索 {real_name})")
        print(f"  可用物品: {[k for k in scene_objects.keys()][:20]}")
        return False
    key = ROBOT_NAME_MAP.get(robot_name)
    if not key:
        print(f"  ⚠️ 未知机器人名 {robot_name}")
        return False
    robot_body = robot_bodies.get(key)
    if robot_body is None:
        print(f"  ⚠️ 找不到 {robot_name} 的 body")
        return False
    obj_pos = p.getBasePositionAndOrientation(obj_body)[0]
    rob_pos = p.getBasePositionAndOrientation(robot_body)[0]
    offset = [obj_pos[i] - rob_pos[i] for i in range(3)]
    constraint = p.createConstraint(
        parentBodyUniqueId=robot_body, parentLinkIndex=-1,
        childBodyUniqueId=obj_body, childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=offset,
        childFramePosition=[0, 0, 0],
    )
    for _ in range(10):
        p.stepSimulation()
    print(f"  ✓ {robot_name} 捡起 {object_name} (#{constraint})")
    return True


def place(robot_name, object_name, target_name):
    """放置物体到桌面"""
    real_name = ITEM_NAME_MAP.get(object_name, object_name)
    obj_body = scene_objects.get(real_name)
    if obj_body is None:
        print(f"  ⚠️ 找不到物体 {object_name}")
        return False
    target_pos = TABLE_POS.get(target_name.lower())
    if target_pos is None:
        target_pos = [8.5, 5.5, 0.86]  # 默认放 table_new_2 桌面
    # 释放约束
    for rk in list(robot_bodies.keys()):
        pass  # 约束通过 body ID 查找
    p.resetBasePositionAndOrientation(obj_body, [target_pos[0], target_pos[1], target_pos[2] + 0.01],
                                       p.getQuaternionFromEuler([0, 0, 0]))
    for _ in range(10):
        p.stepSimulation()
    # 释放约束
    if obj_body in [v for v in globals().get('_constraints', {}).values()]:
        pass  # 约束在 pick 时已创建，这里不做复杂管理
    print(f"  ✓ {robot_name} 放置 {object_name} 到 {target_name}")


def release_constraint(obj_body):
    """释放物体上的所有约束"""
    for i in range(p.getNumConstraints()):
        try:
            info = p.getConstraintInfo(i)
            if info and info[5] == obj_body:
                p.removeConstraint(i)
        except:
            pass


# ========== 动作序列 ==========
print("\n" + "=" * 60)
print("  开始执行动作序列")
print("=" * 60)

# 简化动作序列：按实际 Phase 4 输出整理
actions = [
    # === 第1轮：Alice 去 table_new_1, Bob 捡 bread_0 ===
    ("Alice", "navigate", "table_new_1"),
    ("Bob", "pick", "bread_0"),
    
    # === 第2轮：Alice 捡 bacon, Bob 放 bread_0 → cutting_board ===
    ("Alice", "pick", "bacon"),
    ("Bob", "place_obj", ("bread_0", "cutting_board")),  # Bob 放 bread_0 到 cutting_board
    
    # === 第3轮：Lucy 拿 bread_1 ===
    ("Lucy", "navigate", "table_new_1"),
    ("Lucy", "pick", "bread_1"),
    
    # === 第4轮：Alice 送 bacon 到 Bob 桌 ===
    ("Alice", "navigate", "table_new_2"),
    ("Alice", "place_obj", ("bacon_0", "table_new_2")),
    
    # === 第5轮：Bob 捡 bacon 放 cutting_board ===
    ("Bob", "pick", "bacon_0"),
    ("Bob", "place_obj", ("bacon_0", "cutting_board")),  # 2/3
    
    # === 第6轮：Lucy 送 bread_1 到 Bob 桌 ===
    ("Lucy", "navigate", "table_new_2"),
    ("Lucy", "place_obj", ("bread_1", "table_new_2")),
    
    # === 第7轮：Bob 捡 bread_1 放 cutting_board ===
    ("Bob", "pick", "bread_1"),
    ("Bob", "place_obj", ("bread_1", "cutting_board")),  # 3/3 DONE
]

for i, action_tuple in enumerate(actions):
    robot, cmd, target = action_tuple[0], action_tuple[1], action_tuple[2]
    print(f"\n[{i+1}/{len(actions)}] {robot} → {cmd}({target})")
    time.sleep(1.2)
    
    if cmd == "navigate":
        navigate(robot, target)
    elif cmd == "pick":
        pick(robot, target)
    elif cmd == "place_obj":
        # target 是 (object_name, target_location)
        obj_name, tgt = target
        # 先释放约束
        real_obj = ITEM_NAME_MAP.get(obj_name, obj_name)
        obj_body = scene_objects.get(real_obj)
        if obj_body:
            release_constraint(obj_body)
        place(robot, obj_name, tgt)

print("\n" + "=" * 60)
print("  ✅ 动作序列播放完毕！")
print("  按 Ctrl+C 退出，或关闭 PyBullet 窗口")
print("=" * 60)

# 保持窗口打开
try:
    while True:
        p.stepSimulation()
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n退出...")
    client.disconnect()
