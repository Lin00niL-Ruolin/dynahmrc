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

# ===== 硬编码坐标 =====
# 地面导航点（Z=0，地面机器人用）
NAV = {
    'table_new_1_ground': [8.5, 3.5, 0],     # table_new_1 旁边地面
    'table_new_2_ground': [8.5, 5.0, 0],     # table_new_2 旁边地面
    'table_new_1_front': [7.5, 4.0, 0],       # table_new_1 前方
    'table_new_2_side': [7.5, 5.5, 0],        # table_new_2 侧面
    'bacon_pos': [8.5, 4.0, 0],               # 培根所在桌子旁
    'bread1_pos': [8.6, 4.2, 0],              # bread_1 旁边
}

# 桌面位置（Bob 抓取和物品放置用，Z=0.86）
TABLE_POS = {
    'table_new_1': [8.5, 4, 0.86],
    'table_new_2': [8.5, 5.5, 0.86],
    'cutting_board': [8.5, 5.5, 0.86],
}

# 机器人初始位置（记录手臂 Z 偏移）
ARM_Z_OFFSET = {}
for rk, body_id in robot_bodies.items():
    if rk == 'new_robot_arm':
        pos = p.getBasePositionAndOrientation(body_id)[0]
        ARM_Z_OFFSET['arm'] = pos[2]  # 约 1.02m
        print(f"  手臂初始 Z={pos[2]:.2f}")


def get_pos(name):
    t = name.lower() if isinstance(name, str) else ''
    return GROUND_POS.get(t)


# 预先获取手臂 Z 偏移（只取一次，保持固定）
_arm_z = None
def get_arm_z():
    global _arm_z
    if _arm_z is None:
        arm_body = robot_bodies.get('new_robot_arm')
        if arm_body is not None:
            _arm_z = p.getBasePositionAndOrientation(arm_body)[0][2]
    return _arm_z or 1.02


def navigate_to(robot_name, target_key):
    """导航到硬编码坐标点"""
    key = ROBOT_NAME_MAP.get(robot_name)
    if not key:
        print(f"  ⚠️ 未知机器人名 {robot_name}")
        return False
    body = robot_bodies.get(key)
    if body is None:
        print(f"  ⚠️ 找不到 {robot_name} 的 body")
        return False
    pos = NAV.get(target_key)
    if pos is None:
        print(f"  ⚠️ 未知导航点 {target_key}")
        return False
    
    old_pos = p.getBasePositionAndOrientation(body)[0]
    
    # A* 寻路
    path = planner.plan(old_pos[0], old_pos[1], pos[0], pos[1])
    if path is None:
        p.resetBasePositionAndOrientation(body, pos, p.getQuaternionFromEuler([0, 0, 0]))
        print(f"  ⚠️ A* 无路径，直接瞬移")
    else:
        paired_key = ROBOT_PAIRS.get(key)
        paired_body = robot_bodies.get(paired_key) if paired_key else None
        arm_z = get_arm_z()
        navigate_along_path(p, body, path, paired_body, paired_z=arm_z, steps_per_point=8)
    
    for _ in range(10):
        p.stepSimulation()
    
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

# 动作序列（使用硬编码坐标）
actions = [
    # === 第1轮 ===
    # Alice 到 table_new_1 旁边地面 → Bob 捡 bread_0
    ("navigate", "Alice", "table_new_1_front"),
    ("pick", "Bob", "bread_0"),
    
    # Alice 捡 bacon（她在 table_new_1 旁边）
    ("pick", "Alice", "bacon"),
    
    # Bob 放 bread_0 到 cutting_board
    ("place", "Bob", "bread_0", "cutting_board"),  # 1/3
    
    # === 第2轮 ===
    # Lucy 到 table_new_1 旁边捡 bread_1
    ("navigate", "Lucy", "bread1_pos"),
    ("pick", "Lucy", "bread_1"),
    
    # Alice 送 bacon 到 Bob 桌
    ("navigate", "Alice", "table_new_2_side"),
    ("place", "Alice", "bacon_0", "table_new_2"),
    
    # Bob 捡 bacon 放 cutting_board
    ("pick", "Bob", "bacon_0"),
    ("place", "Bob", "bacon_0", "cutting_board"),  # 2/3
    
    # === 第3轮 ===
    # Lucy 送 bread_1 到 Bob 桌
    ("navigate", "Lucy", "table_new_2_side"),
    ("place", "Lucy", "bread_1", "table_new_2"),
    
    # Bob 捡 bread_1 放 cutting_board
    ("pick", "Bob", "bread_1"),
    ("place", "Bob", "bread_1", "cutting_board"),  # 3/3 DONE
]

for i, action_t in enumerate(actions):
    cmd = action_t[0]
    print(f"\n[{i+1}/{len(actions)}] {' '.join(str(x) for x in action_t)}")
    time.sleep(1.2)
    
    if cmd == "navigate":
        robot, target_key = action_t[1], action_t[2]
        navigate_to(robot, target_key)
    elif cmd == "pick":
        robot, obj = action_t[1], action_t[2]
        pick(robot, obj)
    elif cmd == "place":
        robot, obj_name, tgt = action_t[1], action_t[2], action_t[3]
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
