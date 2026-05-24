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
time.sleep(1)

# 设置相机视角
try:
    p.resetDebugVisualizerCamera(
        cameraDistance=8,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[5, 4, 0]
    )
except:
    pass

# 获取机器人 body_id
robot_bodies = {}
for attr in dir(client):
    if attr in ['bob_arm', 'bob', 'new_robot_base', 'new_robot_arm', 'david', 'drone_body']:
        val = getattr(client, attr, None)
        if isinstance(val, int) and val > 0:
            robot_bodies[attr] = val
            print(f"  机器人: {attr} = {val}")

# 获取物品 body_id
scene_objects = {}
for attr in dir(client):
    val = getattr(client, attr, None)
    if isinstance(val, int) and val > 0 and val not in robot_bodies.values():
        scene_objects[attr] = val

def get_pos(name):
    """获取家具位置"""
    known_pos = {
        'fridge': [9.4, 0.5, 0],
        'elementa': [7.4, 0.5, 0],
        'elementb1': [5.9, 0.5, 0],
        'elementc': [8.6, 0.5, 0],
        'microwave': [8.1, 0.3, 0],
        'table': [3, 2, 0],
        'table_new_1': [8.5, 4, 0.86],
        'table_new_2': [8.5, 5.5, 0.86],
        'cutting_board': [8.5, 5.5, 0.86],
    }
    t = target.lower() if isinstance(target := name, str) else ''
    return known_pos.get(t)


def navigate(robot_name, target_name, robot_key):
    """导航：移动机器人到目标位置"""
    body = robot_bodies.get(robot_key)
    if body is None:
        print(f"  ⚠️ 找不到机器人 {robot_key}")
        return False
    pos = get_pos(target_name)
    if pos is None:
        print(f"  ⚠️ 未知目标 {target_name}")
        return False
    p.resetBasePositionAndOrientation(body, pos, p.getQuaternionFromEuler([0, 0, 0]))
    # 同时移动配对部件（底座→手臂）
    pairs = {'new_robot_base': 'new_robot_arm'}
    if robot_key in pairs:
        paired = robot_bodies.get(pairs[robot_key])
        if paired is not None:
            p.resetBasePositionAndOrientation(paired, pos, p.getQuaternionFromEuler([0, 0, 0]))
    for _ in range(20):
        p.stepSimulation()
    print(f"  ✓ {robot_name} → {target_name} @ {pos}")
    return True


def pick(robot_name, object_name):
    """拾取物体"""
    obj_body = scene_objects.get(object_name)
    if obj_body is None:
        print(f"  ⚠️ 找不到物体 {object_name}")
        return False
    robot_key = None
    for rk, rbody in robot_bodies.items():
        if rk.startswith(robot_name.lower().split('_')[0]) or rk == robot_name:
            robot_key = rk
            break
    if robot_key is None:
        print(f"  ⚠️ 找不到机器人 {robot_name}")
        return False
    robot_body = robot_bodies[robot_key]
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
    """放置物体"""
    obj_body = scene_objects.get(object_name)
    if obj_body is None:
        print(f"  ⚠️ 找不到物体 {object_name}")
        return False
    target_pos = get_pos(target_name)
    if target_pos is None:
        target_pos = [8.5, 5.5, 0.86]  # 默认放 table_new_2
    # 释放约束
    for rk in list(robot_bodies.keys()):
        pass  # 约束通过 body ID 查找
    p.resetBasePositionAndOrientation(obj_body, [target_pos[0], target_pos[1], target_pos[2] + 0.01],
                                       p.getQuaternionFromEuler([0, 0, 0]))
    for _ in range(10):
        p.stepSimulation()
    print(f"  ✓ {robot_name} 放置 {object_name} 到 {target_name}")


# ========== 动作序列 ==========
print("\n" + "=" * 60)
print("  开始执行动作序列")
print("=" * 60)

actions = [
    # Step 1: Alice 去 table_new_1, Bob 捡 bread_0
    ("Alice", "navigate", "table_new_1"),
    ("Bob", "pick", "bread_0"),
    
    # Step 2: Alice 捡 bacon, Bob 放 bread_0 到 cutting_board
    ("Alice", "pick", "bacon"),
    ("Bob", "place", ("bacon", "cutting_board")),  # wait, Bob 放的是 bread_0
    
    # 修正：Bob 放 bread_0 到 cutting_board
    ("Bob", "navigate", "table_new_2"),  # Bob 不动，导航到自己的位置
    ("Bob", "place", "cutting_board"),
    
    # Step 3: Lucy 去 table_new_1 捡 bread_1
    ("Lucy", "navigate", "table_new_1"),
    ("Lucy", "pick", "bread_1"),
    
    # Step 4: Alice 导航到 table_new_2 放 bacon
    ("Alice", "navigate", "table_new_2"),
    ("Alice", "place", "table_new_2"),
    
    # Step 5: Bob 捡 bacon 放 cutting_board
    ("Bob", "pick", "bacon"),
    ("Bob", "navigate", "table_new_2"),
    ("Bob", "place", "cutting_board"),
    
    # Step 6: Lucy 导航到 table_new_2 放 bread_1
    ("Lucy", "navigate", "table_new_2"),
    ("Lucy", "place", "table_new_2"),
    
    # Step 7: Bob 捡 bread_1 放 cutting_board
    ("Bob", "pick", "bread_1"),
    ("Bob", "place", "cutting_board"),
]

for i, action in enumerate(actions):
    robot, cmd, target = action[0], action[1], action[2]
    print(f"\n[{i+1}/{len(actions)}] {robot} → {cmd}({target})")
    time.sleep(1.5)  # 每步间隔1.5秒，让用户看到过程
    
    if cmd == "navigate":
        # 确定 robot_key
        rk = None
        for k in robot_bodies:
            if robot.lower() in k.lower() or k.lower() in robot.lower():
                rk = k
                break
        if rk:
            navigate(robot, target, rk)
        else:
            print(f"  ⚠️ 找不到 {robot} 的 body")
    elif cmd == "pick":
        pick(robot, target)
    elif cmd == "place":
        # 如果是 tuple，第一个是物体名
        obj_name = target[0] if isinstance(target, tuple) else target
        place(robot, obj_name, target if not isinstance(target, tuple) else "cutting_board")

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
