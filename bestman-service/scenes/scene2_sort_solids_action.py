#!/usr/bin/env python3
"""
scene2_sort_solids_action.py
Sort Solids 任务 - 机器人协作动作脚本（硬编码）
在 BestMan PyBullet 场景2中播放机器人协作过程

使用方法: 
  cd /home/developer/.openclaw/workspace/dynahmrc/bestman-service
  python3 scenes/scene2_sort_solids_action.py
"""

import os
import sys
import json
import math
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
# scene2 在 dynahmrc/bestman-service/scenes/
# BestMan 在 workspace/BestMan/
workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
bestman_dir = os.path.join(workspace_dir, 'BestMan')
sys.path.insert(0, bestman_dir)
sys.path.insert(0, os.path.dirname(script_dir))

import pybullet as p
from Env.Client import Client
from scenes.scene2_setup import setup_scene2


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
print("  Sort Solids - 动作回放脚本")
print("=" * 60)

cfg = load_yaml_config('Config/default.yaml')
cfg.Client.enable_GUI = True

print("[1/3] 连接 PyBullet (GUI模式)...")
client = Client(cfg.Client)

print("[2/3] 加载场景2 (kitchen/sort_solids)...")
scene_json = os.path.join(script_dir, "scene2.json")
setup_scene2(client, scene_json)

print("[3/3] 初始化完成，开始播放动作...")
time.sleep(1)

# 设置相机视角
try:
    p.resetDebugVisualizerCamera(
        cameraDistance=8,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[5, 5, 0]
    )
except:
    pass

# 获取机器人 body_id
robot_bodies = {}
for attr in dir(client):
    if attr in ['bob_arm', 'bob', 'new_robot_base', 'new_robot_arm', 'david', 'drone_body',
                'alice_base', 'alice_arm']:
        val = getattr(client, attr, None)
        if isinstance(val, int) and val > 0:
            robot_bodies[attr] = val

# 获取物品 body_id
scene_objects = {}
for attr in dir(client):
    val = getattr(client, attr, None)
    if isinstance(val, int) and val > 0 and val not in robot_bodies.values():
        scene_objects[attr] = val

print(f"  机器人: {list(robot_bodies.keys())}")
print(f"  场景物体: {len(scene_objects)} 个")


def get_pos(name):
    """获取家具位置"""
    known_pos = {
        'elementb1': [1, 0.5, 0],
        'elementa': [2.5, 0.5, 0],
        'microwave': [3.2, 0.3, 0],
        'elementc': [3.7, 0.5, 0],
        'fridge': [4.5, 0.5, 0],
        'table1': [1, 4, 0.86],
        'table2': [3, 5, 0.86],
        'bookcase': [1, 6.5, 0.96],
        'sofa': [7.5, 9, 0],
        'shelf_table': [9.5, 7.5, 0],
        'table_2': [3, 5, 0.86],
        'cube_purple': [3, 5, 0.86],
    }
    t = name.lower()
    return known_pos.get(t)


def navigate(robot_name, target_name):
    """导航：移动机器人到目标位置"""
    body = None
    rk = None
    for k, v in robot_bodies.items():
        if robot_name.lower() in k.lower() or k.lower() in robot_name.lower():
            body = v
            rk = k
            break
    if body is None:
        print(f"  ⚠️ 找不到机器人 {robot_name}")
        return False
    pos = get_pos(target_name)
    if pos is None:
        print(f"  ⚠️ 未知目标 {target_name}")
        return False
    p.resetBasePositionAndOrientation(body, pos, p.getQuaternionFromEuler([0, 0, 0]))
    # 同时移动手臂
    arm_key = None
    if rk == 'new_robot_base':
        arm_key = 'new_robot_arm'
    elif rk == 'alice_base':
        arm_key = 'alice_arm'
    if arm_key and arm_key in robot_bodies:
        p.resetBasePositionAndOrientation(robot_bodies[arm_key], pos, p.getQuaternionFromEuler([0, 0, 0]))
    for _ in range(20):
        p.stepSimulation()
    print(f"  ✓ {robot_name} → {target_name} @ ({pos[0]:.1f}, {pos[1]:.1f})")
    return True


def pick(robot_name, object_name):
    """拾取物体"""
    obj_body = scene_objects.get(object_name)
    if obj_body is None:
        print(f"  ⚠️ 找不到物体 {object_name}")
        return False
    body = None
    rk = None
    for k, v in robot_bodies.items():
        if robot_name.lower() in k.lower() or k.lower() in robot_name.lower():
            body = v
            rk = k
            break
    if body is None:
        print(f"  ⚠️ 找不到机器人 {robot_name}")
        return False
    obj_pos = p.getBasePositionAndOrientation(obj_body)[0]
    rob_pos = p.getBasePositionAndOrientation(body)[0]
    offset = [obj_pos[i] - rob_pos[i] for i in range(3)]
    constraint = p.createConstraint(
        parentBodyUniqueId=body, parentLinkIndex=-1,
        childBodyUniqueId=obj_body, childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=offset,
        childFramePosition=[0, 0, 0],
    )
    for _ in range(10):
        p.stepSimulation()
    print(f"  ✓ {robot_name} 捡起 {object_name}")
    return True


def place(robot_name, object_name):
    """放置物体到桌上（靠近 Bob）"""
    obj_body = scene_objects.get(object_name)
    if obj_body is None:
        print(f"  ⚠️ 找不到物体 {object_name}")
        return False
    # 放 Bob 桌上
    target_pos = [3.3, 5.2, 0.86]
    p.resetBasePositionAndOrientation(obj_body, target_pos, p.getQuaternionFromEuler([0, 0, 0]))
    for _ in range(10):
        p.stepSimulation()
    print(f"  ✓ {robot_name} 放置 {object_name}")


def bob_pick_and_place(object_name):
    """Bob 捡起物体并放到大色块上"""
    obj_body = scene_objects.get(object_name)
    if obj_body is None:
        print(f"  ⚠️ 找不到物体 {object_name}")
        return False
    
    # Bob 捡起
    bob_body = robot_bodies.get('bob_arm') or robot_bodies.get('bob')
    if bob_body is None:
        print("  ⚠️ 找不到 Bob")
        return False
    
    obj_pos = p.getBasePositionAndOrientation(obj_body)[0]
    rob_pos = p.getBasePositionAndOrientation(bob_body)[0]
    offset = [obj_pos[i] - rob_pos[i] for i in range(3)]
    constraint = p.createConstraint(
        parentBodyUniqueId=bob_body, parentLinkIndex=-1,
        childBodyUniqueId=obj_body, childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=offset,
        childFramePosition=[0, 0, 0],
    )
    for _ in range(10):
        p.stepSimulation()
    print(f"  ✓ Bob 捡起 {object_name}")
    
    # 放到大色块上（cube_purple 在 table2 上）
    target_pos = [3.5, 5.3, 0.86]
    p.resetBasePositionAndOrientation(obj_body, target_pos, p.getQuaternionFromEuler([0, 0, 0]))
    # 移除约束
    p.removeConstraint(constraint)
    for _ in range(10):
        p.stepSimulation()
    print(f"  ✓ Bob 放置 {object_name} → cube_purple ✅")


# ========== 动作序列 ==========
print("\n" + "=" * 60)
print("  开始执行动作序列")
print("=" * 60)

# 由于 sort_solids 随机化颜色，这里以 purple 为例
# 如果任务随机到其他颜色，改 object_name 和 cube_name 即可
TARGET_COLOR = "purple"
SMALL_CUBE = f"small_cube_{TARGET_COLOR}"
LARGE_CUBE = f"cube_{TARGET_COLOR}"

actions = [
    # Step 1: Alice 导航到 shelf_table 找小色块
    ("Alice", "navigate", "shelf_table"),
    
    # Step 2: Alice 捡起小色块
    ("Alice", "pick", SMALL_CUBE),
    
    # Step 3: Alice 导航到 Bob 的桌子 (table2) 放下
    ("Alice", "navigate", "table_2"),
    ("Alice", "place", SMALL_CUBE),
    
    # Step 4: Bob 捡起小色块
    ("Bob", "pick", SMALL_CUBE),  # 实际执行 bob_pick_and_place
    
    # Step 5: Bob 放到大色块上完成
    ("Bob", "place", LARGE_CUBE),
]

for i, action in enumerate(actions):
    robot, cmd, target = action
    print(f"\n[{i+1}/{len(actions)}] {robot} → {cmd}({target})")
    time.sleep(1.5)
    
    if cmd == "navigate":
        navigate(robot, target)
    elif cmd == "pick":
        pick(robot, target)
    elif cmd == "place":
        if robot == "Bob":
            # Bob 放置到对应大色块
            obj_to_place = SMALL_CUBE if i == len(actions) - 1 else SMALL_CUBE
            bob_pick_and_place(SMALL_CUBE)
        else:
            place(robot, target)

print("\n" + "=" * 60)
print(f"  ✅ 任务完成！{SMALL_CUBE} → {LARGE_CUBE}")
print("  按 Ctrl+C 退出")
print("=" * 60)

try:
    while True:
        p.stepSimulation()
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n退出...")
    client.disconnect()
