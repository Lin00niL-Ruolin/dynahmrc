#!/usr/bin/env python3
"""
scene3_pack_objects_action.py
Pack Objects 任务 - 机器人协作动作脚本（硬编码）
在 BestMan PyBullet 场景3中播放机器人协作过程

使用方法: 
  cd /home/developer/.openclaw/workspace/dynahmrc/bestman-service
  python3 scenes/scene3_pack_objects_action.py
"""

import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
# scene3 在 dynahmrc/bestman-service/scenes/
# BestMan 在 workspace/BestMan/
workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
bestman_dir = os.path.join(workspace_dir, 'BestMan')
sys.path.insert(0, bestman_dir)
sys.path.insert(0, os.path.dirname(script_dir))

import pybullet as p
from Env.Client import Client
from scenes.path_planner import AStarPathPlanner


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


def setup_scene3_local(client):
    """初始化场景3（living_room）"""
    import json
    scene_json = os.path.join(script_dir, "scene3.json")
    if os.path.exists(scene_json):
        with open(scene_json) as f:
            data = json.load(f)
        for obj in data:
            try:
                name = obj['obj_name']
                path = obj['model_path']
                pos = obj['object_position']
                ori = obj['object_orientation']
                scale = obj.get('scale', 1.0)
                fixed = obj.get('fixed_base', True)
                processed_ori = []
                for val in ori:
                    if isinstance(val, str):
                        processed_ori.append(eval(val))
                    else:
                        processed_ori.append(val)
                oid = client.load_object(
                    obj_name=name, model_path=path,
                    object_position=pos, object_orientation=processed_ori,
                    scale=scale, fixed_base=fixed
                )
                setattr(client, name, oid)
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {obj.get('obj_name', '?')}: {e}")
    # 添加物品（fork_0, apple, book_0, soap）
    items_data = [
        ('fork_0', 'Asset/Scene/Object/Kitchen_world_models/Bottle/3558/mobility.urdf', [1.2, 0.55, 1.1], 2.0),
        ('apple', 'Asset/Scene/Object/Kitchen_world_models/Bottle/3614/mobility.urdf', [4.15, 4, 0.85], 1.0),
        ('book_0', 'Asset/Scene/Object/Kitchen_world_models/Bottle/3574/mobility.urdf', [7.5, 5.5, 1.8], 1.5),
        ('soap', 'Asset/Scene/Object/Kitchen_world_models/Bottle/3615/mobility.urdf', [5.7, 6.0, 1.71], 1.5),
    ]
    for item_name, model_path, pos, scale in items_data:
        try:
            if model_path.startswith("Asset/"):
                full_path = os.path.join(bestman_dir, model_path)
            else:
                full_path = model_path
            oid = p.loadURDF(full_path, pos, globalScaling=scale)
            setattr(client, item_name, oid)
            print(f"  ✓ {item_name}")
        except Exception as e:
            print(f"  ✗ {item_name}: {e}")

    # Bob 机器人
    bob_path = 'Asset/Robot/mobile_manipulator/arm/ufactory/urdf/xarm6.urdf'
    try:
        full = os.path.join(bestman_dir, bob_path) if bob_path.startswith("Asset/") else bob_path
        bob_id = p.loadURDF(full, [2, 4, 0.86], p.getQuaternionFromEuler([0, 0, 0]))
        setattr(client, 'bob_arm', bob_id)
        setattr(client, 'bob', bob_id)
        print("  ✓ Bob")
    except Exception as e:
        print(f"  ✗ Bob: {e}")

    # Alice base + arm
    try:
        alice_base_path = 'Asset/Robot/mobile_manipulator/base/urdf/youbot.urdf'
        alice_arm_path = 'Asset/Robot/mobile_manipulator/arm/ufactory/urdf/xarm6.urdf'
        base_id = p.loadURDF(os.path.join(bestman_dir, alice_base_path), [5, 2, 0], globalScaling=1.0)
        arm_id = p.loadURDF(os.path.join(bestman_dir, alice_arm_path), [5, 2, 0.5], globalScaling=0.8)
        setattr(client, 'alice_base', base_id)
        setattr(client, 'alice_arm', arm_id)
        print("  ✓ Alice")
    except Exception as e:
        print(f"  ✗ Alice: {e}")

    # David
    try:
        david_path = 'Asset/Robot/mobile_manipulator/base/urdf/youbot.urdf'
        david_id = p.loadURDF(os.path.join(bestman_dir, david_path), [8, 6, 0], globalScaling=0.8)
        setattr(client, 'david', david_id)
        print("  ✓ David")
    except Exception as e:
        print(f"  ✗ David: {e}")

    # Lucy drone
    try:
        lucy_path = 'Asset/Robot/mobile_manipulator/arm/ufactory/urdf/xarm6.urdf'
        drone_id = p.loadURDF(os.path.join(bestman_dir, lucy_path), [3, 3, 1.0], globalScaling=0.5)
        setattr(client, 'drone_body', drone_id)
        print("  ✓ Lucy")
    except Exception as e:
        print(f"  ✗ Lucy: {e}")


# ========== 初始化场景 ==========
print("=" * 60)
print("  Pack Objects - 动作回放脚本")
print("=" * 60)

cfg = load_yaml_config('Config/default.yaml')
cfg.Client.enable_GUI = True

print("[1/3] 连接 PyBullet (GUI模式)...")
client = Client(cfg.Client)

print("[2/3] 加载场景3 (living_room)...")
setup_scene3_local(client)

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
for attr in ['bob_arm', 'bob', 'alice_base', 'alice_arm', 'david', 'drone_body']:
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
print(f"  物品: {[k for k in scene_objects.keys() if not k.startswith('_') and k != 'enable_cache'][:20]}")


def get_pos(name):
    """获取 living_room 家具位置"""
    known_pos = {
        'kitchen_cabinet': [1.2, 0.5, 0],
        'kitchen_counter': [3.1, 0.5, 0],
        'microwave': [3.8, 0.3, 0],
        'dishwasher': [4.6, 0.7, 0],
        'fridge': [5.5, 0.5, 0],
        'cabinet_2': [7.3, 0.6, 0],
        'sofa': [8.6, 0.8, 0],
        'packing_table': [8, 3, 0],
        'source_table_1': [2, 4, 0],
        'source_table_2': [4, 4, 0],
        'chair': [2, 3, 0],
        'bookcase': [7.5, 5.5, 0],
        'bathtub': [1.5, 9.4, 0],
        'sink_base': [4.7, 9.5, 0],
        'sink': [4.7, 9.6, 0],
        'tray': [2, 4, 0.87],
        'wall_shelf': [5.7, 6.0, 0],
        'rug': [8, 2.4, 0],
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
    arm_key = None
    if rk == 'alice_base':
        arm_key = 'alice_arm'
    if arm_key and arm_key in robot_bodies:
        p.resetBasePositionAndOrientation(robot_bodies[arm_key], pos, p.getQuaternionFromEuler([0, 0, 0]))
    for _ in range(20):
        p.stepSimulation()
    print(f"  ✓ {robot_name} → {target_name}")
    return True


def pick(robot_name, object_name):
    """拾取物体"""
    obj_body = scene_objects.get(object_name)
    if obj_body is None:
        print(f"  ⚠️ 找不到物体 {object_name}")
        return False
    body = None
    for k, v in robot_bodies.items():
        if robot_name.lower() in k.lower() or k.lower() in robot_name.lower():
            body = v
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


def place_on_bobs_table(robot_name, object_name):
    """放置物体到 Bob 的桌子 (source_table_1)"""
    obj_body = scene_objects.get(object_name)
    if obj_body is None:
        print(f"  ⚠️ 找不到物体 {object_name}")
        return False
    target_pos = [2.3, 4.2, 0.86]
    p.resetBasePositionAndOrientation(obj_body, target_pos, p.getQuaternionFromEuler([0, 0, 0]))
    for _ in range(10):
        p.stepSimulation()
    print(f"  ✓ {robot_name} 放置 {object_name} → Bob's table")


def bob_pick_and_place_in_tray(object_name):
    """Bob 捡起物体并放入 tray"""
    obj_body = scene_objects.get(object_name)
    if obj_body is None:
        print(f"  ⚠️ 找不到物体 {object_name}")
        return False
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
    time.sleep(0.5)
    # 放入 tray
    target_pos = [2.5, 4.0, 0.87]
    p.resetBasePositionAndOrientation(obj_body, target_pos, p.getQuaternionFromEuler([0, 0, 0]))
    p.removeConstraint(constraint)
    for _ in range(10):
        p.stepSimulation()
    print(f"  ✓ Bob 放入 tray ✅")


# ========== 动作序列 ==========
print("\n" + "=" * 60)
print("  开始执行动作序列")
print("=" * 60)

# 并行：Alice 去 source_table_2 拿 apple, David 去 kitchen_cabinet 侦察, Lucy 去 wall_shelf 拿 soap
actions = [
    # 第1轮：Alice 拿 apple, Lucy 拿 soap
    ("Alice", "navigate", "source_table_2"),
    ("Alice", "pick", "apple"),
    ("Alice", "navigate", "source_table_1"),     # Alice 去 Bob 桌
    ("Alice", "place", "apple"),                  # 放 apple 到 Bob 桌
    
    # Bob 把 apple 放入 tray
    ("Bob", "pick", "apple"),
    ("Bob", "place_in_tray", "apple"),            # 1/4
    
    # 第2轮：Alice 去 bookcase 拿 book_0
    ("Alice", "navigate", "bookcase"),
    ("Alice", "pick", "book_0"),
    ("Alice", "navigate", "source_table_1"),
    ("Alice", "place", "book_0"),
    
    # Bob 把 book_0 放入 tray
    ("Bob", "pick", "book_0"),
    ("Bob", "place_in_tray", "book_0"),           # 2/4
    
    # 第3轮：Lucy 拿 soap
    ("Lucy", "navigate", "wall_shelf"),
    ("Lucy", "pick", "soap"),
    ("Lucy", "navigate", "source_table_1"),
    ("Lucy", "place", "soap"),
    
    # Bob 把 soap 放入 tray
    ("Bob", "pick", "soap"),
    ("Bob", "place_in_tray", "soap"),             # 3/4
    
    # 第4轮：Lucy 拿 fork_0
    ("Lucy", "navigate", "kitchen_cabinet"),
    ("Lucy", "pick", "fork_0"),
    ("Lucy", "navigate", "source_table_1"),
    ("Lucy", "place", "fork_0"),
    
    # Bob 把 fork_0 放入 tray
    ("Bob", "pick", "fork_0"),
    ("Bob", "place_in_tray", "fork_0"),           # 4/4 DONE
]

for i, action in enumerate(actions):
    robot, cmd, target = action
    print(f"\n[{i+1}/{len(actions)}] {robot} → {cmd}({target})")
    time.sleep(1.2)
    
    if cmd == "navigate":
        navigate(robot, target)
    elif cmd == "pick":
        pick(robot, target)
    elif cmd == "place":
        place_on_bobs_table(robot, target)
    elif cmd == "place_in_tray":
        bob_pick_and_place_in_tray(target)

print("\n" + "=" * 60)
print("  ✅ 全部 4/4 物品放入 tray！任务完成！")
print("  按 Ctrl+C 退出")
print("=" * 60)

try:
    while True:
        p.stepSimulation()
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n退出...")
    client.disconnect()
