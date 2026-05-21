"""
场景3 设置脚本
"""

import os
import json
import math

import pybullet as p

# 房间参数
ROOM_H = 2.8    # 墙高
WALL_T = 0.15   # 墙厚度

# 颜色
COLOR_WALL = [0.92, 0.92, 0.95, 1.0]  # 米白色
COLOR_FLOOR = [0.85, 0.75, 0.65, 1.0]  # 木色


def create_wall(client, wall_name, x1, y1, x2, y2, color=COLOR_WALL):
    """从两点创建一面墙"""
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx*dx + dy*dy)
    if length == 0:
        return None
    
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    # 水平墙: size = [length, WALL_T, ROOM_H]
    # 垂直墙: size = [WALL_T, length, ROOM_H]
    # (box 默认轴向对齐，不用旋转)
    if dy == 0:  # 水平
        half = [length / 2, WALL_T / 2, ROOM_H / 2]
    else:        # 垂直 (dx == 0)
        half = [WALL_T / 2, length / 2, ROOM_H / 2]

    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[cx, cy, ROOM_H / 2],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
    )
    setattr(client, wall_name, body_id)
    return body_id


def setup_scene3(client, scene_json_path=None):
    """
    场景3
    """
    print("\n" + "=" * 60)
    print("  场景3 搭建开始")
    print("=" * 60)

    # 1. 十面墙
    print("\n--- 搭建十面墙 ---")

    walls = [
        ("wall1",   0,  0,   10, 0,   "底部外墙 (0,0)→(10,0)"),
        ("wall2",   10, 0,   10, 6,   "右下部外墙 (10,0)→(10,6)"),
        ("wall3",   10, 6,   6,  6,   "右上部外墙 (10,6)→(6,6)"),
        ("wall4",   6,  4,   6,  8,   "内部纵墙 (6,4)→(6,8)"),
        ("wall5",   6,  8,   5,  8,   "内部横墙 (6,8)→(5,8)"),
        ("wall6",   5.5,8,   5.5,10,  "上部纵墙 (5.5,8)→(5.5,10)"),
        ("wall7",   5.5,10,  0,  10,  "顶部外墙 (5.5,10)→(0,10)"),
        ("wall8",   0,  10,  0,  0,   "左部外墙 (0,10)→(0,0)"),
        ("wall9",   3,  8,   0,  8,   "左部横墙 (3,8)→(0,8)"),
        ("wall10",  6,  0,   6,  2,   "底部纵墙 (6,0)→(6,2)"),
    ]

    for name, x1, y1, x2, y2, desc in walls:
        create_wall(client, name, x1, y1, x2, y2)
        print(f"[场景] ✅ {name}: {desc}")

    # 2. 地板 (按多边形拐点铺设: (0,0)→(10,0)→(10,6)→(6,6)→(6,8)→(5.5,8)→(5.5,10)→(0,10))
    print("\n--- 铺设地板 ---")
    floor_thickness = 0.01
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tex_path = os.path.join(script_dir, "wood_texture.png")
    tex_id = p.loadTexture(tex_path) if os.path.exists(tex_path) else None

    def make_floor(name, cx, cy, sx, sy):
        half = [sx / 2, sy / 2, floor_thickness / 2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=COLOR_FLOOR)
        fid = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=[cx, cy, floor_thickness / 2 + 0.005],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        if tex_id is not None:
            p.changeVisualShape(fid, -1, textureUniqueId=tex_id)
        setattr(client, name, fid)

    # ① 主区域: (0,0)→(10,0)→(10,6)→(0,6)  10m×6m
    make_floor("floor_main", 5.0, 3.0, 10.0, 6.0)
    # ② 中段: (0,6)→(6,6)→(6,8)→(0,8)  6m×2m
    make_floor("floor_mid", 3.0, 7.0, 6.0, 2.0)
    # ③ 顶段: (0,8)→(5.5,8)→(5.5,10)→(0,10)  5.5m×2m
    make_floor("floor_top", 2.75, 9.0, 5.5, 2.0)

    print(f"[场景] ✅ 地板: 3块 (10m×6m + 6m×2m + 5.5m×2m)")

    # 3. 加载 scene3.json (厨房 + 沙发)
    print("\n--- 加载家具 ---")
    if scene_json_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        scene_json_path = os.path.join(script_dir, "scene3.json")

    if os.path.exists(scene_json_path):
        workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        bestman_dir = os.path.join(workspace_dir, 'BestMan')
        original_cwd = os.getcwd()
        os.chdir(bestman_dir)

        try:
            with open(scene_json_path, 'r') as f:
                scene_data = json.load(f)

            loaded = 0
            for obj in scene_data:
                try:
                    obj_name = obj['obj_name']
                    model_path = obj['model_path']
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

                    obj_id = client.load_object(
                        obj_name=obj_name,
                        model_path=model_path,
                        object_position=pos,
                        object_orientation=processed_ori,
                        scale=scale,
                        fixed_base=fixed
                    )
                    setattr(client, obj_name, obj_id)
                    print(f"  ✓ {obj_name} @ ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.2f})")
                    loaded += 1

                    # 沙发改成深灰色
                    if obj_name == "sofa":
                        p.changeVisualShape(obj_id, -1, rgbaColor=[0.38, 0.38, 0.42, 1])
                        print(f"  🎨 sofa → 深灰色")

                    # 桌子改为中度灰色
                    if obj_name == "packing_table":
                        num_joints = p.getNumJoints(obj_id)
                        p.changeVisualShape(obj_id, -1, rgbaColor=[0.55, 0.55, 0.55, 1])
                        for j in range(num_joints):
                            p.changeVisualShape(obj_id, j, rgbaColor=[0.55, 0.55, 0.55, 1])
                        print(f"  🎨 packing_table → 中度灰色")

                    if obj_name in ["source_table_1", "source_table_2"]:
                        num_joints = p.getNumJoints(obj_id)
                        p.changeVisualShape(obj_id, -1, rgbaColor=[0.76, 0.6, 0.42, 1])
                        for j in range(num_joints):
                            p.changeVisualShape(obj_id, j, rgbaColor=[0.76, 0.6, 0.42, 1])
                        print(f"  🎨 {obj_name} → 浅木色")

                except Exception as e:
                    print(f"  ✗ {obj.get('obj_name', '?')}: {e}")

            print(f"[场景] ✅ 已加载 {loaded}/{len(scene_data)} 个物体")
        finally:
            os.chdir(original_cwd)
    else:
        print("[场景] ⚠️ 未找到 scene3.json")

    # 4. 靠墙台面 (跟场景二相同, 贴在墙4 (6,4)→(6,8) 上)
    print("\n--- 靠墙台面 ---")
    shelf_half = [0.3, 2.0, 0.05]  # 0.6m×4m×0.1m, 紧贴墙4
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=shelf_half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=shelf_half, rgbaColor=[0.92, 0.92, 0.95, 1.0])
    shelf_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[5.7, 6, 1.6],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
    )
    setattr(client, "wall_shelf", shelf_id)
    print(f"[场景] ✅ 靠墙台面 @ (5.7, 6) 高1.6m 紧贴墙4")

    # 5. 固定机械臂 (xArm6) 在 (2,4) 桌子上
    print("\n--- 固定机械臂 ---")
    arm_path = 'Asset/Robot/mobile_manipulator/arm/ufactory/urdf/xarm6.urdf'
    tx, ty = 2, 4
    table_top_z = 0.83  # table scale=1.0
    try:
        # 底座方块
        base_half = [0.12, 0.12, 0.04]
        bc = p.createCollisionShape(p.GEOM_BOX, halfExtents=base_half)
        bv = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half, rgbaColor=[0.3, 0.3, 0.35, 1])
        bob_base = p.createMultiBody(0, bc, bv, [tx, 4.4, table_top_z])
        setattr(client, "bob_base", bob_base)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        bestman_dir = os.path.join(workspace_dir, 'BestMan')
        cwd = os.getcwd()
        os.chdir(bestman_dir)

        bob_id = client.load_object(
            obj_name="bob", model_path=arm_path,
            object_position=[tx, 4.4, table_top_z + 0.04],
            object_orientation=[0, 0, 0], fixed_base=True
        )
        setattr(client, "bob_arm", bob_id)
        print(f"[机器人] ✅ Bob 固定臂 (xArm6) @ (2, 4) 桌面上")
        os.chdir(cwd)
    except Exception as e:
        print(f"[机器人] ⚠️ Bob: {e}")

    # 6. 机器人
    print("\n--- 创建机器人 ---")
    segbot_path = 'Asset/Robot/mobile_manipulator/base/segbot/urdf/segbot.urdf'
    arm_path = 'Asset/Robot/mobile_manipulator/arm/ufactory/urdf/xarm6.urdf'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    bestman_dir = os.path.join(workspace_dir, 'BestMan')
    cwd = os.getcwd()
    os.chdir(bestman_dir)

    # Alice: 移动操作臂 @ (5, 3)
    try:
        base_id = client.load_object(
            obj_name="alice_base", model_path=segbot_path,
            object_position=[5, 3, 0],
            object_orientation=[0, 0, 0], fixed_base=True
        )
        setattr(client, "alice_base", base_id)

        arm_z = 1.02
        arm_id = client.load_object(
            obj_name="alice_arm", model_path=arm_path,
            object_position=[5, 3, arm_z],
            object_orientation=[0, 0, 0], fixed_base=True
        )
        setattr(client, "alice_arm", arm_id)

        # 固定约束
        num_joints = p.getNumJoints(base_id)
        for i in range(num_joints):
            info = p.getJointInfo(base_id, i)
            child_name = info[12].decode('utf-8') if isinstance(info[12], bytes) else info[12]
            if 'plate' in child_name.lower():
                p.createConstraint(
                    parentBodyUniqueId=base_id, parentLinkIndex=i,
                    childBodyUniqueId=arm_id, childLinkIndex=-1,
                    jointType=p.JOINT_FIXED, jointAxis=[0,0,0],
                    parentFramePosition=[0,0,0.05], childFramePosition=[0,0,0],
                )
                print(f"[约束] ✅ Alice xArm6 固定到 plate_link")
                break
        print(f"[机器人] ✅ Alice 移动操作臂 @ (5, 3)")
    except Exception as e:
        print(f"[机器人] ⚠️ Alice: {e}")

    # David: 移动机器人 @ (4, 4)
    try:
        david_id = client.load_object(
            obj_name="david", model_path=segbot_path,
            object_position=[3, 6, 0],
            object_orientation=[0, 0, 0], fixed_base=True
        )
        setattr(client, "david", david_id)
        print(f"[机器人] ✅ David 移动机器人 @ (3, 6)")
    except Exception as e:
        print(f"[机器人] ⚠️ David: {e}")

    # Lucy: 无人机 @ (8,3) 中度灰色桌子上面
    try:
        cx, cy = 8, 3
        hover_z = 0.85
        body_half = [0.05, 0.04, 0.025]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=body_half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=body_half, rgbaColor=[0.2, 0.2, 0.2, 1])
        body_id = p.createMultiBody(0.05, col, vis, [cx, cy, hover_z])
        setattr(client, "drone_body", body_id)

        arm_len = 0.12
        arm_half = [arm_len, 0.01, 0.005]
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            a_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=arm_half)
            a_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=arm_half, rgbaColor=[0.5, 0.5, 0.5, 1])
            arm_id = p.createMultiBody(0, a_col, a_vis, [cx+dx*arm_len, cy+dy*arm_len, hover_z])
            setattr(client, f"drone_arm_{dx}_{dy}", arm_id)

        rotor_half = [0.04, 0.04, 0.003]
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            r_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=rotor_half)
            r_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=rotor_half, rgbaColor=[0.8, 0.8, 0.9, 0.6])
            rotor_id = p.createMultiBody(0, r_col, r_vis, [cx+dx*arm_len, cy+dy*arm_len, hover_z+0.02])
            setattr(client, f"drone_rotor_{dx}_{dy}", rotor_id)
        print(f"[机器人] ✅ Lucy 无人机 @ (8,3) 灰色桌面上方 z={hover_z}")
    except Exception as e:
        print(f"[机器人] ⚠️ Lucy: {e}")

    os.chdir(cwd)

    # 坐标轴
    axis_len = 1.5
    p.addUserDebugLine([0, 0, 0], [axis_len, 0, 0], [1, 0, 0])
    p.addUserDebugLine([0, 0, 0], [0, axis_len, 0], [0, 1, 0])
    p.addUserDebugLine([0, 0, 0], [0, 0, axis_len], [0, 0, 1])
    p.addUserDebugText("X", [axis_len + 0.3, 0, 0], [1, 0, 0])
    p.addUserDebugText("Y", [0, axis_len + 0.3, 0], [0, 1, 0])
    p.addUserDebugText("Z", [0, 0, axis_len + 0.3], [0, 0, 1])
    print("[场景] ✅ 坐标轴指示器")

    for _ in range(50):
        p.stepSimulation()

    print("\n" + "=" * 60)
    print("  场景3 搭建完成")
    print("=" * 60)


if __name__ == "__main__":
    print("请在 DynaHMRC 框架下调用此模块")
