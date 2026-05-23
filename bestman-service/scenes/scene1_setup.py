"""
场景1 设置脚本 - 四面墙 (10m × 8m)
"""

import os
import json
import math

import pybullet as p

# 房间参数
ROOM_X = 10.0   # 横向长度
ROOM_Y = 8.0    # 纵向宽度
ROOM_H = 2.8    # 墙高
WALL_T = 0.15   # 墙厚度

# 颜色
COLOR_WALL = [0.92, 0.92, 0.95, 1.0]  # 米白色


def resolve_asset_path(relative_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    bestman_dir = os.path.join(workspace_dir, 'BestMan')
    if relative_path.startswith("Asset/"):
        return os.path.join(bestman_dir, relative_path)
    return os.path.join(bestman_dir, "Asset", relative_path)


def create_wall(client, wall_name, x, y, z, size, color=COLOR_WALL, rpy=(0, 0, 0)):
    """用 pybullet 几何体创建一面墙"""
    half = [s / 2 for s in size]

    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[x, y, z],
        baseOrientation=p.getQuaternionFromEuler(rpy)
    )
    setattr(client, wall_name, body_id)
    return body_id


COLOR_FLOOR = [0.85, 0.75, 0.65, 1.0]      # 木色地板

def create_wood_floor(client):
    """在四面墙内侧铺木地板，几乎贴满墙根"""
    floor_thickness = 0.01
    # 只留极小的缝隙防 z-fighting
    gap = 0.005
    floor_x = ROOM_X - 2 * gap
    floor_y = ROOM_Y - 2 * gap

    half = [floor_x / 2, floor_y / 2, floor_thickness / 2]
    center = [ROOM_X / 2, ROOM_Y / 2, floor_thickness / 2 + 0.005]

    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=COLOR_FLOOR)
    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=center,
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
    )
    
    # 加载木纹纹理贴图
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tex_path = os.path.join(script_dir, "wood_texture.png")
    if os.path.exists(tex_path):
        tex_id = p.loadTexture(tex_path)
        p.changeVisualShape(body_id, -1, textureUniqueId=tex_id)
        print(f"[场景] ✅ 木纹纹理已加载")
    
    setattr(client, "wood_floor", body_id)
    print(f"[场景] ✅ 木地板已铺设 ({floor_x:.1f}m × {floor_y:.1f}m)")
    return body_id


def setup_scene1(client, scene_json_path=None):
    """
    场景1：四面墙 (10m × 8m)，+ scene1.json 家具
    """
    print("\n" + "=" * 60)
    print("  场景1 搭建开始 (10m × 8m 四面墙)")
    print("=" * 60)

    # 1. 四面外墙
    print("\n--- 搭建四面墙 ---")

    # 下墙 (y ≈ 0)
    create_wall(client, "wall_bottom",
                ROOM_X / 2, -WALL_T / 2, ROOM_H / 2,
                [ROOM_X + 2 * WALL_T, WALL_T, ROOM_H])

    # 上墙 (y ≈ ROOM_Y)
    create_wall(client, "wall_top",
                ROOM_X / 2, ROOM_Y + WALL_T / 2, ROOM_H / 2,
                [ROOM_X + 2 * WALL_T, WALL_T, ROOM_H])

    # 左墙 (x ≈ 0)
    create_wall(client, "wall_left",
                -WALL_T / 2, ROOM_Y / 2, ROOM_H / 2,
                [WALL_T, ROOM_Y, ROOM_H])

    # 右墙 (x ≈ ROOM_X)
    create_wall(client, "wall_right",
                ROOM_X + WALL_T / 2, ROOM_Y / 2, ROOM_H / 2,
                [WALL_T, ROOM_Y, ROOM_H])

    print("[场景] ✅ 四面外墙 (10m × 8m) 已创建")

    # 1.2 内部隔墙
    print("\n--- 搭建内部隔墙 ---")

    # 墙1: (5,0) → (5,5)  垂直墙，沿 Y 方向 5m
    create_wall(client, "wall_inner_v1",
                5, 2.5, ROOM_H / 2,
                [WALL_T, 5, ROOM_H])
    print(f"[场景] ✅ 垂直墙1: (5,0) → (5,5) 长度=5m")

    # 墙2: (3,4) → (0,4)  水平墙，沿 X 方向 3m
    create_wall(client, "wall_inner_h1",
                1.5, 4, ROOM_H / 2,
                [3, WALL_T, ROOM_H])
    print(f"[场景] ✅ 水平墙2: (3,4) → (0,4) 长度=3m")

    # 墙3: (5,7) → (5,8)  垂直短墙，沿 Y 方向 1m
    create_wall(client, "wall_inner_v2",
                5, 7.5, ROOM_H / 2,
                [WALL_T, 1, ROOM_H])
    print(f"[场景] ✅ 垂直墙3: (5,7) → (5,8) 长度=1m")

    # 1.3 木地板
    print("\n--- 铺设木地板 ---")
    create_wood_floor(client)

    # 1.5 坐标轴指示器 (红X 绿Y 蓝Z)
    axis_len = 1.5
    p.addUserDebugLine([0, 0, 0], [axis_len, 0, 0], [1, 0, 0])     # X 红
    p.addUserDebugLine([0, 0, 0], [0, axis_len, 0], [0, 1, 0])     # Y 绿
    p.addUserDebugLine([0, 0, 0], [0, 0, axis_len], [0, 0, 1])     # Z 蓝
    # 加文字标签
    p.addUserDebugText("X", [axis_len + 0.3, 0, 0], [1, 0, 0])
    p.addUserDebugText("Y", [0, axis_len + 0.3, 0], [0, 1, 0])
    p.addUserDebugText("Z", [0, 0, axis_len + 0.3], [0, 0, 1])
    print("[场景] ✅ 坐标轴指示器 (红X 绿Y 蓝Z)")

    # 2. 加载 scene1.json 中的物体
    if scene_json_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        scene_json_path = os.path.join(script_dir, "scene1.json")

    if os.path.exists(scene_json_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
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
                    print(f"  ✓ {obj_name} @ ({pos[0]}, {pos[1]}, {pos[2]})")
                    loaded += 1
                except Exception as e:
                    print(f"  ✗ {obj.get('obj_name', '?')}: {e}")

            print(f"[场景] ✅ 已加载 {loaded}/{len(scene_data)} 个物体")
        finally:
            os.chdir(original_cwd)
    else:
        print("[场景] ⚠️ 未找到 scene1.json，仅创建了四面墙")

    # 6. 固定机械臂 (UR5e) 在 table_new_2 (8.5, 5.8) 上
    print("\n--- 固定机械臂 ---")
    bob_arm_path = 'Asset/Robot/mobile_manipulator/arm/ufactory/urdf/xarm6.urdf'
    tx, ty = 8.5, 5.85
    table_top_z = 0.86
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        bestman_dir = os.path.join(workspace_dir, 'BestMan')
        cwd = os.getcwd()
        os.chdir(bestman_dir)

        bob_id = client.load_object(
            obj_name="bob", model_path=bob_arm_path,
            object_position=[tx, ty, table_top_z],
            object_orientation=[0, 0, 0],
            scale=1.0, fixed_base=True
        )
        setattr(client, "bob_arm", bob_id)
        print(f"[机器人] ✅ Bob 固定臂 (UR5e) @ ({tx}, {ty}) 桌面上")
        os.chdir(cwd)
    except Exception as e:
        print(f"[机器人] ⚠️ Bob: {e}")

    # 7. 移动操作机械臂在 (7, 7)
    print("\n--- 移动操作机械臂 ---")
    segbot_path = 'Asset/Robot/mobile_manipulator/base/segbot/urdf/segbot.urdf'
    new_arm_path = 'Asset/Robot/mobile_manipulator/arm/ufactory/urdf/xarm6.urdf'
    rx, ry = 6.5, 7
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        bestman_dir = os.path.join(workspace_dir, 'BestMan')
        cwd = os.getcwd()
        os.chdir(bestman_dir)

        # Segbot base
        base_id = client.load_object(
            obj_name="new_robot", model_path=segbot_path,
            object_position=[rx, ry, 0],
            object_orientation=[0, 0, 0],
            scale=1.0, fixed_base=True
        )
        setattr(client, "new_robot_base", base_id)
        print(f"  ✓ 底座 @ ({rx}, {ry})")

        # xArm6 arm on top of segbot (height 1.02 matching scene3)
        arm_z = 1.02
        arm_id = client.load_object(
            obj_name="new_robot_arm", model_path=new_arm_path,
            object_position=[rx, ry, arm_z],
            object_orientation=[0, 0, 0],
            scale=1.0, fixed_base=True
        )
        setattr(client, "new_robot_arm", arm_id)
        print(f"  ✓ xArm6 机械臂 @ ({rx}, {ry}, {arm_z})")

        # 固定约束：将机械臂固定在底座上
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
                print(f"  ✓ 约束: arm → {child_name}")

        print(f"[机器人] ✅ 移动操作机械臂 @ ({rx}, {ry})")
        os.chdir(cwd)
    except Exception as e:
        print(f"[机器人] ⚠️ 移动操作臂: {e}")

    # 8. 用 PyBullet 几何体创建案板（长方体）
    print("\n--- 案板 ---")
    try:
        board_half = [0.35, 0.25, 0.02]  # 70cm x 50cm x 4cm
        board_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=board_half)
        board_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=board_half,
            rgbaColor=[0.8, 0.7, 0.5, 1.0]  # 浅木色底
        )
        board_id = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=board_col,
            baseVisualShapeIndex=board_vis,
            basePosition=[8.5, 5.5, 0.86],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        # 加载木纹纹理
        tex_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wood_texture.png")
        if os.path.exists(tex_path):
            tex_id = p.loadTexture(tex_path)
            p.changeVisualShape(board_id, -1, textureUniqueId=tex_id)
        setattr(client, "cutting_board", board_id)
        print(f"  ✓ 案板 (70cmx50cmx4cm) 带木纹 @ (8.5, 5.5, 0.86)")
    except Exception as e:
        print(f"  ⚠️ 案板: {e}")

    # 9. 食物物品（3件，堆叠到案板上）
    # 桌子二上的物品（Bob 可以够到）
    print("\n--- ham_bottom (桌子二) ---")
    try:
        meat_half = [0.08, 0.03, 0.005]
        meat_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=meat_half)
        ham_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=meat_half, rgbaColor=[0.92, 0.25, 0.15, 1.0])
        ham_bottom = p.createMultiBody(0.1, meat_col, ham_vis, [8.2, 5.5, 0.86], p.getQuaternionFromEuler([0, 0, -0.1]))
        setattr(client, "ham_bottom", ham_bottom)
        print(f"  ✓ ham_bottom @ (8.2, 5.5, 0.86) — 桌子二")
    except Exception as e:
        print(f"  ⚠️ ham_bottom: {e}")

    # 桌子一上的物品（需要其他机器人送到案板，Bob 够不到）
    print("\n--- bacon (桌子一) ---")
    try:
        bacon_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=meat_half, rgbaColor=[0.92, 0.25, 0.15, 1.0])
        bacon = p.createMultiBody(0.1, meat_col, bacon_vis, [8.5, 4, 0.86], p.getQuaternionFromEuler([0, 0, 0.3]))
        setattr(client, "bacon", bacon)
        print(f"  ✓ bacon @ (8.5, 4, 0.86) — 桌子一")
    except Exception as e:
        print(f"  ⚠️ bacon: {e}")

    print("\n--- ham_top (桌子一) ---")
    try:
        ham_top_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=meat_half, rgbaColor=[0.85, 0.30, 0.20, 1.0])
        ham_top = p.createMultiBody(0.1, meat_col, ham_top_vis, [8.55, 4.2, 0.86], p.getQuaternionFromEuler([0, 0, 0.1]))
        setattr(client, "ham_top", ham_top)
        print(f"  ✓ ham_top @ (8.55, 4.2, 0.86) — 桌子一")
    except Exception as e:
        print(f"  ⚠️ ham_top: {e}")
    except Exception as e:
        print(f"  ⚠️ 面包片: {e}")

    # 11. David — 纯移动底座 (segbot 无手臂) @ (4, 6)
    print("\n--- David 移动底座 ---")
    dx, dy = 4.0, 6.0
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        bestman_dir = os.path.join(workspace_dir, 'BestMan')
        cwd = os.getcwd()
        os.chdir(bestman_dir)

        david_id = client.load_object(
            obj_name="david",
            model_path='Asset/Robot/mobile_manipulator/base/segbot/urdf/segbot.urdf',
            object_position=[dx, dy, 0],
            object_orientation=[0, 0, 0],
            scale=1.0, fixed_base=True
        )
        setattr(client, "david", david_id)
        print(f"[机器人] ✅ David 移动底座 @ ({dx}, {dy})")
        os.chdir(cwd)
    except Exception as e:
        print(f"[机器人] ⚠️ David: {e}")

    # 12. Lucy — 无人机 (场景三风格, 放在 obj_name=table 的桌子上面)
    print("\n--- Lucy 无人机 (场景三风格) ---")
    cx, cy = 3.0, 2.0
    hover_z = 0.86  # 桌子高度
    try:
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
        print(f"[机器人] ✅ Lucy 无人机 (场景三风格) @ ({cx}, {cy}) 桌面上方 z={hover_z}")
    except Exception as e:
        print(f"[机器人] ⚠️ Lucy: {e}")

    for _ in range(50):
        p.stepSimulation()

    print("\n" + "=" * 60)
    print("  场景1 搭建完成！(Alice @ new_robot, Bob @ bob_arm, David, Lucy)")
    print("=" * 60)


if __name__ == "__main__":
    print("请在 DynaHMRC 框架下调用此模块")
