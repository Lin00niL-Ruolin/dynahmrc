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
                [WALL_T, 5, ROOM_H],
                color=[0.7, 0.7, 0.72, 1.0])  # 灰色
    print(f"[场景] ✅ 垂直墙1: (5,0) → (5,5) 长度=5m")

    # 墙2: (3,4) → (0,4)  水平墙，沿 X 方向 3m
    create_wall(client, "wall_inner_h1",
                1.5, 4, ROOM_H / 2,
                [3, WALL_T, ROOM_H],
                color=[0.7, 0.7, 0.72, 1.0])  # 灰色
    print(f"[场景] ✅ 水平墙2: (3,4) → (0,4) 长度=3m")

    # 墙3: (5,7) → (5,8)  垂直短墙，沿 Y 方向 1m
    create_wall(client, "wall_inner_v2",
                5, 7.5, ROOM_H / 2,
                [WALL_T, 1, ROOM_H],
                color=[0.7, 0.7, 0.72, 1.0])  # 灰色
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

    for _ in range(50):
        p.stepSimulation()

    print("\n" + "=" * 60)
    print("  场景1 搭建完成！")
    print("=" * 60)


if __name__ == "__main__":
    print("请在 DynaHMRC 框架下调用此模块")
