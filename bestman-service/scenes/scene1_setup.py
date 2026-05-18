"""
场景1 设置脚本 (新)
房间：10m × 8m，四面外墙 + 两面内墙

外墙布局（10m × 8m 房间）：
  左上 (0,0)  右上 (10,0)
  左下 (0,8)  右下 (10,8)

内墙布局：
  水平墙: (7,4) → (10,4)  在 y=4 处，x 从 7 到 10
  垂直墙: (6,0) → (6,5)  在 x=6 处，y 从 0 到 5

家具：
  冰箱 (1,1)、烤箱 (3,0.5)、小橱柜 (5,1)
  table1(2.5,4)+chair1、table2(2.5,6)+chair2(+固定臂)
  移动臂 (4,7)、书架 (8,1)+无人机、桌子(7,2)+两椅子、洗手池(9,6)
"""

import os
import sys
import json
import math

import pybullet as p

# 房间参数
ROOM_X = 10.0   # 横向长度
ROOM_Y = 8.0    # 纵向宽度
ROOM_H = 2.8    # 墙高
WALL_T = 0.15   # 墙厚度

# 颜色定义
COLOR_WALL = [0.92, 0.92, 0.95, 1.0]       # 米白色墙
COLOR_INNER_WALL = [0.7, 0.7, 0.72, 1.0]   # 灰色内墙
COLOR_FLOOR = [0.85, 0.75, 0.65, 1.0]      # 木色地板


def resolve_asset_path(relative_path):
    """从项目根目录解析资产路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    bestman_dir = os.path.join(workspace_dir, 'BestMan')
    
    if relative_path.startswith("Asset/"):
        return os.path.join(bestman_dir, relative_path)
    return os.path.join(bestman_dir, "Asset", relative_path)


def create_wall(client, wall_name, x, y, z, size, color=COLOR_WALL, rpy=(0, 0, 0)):
    """用 pybullet 几何体创建一面墙"""
    half = [s/2 for s in size]
    
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


def create_floor(client):
    """创建 10m × 8m 的木色地板（比默认 plane 高一点避免闪烁）"""
    floor_z = 0.01  # 略高于默认 plane
    
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[ROOM_X/2, ROOM_Y/2, 0.01])
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[ROOM_X/2, ROOM_Y/2, 0.01],
                                  rgbaColor=COLOR_FLOOR)
    floor_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[ROOM_X/2, ROOM_Y/2, floor_z],
        baseOrientation=[0, 0, 0, 1]
    )
    setattr(client, "floor_10x8", floor_id)
    print(f"[场景] ✅ 地板已创建: 10m × 8m, 位置({ROOM_X/2}, {ROOM_Y/2}, {floor_z})")
    return floor_id


def setup_scene1(client, scene_json_path=None):
    """
    设置场景1的完整布局
    
    Args:
        client: BestMan.Client 实例
        scene_json_path: scene1.json 路径，None 则自动定位
    """
    print("\n" + "="*60)
    print("  场景1 搭建开始 (10m × 8m)")
    print("="*60)

    # 1. 创建房间墙壁
    print("\n--- 搭建房间结构 ---")
    
    # 下墙 (y ≈ 0)
    create_wall(client, "wall_bottom", ROOM_X/2, -WALL_T/2, ROOM_H/2, 
                [ROOM_X + 2*WALL_T, WALL_T, ROOM_H])
    
    # 上墙 (y ≈ ROOM_Y)
    create_wall(client, "wall_top", ROOM_X/2, ROOM_Y + WALL_T/2, ROOM_H/2,
                [ROOM_X + 2*WALL_T, WALL_T, ROOM_H])
    
    # 左墙 (x ≈ 0)
    create_wall(client, "wall_left", -WALL_T/2, ROOM_Y/2, ROOM_H/2,
                [WALL_T, ROOM_Y, ROOM_H])
    
    # 右墙 (x ≈ ROOM_X)
    create_wall(client, "wall_right", ROOM_X + WALL_T/2, ROOM_Y/2, ROOM_H/2,
                [WALL_T, ROOM_Y, ROOM_H])
    
    print("[场景] ✅ 四面外墙已创建")

    # 2. 创建内部隔墙
    print("\n--- 搭建内部隔墙 ---")
    
    # 水平内墙: (7,4) → (10,4)，3m 长
    inner_wall_h_len = 3.0
    create_wall(client, "wall_inner_h", 
                7 + inner_wall_h_len/2, 4, ROOM_H/2,
                [inner_wall_h_len, WALL_T, ROOM_H],
                color=COLOR_INNER_WALL)
    print(f"[场景] ✅ 水平内墙: (7,4) → (10,4) 长度={inner_wall_h_len}m")
    
    # 垂直内墙: (6,0) → (6,5)，5m 长
    inner_wall_v_len = 5.0
    create_wall(client, "wall_inner_v",
                6, inner_wall_v_len/2, ROOM_H/2,
                [WALL_T, inner_wall_v_len, ROOM_H],
                color=COLOR_INNER_WALL)
    print(f"[场景] ✅ 垂直内墙: (6,0) → (6,5) 长度={inner_wall_v_len}m")

    # 3. 创建地板 (覆盖默认 plane，10m×8m)
    print("\n--- 创建地板 ---")
    create_floor(client)

    # 4. 加载场景 JSON (URDF 家具)
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
        print(f"[场景] ⚠️ 场景文件未找到: {scene_json_path}")

    # 5. 运行几步让物理引擎稳定
    for _ in range(50):
        p.stepSimulation()
    
    print("\n" + "="*60)
    print("  场景1 搭建完成！")
    print("="*60)


if __name__ == "__main__":
    print("请在 DynaHMRC 框架下调用此模块")
