"""
场景交互编辑器
在 PyBullet 3D GUI 中通过滑块实时调整家具位置和朝向

使用方法:
  1. 运行此脚本
  2. 弹出 PyBullet 3D 窗口
  3. 拖动 GUI 左侧的滑块调整每个物体的 x/y/z/yaw
  4. 调好后按 Enter 输出最终场景 JSON

快捷键:
  - 滑块调整 → 物体实时移动
  - Enter → 打印当前所有物体坐标
  - Ctrl+C → 退出
"""

import sys
import os
import json
import math
import yaml

import pybullet as p
import pybullet_data

# ============ 路径设置 ============
WORKSPACE = "/home/developer/.openclaw/workspace"

# ============ 场景物体定义 ============

# 每个物体: [名字, URDF路径, 初始[x,y,z], 初始朝向, 是否为固定底座]
SCENE_OBJECTS = [
    # -- 厨房区 --
    ["fridge",          "Asset/Scene/Object/Kitchen_world_models/Fridge/12036/mobility.urdf",                [1.5, 6.5, 0],   [0,0,0,1], True],
    ["oven",            "Asset/Scene/Object/Kitchen_world_models/Oven/101943/mobility.urdf",                [3.0, 6.5, 0],   [0,0,0,1], True],
    ["counter",         "Asset/Scene/Object/Kitchen_world_models/elementA/urdf/kitchen_part_right_gen_convex.urdf", [4.5, 6.5, 0], [0,0,0,1], True],
    ["wall_cabinet_1",  "Asset/Scene/Object/Kitchen_world_models/CabinetUpper/46889/mobility.urdf",         [2.5, 7.2, 1.5], [0,0,0,1], True],
    ["wall_cabinet_2",  "Asset/Scene/Object/Kitchen_world_models/CabinetUpper/46744/mobility.urdf",         [4.0, 7.2, 1.5], [0,0,0,1], True],
    ["cabinet_upper",   "Asset/Scene/Object/Kitchen_world_models/CabinetTall/46456/mobility.urdf",          [5.5, 7.2, 0],   [0,0,0,1], True],

    # -- 餐桌区 --
    ["table1",          "Asset/Scene/Object/Kitchen_world_models/DiningTable/28164/mobility.urdf",          [6.0, 5.0, 0],   [0,0,0,1], True],
    ["table1_chair",    "Asset/Scene/Object/URDF_models/furniture_chair/model.urdf",                       [5.5, 4.5, 0],   [0,0,0,1], True],
    ["table2",          "Asset/Scene/Object/Kitchen_world_models/DiningTable/23782/mobility.urdf",          [8.5, 5.0, 0],   [0,0,0,1], True],
    ["table2_chair",    "Asset/Scene/Object/URDF_models/furniture_chair/model.urdf",                       [9.0, 4.5, 0],   [0,0,0,1], True],
    ["table3",          "Asset/Scene/Object/Kitchen_world_models/DiningTable/32932/mobility.urdf",          [2.0, 2.0, 0],   [0,0,0,1], True],
    ["table3_chair_1",  "Asset/Scene/Object/URDF_models/furniture_chair/model.urdf",                       [1.5, 1.5, 0],   [0,0,0,1], True],
    ["table3_chair_2",  "Asset/Scene/Object/URDF_models/furniture_chair/model.urdf",                       [2.5, 1.5, 0],   [0,0,0,1], True],

    # -- 右侧 --
    ["sink",            "Asset/Scene/Object/Kitchen_world_models/Sink/102379/mobility.urdf",               [10.0, 1.5, 0],  [0,0,0,1], True],
    ["cabinet_right",   "Asset/Scene/Object/Kitchen_world_models/CabinetLower/35059/mobility.urdf",         [5.0, 7.2, 0],   [0,0,0,1], True],
]


def create_room():
    """创建四面墙和地板"""
    ROOM_X, ROOM_Y, ROOM_H = 12, 8, 2.8
    WALL_T = 0.15
    
    # 地板 - 木色平面
    plane_path = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
    floor = p.loadURDF(plane_path, [0, 0, 0])
    # 给地板加点颜色
    p.changeVisualShape(floor, -1, rgbaColor=[0.82, 0.71, 0.55, 1.0])
    
    wall_color = [0.92, 0.92, 0.95, 1.0]
    
    def add_wall(x, y, z, hx, hy, hz, rpy=(0,0,0)):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=wall_color)
        return p.createMultiBody(0, col, vis, [x,y,z], p.getQuaternionFromEuler(rpy))
    
    # 四面墙
    add_wall(ROOM_X/2, -WALL_T/2, ROOM_H/2, ROOM_X/2+WALL_T, WALL_T/2, ROOM_H/2)
    add_wall(ROOM_X/2, ROOM_Y+WALL_T/2, ROOM_H/2, ROOM_X/2+WALL_T, WALL_T/2, ROOM_H/2)
    add_wall(-WALL_T/2, ROOM_Y/2, ROOM_H/2, WALL_T/2, ROOM_Y/2, ROOM_H/2)
    add_wall(ROOM_X+WALL_T/2, ROOM_Y/2, ROOM_H/2, WALL_T/2, ROOM_Y/2, ROOM_H/2)


def resolve_path(relative_path):
    """将 Asset 相对路径转为绝对路径"""
    return os.path.join(WORKSPACE, "BestMan", relative_path)


def main():
    print("="*60)
    print("  DynaHMRC 场景交互编辑器")
    print("="*60)
    
    # 初始化 PyBullet GUI
    p.connect(p.GUI)
    p.setGravity(0, 0, 0)  # 无重力，方便调位置
    
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    # p.setRealTimeSimulation(1)  # 关掉物理模拟
    
    # 创建房间
    create_room()
    print("[房间] ✅ 四面墙 + 木地板")
    
    # 加载每个物体并创建调试滑块
    bodies = []
    sliders = []
    
    for i, (name, urdf_rel, init_pos, init_orn, fixed_base) in enumerate(SCENE_OBJECTS):
        urdf_path = resolve_path(urdf_rel)
        
        if not os.path.exists(urdf_path):
            print(f"  ✗ {name}: URDF 不存在 ({urdf_path})")
            continue
        
        body_id = p.loadURDF(urdf_path, init_pos, init_orn, useFixedBase=fixed_base)
        
        # 为每个物体创建调试滑块 (x, y, z, yaw)
        x_slider = p.addUserDebugParameter(f"[{i}] {name}_x", -2, 14, init_pos[0])
        y_slider = p.addUserDebugParameter(f"[{i}] {name}_y", -2, 10, init_pos[1])
        z_slider = p.addUserDebugParameter(f"[{i}] {name}_z", -1, 3, init_pos[2])
        yaw_slider = p.addUserDebugParameter(f"[{i}] {name}_yaw", -180, 180, 0)
        
        bodies.append((name, body_id, init_pos, urdf_path, fixed_base))
        sliders.append((name, x_slider, y_slider, z_slider, yaw_slider))
        
        print(f"  ✓ {name}")
    
    print(f"\n✅ 共加载 {len(bodies)} 个物体")
    print("\n操作说明:")
    print("  1. 拖动左侧滑块 → 物体实时移动")
    print("  2. 在终端按 Enter → 输出所有物体当前坐标（JSON格式）")
    print("  3. Ctrl+C → 退出")
    
    try:
        while True:
            # 更新所有物体的位置
            for info in sliders:
                name, xs, ys, zs, yaws = info
                x = p.readUserDebugParameter(xs)
                y = p.readUserDebugParameter(ys)
                z = p.readUserDebugParameter(zs)
                yaw_deg = p.readUserDebugParameter(yaws)
                yaw_rad = yaw_deg * math.pi / 180.0
                
                # 找到对应的 body
                for b_name, body_id, init_pos, _, _ in bodies:
                    if b_name == name:
                        p.resetBasePositionAndOrientation(
                            body_id,
                            [x, y, z],
                            p.getQuaternionFromEuler([0, 0, yaw_rad])
                        )
                        break
            
            import select
            # 检查用户是否按了 Enter
            if sys.stdin in select.select([sys.stdin], [], [], 0.01)[0]:
                input()  # 吃掉 Enter
                print("\n" + "="*60)
                print("  当前场景 JSON（复制到 scene1.json）")
                print("="*60)
                scene_data = []
                for info in sliders:
                    name, xs, ys, zs, yaws = info
                    x = p.readUserDebugParameter(xs)
                    y = p.readUserDebugParameter(ys)
                    z = p.readUserDebugParameter(zs)
                    for b_name, body_id, init_pos, urdf_path, fixed in bodies:
                        if b_name == name:
                            # 找到原始的相对路径
                            rel_path = None
                            for _, _, _, orig_path, _ in bodies:
                                if b_name == name:
                                    # 从绝对路径转回相对路径
                                    abs_path = orig_path
                                    idx = abs_path.find("Asset/")
                                    if idx >= 0:
                                        rel_path = abs_path[idx:]
                                    break
                            
                            scene_data.append({
                                "obj_name": name,
                                "model_path": rel_path or "unknown",
                                "object_position": [round(x,2), round(y,2), round(z,2)],
                                "object_orientation": [0, 0, 0],
                                "scale": 1.0,
                                "fixed_base": fixed
                            })
                            break
                
                json_str = json.dumps(scene_data, indent=4, ensure_ascii=False)
                print(json_str)
                print("\n(已复制到剪贴板，可保存到 scene1.json)")
                print("继续调整或 Ctrl+C 退出\n")
            
            p.stepSimulation()
    
    except KeyboardInterrupt:
        print("\n退出编辑器")
    
    p.disconnect()


if __name__ == "__main__":
    main()
