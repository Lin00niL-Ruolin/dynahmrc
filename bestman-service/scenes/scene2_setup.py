"""
场景2 设置脚本 - 六面墙 (L型房间)
适合固体分类任务 (sort_solids)
"""

import os
import json
import math

import pybullet as p

# 房间参数
ROOM_X = 10.0   # 横向总长度
ROOM_Y = 10.0   # 纵向总宽度
ROOM_H = 2.8    # 墙高
WALL_T = 0.15   # 墙厚度

# 颜色
COLOR_WALL = [0.92, 0.92, 0.95, 1.0]  # 米白色
COLOR_FLOOR = [0.85, 0.75, 0.65, 1.0]  # 木色


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


def create_wood_floor(client):
    """铺木地板（覆盖整个 L 型区域）"""
    floor_thickness = 0.01
    gap = 0.005

    # 用两块地板组成 L 型
    # 主区域: 10m × 8m (x:0→10, y:0→8)
    main_x = ROOM_X - 2 * gap
    main_y = 8 - 2 * gap
    half1 = [main_x / 2, main_y / 2, floor_thickness / 2]
    center1 = [ROOM_X / 2, 4, floor_thickness / 2 + 0.005]

    col1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=half1)
    vis1 = p.createVisualShape(p.GEOM_BOX, halfExtents=half1, rgbaColor=COLOR_FLOOR)
    body1 = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=col1, baseVisualShapeIndex=vis1,
        basePosition=center1, baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
    )

    # 右上角小区域: 4m × 2m (x:6→10, y:8→10)
    alcove_x = 4 - 2 * gap
    alcove_y = 2 - 2 * gap
    half2 = [alcove_x / 2, alcove_y / 2, floor_thickness / 2]
    center2 = [8, 9, floor_thickness / 2 + 0.005]

    col2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=half2)
    vis2 = p.createVisualShape(p.GEOM_BOX, halfExtents=half2, rgbaColor=COLOR_FLOOR)
    body2 = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=col2, baseVisualShapeIndex=vis2,
        basePosition=center2, baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
    )

    # 加载木纹纹理
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tex_path = os.path.join(script_dir, "wood_texture.png")
    if os.path.exists(tex_path):
        tex_id = p.loadTexture(tex_path)
        p.changeVisualShape(body1, -1, textureUniqueId=tex_id)
        p.changeVisualShape(body2, -1, textureUniqueId=tex_id)

    setattr(client, "wood_floor", body1)
    setattr(client, "wood_floor_alcove", body2)
    print(f"[场景] ✅ 木地板已铺设 (10m×8m + 右上 4m×2m)")


def create_bed(client):
    """用几何体搭建一张3D床 (带枕头)"""
    s = 2.1
    bed_color = [0.25, 0.35, 0.45, 1.0]
    mattress_color = [0.85, 0.85, 0.90, 1.0]
    
    cx, cy = 8.5, 2.4
    yaw = 0
    
    def place(x, y, z):
        # 绕中心旋转 (cx, cy) 后 + 平移
        dx, dy = x - cx, y - cy
        rx = dx * math.cos(yaw) - dy * math.sin(yaw) + cx
        ry = dx * math.sin(yaw) + dy * math.cos(yaw) + cy
        return [rx, ry, z]
    
    fw, fl, fh = 0.75*s, 1.0*s, 0.15*s  # 床架
    mw, ml, mh = 0.7*s, 0.9*s, 0.1*s     # 床垫
    hw, hl, hh = 0.75*s, 0.05*s, 0.3*s   # 床头板
    bw, bl, bh = 0.7*s, 0.85*s, 0.04*s   # 被子
    
    # 床架
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[fw, fl, fh])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[fw, fl, fh], rgbaColor=bed_color)
    setattr(client, "bed_frame", p.createMultiBody(0, col, vis, place(cx, cy, fh), p.getQuaternionFromEuler([0, 0, yaw])))
    
    # 床垫
    col2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[mw, ml, mh])
    vis2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[mw, ml, mh], rgbaColor=mattress_color)
    setattr(client, "bed_mattress", p.createMultiBody(0, col2, vis2, place(cx, cy, fh*2 + mh), p.getQuaternionFromEuler([0, 0, yaw])))
    
    # 被子
    col3 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[bw, bl, bh])
    vis3 = p.createVisualShape(p.GEOM_BOX, halfExtents=[bw, bl, bh], rgbaColor=[0.7, 0.8, 0.9, 1.0])
    setattr(client, "bed_blanket", p.createMultiBody(0, col3, vis3, place(cx, cy, fh*2 + mh*2 + bh), p.getQuaternionFromEuler([0, 0, yaw])))
    
    # 床头板 (在床的 -Y 方向，旋转后)
    head_pos = place(cx, cy - 1.0*s - hl, fh + hh)
    col4 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hw, hl, hh])
    vis4 = p.createVisualShape(p.GEOM_BOX, halfExtents=[hw, hl, hh], rgbaColor=bed_color)
    setattr(client, "bed_headboard", p.createMultiBody(0, col4, vis4, head_pos, p.getQuaternionFromEuler([0, 0, yaw])))
    
    # 枕头 (长=床宽=2*fw, 宽=0.5, 放在被子上方)
    pw, pl, ph = fw, 0.4, 0.05
    pillow_pos = place(cx, cy - 1.0*s + 0.3, fh*2 + mh*2 + bh*2 + ph)
    col5 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[pw, pl, ph])
    vis5 = p.createVisualShape(p.GEOM_BOX, halfExtents=[pw, pl, ph], rgbaColor=[1, 1, 1, 1])
    setattr(client, "bed_pillow", p.createMultiBody(0, col5, vis5, pillow_pos, p.getQuaternionFromEuler([0, 0, yaw])))
    
    print(f"[场景] ✅ 3D床已搭建 @ ({cx}, {cy}) 旋转180° 放大{s}x")


def create_sort_objects(client):
    """在 (3,5) 桌子上放固定机械臂和6个彩色分类方块"""
    tx, ty = 3, 5
    table_top_z = 1.156
    box_size = 0.2
    
    # 切换到 BestMan 目录以便 load_object 能找到 Asset 路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    bestman_dir = os.path.join(workspace_dir, 'BestMan')
    original_cwd = os.getcwd()
    os.chdir(bestman_dir)
    
    # Bob 固定臂机器人 (Franka Panda + 底座)
    bob_path = 'Asset/Robot/mobile_manipulator/arm/ufactory/urdf/xarm6.urdf'
    try:
        # 先加一个底座方块
        base_half = [0.15, 0.15, 0.05]
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=base_half)
        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half, rgbaColor=[0.3, 0.3, 0.35, 1])
        bob_base = p.createMultiBody(0, base_col, base_vis, [tx, ty + 0.7, table_top_z])
        setattr(client, "bob_base", bob_base)
        
        # 机械臂装在底座上
        bob_id = client.load_object(
            obj_name="bob", model_path=bob_path,
            object_position=[tx, ty + 0.7, table_top_z + 0.05],
            object_orientation=[0, 0, 0], fixed_base=True
        )
        setattr(client, "bob_arm", bob_id)
        print(f"[分类] ✅ Bob 固定臂机器人 @ ({tx}, {ty+0.7})")
    except Exception as e:
        print(f"[分类] ⚠️ Bob: {e}")
    
    # 6个彩色方块 (2排3列)
    colors = [
        [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1],
        [1, 1, 0, 1], [0.5, 0, 0.5, 1], [1, 0.5, 0, 1],
    ]
    names = ["red", "green", "blue", "yellow", "purple", "orange"]
    
    for i, (color, name) in enumerate(zip(colors, names)):
        row, col = i // 3, i % 3
        x = tx - 0.3 + col * 0.3
        y = ty - 0.2 + row * 0.3
        half = box_size / 2
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=color)
        body_id = p.createMultiBody(
            baseMass=0.1, baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[x, y, table_top_z + box_size/2],
        )
        setattr(client, f"cube_{name}", body_id)
    print(f"[分类] ✅ 6个彩色方块 @ ({tx}, {ty}) 桌面上")
    
    # 6个小方块 (大的一半, 放在各家具上)
    small_size = box_size / 2
    # [x, y, z] 各小方块的位置
    small_positions = [
        [9.5, 6.0, 1.85],    # 红 → 台面上
        [1.0, 6.5, 2.2],      # 绿 → 书架最顶层
        [7.0, 8.6, 0.80],     # 蓝 → 沙发前边缘
        [8.5, 2.8, 1.3],      # 黄 → 床上
        [9.5, 7.5, 1.85],     # 紫 → 台面上(分开)
        [6, 2, 0.1],          # 橙 → 地毯上
    ]
    for i, (color, name) in enumerate(zip(colors, names)):
        x, y, z = small_positions[i]
        half = small_size / 2
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, half])
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=color)
        body_id = p.createMultiBody(
            baseMass=0.1, baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[x, y, z],
        )
        setattr(client, f"small_cube_{name}", body_id)
        print(f"[分类] ✅ 小{name}方块 @ ({x:.1f}, {y:.1f}, z={z:.2f})")
    
    os.chdir(original_cwd)


def create_robots(client):
    """添加移动操作机械臂机器人 Alice"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    bestman_dir = os.path.join(workspace_dir, 'BestMan')
    original_cwd = os.getcwd()
    os.chdir(bestman_dir)
    
    # Alice: Segbot 底盘 + Franka Panda 臂 @ (8, 6)
    segbot_path = 'Asset/Robot/mobile_manipulator/base/segbot/urdf/segbot.urdf'
    arm_path = 'Asset/Robot/mobile_manipulator/arm/ufactory/urdf/xarm6.urdf'
    
    try:
        # 底盘
        base_id = client.load_object(
            obj_name="alice_base", model_path=segbot_path,
            object_position=[8, 6, 0],
            object_orientation=[0, 0, 0], fixed_base=True
        )
        setattr(client, "alice_base", base_id)
        
        # 分析 Segbot URDF: base_footprint(z=0) → base_link(z=0.2165)→ cylinder→ plate
        # plate_link 顶面约在 z=1.02, Panda arm 装在此之上
        arm_z = 1.02
        
        # 手臂 (装在底盘上面的安装板上)
        arm_id = client.load_object(
            obj_name="alice_arm", model_path=arm_path,
            object_position=[8, 6, arm_z],
            object_orientation=[0, 0, 0], fixed_base=True
        )
        setattr(client, "alice_arm", arm_id)
        
        # 固定约束: 将 Panda 底座连接到 Segbot plate_link
        try:
            num_joints = p.getNumJoints(base_id)
            plate_idx = None
            for i in range(num_joints):
                info = p.getJointInfo(base_id, i)
                child_name = info[12].decode('utf-8') if isinstance(info[12], bytes) else info[12]
                if 'plate' in child_name.lower():
                    plate_idx = i
                    break
            if plate_idx is not None:
                p.createConstraint(
                    parentBodyUniqueId=base_id,
                    parentLinkIndex=plate_idx,
                    childBodyUniqueId=arm_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0.05],
                    childFramePosition=[0, 0, 0],
                )
                print(f"[约束] ✅ Panda 臂已固定到 Segbot plate_link")
        except Exception as ce:
            print(f"[约束] ⚠️ 固定约束失败: {ce}")
        
        print(f"[机器人] ✅ Alice 移动操作臂 @ (8, 6)")
        
        # David: Segbot 底盘 (纯移动, 无手臂) @ (8, 7)
        david_id = client.load_object(
            obj_name="david", model_path=segbot_path,
            object_position=[8, 7, 0],
            object_orientation=[0, 0, 0], fixed_base=True
        )
        setattr(client, "david", david_id)
        print(f"[机器人] ✅ David 移动机器人 @ (8, 7)")
    except Exception as e:
        print(f"[机器人] ⚠️ Alice: {e}")
    
    os.chdir(original_cwd)


def create_drone(client):
    """用几何体搭一个无人机, 停在桌子一 (1,4) 上方"""
    cx, cy = 1, 4
    hover_z = 1.5  # 悬浮高度
    
    # 机身 (小立方体)
    body_half = [0.1, 0.08, 0.05]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=body_half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=body_half, rgbaColor=[0.2, 0.2, 0.2, 1])
    body_id = p.createMultiBody(0.1, col, vis, [cx, cy, hover_z])
    setattr(client, "drone_body", body_id)
    
    # 4个旋臂 (十字形)
    arm_len = 0.25
    arm_half = [arm_len, 0.02, 0.01]
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        a_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=arm_half)
        a_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=arm_half, rgbaColor=[0.5, 0.5, 0.5, 1])
        arm_id = p.createMultiBody(0, a_col, a_vis, [cx + dx*arm_len, cy + dy*arm_len, hover_z])
        setattr(client, f"drone_arm_{dx}_{dy}", arm_id)
    
    # 4个旋翼 (小圆盘, 用薄长方体代替)
    rotor_half = [0.08, 0.08, 0.005]
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        r_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=rotor_half)
        r_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=rotor_half, rgbaColor=[0.8, 0.8, 0.9, 0.6])
        rotor_id = p.createMultiBody(0, r_col, r_vis, [cx + dx*arm_len, cy + dy*arm_len, hover_z + 0.03])
        setattr(client, f"drone_rotor_{dx}_{dy}", rotor_id)
    
    print(f"[场景] ✅ 无人机已搭建 @ ({cx}, {cy}) 悬停 z={hover_z}")


def setup_scene2(client, scene_json_path=None):
    """
    场景2：六面墙 L 型房间 + scene2.json 家具
    适合固体分类任务
    """
    print("\n" + "=" * 60)
    print("  场景2 搭建开始 (L型 10m×10m)")
    print("=" * 60)

    # 1. 六面墙
    print("\n--- 搭建六面墙 ---")

    # 墙1: (0,0) → (10,0)  下墙，沿 X 方向 10m
    create_wall(client, "wall_bottom",
                ROOM_X / 2, -WALL_T / 2, ROOM_H / 2,
                [ROOM_X + 2 * WALL_T, WALL_T, ROOM_H])
    print(f"[场景] ✅ 墙1: (0,0) → (10,0) 长度=10m")

    # 墙2: (10,0) → (10,10)  右墙，沿 Y 方向 10m
    create_wall(client, "wall_right",
                ROOM_X + WALL_T / 2, ROOM_Y / 2, ROOM_H / 2,
                [WALL_T, ROOM_Y, ROOM_H])
    print(f"[场景] ✅ 墙2: (10,0) → (10,10) 长度=10m")

    # 墙3: (10,10) → (6,10)  右上短墙，沿 X 方向 4m
    create_wall(client, "wall_top_right",
                8, ROOM_Y + WALL_T / 2, ROOM_H / 2,
                [4, WALL_T, ROOM_H])
    print(f"[场景] ✅ 墙3: (10,10) → (6,10) 长度=4m")

    # 墙4: (6,10) → (6,8)  内竖墙，沿 Y 方向 2m
    create_wall(client, "wall_inner_v",
                6, 9, ROOM_H / 2,
                [WALL_T, 2, ROOM_H],
                color=[0.7, 0.7, 0.72, 1.0])  # 灰色
    print(f"[场景] ✅ 墙4: (6,10) → (6,8) 长度=2m")

    # 墙5: (6,8) → (0,8)  内横墙，沿 X 方向 6m
    create_wall(client, "wall_inner_h",
                3, 8, ROOM_H / 2,
                [6, WALL_T, ROOM_H],
                color=[0.7, 0.7, 0.72, 1.0])  # 灰色
    print(f"[场景] ✅ 墙5: (6,8) → (0,8) 长度=6m")

    # 墙7: (5,0) → (5,4)  内竖墙，沿 Y 方向 4m
    create_wall(client, "wall_inner_v2",
                5, 2, ROOM_H / 2,
                [WALL_T, 4, ROOM_H],
                color=[0.7, 0.7, 0.72, 1.0])  # 灰色
    print(f"[场景] ✅ 墙7: (5,0) → (5,4) 长度=4m")

    # 墙6: (0,0) → (0,8)  左墙，沿 Y 方向 8m
    create_wall(client, "wall_left",
                -WALL_T / 2, 4, ROOM_H / 2,
                [WALL_T, 8, ROOM_H])
    print(f"[场景] ✅ 墙6: (0,0) → (0,8) 长度=8m")

    # 2. 黑色长方形板 (3m×4m, 贴底墙, 高1.6m)
    print("\n--- 搭建黑板 ---")
    board_half = [0.5, 1.0, 0.05]  # 局部: X=1m, Y=2m→Z高, Z=薄
    board_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=board_half)
    board_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=board_half, rgbaColor=[0, 0, 0, 1])
    board_id = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=board_col,
        baseVisualShapeIndex=board_vis,
        basePosition=[7, 0.05, 2.1],
        baseOrientation=p.getQuaternionFromEuler([math.pi/2, math.pi/2, 0])
    )
    setattr(client, "black_board", board_id)
    print(f"[场景] ✅ 黑色板: 1m×2m roll90° @ (7, 0) 高1.7m")

    # 2.5. 靠墙台面 (5m×1m, 高1.8m)
    print("\n--- 搭建靠墙台面 ---")
    # 平台从 x=9 延伸到 x=10（1m宽），y=5到y=10（5m长）
    shelf_half = [0.5, 2.5, 0.05]
    shelf_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=shelf_half)
    shelf_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=shelf_half, rgbaColor=[0.92, 0.92, 0.95, 1.0])
    shelf_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=shelf_col,
        baseVisualShapeIndex=shelf_vis,
        basePosition=[9.5, 7.5, 1.8],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
    )
    setattr(client, "shelf_table", shelf_id)
    print(f"[场景] ✅ 靠墙台面: (10,5)→(10,10) 5m×1m 高1.8m")

    # 3. 木地板
    print("\n--- 铺设木地板 ---")
    create_wood_floor(client)

    # 2.5 用几何体搭一个3D床 (放在右上角 alcove 区域)
    print("\n--- 搭建3D床 ---")
    create_bed(client)
    
    # 3. 坐标轴指示器
    axis_len = 1.5
    p.addUserDebugLine([0, 0, 0], [axis_len, 0, 0], [1, 0, 0])
    p.addUserDebugLine([0, 0, 0], [0, axis_len, 0], [0, 1, 0])
    p.addUserDebugLine([0, 0, 0], [0, 0, axis_len], [0, 0, 1])
    p.addUserDebugText("X", [axis_len + 0.3, 0, 0], [1, 0, 0])
    p.addUserDebugText("Y", [0, axis_len + 0.3, 0], [0, 1, 0])
    p.addUserDebugText("Z", [0, 0, axis_len + 0.3], [0, 0, 1])
    print("[场景] ✅ 坐标轴指示器")

    # 4. 加载 scene2.json
    if scene_json_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        scene_json_path = os.path.join(script_dir, "scene2.json")

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

                    client.load_object(
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
        print("[场景] ⚠️ 未找到 scene2.json")

    # 5. 创建分类物品 (Bob + 彩色方块)
    print("\n--- 搭建分类任务物品 ---")
    create_sort_objects(client)
    create_robots(client)
    create_drone(client)

    for _ in range(50):
        p.stepSimulation()

    print("\n" + "=" * 60)
    print("  场景2 搭建完成！")
    print("=" * 60)


if __name__ == "__main__":
    print("请在 DynaHMRC 框架下调用此模块")
