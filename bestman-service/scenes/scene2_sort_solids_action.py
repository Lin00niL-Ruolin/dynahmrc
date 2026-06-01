#!/usr/bin/env python3
"""
scene2_sort_solids_action.py — v3
绿色+蓝色方块分类任务

流程：
1. David（移动机器人）→ 书架旁停留2秒 → 回起点
2a. Lucy（无人机）→ 慢速起飞 → 书架取绿 → 放桌左
2b. Alice（移动操作臂）→ 捡蓝 → 放桌右
3. Bob（固定臂）→ 自然伸缩放置绿到 cube_green → 放置蓝到 cube_blue
"""
import os, sys, math, time, pybullet as p

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
bestman_dir = os.path.join(workspace_dir, 'BestMan')
sys.path.insert(0, bestman_dir)
sys.path.insert(0, os.path.dirname(script_dir))
from Env.Client import Client
from scenes.scene2_setup import setup_scene2
from scenes.path_planner import AStarPathPlanner

_grasp = {}


def load_yaml_config(cfg_path):
    import yaml
    from types import SimpleNamespace
    def d2ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: d2ns(v) for k, v in d.items()})
        return d
    fp = cfg_path
    if not os.path.exists(fp):
        fp = os.path.join(bestman_dir, cfg_path)
    with open(fp) as f:
        return d2ns(yaml.safe_load(f))


# ============================================================
#  启动
# ============================================================
print("=" * 60)
print("  Scene 2: Sort Solids v3")
print("=" * 60)

cfg = load_yaml_config('Config/default.yaml')
cfg.Client.enable_GUI = bool(os.environ.get('DISPLAY'))
print(f"[1] PyBullet ({'GUI' if cfg.Client.enable_GUI else 'DIRECT'})…")
client = Client(cfg.Client)

print("[2] 加载场景2…")
setup_scene2(client, os.path.join(script_dir, 'scene2.json'))
for _ in range(50):
    p.stepSimulation()

print("[3] 路径规划器…")
planner = AStarPathPlanner(scene='kitchen', grid_size=0.15, robot_radius=0.3)
time.sleep(2)

# ============================================================
#  扫描
# ============================================================
robot = {}
for a in ['bob_arm', 'alice_base', 'alice_arm', 'david',
          'drone_body', 'drone_arm_1_0', 'drone_arm_-1_0',
          'drone_arm_0_1', 'drone_arm_0_-1',
          'drone_rotor_1_0', 'drone_rotor_-1_0',
          'drone_rotor_0_1', 'drone_rotor_0_-1']:
    v = getattr(client, a, None)
    if isinstance(v, int) and v > 0:
        robot[a] = v

objects = {}
for a in dir(client):
    v = getattr(client, a, None)
    if (isinstance(v, int) and v > 0
            and v not in robot.values()
            and not a.startswith('wall_')
            and a not in ('wood_floor', 'wood_floor_alcove', 'enable_cache',
                          'black_board', 'shelf_table')):
        if 'cube' in a or 'tray' in a:
            objects[a] = v

print(f"  机器人: {list(robot.keys())}")
print(f"  物品:   {list(objects.keys())}")

# ============================================================
#  常量
# ============================================================
TX, TY = 3, 5
TABLE_TOP_Z = 1.156
BOX_SIZE = 0.2
SMALL_SIZE = 0.1

SMALL_GREEN_POS = [1.0, 6.5, 2.2]
SMALL_BLUE_POS  = [7.0, 8.6, 0.80]

# 大方块
LARGE_POS = {}
for i, name in enumerate(['red', 'green', 'blue', 'yellow', 'purple', 'orange']):
    row, col = i // 3, i % 3
    x = TX - 0.3 + col * 0.3
    y = TY - 0.2 + row * 0.3
    LARGE_POS[name] = [x, y, TABLE_TOP_Z + BOX_SIZE/2]

LARGE_SHIFT_Y = 0.20  # 往 Bob 方向移（但离臂远一点点）
for k in LARGE_POS:
    LARGE_POS[k][1] += LARGE_SHIFT_Y
print(f"  大方块: green={LARGE_POS['green']}  blue={LARGE_POS['blue']}")

DROP_LEFT  = [TX - 0.6, TY + 0.5, TABLE_TOP_Z + SMALL_SIZE/2]
DROP_RIGHT = [TX + 0.6, TY + 0.5, TABLE_TOP_Z + SMALL_SIZE/2]

# 无人机零件相对 body 的初始偏移（飞行时保持相对位置）
DRONE_OFFSETS: dict[int, list[float]] = {}

NAV = {
    'alice_start': (8, 6), 'alice_end': (8, 6),
    'david_start': (8, 7), 'david_end': (8, 7),
    'bookcase_front': (3, 6.5),
    'corridor': (5.5, 5.5),
    'blue_spot': (6.5, 8.2),
    'table_west': (2.0, 6.0),
}

BOB_XY = (3, 5.7)
BOB_Z = TABLE_TOP_Z + 0.05
SAFE_Z = 1.60

ALICE_ARM_Z = 1.02

DRONE_HOVER = 1.5
DRONE_FLY_Z_BOOK = 2.5


# ============================================================
#  工具
# ============================================================
def _release(obj):
    uid = _grasp.pop(obj, None)
    if uid is not None:
        try: p.removeConstraint(uid)
        except: pass

# 各机械臂的末端执行器 link index
ARM_EE = {'alice_arm': 6, 'bob_arm': 6}  # xarm6 都是第6号link

def _grasp_obj(pid, cid, ee_link=None):
    """将物品约束到机械臂的末端执行器上。ee_link=None 则绑到基座（无人机用）"""
    parent_link = ee_link if ee_link is not None else -1
    cid2 = p.createConstraint(
        pid, parent_link, cid, -1, p.JOINT_FIXED,
        [0, 0, 0], [0, 0, 0], [0, 0, 0])
    _grasp[cid] = cid2

def _tp(body, pos, yaw=None):
    orn = p.getQuaternionFromEuler([0, 0, yaw] if yaw is not None else [0, 0, 0])
    p.resetBasePositionAndOrientation(body, list(pos), orn)

def _lock(oid, pos):
    """将物体稳固定在某个位置（质量0，不参与物理）"""
    _release(oid)
    _tp(oid, pos)
    p.changeDynamics(oid, -1, mass=0, lateralFriction=0.8,
                     activationState=p.ACTIVATION_STATE_SLEEP)
    _step(5)

def _step(n):
    for _ in range(n): p.stepSimulation()


# ============================================================
#  Alice 导航
# ============================================================
def _route_arounds(src, dst):
    """北绕路径绕过 table2"""
    sx, sy = src; dx, dy = dst
    t_rx, t_ty = 4.2 + 0.3, 5.9 + 0.3
    src_left = sx < 2.1
    dst_right = dx > t_rx
    crossing = (src_left and not dst_right) or (dst_right and not src_left)
    if not crossing:
        return [dst]
    north_y = 8.3
    wps = []
    if sy < north_y: wps.append((sx, north_y))
    wps.append((dx, north_y))
    if dy < north_y: wps.append((dx, dy))
    return wps


def _other_robot_positions(exclude_base=None):
    pos = []
    for rname, rid in robot.items():
        if exclude_base and rid == exclude_base: continue
        if 'rotor' in rname or 'arm' in rname: continue
        try:
            rp = p.getBasePositionAndOrientation(rid)[0]
            pos.append((rp[0], rp[1]))
        except: pass
    return pos


def _drive_toward(base, arm, target_xy, arm_z, speed=0.04, step_delay=0.008):
    cx, cy = p.getBasePositionAndOrientation(base)[0][:2]
    ex, ey = target_xy
    dx, dy = ex - cx, ey - cy
    dist = math.hypot(dx, dy)
    if dist < 0.08:
        _tp(base, [ex, ey, 0])
        if arm: _tp(arm, [ex, ey, arm_z])
        _step(20)
        return True
    steps = max(int(dist / speed), 12)
    cur_yaw = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(base)[1])[2]
    tgt_yaw = math.atan2(dy, dx)
    ang_diff = tgt_yaw - cur_yaw
    while ang_diff > math.pi: ang_diff -= 2*math.pi
    while ang_diff < -math.pi: ang_diff += 2*math.pi
    rot_st = min(steps, max(int(steps*0.3), 6))
    for s in range(steps):
        t = (s+1)/steps
        yaw = cur_yaw + ang_diff*min((s+1)/rot_st, 1.0) if s < rot_st else tgt_yaw
        nx = cx + dx*t; ny = cy + dy*t
        if planner.is_collision(nx, ny):
            print(f"  ⚠️ 前方障碍! 停 ({nx:.2f},{ny:.2f})")
            break
        collide = False
        for rx, ry in _other_robot_positions(base):
            if math.hypot(nx-rx, ny-ry) < 0.45:
                print(f"  ⚠️ 机器人太近! 停 ({nx:.2f},{ny:.2f})")
                collide = True; break
        if collide: break
        _tp(base, [nx, ny, 0], yaw)
        if arm: _tp(arm, [nx, ny, arm_z], yaw)
        _step(4); time.sleep(step_delay)
    _step(20)
    return True


def alice_drive(target_xy):
    base = robot['alice_base']; arm = robot.get('alice_arm')
    cx, cy = p.getBasePositionAndOrientation(base)[0][:2]
    if planner.is_collision(target_xy[0], target_xy[1]):
        print(f"  ⚠️ 目标({target_xy[0]:.1f},{target_xy[1]:.1f})在障碍物中，绕行…")
        wps = _route_arounds((cx, cy), target_xy)
        for wp in wps:
            if not planner.is_collision(wp[0], wp[1]):
                path = planner.plan(cx, cy, wp[0], wp[1])
                if path and len(path) >= 2:
                    for i in range(len(path)-1):
                        _drive_toward(base, arm, path[i+1], ALICE_ARM_Z, 0.04, 0.008)
                        cx, cy = p.getBasePositionAndOrientation(base)[0][:2]
                else:
                    _drive_toward(base, arm, wp, ALICE_ARM_Z, 0.03, 0.01)
        _drive_toward(base, arm, target_xy, ALICE_ARM_Z, 0.02, 0.012)
    else:
        path = planner.plan(cx, cy, target_xy[0], target_xy[1])
        if path and len(path) >= 2:
            for i in range(len(path)-1):
                _drive_toward(base, arm, path[i+1], ALICE_ARM_Z, 0.04, 0.008)
        else:
            print(f"  ⚠️ A*无路径，绕行…")
            wps = _route_arounds((cx, cy), target_xy)
            for wp in wps:
                path = planner.plan(cx, cy, wp[0], wp[1])
                if path and len(path) >= 2:
                    for i in range(len(path)-1):
                        _drive_toward(base, arm, path[i+1], ALICE_ARM_Z, 0.04, 0.008)
                        cx, cy = p.getBasePositionAndOrientation(base)[0][:2]
    _step(30)
    final = p.getBasePositionAndOrientation(base)[0]
    print(f"  ✓ Alice → ({final[0]:.1f},{final[1]:.1f})")
    return True


def david_drive(target_xy):
    base = robot.get('david') or robot.get('david_base')
    if base is None: return False
    cx, cy = p.getBasePositionAndOrientation(base)[0][:2]
    if planner.is_collision(target_xy[0], target_xy[1]):
        print(f"  ⚠️ David目标({target_xy[0]:.1f},{target_xy[1]:.1f})在障碍物中")
        bx, by, bw, bh = planner.bounds
        best_pt = None; best_d = float('inf')
        for dx in range(int(bw*4)):
            for dy in range(int(bh*4)):
                tx = bx + dx*0.25; ty = by + dy*0.25
                if not planner.is_collision(tx, ty):
                    d = math.hypot(tx-target_xy[0], ty-target_xy[1])
                    if d < best_d: best_d = d; best_pt = (tx, ty)
        if best_pt: target_xy = best_pt
        else: return False
    path = planner.plan(cx, cy, target_xy[0], target_xy[1])
    if path and len(path) >= 2:
        for i in range(len(path)-1):
            _drive_toward(base, None, path[i+1], 0, 0.04, 0.006)
    else:
        print(f"  ⚠️ A*无路径，直线移动")
        _drive_toward(base, None, target_xy, 0, 0.03, 0.008)
    _step(30)
    final = p.getBasePositionAndOrientation(base)[0]
    print(f"  ✓ David → ({final[0]:.1f},{final[1]:.1f})")
    return True


# ============================================================
#  Alice xarm6 IK
# ============================================================
AAJ = list(range(6)); AEE = 6

def alice_ik(body, tgt, steps=40):
    rest = [0, -0.3, -0.5, 0, 0.3, 0]
    ik = p.calculateInverseKinematics(
        body, AEE, list(tgt),
        lowerLimits=[-6.28]*6, upperLimits=[6.28]*6,
        jointRanges=[12.56]*6, restPoses=rest,
        maxNumIterations=300, residualThreshold=0.0005)
    cur = [p.getJointState(body, j)[0] for j in AAJ]
    for s in range(1, steps+1):
        t = s/steps; ease = t*t*(3-2*t)
        for j in range(6):
            p.resetJointState(body, j, cur[j] + (ik[j]-cur[j])*ease)
        _step(4); time.sleep(0.008)
    for j in range(6): p.resetJointState(body, j, ik[j])
    _step(20)

def alice_grip(body, close):
    v = 0.0 if close else 0.6
    for j in [7, 8, 11]:
        p.setJointMotorControl2(body, j, p.POSITION_CONTROL, targetPosition=v, force=25)
    _step(40)

def alice_neutral(body):
    n = [0, -0.5, 0, 0, 0.5, 0]
    cur = [p.getJointState(body, j)[0] for j in range(6)]
    for s in range(1, 35):
        t = s/35; ease = t*t*(3-2*t)
        for j in range(6): p.resetJointState(body, j, cur[j] + (n[j]-cur[j])*ease)
        _step(4); time.sleep(0.008)
    _step(20)

def alice_pick(pos, obj_name):
    arm = robot.get('alice_arm'); obj_id = objects.get(obj_name)
    if not arm or obj_id is None: return False
    alice_neutral(arm); alice_grip(arm, False)
    # 三段式：先高绕过 → 接近 → 到位（避免臂杆碰触物块）
    high = [pos[0], pos[1], pos[2] + 0.40]   # 高高越过物块
    above = [pos[0], pos[1], pos[2] + 0.12]   # 接近
    alice_ik(arm, high, 40)
    alice_ik(arm, above, 25)
    alice_ik(arm, list(pos), 20)
    # 把物品瞬移到夹爪位置
    ee_pos = p.getLinkState(arm, AEE)[0]
    _release(obj_id)
    _tp(obj_id, [ee_pos[0], ee_pos[1], ee_pos[2] - 0.01])
    _step(5)
    alice_grip(arm, True)
    _release(obj_id)
    _grasp_obj(arm, obj_id, ee_link=AEE)
    alice_ik(arm, high, 30)
    print(f"  ✓ Alice 捡 {obj_name}")
    return True

def alice_place(pos, obj_name):
    arm = robot.get('alice_arm'); obj_id = objects.get(obj_name)
    if not arm or obj_id is None: return False
    above = [pos[0], pos[1], pos[2] + 0.12]
    alice_ik(arm, above, 35); alice_ik(arm, list(pos), 30)
    _release(obj_id); alice_grip(arm, False); _lock(obj_id, list(pos))
    alice_ik(arm, above, 25); alice_neutral(arm)
    print(f"  ✓ Alice 放 {obj_name}")
    return True


# ============================================================
#  Lucy 无人机
# ============================================================
def drone_fly_to(target_xy, target_z=None, speed_mult=1.0):
    """飞行：body 到目标，其他零件跟 body 保持偏移"""
    base = robot['drone_body']
    cz = p.getBasePositionAndOrientation(base)[0][2]
    tz = target_z if target_z is not None else cz
    cx, cy = p.getBasePositionAndOrientation(base)[0][:2]
    dx = target_xy[0]-cx; dy = target_xy[1]-cy; dz = tz-cz
    dist = math.hypot(dx, dy, dz)
    if dist < 0.02: return True
    step_per_m = 60 / max(speed_mult, 0.1)
    steps = max(int(dist * step_per_m), 20)
    for s in range(1, steps+1):
        t = s / steps
        # body 直接到中间位置
        bx = cx + dx * t
        by = cy + dy * t
        bz = cz + dz * t
        _tp(base, [bx, by, bz])
        # 其他零件按偏移跟随
        for pid, off in DRONE_OFFSETS.items():
            pp = p.getBasePositionAndOrientation(pid)[0]
            _tp(pid, [bx + off[0], by + off[1], bz + off[2]])
        _step(1); time.sleep(0.004 / max(speed_mult, 0.1))
    # 精确到位（只步进2步避免碰撞弹开零件）
    _tp(base, [target_xy[0], target_xy[1], tz])
    for pid, off in DRONE_OFFSETS.items():
        _tp(pid, [target_xy[0]+off[0], target_xy[1]+off[1], tz+off[2]])
    _step(2)
    pos = p.getBasePositionAndOrientation(base)[0]
    print(f"  ✓ Drone → ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.2f})")
    return True

def drone_grab(obj_name):
    base = robot['drone_body']; obj_id = objects.get(obj_name)
    if obj_id is None: return False
    dp = p.getBasePositionAndOrientation(base)[0]
    grip_z = dp[2] - 0.12
    _release(obj_id); _tp(obj_id, [dp[0], dp[1], grip_z]); _step(10)
    _grasp_obj(base, obj_id)
    print(f"  ✓ Drone 吸 {obj_name}"); return True

def drone_release(obj_name, target_pos):
    base = robot['drone_body']; obj_id = objects.get(obj_name)
    if obj_id is None: return False
    above = [target_pos[0], target_pos[1], target_pos[2] + 0.03]
    cur = p.getBasePositionAndOrientation(base)[0]
    dx = above[0]-cur[0]; dy = above[1]-cur[1]; dz = above[2]-cur[2]
    dist = math.hypot(dx, dy, dz)
    steps = max(int(dist/0.015), 25)
    for s in range(1, steps+1):
        t = s / steps
        bx = cur[0] + dx * t
        by = cur[1] + dy * t
        bz = cur[2] + dz * t
        _tp(base, [bx, by, bz])
        for pid, off in DRONE_OFFSETS.items():
            _tp(pid, [bx+off[0], by+off[1], bz+off[2]])
        _step(1); time.sleep(0.008)
    _step(2)
    _release(obj_id); _lock(obj_id, list(target_pos))
    print(f"  ✓ Drone 放 {obj_name}"); return True


# ============================================================
#  Bob xarm6 — 自然伸缩 + 碰撞避免
# ============================================================
BAJ = list(range(6)); BEE = 6

def _bob_ik_smooth(body, tgt, steps=50):
    """平滑 IK：预先计算完整轨迹，分步插值"""
    rest = [0, 0, 0, 0, 0, 0]
    ik = p.calculateInverseKinematics(
        body, BEE, list(tgt),
        lowerLimits=[-6.28]*6, upperLimits=[6.28]*6,
        jointRanges=[12.56]*6, restPoses=rest,
        maxNumIterations=600, residualThreshold=0.0002)
    cur = [p.getJointState(body, j)[0] for j in BAJ]
    for s in range(1, steps+1):
        t = s/steps; ease = t*t*(3-2*t)
        for j in range(6):
            p.resetJointState(body, j, cur[j] + (ik[j]-cur[j])*ease)
        _step(4)
        time.sleep(0.006)
    for j in range(6): p.resetJointState(body, j, ik[j])
    _step(15)

def _bob_check_collision(body):
    """检查 Bob 臂是否与其他物体碰撞（返回 True=有碰撞）"""
    for obj_name, obj_id in objects.items():
        if obj_id in robot.values():
            continue
        pts = p.getClosestPoints(bodyA=body, bodyB=obj_id, distance=0.01)
        for pt in pts:
            if pt[8] < 0.005:  # contact distance < 0.5cm
                return True
    return False

def _bob_trajectory(body, waypoints, label=""):
    """沿路径点序列平滑移动，遇碰撞则跳过后续"""
    for i, wp in enumerate(waypoints):
        _bob_ik_smooth(body, wp, 30)
        if _bob_check_collision(body):
            print(f"  ⚠️ Bob {label} 路径{i}碰撞，尝试绕行…")
            # 升高再过去
            above = [wp[0], wp[1], wp[2] + 0.15]
            _bob_ik_smooth(body, above, 20)
            _bob_ik_smooth(body, wp, 25)
    return True

def bob_grip(body, close):
    v = 0.0 if close else 0.6
    for j in [7, 8, 11]:
        p.setJointMotorControl2(body, j, p.POSITION_CONTROL, targetPosition=v, force=25)
    _step(40)

def bob_neutral(body):
    n = [0, -0.5, 0, 0, 0.5, 0]
    cur = [p.getJointState(body, j)[0] for j in BAJ]
    for s in range(1, 40):
        t = s/40; ease = t*t*(3-2*t)
        for j in range(6): p.resetJointState(body, j, cur[j] + (n[j]-cur[j])*ease)
        _step(4); time.sleep(0.008)
    _step(20)

def bob_pick(pos, obj_name):
    """Bob 捡：升→伸→降→抓→升→收（无碰撞检测，直接IK）"""
    body = robot['bob_arm']; obj_id = objects.get(obj_name)
    if obj_id is None: return False
    bob_neutral(body); bob_grip(body, False)
    bx, by = BOB_XY; sz = SAFE_Z
    # 升到安全高 → 伸到物块上方 → 降到位
    _bob_ik_smooth(body, [bx, by, sz], 25)
    _bob_ik_smooth(body, [pos[0], pos[1], sz], 30)
    _bob_ik_smooth(body, [pos[0], pos[1], pos[2] + 0.08], 20)
    _bob_ik_smooth(body, list(pos), 15)
    # 抓
    ee_pos = p.getLinkState(body, BEE)[0]
    _release(obj_id)
    _tp(obj_id, [ee_pos[0], ee_pos[1], ee_pos[2] - 0.01])
    _step(8); bob_grip(body, True)
    _release(obj_id); _grasp_obj(body, obj_id, ee_link=BEE)
    # 升回
    _bob_ik_smooth(body, [pos[0], pos[1], sz], 20)
    _bob_ik_smooth(body, [bx, by, sz], 25)
    print(f"  ✓ Bob 捡 {obj_name}")
    return True


def bob_place(pos, obj_name):
    """Bob 放：升→伸→降→放→升→收"""
    body = robot['bob_arm']; obj_id = objects.get(obj_name)
    if obj_id is None: return False
    bx, by = BOB_XY; sz = SAFE_Z
    # 升 → 伸到放置点上方 → 降
    _bob_ik_smooth(body, [bx, by, sz], 20)
    _bob_ik_smooth(body, [pos[0], pos[1], sz], 25)
    _bob_ik_smooth(body, [pos[0], pos[1], pos[2] + 0.08], 15)
    _bob_ik_smooth(body, list(pos), 15)
    # 放
    _release(obj_id); bob_grip(body, False)
    _step(5)
    _lock(obj_id, list(pos))
    # 升回
    _bob_ik_smooth(body, [pos[0], pos[1], sz], 15)
    _bob_ik_smooth(body, [bx, by, sz], 20)
    bob_neutral(body)
    print(f"  ✓ Bob 放 {obj_name}")
    return True


# ============================================================
#  初始位置
# ============================================================
print("\n[4] 初始位置…")
_tp(robot['alice_base'], [8, 6, 0], -math.pi/2)
if robot.get('alice_arm'):
    _tp(robot['alice_arm'], [8, 6, ALICE_ARM_Z], -math.pi/2)
if robot.get('david'):
    _tp(robot['david'], [8, 7, 0])
# 无人机：记录偏移 + 禁用零件间碰撞 + 去质量防下落
body = robot['drone_body']
body_pos = list(p.getBasePositionAndOrientation(body)[0])
drone_parts = [(k,v) for k,v in robot.items() if k.startswith('drone_')]
drone_all_ids = [v for _,v in drone_parts]
for i, (n1, id1) in enumerate(drone_parts):
    DRONE_OFFSETS[id1] = [0,0,0] if id1 == body else [
        p.getBasePositionAndOrientation(id1)[0][j] - body_pos[j] for j in range(3)]
    # 设质量为0（不受重力影响，靠_tp定位）
    p.changeDynamics(id1, -1, mass=0)
    # 禁用此零件与其他所有无人机零件的碰撞
    for id2 in drone_all_ids:
        if id1 != id2:
            p.setCollisionFilterPair(id1, id2, -1, -1, 0)
print(f"  无人机: {len(DRONE_OFFSETS)} 零件 (碰撞禁用+无质量)")
for _ in range(30): p.stepSimulation()
print("  ✅ 就位")

# 把大方块物理移到 LARGE_SHIFT_Y 偏移后的位置
print("  移动大方块…")
for k in LARGE_POS:
    obj_name = f'cube_{k}'
    if obj_name in objects:
        oid = objects[obj_name]
        old_pos = list(p.getBasePositionAndOrientation(oid)[0])
        _tp(oid, [old_pos[0], old_pos[1] + LARGE_SHIFT_Y, old_pos[2]])
for _ in range(20): p.stepSimulation()

# ============================================================
#  🎬 执行
# ============================================================
print("\n" + "="*60 + "\n  🎬 开始\n" + "="*60)
t0 = time.time()

# ── Step 1: David 书架巡游 ──
print("\n--- Step 1: David 书架巡游 ---")
david_drive(NAV['bookcase_front'])
print("  ⏳ 停留 2 秒…"); time.sleep(2)
david_drive(NAV['david_end'])

# ── Step 2a+2b: Lucy↔Alice 同时 ──
print("\n--- Step 2: Lucy↔Alice 同时行动 ---")
alice_drive(NAV['corridor'])
drone_fly_to((1, 4), 2.0, speed_mult=0.3)
alice_drive(NAV['blue_spot'])

# 各自拿取
alice_pick(SMALL_BLUE_POS, 'small_cube_blue')
drone_fly_to((1, 6.5), DRONE_FLY_Z_BOOK, speed_mult=0.4)
sg = objects.get('small_cube_green')
if sg: _lock(sg, SMALL_GREEN_POS); _step(10)
drone_fly_to((1, 6.5), SMALL_GREEN_POS[2] + 0.12, speed_mult=0.2)
drone_grab('small_cube_green')
drone_fly_to((1, 6.5), DRONE_FLY_Z_BOOK, speed_mult=0.3)
# 直接去 Bob 桌子，不回 (1,4)
alice_drive(NAV['corridor'])
drone_fly_to((TX, TY), DRONE_FLY_Z_BOOK, speed_mult=0.4)
alice_drive(NAV['table_west'])

# 放置
drone_release('small_cube_green', DROP_LEFT)
alice_place(DROP_RIGHT, 'small_cube_blue')

# 回位
drone_fly_to((TX, TY), DRONE_FLY_Z_BOOK, speed_mult=0.3)
drone_fly_to((1, 4), DRONE_FLY_Z_BOOK, speed_mult=0.4)
drone_fly_to((1, 4), DRONE_HOVER, speed_mult=0.3)
alice_drive(NAV['corridor'])
alice_drive(NAV['alice_end'])

# ── Step 3: Bob 放置 ──
print("\n--- Step 3: Bob 精确放置 ---")
bob_pick(DROP_LEFT, 'small_cube_green')
gp = LARGE_POS['green']
gp_top = gp[2] + BOX_SIZE/2
place_green_z = gp_top + SMALL_SIZE/2 + 0.01
bob_place([gp[0], gp[1], place_green_z], 'small_cube_green')

bob_pick(DROP_RIGHT, 'small_cube_blue')
bp = LARGE_POS['blue']
bp_top = bp[2] + BOX_SIZE/2
place_blue_z = bp_top + SMALL_SIZE/2 + 0.01
bob_place([bp[0], bp[1], place_blue_z], 'small_cube_blue')

# ── 完成 ──
elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"  ✅ 完成！⏱ {elapsed:.0f}s")
print(f"  small_cube_green → cube_green")
print(f"  small_cube_blue  → cube_blue")
print(f"{'='*60}")

try:
    while True:
        p.stepSimulation(); time.sleep(0.05)
except KeyboardInterrupt:
    print("\n退出…"); client.disconnect()
