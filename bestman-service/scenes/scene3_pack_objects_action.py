#!/usr/bin/env python3
"""
scene3_pack_objects_action.py — 打包任务
Lucy: 书+肥皂, Alice: 叉子+杯子, Bob: 放托盘
"""
import os, sys, math, time, pybullet as p

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
bestman_dir = os.path.join(workspace_dir, 'BestMan')
sys.path.insert(0, bestman_dir)
sys.path.insert(0, os.path.dirname(script_dir))
from Env.Client import Client
from scenes.scene3_setup import setup_scene3
from scenes.path_planner import AStarPathPlanner

_grasp = {}

def load_yaml_config(cfg_path):
    import yaml
    from types import SimpleNamespace
    def d2ns(d):
        if isinstance(d, dict): return SimpleNamespace(**{k: d2ns(v) for k,v in d.items()})
        return d
    fp = cfg_path
    if not os.path.exists(fp): fp = os.path.join(bestman_dir, cfg_path)
    with open(fp) as f: return d2ns(yaml.safe_load(f))

print("="*60+"\n  Scene 3: Pack Objects\n"+"="*60)
cfg = load_yaml_config('Config/default.yaml')
cfg.Client.enable_GUI = bool(os.environ.get('DISPLAY'))
print(f"[1] PyBullet ({'GUI'if cfg.Client.enable_GUI else'DIRECT'})…")
client = Client(cfg.Client)
print("[2] 加载场景3…")
setup_scene3(client, os.path.join(script_dir, 'scene3.json'))
for _ in range(50): p.stepSimulation()
print("[3] 路径规划器…")
planner = AStarPathPlanner(scene='living_room', grid_size=0.15, robot_radius=0.3)
time.sleep(2)

# 扫描机器人 & 物品
robot = {}
for a in ['bob_arm','alice_base','alice_arm','david',
          'drone_body','drone_arm_1_0','drone_arm_-1_0',
          'drone_arm_0_1','drone_arm_0_-1',
          'drone_rotor_1_0','drone_rotor_-1_0',
          'drone_rotor_0_1','drone_rotor_0_-1']:
    v = getattr(client, a, None)
    if isinstance(v,int) and v>0: robot[a]=v

objects = {}
known_items = {'fork_0','apple','book_0','soap','cup','blue_bowl','lemon','tray'}
for a in dir(client):
    v = getattr(client, a, None)
    if isinstance(v,int) and v>0 and v not in robot.values() and a in known_items:
        objects[a] = v
print(f"  机器人: {list(robot.keys())}\n  物品:   {list(objects.keys())}")

# ============ 常量 ============
# Bob 在 source_table_1 (2,4)，桌子顶面 Z=0.83
TX, TY = 2, 4
TABLE_Z = 0.83
BOB_XY = (2, 4.4)
BOB_Z = TABLE_Z + 0.04  # 0.87
SAFE_Z = 1.30
ALICE_ARM_Z = 1.02

# 物品位置
ITEM_POS = {
    'fork_0':  [1.2, 0.5, 1.10],   # kitchen_cabinet 上
    'book_0':  [7.5, 5.5, 1.80],   # bookcase 上
    'soap':    [5.7, 6.0, 1.71],   # wall_shelf 上
    'cup':     [4.0, 4.0, 0.85],   # source_table_2 上
}

# Bob 桌子上放置点（左右两侧）
DROP_LEFT  = [TX-0.3, TY+0.3, TABLE_Z + 0.04]
DROP_RIGHT = [TX+0.3, TY+0.3, TABLE_Z + 0.04]

# 托盘在 Bob 桌子 (2,4,0.87)，Bob 上面放物块位置
TRAY_POS = [2, 4, 0.87]

# 托盘内 4 个物块的放置位置（2排2列）
TRAY_SLOTS = [
    [TX-0.06, TY-0.06, TRAY_POS[2] + 0.03],
    [TX+0.06, TY-0.06, TRAY_POS[2] + 0.03],
    [TX-0.06, TY+0.06, TRAY_POS[2] + 0.03],
    [TX+0.06, TY+0.06, TRAY_POS[2] + 0.03],
]

# 导航点
NAV = {
    'alice_start': (5, 3), 'alice_end': (5, 3),
    'david_start': (3, 6), 'david_end': (3, 6),
    'corridor': (5.5, 3.0),     # 走廊中间（x=6 的通道南侧）
    'fork_spot': (1.8, 1.2),    # 叉子附近
    'cup_spot': (4.0, 3.5),     # cup 桌子南侧
    'bob_table': (2.0, 3.5),    # Bob 桌子南侧
}

DRONE_HOVER = 0.85
DRONE_FLY_Z = 2.0

# ============ 工具 ============
DRONE_OFFSETS = {}
ARM_EE = {'alice_arm':6, 'bob_arm':6}

def _release(obj):
    uid = _grasp.pop(obj,None)
    if uid is not None:
        try: p.removeConstraint(uid)
        except: pass

def _grasp_obj(pid,cid,ee_link=None):
    pl = ee_link if ee_link is not None else -1
    cid2 = p.createConstraint(pid,pl,cid,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,0])
    _grasp[cid]=cid2

def _tp(body,pos,yaw=None):
    orn = p.getQuaternionFromEuler([0,0,yaw]if yaw is not None else[0,0,0])
    p.resetBasePositionAndOrientation(body,list(pos),orn)

def _lock(oid,pos):
    _release(oid); _tp(oid,pos)
    p.changeDynamics(oid,-1,mass=0,lateralFriction=0.8,activationState=p.ACTIVATION_STATE_SLEEP)
    _step(5)

def _step(n):
    for _ in range(n): p.stepSimulation()

# ============ 导航 ============
def _other_robot_positions(exclude=None):
    pos = []
    for rn,rid in robot.items():
        if exclude and rid==exclude: continue
        if 'rotor' in rn or 'arm' in rn: continue
        try: pos.append(p.getBasePositionAndOrientation(rid)[0][:2])
        except: pass
    return pos

def _drive_toward(base,arm,target_xy,arm_z,speed=0.04,delay=0.008):
    cx,cy = p.getBasePositionAndOrientation(base)[0][:2]
    ex,ey = target_xy
    dx,dy = ex-cx, ey-cy
    dist = math.hypot(dx,dy)
    if dist < 0.08:
        _tp(base,[ex,ey,0]); _step(20)
        if arm: _tp(arm,[ex,ey,arm_z])
        return True
    steps = max(int(dist/speed),12)
    for s in range(steps):
        t = (s+1)/steps; nx=cx+dx*t; ny=cy+dy*t
        if planner.is_collision(nx,ny):
            print(f"  ⚠️ 障碍! 停 ({nx:.2f},{ny:.2f})"); break
        collide=False
        for rx,ry in _other_robot_positions(base):
            if math.hypot(nx-rx,ny-ry)<0.45: collide=True; break
        if collide: break
        _tp(base,[nx,ny,0])
        if arm: _tp(arm,[nx,ny,arm_z])
        _step(3); time.sleep(delay)
    _step(20); return True

def robot_drive(name,target_xy):
    """通用导航：先A*规划再移动"""
    base = robot.get(name) or robot.get(f'{name}_base')
    arm = robot.get(f'{name}_arm') if name not in ('david','drone_body') else None
    arm_z = ALICE_ARM_Z if name=='alice' else 0
    if base is None: return False
    cx,cy = p.getBasePositionAndOrientation(base)[0][:2]
    path = planner.plan(cx,cy,target_xy[0],target_xy[1])
    if path and len(path)>=2:
        for i in range(len(path)-1):
            _drive_toward(base,arm,path[i+1],arm_z,0.04,0.008)
    else:
        _drive_toward(base,arm,target_xy,arm_z,0.03,0.01)
    final = p.getBasePositionAndOrientation(base)[0]
    print(f"  ✓ {name} → ({final[0]:.1f},{final[1]:.1f})")
    return True

# ============ Alice xarm6 IK ============
AAJ=list(range(6)); AEE=6
def alice_ik(body,tgt,steps=40):
    rest=[0,-0.3,-0.5,0,0.3,0]
    ik=p.calculateInverseKinematics(body,AEE,list(tgt),
        lowerLimits=[-6.28]*6,upperLimits=[6.28]*6,
        jointRanges=[12.56]*6,restPoses=rest,
        maxNumIterations=300,residualThreshold=0.0005)
    cur=[p.getJointState(body,j)[0] for j in AAJ]
    for s in range(1,steps+1):
        t=s/steps;ease=t*t*(3-2*t)
        for j in range(6): p.resetJointState(body,j,cur[j]+(ik[j]-cur[j])*ease)
        _step(4);time.sleep(0.008)
    for j in range(6): p.resetJointState(body,j,ik[j])
    _step(20)

def alice_grip(body,close):
    v=0.0 if close else 0.6
    for j in [7,8,11]: p.setJointMotorControl2(body,j,p.POSITION_CONTROL,targetPosition=v,force=25)
    _step(40)

def alice_neutral(body):
    n=[0,-0.5,0,0,0.5,0]
    cur=[p.getJointState(body,j)[0] for j in range(6)]
    for s in range(1,35):
        t=s/35;ease=t*t*(3-2*t)
        for j in range(6): p.resetJointState(body,j,cur[j]+(n[j]-cur[j])*ease)
        _step(4);time.sleep(0.008)
    _step(20)

def alice_pick(pos,obj_name):
    arm=robot.get('alice_arm'); oid=objects.get(obj_name)
    if not arm or oid is None: return False
    alice_neutral(arm); alice_grip(arm,False)
    high=[pos[0],pos[1],pos[2]+0.35]
    above=[pos[0],pos[1],pos[2]+0.10]
    alice_ik(arm,high,35); alice_ik(arm,above,20); alice_ik(arm,list(pos),20)
    ee=p.getLinkState(arm,AEE)[0]
    _release(oid); _tp(oid,[ee[0],ee[1],ee[2]-0.01]); _step(5)
    alice_grip(arm,True); _release(oid); _grasp_obj(arm,oid,ee_link=AEE)
    alice_ik(arm,high,25)
    print(f"  ✓ Alice 捡 {obj_name}"); return True

def alice_place(pos,obj_name):
    arm=robot.get('alice_arm'); oid=objects.get(obj_name)
    if not arm or oid is None: return False
    high=[pos[0],pos[1],pos[2]+0.30]
    alice_ik(arm,high,30); alice_ik(arm,list(pos),25)
    _release(oid); alice_grip(arm,False); _lock(oid,list(pos))
    alice_ik(arm,high,20); alice_neutral(arm)
    print(f"  ✓ Alice 放 {obj_name}"); return True

# ============ Lucy 无人机 ============
def drone_fly_to(target_xy,target_z=None,speed_mult=0.5):
    base=robot['drone_body']; cz=p.getBasePositionAndOrientation(base)[0][2]
    tz=target_z if target_z is not None else cz
    cx,cy=p.getBasePositionAndOrientation(base)[0][:2]
    dx=target_xy[0]-cx; dy=target_xy[1]-cy; dz=tz-cz
    dist=math.hypot(dx,dy,dz)
    if dist<0.02: return True
    spm=60/max(speed_mult,0.1); steps=max(int(dist*spm),20)
    for s in range(1,steps+1):
        t=s/steps; bx=cx+dx*t; by=cy+dy*t; bz=cz+dz*t
        _tp(base,[bx,by,bz])
        for pid,off in DRONE_OFFSETS.items():
            _tp(pid,[bx+off[0],by+off[1],bz+off[2]])
        _step(1); time.sleep(0.004/max(speed_mult,0.1))
    _tp(base,[target_xy[0],target_xy[1],tz])
    for pid,off in DRONE_OFFSETS.items():
        _tp(pid,[target_xy[0]+off[0],target_xy[1]+off[1],tz+off[2]])
    _step(2); return True

def drone_grab(obj_name):
    base=robot['drone_body']; oid=objects.get(obj_name)
    if oid is None: return False
    dp=p.getBasePositionAndOrientation(base)[0]; gz=dp[2]-0.10
    _release(oid); _tp(oid,[dp[0],dp[1],gz]); _step(8)
    _grasp_obj(base,oid); print(f"  ✓ Drone 吸 {obj_name}"); return True

def drone_release(obj_name,target_pos):
    base=robot['drone_body']; oid=objects.get(obj_name)
    if oid is None: return False
    above=[target_pos[0],target_pos[1],target_pos[2]+0.03]
    cur=p.getBasePositionAndOrientation(base)[0]
    dx=above[0]-cur[0]; dy=above[1]-cur[1]; dz=above[2]-cur[2]
    steps=max(int(math.hypot(dx,dy,dz)/0.015),20)
    for s in range(1,steps+1):
        t=s/steps; bx=cur[0]+dx*t; by=cur[1]+dy*t; bz=cur[2]+dz*t
        _tp(base,[bx,by,bz])
        for pid,off in DRONE_OFFSETS.items():
            _tp(pid,[bx+off[0],by+off[1],bz+off[2]])
        _step(1); time.sleep(0.006)
    _step(2); _release(oid); _lock(oid,list(target_pos))
    print(f"  ✓ Drone 放 {obj_name}"); return True

# ============ Bob xarm6 ============
BAJ=list(range(6)); BEE=6
def bob_ik(body,tgt,steps=40):
    rest=[0,0,0,0,0,0]
    ik=p.calculateInverseKinematics(body,BEE,list(tgt),
        lowerLimits=[-6.28]*6,upperLimits=[6.28]*6,
        jointRanges=[12.56]*6,restPoses=rest,
        maxNumIterations=500,residualThreshold=0.0003)
    cur=[p.getJointState(body,j)[0] for j in BAJ]
    for s in range(1,steps+1):
        t=s/steps;ease=t*t*(3-2*t)
        for j in range(6): p.resetJointState(body,j,cur[j]+(ik[j]-cur[j])*ease)
        _step(4);time.sleep(0.008)
    for j in range(6): p.resetJointState(body,j,ik[j])
    _step(15)

def bob_grip(body,close):
    v=0.0 if close else 0.6
    for j in [7,8,11]: p.setJointMotorControl2(body,j,p.POSITION_CONTROL,targetPosition=v,force=25)
    _step(40)

def bob_neutral(body):
    n=[0,-0.5,0,0,0.5,0]
    cur=[p.getJointState(body,j)[0] for j in BAJ]
    for s in range(1,35):
        t=s/35;ease=t*t*(3-2*t)
        for j in range(6): p.resetJointState(body,j,cur[j]+(n[j]-cur[j])*ease)
        _step(4);time.sleep(0.008)
    _step(20)

def bob_pick(pos,obj_name):
    body=robot['bob_arm']; oid=objects.get(obj_name)
    if oid is None: return False
    bob_neutral(body); bob_grip(body,False)
    bx,by=BOB_XY; sz=SAFE_Z
    bob_ik(body,[bx,by,sz],20)
    bob_ik(body,[pos[0],pos[1],sz],25)
    bob_ik(body,[pos[0],pos[1],pos[2]+0.08],15)
    bob_ik(body,list(pos),15)
    ee=p.getLinkState(body,BEE)[0]
    _release(oid); _tp(oid,[ee[0],ee[1],ee[2]-0.01]); _step(8)
    bob_grip(body,True); _release(oid); _grasp_obj(body,oid,ee_link=BEE)
    bob_ik(body,[pos[0],pos[1],sz],15)
    bob_ik(body,[bx,by,sz],20)
    print(f"  ✓ Bob 捡 {obj_name}"); return True

def bob_place(pos,obj_name):
    body=robot['bob_arm']; oid=objects.get(obj_name)
    if oid is None: return False
    bx,by=BOB_XY; sz=SAFE_Z
    bob_ik(body,[bx,by,sz],15)
    bob_ik(body,[pos[0],pos[1],sz],20)
    bob_ik(body,[pos[0],pos[1],pos[2]+0.06],12)
    bob_ik(body,list(pos),12)
    _release(oid); bob_grip(body,False); _step(5)
    _lock(oid,list(pos))
    bob_ik(body,[bx,by,sz],15); bob_neutral(body)
    print(f"  ✓ Bob 放 {obj_name} → 托盘"); return True

# ============ 初始位置 ============
print("\n[4] 初始位置…")
# Alice
_tp(robot['alice_base'],[5,3,0],-math.pi/2)
if robot.get('alice_arm'): _tp(robot['alice_arm'],[5,3,ALICE_ARM_Z],-math.pi/2)
# David
if robot.get('david'): _tp(robot['david'],[3,6,0])
# 无人机
drone_parts=[(k,v) for k,v in robot.items() if k.startswith('drone_')]
body=robot['drone_body']; bp=p.getBasePositionAndOrientation(body)[0]
for name,pid in drone_parts:
    DRONE_OFFSETS[pid]=[0,0,0] if pid==body else [
        p.getBasePositionAndOrientation(pid)[0][j]-bp[j] for j in range(3)]
    p.changeDynamics(pid,-1,mass=0)
    for id2 in [v for _,v in drone_parts]:
        if pid!=id2: p.setCollisionFilterPair(pid,id2,-1,-1,0)
print(f"  无人机: {len(DRONE_OFFSETS)} 零件")
# 肥皂去质量防掉落（不固定，可拿取）
soap_id=objects.get('soap')
if soap_id:
    _tp(soap_id,ITEM_POS['soap'])
    p.changeDynamics(soap_id,-1,mass=0)
print("  肥皂已去质量防掉落")
for _ in range(30): p.stepSimulation()
print("  ✅ 就位")

# ============ 执行 ============
print("\n"+"="*60+"\n  🎬 开始\n"+"="*60)
t0=time.time()
slot_idx=0

# ── Step 1a: Lucy 取书放到 Bob 桌 ──
print("\n--- Step 1a: Lucy 取书 ---")
drone_fly_to((8,3),DRONE_FLY_Z,0.3)          # 升空
drone_fly_to((7.5,5.5),DRONE_FLY_Z,0.4)       # 飞到书架
# 重置书的位置再拿
book_obj=objects.get('book_0')
if book_obj: _lock(book_obj,ITEM_POS['book_0']); _step(10)
drone_fly_to((7.5,5.5),ITEM_POS['book_0'][2]+0.10,0.2)  # 降到书旁边
drone_grab('book_0')
drone_fly_to((7.5,5.5),DRONE_FLY_Z,0.3)       # 升起
drone_fly_to((2,4),DRONE_FLY_Z,0.4)            # 飞到 Bob 桌
drone_release('book_0',DROP_LEFT)              # 放下书

# ── Step 1b: Lucy 取肥皂放到 Bob 桌 ──
print("\n--- Step 1b: Lucy 取肥皂 ---")
drone_fly_to((2,4),DRONE_FLY_Z,0.3)
drone_fly_to((5.7,6),DRONE_FLY_Z,0.4)          # 飞到台面
soap_obj=objects.get('soap')
if soap_obj: _lock(soap_obj,ITEM_POS['soap']); _step(10)
drone_fly_to((5.7,6),ITEM_POS['soap'][2]+0.10,0.2)  # 降到肥皂旁
drone_grab('soap')
drone_fly_to((5.7,6),DRONE_FLY_Z,0.3)
# 同时 Alice 出发去拿叉子
robot_drive('alice',NAV['corridor'])
drone_fly_to((2,4),DRONE_FLY_Z,0.4)            # 飞回 Bob 桌
robot_drive('alice',NAV['fork_spot'])
drone_release('soap',DROP_RIGHT)               # 放下肥皂

# ── Step 2a: Alice 拿叉子 ──
print("\n--- Step 2a: Alice 拿叉子 ---")
fork_obj=objects.get('fork_0')
if fork_obj: _lock(fork_obj,ITEM_POS['fork_0']); _step(10)
alice_pick(ITEM_POS['fork_0'],'fork_0')
robot_drive('alice',NAV['bob_table'])
alice_place(DROP_LEFT,'fork_0')

# ── Step 2b: Alice 拿杯子 ──
print("\n--- Step 2b: Alice 拿杯子 ---")
robot_drive('alice',NAV['cup_spot'])
cup_obj=objects.get('cup')
if cup_obj: _lock(cup_obj,ITEM_POS['cup']); _step(10)
alice_pick(ITEM_POS['cup'],'cup')
robot_drive('alice',NAV['bob_table'])
alice_place(DROP_RIGHT,'cup')
robot_drive('alice',NAV['alice_end'])

# ── Step 3: Bob 所有物品放托盘 ──
print("\n--- Step 3: Bob 放入托盘 ---")
# 顺序：fork_0, book_0, soap, cup
order = ['fork_0','book_0','soap','cup']
for item_name in order:
    pos = list(ITEM_POS[item_name])
    # 找到物品当前在 Bob 桌子上的哪个 DROP 位置
    drop_pos = DROP_LEFT if item_name in ('book_0','fork_0') else DROP_RIGHT
    bob_pick(drop_pos, item_name)
    bob_place(TRAY_SLOTS[slot_idx], item_name)
    slot_idx += 1

# ── 完成 ──
elapsed = time.time()-t0
print(f"\n{'='*60}")
print(f"  ✅ 完成！⏱ {elapsed:.0f}s")
print(f"  放入托盘: book_0, soap, fork_0, cup")
print(f"{'='*60}")

try:
    while True: p.stepSimulation(); time.sleep(0.05)
except KeyboardInterrupt: print("\n退出…"); client.disconnect()
