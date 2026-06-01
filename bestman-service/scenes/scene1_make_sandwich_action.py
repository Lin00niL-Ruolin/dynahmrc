#!/usr/bin/env python3
"""
scene1_make_sandwich_action.py — v8
三明治制作：Bob(IK) + Alice(导航+臂)
"""
import os,sys,math,time,pybullet as p

script_dir=os.path.dirname(os.path.abspath(__file__))
workspace_dir=os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
bestman_dir=os.path.join(workspace_dir,'BestMan')
sys.path.insert(0,bestman_dir);sys.path.insert(0,os.path.dirname(script_dir))
from Env.Client import Client
from scenes.scene1_setup import setup_scene1
from scenes.path_planner import AStarPathPlanner

_grasp={}
def load_yaml_config(cfg_path):
    import yaml;from types import SimpleNamespace
    def d2ns(d):
        if isinstance(d,dict):return SimpleNamespace(**{k:d2ns(v) for k,v in d.items()})
        return d
    fp=cfg_path
    if not os.path.exists(fp):fp=os.path.join(bestman_dir,cfg_path)
    with open(fp) as f:return d2ns(yaml.safe_load(f))

print("="*60+"\n  Make Sandwich v8\n"+"="*60)
cfg=load_yaml_config('Config/default.yaml')
cfg.Client.enable_GUI=bool(os.environ.get('DISPLAY'))
print(f"[1] PyBullet ({'GUI'if cfg.Client.enable_GUI else'DIRECT'})…")
client=Client(cfg.Client)
print("[2] 加载场景1…")
setup_scene1(client,os.path.join(script_dir,"scene1.json"))
for _ in range(30):p.stepSimulation()
print("[3] 路径规划器…")
planner=AStarPathPlanner(scene='scene1',grid_size=0.15,robot_radius=0.3)
time.sleep(2)

robot={}
for a in['bob_arm','new_robot_base','new_robot_arm']:
    v=getattr(client,a,None)
    if isinstance(v,int)and v>0:robot[a]=v
ITEMS={'bread_0':'bread_0','bread_1':'bread_1','bacon_0':'bacon_0'}
objects={}
for a in dir(client):
    v=getattr(client,a,None)
    if(isinstance(v,int)and v>0 and v not in robot.values()
       and not a.startswith('wall_')and a not in('wood_floor','enable_cache')
       and a in(*ITEMS.values(),'cutting_board')):objects[a]=v
print(f"  机器人:{list(robot.keys())}\n  物品:{list(objects.keys())}")

# 坐标
T=0.826; BH=0.025; BAH=0.010; BOARD_TOP=0.880
# 堆叠在案板上(8.5,5.5) —— 三明治在案板上制作
# 案板顶面 z=0.86+0.02=0.88，bread 半高 0.025
BOARD_TOP = 0.880
ST=[
    (8.5, 5.5, BOARD_TOP+BH),              # bread_0 中心
    (8.5, 5.5, BOARD_TOP+BH*2+BAH),        # bacon 中心在 bread_0 上
    (8.5, 5.5, BOARD_TOP+BH*2+BAH*2+BH+0.005), # bread_1 中心在 bacon 上
]
DROP=(8.5,5.8,T)           # Alice 临时放桌子二上（不在案板）
NAV={'s':(6.5,7),'t1':(7.7,4.0),'t2':(6.8,5.5)}
POS={
    'bread_0':[8.2,5.85,T+BH],   # 物品中心(桌面+半高)
    'bread_1':[8.55,4.2,T+BH],
    'bacon_0':[8.5,4.0,T+BAH],
}

BOB=robot['bob_arm'];AARM=robot.get('new_robot_arm')
ABASE=robot['new_robot_base'];AJ=list(range(6));AE=6
p.resetBasePositionAndOrientation(ABASE,[6.5,7,0],p.getQuaternionFromEuler([0,0,-math.pi/2]))
if AARM:p.resetBasePositionAndOrientation(AARM,[6.5,7,1.02],p.getQuaternionFromEuler([0,0,-math.pi/2]))

# ── 工具 ──
def _release(obj):
    uid=_grasp.pop(obj,None)
    if uid is not None:
        try:p.removeConstraint(uid)
        except:pass
def _grasp_obj(pid,cid):
    _grasp[cid]=p.createConstraint(pid,AE,cid,-1,p.JOINT_FIXED,[0,0,0],
                                    parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
def _tp(body,pos,yaw=None):
    orn=p.getQuaternionFromEuler([0,0,yaw]if yaw is not None else[0,0,0])
    p.resetBasePositionAndOrientation(body,list(pos),orn)
def _step(n):
    for _ in range(n):p.stepSimulation()
def _lock(oid,pos):
    _release(oid);_tp(oid,pos)
    p.changeDynamics(oid,-1,mass=0.001,lateralFriction=0.8,activationState=p.ACTIVATION_STATE_SLEEP)
    _step(20)

# ── IK（resetJointState 绕过URDF限制）──
def _ik(body,tgt,steps=50):
    rest=[0,-0.3,-0.5,0,0.3,0]
    ik=p.calculateInverseKinematics(body,AE,list(tgt),
        lowerLimits=[-6.28]*6,upperLimits=[6.28]*6,
        jointRanges=[12.56]*6,restPoses=rest,
        maxNumIterations=300,residualThreshold=0.0005)
    cur=[p.getJointState(body,j)[0]for j in AJ]
    for s in range(1,steps+1):
        t=s/steps;ease=t*t*(3-2*t)
        for j in range(6):
            p.resetJointState(body,j,cur[j]+(ik[j]-cur[j])*ease)
        _step(6);time.sleep(0.012)
    for j in range(6):p.resetJointState(body,j,ik[j])
    _step(30)
def _grip(body,close):
    v=0.0 if close else 0.6
    for j in[7,8,11]:p.setJointMotorControl2(body,j,p.POSITION_CONTROL,targetPosition=v,force=25)
    _step(40)
def _neutral(body):
    n=[0,-0.5,0,0,0.5,0];cur=[p.getJointState(body,j)[0]for j in AJ]
    for s in range(1,45):
        t=s/45;ease=t*t*(3-2*t)
        for j in range(6):p.resetJointState(body,j,cur[j]+(n[j]-cur[j])*ease)
        _step(6);time.sleep(0.012)
    _step(30)

# Bob 固定基座坐标
BOB_XY = (8.5, 5.85)
SAFE_Z = 1.10      # 安全提升高度（远高于桌面）

def bob_pick(pos,name):
    o=objects[ITEMS[name]];_neutral(BOB);_grip(BOB,False)
    # 三段：直上Bob上方→水平伸到物品上方→直降到物品
    above_bob   = [BOB_XY[0], BOB_XY[1], SAFE_Z]    # Bob 正上方
    above_item  = [pos[0], pos[1], SAFE_Z]           # 物品正上方
    _ik(BOB, above_bob, 25)      # ↑ 直上（不撞案板）
    _ik(BOB, above_item, 30)     # → 水平伸出
    _ik(BOB, list(pos), 35)      # ↓ 直降到物品
    _grip(BOB,True);_release(o);_grasp_obj(BOB,o)
    _ik(BOB, above_item, 25)     # ↑ 直上
    print(f"  ✓ Bob 捡 {name}");return True

def bob_place(pos,name):
    o=objects[ITEMS[name]]
    # 三段运动：竖直提升→水平移动→竖直下降
    current_pos = p.getLinkState(BOB,AE)[0]
    lift = [BOB_XY[0], BOB_XY[1], SAFE_Z]     # 先升到 Bob 上方
    target_above = [pos[0], pos[1], SAFE_Z]    # 移到目标上方
    _ik(BOB, lift, 25)                          # ↑ 竖直提升
    _ik(BOB, target_above, 30)                  # → 水平移动
    _ik(BOB, list(pos), 35)                     # ↓ 竖直下降
    _release(o);_grip(BOB,False);_lock(o,pos)
    _ik(BOB, target_above, 25)                  # ↑ 竖直抬升
    _neutral(BOB)
    print(f"  ✓ Bob 放 {name}");return True

# ── Alice 导航（硬编码安全路径绕过椅子）──
# 安全路径（绕过椅子左下方再右移到桌子）
SAFE_WP={
    's_t1':[(6.5,7),(6.5,5.8),(6.5,4.5),(7.0,4.2),(7.7,4.0)],
    's_t2':[(6.5,7),(6.5,5.8),(7.5,5.8)],
    't1_t2':[(7.7,4.0),(7.0,4.2),(6.5,4.5),(6.5,5.8),(7.5,5.8)],
    't2_t1':[(7.5,5.8),(6.5,5.8),(6.5,4.5),(7.0,4.2),(7.7,4.0)],
    't1_s':[(7.7,4.0),(7.0,4.2),(6.5,4.5),(6.5,5.8),(6.5,7)],
    't2_s':[(7.5,5.8),(6.5,5.8),(6.5,7)],
}
def alice_drive(wp):
    cur_yaw=p.getEulerFromQuaternion(p.getBasePositionAndOrientation(ABASE)[1])[2]
    for wi in range(len(wp)-1):
        sx,sy=wp[wi];ex,ey=wp[wi+1];dx,dy=ex-sx,ey-sy;d=math.hypot(dx,dy)
        if d<0.001:continue
        tgt_yaw=math.atan2(dy,dx);st=max(int(d/0.02),20)
        ang_diff=tgt_yaw-cur_yaw
        while ang_diff>math.pi:ang_diff-=2*math.pi
        while ang_diff<-math.pi:ang_diff+=2*math.pi
        rot_st=min(st,max(int(st*0.3),6))
        for s in range(st):
            t=(s+1)/st
            yaw=cur_yaw+ang_diff*min((s+1)/rot_st,1.0)if s<rot_st else tgt_yaw
            _tp(ABASE,[sx+dx*t,sy+dy*t,0],yaw)
            if AARM:_tp(AARM,[sx+dx*t,sy+dy*t,1.02],yaw)
            _step(6);time.sleep(0.01)
        cur_yaw=tgt_yaw
    _step(30)
    pos=p.getBasePositionAndOrientation(ABASE)[0]
    print(f"  ✓ Alice → ({pos[0]:.1f},{pos[1]:.1f})");return True
def a_nav(wp_key):
    wp=SAFE_WP.get(wp_key)
    if not wp:return False
    return alice_drive(wp)

# ── Alice 臂──
def alice_grab(xyz,name):
    o=objects[ITEMS[name]]
    if not AARM:return False
    _neutral(AARM);_grip(AARM,False)
    bp=p.getBasePositionAndOrientation(ABASE)[0]
    dx=xyz[0]-bp[0];dy=xyz[1]-bp[1];dz=xyz[2]-1.02;d=math.hypot(dx,dy,dz)
    MAX=0.85
    if d>MAX:s=MAX*0.95/d;at=[bp[0]+dx*s,bp[1]+dy*s,1.02+dz*s]
    else:at=list(xyz)
    ab=[at[0],at[1],at[2]+0.08];_ik(AARM,ab,45);_ik(AARM,at,40)
    # 物品滑到爪下
    ee=p.getLinkState(AARM,AE)[0];_release(o)
    _tp(o,[ee[0],ee[1],at[2]-0.02]);_step(5)
    _grip(AARM,True);_release(o);_grasp_obj(AARM,o);_ik(AARM,ab,30)
    print(f"  ✓ Alice 捡 {name}");return True
def alice_place(pos,name):
    o=objects[ITEMS[name]];ab=[pos[0],pos[1],pos[2]+0.08]
    _ik(AARM,ab,45);_ik(AARM,pos,40);_release(o);_grip(AARM,False);_lock(o,pos)
    _ik(AARM,ab,40);_neutral(AARM)
    print(f"  ✓ Alice 放 {name}");return True

# ── 执行 ──
print("\n"+"="*60+"\n  🎬 开始\n");t0=time.time()
bob_pick(POS['bread_0'],'bread_0');bob_place(ST[0],'bread_0')
a_nav('s_t1');alice_grab(POS['bacon_0'],'bacon_0')
a_nav('t1_t2');alice_place(DROP,'bacon_0')
bob_pick(DROP,'bacon_0');bob_place(ST[1],'bacon_0')
a_nav('t2_t1');alice_grab(POS['bread_1'],'bread_1')
a_nav('t1_t2');alice_place(DROP,'bread_1')
bob_pick(DROP,'bread_1');bob_place(ST[2],'bread_1')
a_nav('t2_s')
print(f"\n⏱ {time.time()-t0:.0f}s\n"+"="*60+"\n  🥪 完成！Ctrl+C 退出\n"+"="*60)
try:
    while True:p.stepSimulation();time.sleep(0.05)
except KeyboardInterrupt:print("\n退出…");client.disconnect()
