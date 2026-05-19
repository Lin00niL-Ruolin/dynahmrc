"""
场景一截图脚本
在 headless 模式下加载 scene1 并用 pybullet 相机渲染多角度截图
无 PIL 依赖，保存为 PNG 格式（使用 Python 内置的 struct/zlib）
"""

import os
import sys
import json
import math
import struct
import zlib

import pybullet as p

# 路径设置
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(os.path.dirname(script_dir))
bestman_dir = os.path.join(workspace_dir, 'BestMan')
sys.path.insert(0, bestman_dir)

from Env.Client import Client
from scenes.scene1_setup import setup_scene1

# 加载配置
import yaml

config_path = os.path.join(bestman_dir, "Config", "default.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

class SimpleNamespace:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, SimpleNamespace(v))
            else:
                setattr(self, k, v)

cfg_ns = SimpleNamespace(cfg)
client_cfg = cfg_ns.Client

# 强制 DIRECT 模式
client_cfg.enable_GUI = False
client_cfg.enable_Debug = False

print("=" * 60)
print("  DynaHMRC 场景一 渲染截图")
print("=" * 60)

client = Client(client_cfg)
print("[✅] Client 创建成功 (DIRECT)")

scene_json = os.path.join(script_dir, "scenes", "scene1.json")
setup_scene1(client, scene_json)

for _ in range(100):
    p.stepSimulation()


def save_png(path, flat_rgba_tup, w, h):
    """纯 Python 保存 flat RGBA tuple 为 PNG（无外部依赖）"""
    arr = np.array(flat_rgba_tup, dtype=np.uint8).reshape(h, w, 4)
    
    # RGBA → RGB (合成到白色背景)
    alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
    rgb = arr[:, :, :3].astype(np.float32)
    rgb = rgb * alpha + (1.0 - alpha) * 255.0
    rgb = rgb.astype(np.uint8)
    
    # PNG raw data: filter byte + pixel rows
    raw_data = b''
    for y in range(h):
        raw_data += b'\x00'  # filter byte (None)
        raw_data += rgb[y].tobytes()
    
    compressed = zlib.compress(raw_data)
    
    def make_chunk(chunk_type, data):
        chunk = chunk_type + data
        crc = struct.pack('>I', zlib.crc32(chunk) & 0xffffffff)
        return struct.pack('>I', len(data)) + chunk + crc
    
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    
    with open(path, 'wb') as f:
        f.write(sig)
        f.write(make_chunk(b'IHDR', ihdr_data))
        f.write(make_chunk(b'IDAT', compressed))
        f.write(make_chunk(b'IEND', b''))


import numpy as np

output_dir = os.path.join(script_dir, "screenshots")
os.makedirs(output_dir, exist_ok=True)

cameras = [
    {
        "name": "overview",
        "cam_pos": [7, 5, 6],
        "target": [5, 4, 0],
        "desc": "俯瞰全景"
    },
    {
        "name": "kitchen",
        "cam_pos": [2, -1, 2],
        "target": [3, 1.5, 0],
        "desc": "厨房区 (冰箱+台面+小柜)"
    },
    {
        "name": "dining",
        "cam_pos": [1, 5.5, 2.5],
        "target": [2.5, 4.5, 0],
        "desc": "餐桌区"
    },
    {
        "name": "table_center",
        "cam_pos": [6, -0.5, 2.5],
        "target": [7, 2, 0],
        "desc": "中央桌子+椅子"
    },
    {
        "name": "bathroom",
        "cam_pos": [8, 8, 3],
        "target": [9, 5.5, 0],
        "desc": "卫生间 (洗手池+马桶)"
    },
]

for cam in cameras:
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam["cam_pos"],
        cameraTargetPosition=cam["target"],
        cameraUpVector=[0, 0, 1]
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=1.6,
        nearVal=0.1,
        farVal=100
    )

    w, h = 960, 600
    _, _, rgba, depth, seg = p.getCameraImage(
        w, h,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    path = os.path.join(output_dir, f"{cam['name']}.png")
    save_png(path, rgba, w, h)
    print(f"[📸] {cam['desc']} → {cam['name']}.png ({w}x{h})")

print(f"\n✅ 所有截图已保存到: {output_dir}")

p.disconnect()
