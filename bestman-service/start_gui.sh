#!/bin/bash
# 在云桌面终端运行：bash start_gui.sh
# 这会打开 PyBullet GUI 窗口，同时启动服务
cd "$(dirname "$0")"
echo "启动 BestMan 3D GUI 服务..."
echo "DISPLAY=$DISPLAY"
/home/developer/miniconda/bin/python3 service.py
