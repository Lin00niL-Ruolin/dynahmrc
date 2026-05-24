#!/usr/bin/env python3
"""
path_planner.py
A* 路径规划器 + 平滑移动
用于 BestMan 仿真中机器人的路径规划和导航
"""

import math
import heapq


class AStarPathPlanner:
    """A* 路径规划器"""
    
    # 场景墙壁定义
    SCENE_WALLS = {
        'scene1': [
            # 外墙（边界）
            {'type': 'rect', 'x': 5, 'y': 0, 'w': 10, 'h': 0.2},    # bottom
            {'type': 'rect', 'x': 5, 'y': 8, 'w': 10, 'h': 0.2},    # top
            {'type': 'rect', 'x': 0, 'y': 4, 'w': 0.2, 'h': 8},     # left
            {'type': 'rect', 'x': 10, 'y': 4, 'w': 0.2, 'h': 8},    # right
            # 内墙
            {'type': 'rect', 'x': 5, 'y': 2.5, 'w': 0.2, 'h': 5.0},  # 垂直墙 (5,0)→(5,5)
            {'type': 'rect', 'x': 1.5, 'y': 4, 'w': 3.0, 'h': 0.2},  # 水平墙 (0,4)→(3,4)
            {'type': 'rect', 'x': 5, 'y': 7.5, 'w': 0.2, 'h': 1.0},  # 垂直墙 (5,7)→(5,8)
        ],
        'kitchen': [
            {'type': 'rect', 'x': 5, 'y': 0, 'w': 10, 'h': 0.2},
            {'type': 'rect', 'x': 5, 'y': 10, 'w': 10, 'h': 0.2},
            {'type': 'rect', 'x': 0, 'y': 5, 'w': 0.2, 'h': 10},
            {'type': 'rect', 'x': 10, 'y': 5, 'w': 0.2, 'h': 10},
            # L 型墙
            {'type': 'rect', 'x': 5, 'y': 2, 'w': 0.2, 'h': 4.0},
            {'type': 'rect', 'x': 3, 'y': 8, 'w': 6.0, 'h': 0.2},
            {'type': 'rect', 'x': 6, 'y': 9, 'w': 0.2, 'h': 2.0},
        ],
        'living_room': [
            {'type': 'rect', 'x': 5, 'y': 0, 'w': 10, 'h': 0.2},
            {'type': 'rect', 'x': 5, 'y': 10, 'w': 10, 'h': 0.2},
            {'type': 'rect', 'x': 0, 'y': 5, 'w': 0.2, 'h': 10},
            {'type': 'rect', 'x': 10, 'y': 5, 'w': 0.2, 'h': 10},
            # 隔墙
            {'type': 'rect', 'x': 6, 'y': 1, 'w': 0.2, 'h': 2.0},
            {'type': 'rect', 'x': 6, 'y': 6, 'w': 0.2, 'h': 4.0},
            {'type': 'rect', 'x': 8, 'y': 6, 'w': 4.0, 'h': 0.2},
            {'type': 'rect', 'x': 5.5, 'y': 9, 'w': 0.2, 'h': 2.0},
            {'type': 'rect', 'x': 1.5, 'y': 8, 'w': 3.0, 'h': 0.2},
        ],
    }
    
    def __init__(self, scene='scene1', grid_size=0.3, robot_radius=0.3):
        """
        scene: 'scene1', 'kitchen', 'living_room'
        grid_size: 每个格子的尺寸(m)
        robot_radius: 机器人半径(m)，用于避障
        """
        self.grid_size = grid_size
        self.robot_radius = robot_radius
        self.walls = self.SCENE_WALLS.get(scene, self.SCENE_WALLS['scene1'])
        
        # 场景边界
        self.bounds = {'x_min': 0, 'x_max': 10, 'y_min': 0, 'y_max': 10}
        if scene == 'scene1':
            self.bounds = {'x_min': 0, 'x_max': 10, 'y_min': 0, 'y_max': 8}
        
        # 构建网格
        self.grid_w = int((self.bounds['x_max'] - self.bounds['x_min']) / grid_size) + 1
        self.grid_h = int((self.bounds['y_max'] - self.bounds['y_min']) / grid_size) + 1
        
    def _is_collision(self, x, y):
        """检查 (x, y) 是否与墙壁碰撞（带机器人半径）"""
        # 边界碰撞
        margin = self.robot_radius
        if (x - margin < self.bounds['x_min'] or x + margin > self.bounds['x_max'] or
            y - margin < self.bounds['y_min'] or y + margin > self.bounds['y_max']):
            return True
        
        # 墙壁碰撞（带机器人半径）
        for wall in self.walls:
            wx, wy, ww, wh = wall['x'], wall['y'], wall['w'], wall['h']
            # 墙壁的 AABB
            wall_min_x = wx - ww/2 - margin
            wall_max_x = wx + ww/2 + margin
            wall_min_y = wy - wh/2 - margin
            wall_max_y = wy + wh/2 + margin
            if wall_min_x < x < wall_max_x and wall_min_y < y < wall_max_y:
                return True
        return False
    
    def _grid_to_world(self, gx, gy):
        """网格坐标 → 世界坐标"""
        wx = self.bounds['x_min'] + gx * self.grid_size + self.grid_size / 2
        wy = self.bounds['y_min'] + gy * self.grid_size + self.grid_size / 2
        return wx, wy
    
    def _world_to_grid(self, wx, wy):
        """世界坐标 → 网格坐标"""
        gx = int((wx - self.bounds['x_min']) / self.grid_size)
        gy = int((wy - self.bounds['y_min']) / self.grid_size)
        return max(0, min(gx, self.grid_w - 1)), max(0, min(gy, self.grid_h - 1))
    
    def plan(self, start_x, start_y, end_x, end_y):
        """
        A* 寻路
        返回路径点列表 [(x1,y1), (x2,y2), ...]，包含起点和终点
        如果找不到路径，返回 None
        """
        start_gx, start_gy = self._world_to_grid(start_x, start_y)
        end_gx, end_gy = self._world_to_grid(end_x, end_y)
        
        # 检查起点终点是否可行
        if self._is_collision(start_x, start_y):
            print(f"  ⚠️ 起点 ({start_x:.1f},{start_y:.1f}) 在障碍物中")
            return None
        if self._is_collision(end_x, end_y):
            print(f"  ⚠️ 终点 ({end_x:.1f},{end_y:.1f}) 在障碍物中")
            return None
        
        def heuristic(gx, gy):
            return math.sqrt((gx - end_gx)**2 + (gy - end_gy)**2)
        
        open_set = [(0, start_gx, start_gy)]
        came_from = {}
        g_score = {(start_gx, start_gy): 0}
        f_score = {(start_gx, start_gy): heuristic(start_gx, start_gy)}
        
        while open_set:
            _, cx, cy = heapq.heappop(open_set)
            
            if (cx, cy) == (end_gx, end_gy):
                # 重建路径
                path = []
                while (cx, cy) in came_from:
                    wx, wy = self._grid_to_world(cx, cy)
                    path.append((wx, wy))
                    cx, cy = came_from[(cx, cy)]
                wx, wy = self._grid_to_world(cx, cy)
                path.append((wx, wy))
                path.reverse()
                # 加上精确终点
                path.append((end_x, end_y))
                return path
            
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
                ngx, ngy = cx + dx, cy + dy
                if 0 <= ngx < self.grid_w and 0 <= ngy < self.grid_h:
                    nwx, nwy = self._grid_to_world(ngx, ngy)
                    if self._is_collision(nwx, nwy):
                        continue
                    move_cost = math.sqrt(dx**2 + dy**2)
                    new_g = g_score.get((cx, cy), float('inf')) + move_cost
                    if new_g < g_score.get((ngx, ngy), float('inf')):
                        came_from[(ngx, ngy)] = (cx, cy)
                        g_score[(ngx, ngy)] = new_g
                        f = new_g + heuristic(ngx, ngy)
                        f_score[(ngx, ngy)] = f
                        heapq.heappush(open_set, (f, ngx, ngy))
        
        print(f"  ⚠️ A* 找不到路径: ({start_x:.1f},{start_y:.1f}) → ({end_x:.1f},{end_y:.1f})")
        return None


def smooth_move(pybullet_module, robot_body, target_pos, speed=0.5):
    """
    平滑移动机器人到目标位置
    使用线性插值，每步小幅度移动
    
    Args:
        pybullet_module: pybullet 模块引用
        robot_body: 机器人的 body ID
        target_pos: 目标位置 [x, y, z]
        speed: 移动速度 (m/s)，步进后自动调节
    
    Returns:
        bool: 是否成功到达
    """
    p = pybullet_module
    current_pos, current_orn = p.getBasePositionAndOrientation(robot_body)
    start_pos = list(current_pos)
    target = list(target_pos)
    
    # 距离
    dx = target[0] - start_pos[0]
    dy = target[1] - start_pos[1]
    dz = target[2] - start_pos[2] if len(target) > 2 else 0
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    if dist < 0.05:
        return True  # 已经在了
    
    # 插值步数（每步约0.1m）
    steps = max(int(dist / 0.08), 5)
    
    for i in range(1, steps + 1):
        t = i / steps
        nx = start_pos[0] + dx * t
        ny = start_pos[1] + dy * t
        nz = (start_pos[2] + dz * t) if len(target) > 2 else start_pos[2]
        p.resetBasePositionAndOrientation(robot_body, [nx, ny, nz], p.getQuaternionFromEuler([0, 0, 0]))
        # 步进仿真
        for _ in range(3):
            p.stepSimulation()
    
    return True


def navigate_along_path(p, robot_body, robot_arm_body, path, paired_body=None):
    """
    沿 A* 路径导航，平滑移动
    
    Args:
        p: pybullet 模块引用
        robot_body: 机器人底座 body ID
        robot_arm_body: 机器人手臂 body ID（如有）
        path: A* 返回的路径点列表
        paired_body: 配对的 body（如手臂跟随底座）
    """
    if not path:
        return False
    
    for i, (px, py) in enumerate(path):
        # 保持 Z=0（地面）
        p.resetBasePositionAndOrientation(robot_body, [px, py, 0], p.getQuaternionFromEuler([0, 0, 0]))
        if paired_body is not None:
            try:
                p.resetBasePositionAndOrientation(paired_body, [px, py, 0], p.getQuaternionFromEuler([0, 0, 0]))
            except:
                pass
        # 每步仿真
        for _ in range(2):
            p.stepSimulation()
    
    return True
