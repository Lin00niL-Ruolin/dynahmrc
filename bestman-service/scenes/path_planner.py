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
            {'x': 5, 'y': 0, 'w': 10, 'h': 0.15},
            {'x': 5, 'y': 8, 'w': 10, 'h': 0.15},
            {'x': 0, 'y': 4, 'w': 0.15, 'h': 8},
            {'x': 10, 'y': 4, 'w': 0.15, 'h': 8},
            # 内墙
            {'x': 5, 'y': 2.5, 'w': 0.15, 'h': 5.0},
            {'x': 1.5, 'y': 4, 'w': 3.0, 'h': 0.15},
            {'x': 5, 'y': 7.5, 'w': 0.15, 'h': 1.0},
            # 场景1家具（必须避让）
            {'x': 9.4, 'y': 0.5, 'w': 0.8, 'h': 0.8},    # fridge
            {'x': 7.4, 'y': 0.5, 'w': 0.8, 'h': 0.6},    # elementA
            {'x': 5.9, 'y': 0.5, 'w': 0.8, 'h': 0.6},    # elementB1
            {'x': 8.6, 'y': 0.5, 'w': 0.6, 'h': 0.6},    # elementC
            {'x': 8.1, 'y': 0.3, 'w': 0.6, 'h': 0.6},    # microwave
            {'x': 3, 'y': 2, 'w': 1.2, 'h': 0.8},        # table_dining
            {'x': 8.5, 'y': 4, 'w': 1.2, 'h': 0.8},      # table_new_1
            {'x': 8.5, 'y': 5.5, 'w': 1.2, 'h': 0.8},    # table_new_2
            {'x': 0.5, 'y': 0.5, 'w': 0.8, 'h': 0.6},    # bookshelf_1
            {'x': 0.5, 'y': 1.5, 'w': 0.8, 'h': 0.6},    # bookshelf_2
            {'x': 0.5, 'y': 2.5, 'w': 0.8, 'h': 0.6},    # bookshelf_3
        ],
        'kitchen': [
            {'x': 5, 'y': 0, 'w': 10, 'h': 0.15},
            {'x': 5, 'y': 10, 'w': 10, 'h': 0.15},
            {'x': 0, 'y': 5, 'w': 0.15, 'h': 10},
            {'x': 10, 'y': 5, 'w': 0.15, 'h': 10},
            {'x': 5, 'y': 2, 'w': 0.15, 'h': 4.0},
            {'x': 3, 'y': 8, 'w': 6.0, 'h': 0.15},
            {'x': 6, 'y': 9, 'w': 0.15, 'h': 2.0},
            # kitchen 家具
            {'x': 1, 'y': 0.5, 'w': 0.8, 'h': 0.6},     # elementB1
            {'x': 2.5, 'y': 0.5, 'w': 0.8, 'h': 0.6},   # elementA
            {'x': 3.2, 'y': 0.3, 'w': 0.6, 'h': 0.4},   # microwave
            {'x': 3.7, 'y': 0.5, 'w': 0.6, 'h': 0.4},   # elementC
            {'x': 4.5, 'y': 0.5, 'w': 0.8, 'h': 0.8},   # fridge
            {'x': 1, 'y': 4, 'w': 1.4, 'h': 1.0},        # table1
            {'x': 3, 'y': 5, 'w': 1.4, 'h': 1.0},        # table2
            {'x': 1, 'y': 6.5, 'w': 0.8, 'h': 0.6},      # bookcase
            {'x': 7.5, 'y': 9, 'w': 1.5, 'h': 0.8},      # sofa
        ],
        'living_room': [
            {'x': 5, 'y': 0, 'w': 10, 'h': 0.15},
            {'x': 5, 'y': 10, 'w': 10, 'h': 0.15},
            {'x': 0, 'y': 5, 'w': 0.15, 'h': 10},
            {'x': 10, 'y': 5, 'w': 0.15, 'h': 10},
            {'x': 6, 'y': 1, 'w': 0.15, 'h': 2.0},
            {'x': 6, 'y': 6, 'w': 0.15, 'h': 4.0},
            {'x': 8, 'y': 6, 'w': 4.0, 'h': 0.15},
            {'x': 5.5, 'y': 9, 'w': 0.15, 'h': 2.0},
            {'x': 1.5, 'y': 8, 'w': 3.0, 'h': 0.15},
            # living_room 家具
            {'x': 1.2, 'y': 0.5, 'w': 0.8, 'h': 0.6},   # kitchen_cabinet
            {'x': 3.1, 'y': 0.5, 'w': 0.8, 'h': 0.6},   # kitchen_counter
            {'x': 3.8, 'y': 0.3, 'w': 0.6, 'h': 0.4},   # microwave
            {'x': 4.6, 'y': 0.7, 'w': 0.6, 'h': 0.4},   # dishwasher
            {'x': 5.5, 'y': 0.5, 'w': 0.8, 'h': 0.8},   # fridge
            {'x': 7.3, 'y': 0.6, 'w': 0.8, 'h': 0.6},   # cabinet_2
            {'x': 8.6, 'y': 0.8, 'w': 1.5, 'h': 0.8},   # sofa
            {'x': 8, 'y': 3, 'w': 1.4, 'h': 1.0},        # packing_table
            {'x': 2, 'y': 4, 'w': 1.2, 'h': 1.0},        # source_table_1
            {'x': 4, 'y': 4, 'w': 1.2, 'h': 1.0},        # source_table_2
            {'x': 7.5, 'y': 5.5, 'w': 0.8, 'h': 0.6},   # bookcase
            {'x': 1.5, 'y': 9.4, 'w': 1.0, 'h': 0.6},   # bathtub
            {'x': 5.7, 'y': 6.0, 'w': 0.8, 'h': 0.3},   # wall_shelf
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


def navigate_along_path(p, robot_body, path, paired_body=None, paired_z=None, steps_per_point=5):
    """
    沿 A* 路径平滑移动（慢速）
    配对 body（如手臂）保持原始 Z 坐标不变，只移动 XY
    
    Args:
        p: pybullet 模块引用
        robot_body: 机器人底座 body ID
        path: A* 返回的路径点列表 [(x1,y1), (x2,y2), ...]
        paired_body: 配对的 body（如手臂跟随底座）
        paired_z: 配对 body 的 Z 坐标（如不传则自动获取）
        steps_per_point: 每路径点的仿真步数（越大越慢）
    """
    if not path or len(path) < 2:
        return False
    
    # 如果配对 body 存在，获取/记录其 Z 坐标（保持不动）
    arm_z = None
    if paired_body is not None:
        try:
            arm_z = paired_z if paired_z is not None else p.getBasePositionAndOrientation(paired_body)[0][2]
        except:
            pass
    
    for i in range(1, len(path)):
        px, py = path[i]
        p.resetBasePositionAndOrientation(robot_body, [px, py, 0], p.getQuaternionFromEuler([0, 0, 0]))
        if paired_body is not None and arm_z is not None:
            try:
                # 只移动 XY，保持 Z 不变！
                p.resetBasePositionAndOrientation(paired_body, [px, py, arm_z], p.getQuaternionFromEuler([0, 0, 0]))
            except:
                pass
        for _ in range(steps_per_point):
            p.stepSimulation()
    
    return True
