#!/usr/bin/env python3
"""
path_planner.py
A* 路径规划器 + 碰撞检测
用于 BestMan 仿真中机器人的路径规划和避障
"""

import math
import heapq


class AStarPathPlanner:
    """A* 路径规划器（网格密度 0.15m，严格避障）"""

    # 各场景障碍物：墙壁 + 家具
    OBSTACLES = {
        'scene1': [
            # 外墙
            (0.0, 0.0, 10.0, 0.2),   # 下
            (0.0, 7.8, 10.0, 0.2),  # 上
            (0.0, 0.0, 0.2, 8.0),   # 左
            (9.8, 0.0, 0.2, 8.0),   # 右
            # 内墙
            (4.925, 0.0, 0.15, 5.0),  # 垂直 (5,0)→(5,5)
            (0.0, 3.925, 3.0, 0.15),  # 水平 (0,4)→(3,4)
            (4.925, 7.0, 0.15, 1.0),  # 垂直 (5,7)→(5,8)
            # 家具（x, y, w, h）— 比实际尺寸大 0.3m 作为安全距离
            (9.0, 0.1, 1.2, 1.2),     # fridge
            (7.0, 0.2, 1.2, 1.0),    # elementA
            (5.5, 0.2, 1.2, 1.0),    # elementB1
            (8.3, 0.1, 1.0, 1.0),    # elementC
            (7.8, 0.0, 1.0, 0.9),    # microwave
            (2.5, 1.6, 1.6, 1.2),    # table_dining
            (8.0, 3.6, 1.6, 1.2),    # table_new_1
            (8.0, 5.1, 1.6, 1.2),    # table_new_2
            (0.1, 0.1, 1.2, 1.0),    # bookshelf_1
            (0.1, 1.1, 1.2, 1.0),    # bookshelf_2
            (0.1, 2.1, 1.2, 1.0),    # bookshelf_3
        ],
        'kitchen': [
            (0.0, 0.0, 10.0, 0.2),
            (0.0, 9.8, 10.0, 0.2),
            (0.0, 0.0, 0.2, 10.0),
            (9.8, 0.0, 0.2, 10.0),
            (4.925, 0.0, 0.15, 4.0),
            (0.0, 7.925, 6.0, 0.15),
            (5.925, 8.0, 0.15, 2.0),
            # 家具
            (0.6, 0.1, 1.2, 1.0),    # elementB1
            (2.1, 0.1, 1.2, 1.0),    # elementA
            (2.9, 0.0, 1.0, 0.8),    # microwave
            (3.4, 0.1, 1.0, 0.8),    # elementC
            (4.1, 0.1, 1.2, 1.2),    # fridge
            (0.4, 3.5, 1.8, 1.4),    # table1
            (2.4, 4.5, 1.8, 1.4),    # table2
            (0.6, 6.1, 1.2, 1.0),    # bookcase
            (7.0, 8.6, 1.9, 1.2),    # sofa
        ],
        'living_room': [
            (0.0, 0.0, 10.0, 0.2),
            (0.0, 9.8, 10.0, 0.2),
            (0.0, 0.0, 0.2, 10.0),
            (9.8, 0.0, 0.2, 10.0),
            (5.925, 0.0, 0.15, 2.0),
            (5.925, 4.0, 0.15, 4.0),
            (6.0, 5.925, 4.0, 0.15),
            (5.425, 8.0, 0.15, 2.0),
            (0.0, 7.925, 3.0, 0.15),
            # 家具
            (0.8, 0.1, 1.2, 1.0),    # kitchen_cabinet
            (2.7, 0.1, 1.2, 1.0),    # kitchen_counter
            (3.5, 0.0, 1.0, 0.8),    # microwave
            (4.3, 0.4, 1.0, 0.8),    # dishwasher
            (5.1, 0.1, 1.2, 1.2),    # fridge
            (6.9, 0.3, 1.2, 1.0),    # cabinet_2
            (8.2, 0.4, 1.9, 1.2),    # sofa
            (7.5, 2.5, 1.8, 1.4),    # packing_table
            (1.5, 3.5, 1.6, 1.4),    # source_table_1
            (3.5, 3.5, 1.6, 1.4),    # source_table_2
            (7.1, 5.1, 1.2, 1.0),    # bookcase
            (1.1, 9.0, 1.4, 1.0),    # bathtub
            (5.3, 5.7, 1.2, 0.7),    # wall_shelf
        ],
    }

    def __init__(self, scene='scene1', grid_size=0.15, robot_radius=0.25):
        self.grid_size = grid_size
        self.robot_radius = robot_radius
        
        obstacles = self.OBSTACLES.get(scene, self.OBSTACLES['scene1'])
        
        # 场景边界
        if scene == 'scene1':
            self.bounds = (0.0, 0.0, 10.0, 8.0)
        else:
            self.bounds = (0.0, 0.0, 10.0, 10.0)
        
        bx, by, bw, bh = self.bounds
        self.grid_w = int(bw / grid_size) + 1
        self.grid_h = int(bh / grid_size) + 1
        
        # 预计算碰撞网格（0=可通行，1=障碍）
        self.collision_grid = [[0] * self.grid_h for _ in range(self.grid_w)]
        margin = robot_radius
        
        for gx in range(self.grid_w):
            for gy in range(self.grid_h):
                wx = bx + gx * grid_size + grid_size / 2
                wy = by + gy * grid_size + grid_size / 2
                
                # 边界检查
                if (wx - margin < bx or wx + margin > bx + bw or
                    wy - margin < by or wy + margin > by + bh):
                    self.collision_grid[gx][gy] = 1
                    continue
                
                # 障碍物检查
                for ox, oy, ow, oh in obstacles:
                    obj_min_x = ox - margin
                    obj_max_x = ox + ow + margin
                    obj_min_y = oy - margin
                    obj_max_y = oy + oh + margin
                    if obj_min_x < wx < obj_max_x and obj_min_y < wy < obj_max_y:
                        self.collision_grid[gx][gy] = 1
                        break

    def is_collision(self, x, y):
        """检查世界坐标 (x,y) 是否碰撞"""
        bx, by, bw, bh = self.bounds
        margin = self.robot_radius
        if (x - margin < bx or x + margin > bx + bw or
            y - margin < by or y + margin > by + bh):
            return True
        gx = int((x - bx) / self.grid_size)
        gy = int((y - by) / self.grid_size)
        if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
            return self.collision_grid[gx][gy] == 1
        return True

    def plan(self, start_x, start_y, end_x, end_y):
        """A* 寻路，返回路径点列表[(x1,y1),...]"""
        bx, by, bw, bh = self.bounds
        
        if self.is_collision(start_x, start_y):
            print(f"  ⚠️ 起点({start_x:.1f},{start_y:.1f})在障碍物中")
            return None
        if self.is_collision(end_x, end_y):
            print(f"  ⚠️ 终点({end_x:.1f},{end_y:.1f})在障碍物中")
            return None
        
        sgx = int((start_x - bx) / self.grid_size)
        sgy = int((start_y - by) / self.grid_size)
        egx = int((end_x - bx) / self.grid_size)
        egy = int((end_y - by) / self.grid_size)
        
        def heuristic(gx, gy):
            return math.sqrt((gx - egx)**2 + (gy - egy)**2)
        
        open_set = [(0, sgx, sgy)]
        came_from = {}
        g_score = {(sgx, sgy): 0}
        
        while open_set:
            _, cx, cy = heapq.heappop(open_set)
            if (cx, cy) == (egx, egy):
                path = []
                while (cx, cy) in came_from:
                    path.append((bx + cx * self.grid_size + self.grid_size/2,
                                 by + cy * self.grid_size + self.grid_size/2))
                    cx, cy = came_from[(cx, cy)]
                path.append((bx + cx * self.grid_size + self.grid_size/2,
                             by + cy * self.grid_size + self.grid_size/2))
                path.reverse()
                path.append((end_x, end_y))
                return path
            
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]:
                ngx, ngy = cx + dx, cy + dy
                if 0 <= ngx < self.grid_w and 0 <= ngy < self.grid_h:
                    if self.collision_grid[ngx][ngy]:
                        continue
                    move_cost = math.sqrt(dx**2 + dy**2)
                    new_g = g_score.get((cx, cy), float('inf')) + move_cost
                    if new_g < g_score.get((ngx, ngy), float('inf')):
                        came_from[(ngx, ngy)] = (cx, cy)
                        g_score[(ngx, ngy)] = new_g
                        heapq.heappush(open_set, (new_g + heuristic(ngx, ngy), ngx, ngy))
        
        print(f"  ⚠️ A* 无路径: ({start_x:.1f},{start_y:.1f})→({end_x:.1f},{end_y:.1f})")
        return None
