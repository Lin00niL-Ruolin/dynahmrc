"""
路径规划模块
包含 A* 全局路径规划和 DWA 局部避障算法
"""

import math
import heapq
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
import pybullet as p


class Node:
    """A* 节点"""
    def __init__(self, x: int, y: int, g: float = 0.0, h: float = 0.0):
        self.x = x
        self.y = y
        self.g = g  # 从起点到当前节点的代价
        self.h = h  # 启发式代价（到终点的估计）
        self.f = g + h  # 总代价
        self.parent = None
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))


class AStarPlanner:
    """A* 全局路径规划器 - 支持动态障碍物和智能点转换"""
    
    def __init__(self, resolution: float = 0.1, robot_radius: float = 0.3):
        """
        初始化 A* 规划器
        
        Args:
            resolution: 栅格地图分辨率（米）
            robot_radius: 机器人半径（米）
        """
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.static_obstacles: Set[Tuple[int, int]] = set()  # 静态障碍物
        self.dynamic_obstacles: Dict[str, Tuple[int, int]] = {}  # 动态障碍物 {robot_id: (x, y)}
        self.obstacles: Set[Tuple[int, int]] = set()  # 合并后的障碍物
        
        # 动态规划缓存
        self.path_cache: Dict[str, List[List[float]]] = {}  # 路径缓存
        self.cache_timestamp: Dict[str, float] = {}  # 缓存时间戳
        self.cache_validity = 5.0  # 缓存有效期（秒）
        
        # 异常统计
        self.stats = {
            'planning_attempts': 0,
            'planning_success': 0,
            'planning_failures': 0,
            'point_conversions': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def update_static_obstacles(self, obstacle_positions: List[List[float]], 
                                 obstacle_sizes: Optional[List[float]] = None, 
                                 client_id: int = 0):
        """
        更新静态障碍物地图
        
        Args:
            obstacle_positions: 障碍物位置列表 [[x, y, z], ...]
            obstacle_sizes: 障碍物尺寸列表（半径），可选
            client_id: PyBullet 客户端 ID
        """
        self.static_obstacles.clear()
        
        print(f"[A*] 更新静态障碍物地图: {len(obstacle_positions)} 个障碍物")
        
        for i, pos in enumerate(obstacle_positions):
            try:
                # 将障碍物位置转换为栅格坐标
                grid_x = int(pos[0] / self.resolution)
                grid_y = int(pos[1] / self.resolution)
                
                # 获取障碍物尺寸（如果有）
                obj_radius = obstacle_sizes[i] if obstacle_sizes and i < len(obstacle_sizes) else 0.0
                
                # 添加障碍物及其周围区域（考虑机器人半径 + 障碍物尺寸）
                # 减小膨胀半径，只使用机器人半径 + 一半物体半径 + 安全距离
                total_radius = self.robot_radius + obj_radius * 0.5 + 0.05
                radius_cells = int(total_radius / self.resolution) + 1
                
                if i < 5:  # 打印前5个障碍物的详细信息
                    print(f"[A*] 静态障碍物 {i}: 位置 ({pos[0]:.2f}, {pos[1]:.2f}) -> 栅格 ({grid_x}, {grid_y}), "
                          f"物体半径 {obj_radius:.2f}m, 膨胀半径 {total_radius:.2f}m ({radius_cells} 格)")
                
                for dx in range(-radius_cells, radius_cells + 1):
                    for dy in range(-radius_cells, radius_cells + 1):
                        if dx * dx + dy * dy <= radius_cells * radius_cells:
                            self.static_obstacles.add((grid_x + dx, grid_y + dy))
            except Exception as e:
                print(f"[A*] 警告: 处理障碍物 {i} 时出错: {e}")
                continue
        
        self._merge_obstacles()
        print(f"[A*] 静态障碍物地图更新完成: 共 {len(self.static_obstacles)} 个栅格, "
              f"分辨率 {self.resolution}m, 机器人半径 {self.robot_radius}m")
    
    def update_dynamic_obstacles(self, robot_positions: Dict[str, List[float]]):
        """
        更新动态障碍物（其他机器人位置）
        
        Args:
            robot_positions: 机器人位置字典 {robot_id: [x, y, z], ...}
        """
        self.dynamic_obstacles.clear()
        
        for robot_id, pos in robot_positions.items():
            try:
                grid_x = int(pos[0] / self.resolution)
                grid_y = int(pos[1] / self.resolution)
                self.dynamic_obstacles[robot_id] = (grid_x, grid_y)
            except Exception as e:
                print(f"[A*] 警告: 更新动态障碍物 {robot_id} 时出错: {e}")
                continue
        
        self._merge_obstacles()
    
    def _merge_obstacles(self):
        """合并静态和动态障碍物"""
        self.obstacles = self.static_obstacles.copy()
        
        # 为动态障碍物添加膨胀区域
        for robot_id, (grid_x, grid_y) in self.dynamic_obstacles.items():
            # 动态障碍物使用更大的安全距离
            radius_cells = int((self.robot_radius * 2) / self.resolution) + 1
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy <= radius_cells * radius_cells:
                        self.obstacles.add((grid_x + dx, grid_y + dy))
    
    def update_obstacles(self, obstacle_positions: List[List[float]], 
                        obstacle_sizes: Optional[List[float]] = None, 
                        client_id: int = 0):
        """兼容旧接口，更新静态障碍物"""
        self.update_static_obstacles(obstacle_positions, obstacle_sizes, client_id)
    
    def plan(self, start: List[float], goal: List[float], 
             robot_id: str = None,
             use_cache: bool = True,
             max_retries: int = 3) -> Optional[List[List[float]]]:
        """
        规划路径 - 支持动态障碍物、缓存和智能点转换
        
        Args:
            start: 起点 [x, y]
            goal: 终点 [x, y]
            robot_id: 机器人ID（用于动态障碍物排除和缓存）
            use_cache: 是否使用缓存
            max_retries: 最大重试次数
        
        Returns:
            路径点列表 [[x, y], ...]，如果无法到达返回 None
        """
        import time
        self.stats['planning_attempts'] += 1
        start_time = time.time()
        
        # 生成缓存键
        cache_key = None
        if robot_id and use_cache:
            cache_key = f"{robot_id}_{start[0]:.2f}_{start[1]:.2f}_{goal[0]:.2f}_{goal[1]:.2f}"
            
            # 检查缓存是否有效
            if cache_key in self.path_cache:
                cache_age = time.time() - self.cache_timestamp.get(cache_key, 0)
                if cache_age < self.cache_validity:
                    self.stats['cache_hits'] += 1
                    print(f"[A*] 使用缓存路径 (年龄: {cache_age:.1f}s)")
                    return self.path_cache[cache_key].copy()
                else:
                    # 缓存过期
                    del self.path_cache[cache_key]
                    del self.cache_timestamp[cache_key]
            
            self.stats['cache_misses'] += 1
        
        # 转换为栅格坐标
        start_node = Node(int(start[0] / self.resolution), int(start[1] / self.resolution))
        goal_node = Node(int(goal[0] / self.resolution), int(goal[1] / self.resolution))
        
        # 智能点转换：将不合理的点（在障碍物中）转换为合理的点
        start_converted = False
        goal_converted = False
        
        if (start_node.x, start_node.y) in self.obstacles:
            print(f"[A*] 起点在障碍物中: ({start_node.x}, {start_node.y})，尝试转换...")
            converted = self._convert_invalid_point(start_node.x, start_node.y, 'start')
            if converted:
                start_node.x, start_node.y = converted
                start_converted = True
                self.stats['point_conversions'] += 1
                print(f"[A*] 起点已转换到: ({start_node.x}, {start_node.y})")
            else:
                # 无法转换，清除附近障碍物
                self._clear_nearby_obstacles(start_node.x, start_node.y, radius=2)
        
        if (goal_node.x, goal_node.y) in self.obstacles:
            print(f"[A*] 终点在障碍物中: ({goal_node.x}, {goal_node.y})，尝试转换...")
            converted = self._convert_invalid_point(goal_node.x, goal_node.y, 'goal')
            if converted:
                goal_node.x, goal_node.y = converted
                goal_converted = True
                self.stats['point_conversions'] += 1
                print(f"[A*] 终点已转换到: ({goal_node.x}, {goal_node.y})")
            else:
                # 无法转换，清除附近障碍物
                self._clear_nearby_obstacles(goal_node.x, goal_node.y, radius=2)
        
        # A* 算法
        open_set = []
        heapq.heappush(open_set, start_node)
        closed_set: Set[Tuple[int, int]] = set()
        open_set_dict: Dict[Tuple[int, int], Node] = {(start_node.x, start_node.y): start_node}
        
        print(f"[A*] 开始规划: 起点 ({start_node.x}, {start_node.y}) -> 终点 ({goal_node.x}, {goal_node.y})")
        print(f"[A*] 距离: {math.sqrt((start_node.x - goal_node.x)**2 + (start_node.y - goal_node.y)**2):.1f} 格")
        print(f"[A*] 障碍物数量: {len(self.obstacles)} 个栅格")
        
        # 打印障碍物边界框
        if self.obstacles:
            min_x = min(x for x, y in self.obstacles)
            max_x = max(x for x, y in self.obstacles)
            min_y = min(y for x, y in self.obstacles)
            max_y = max(y for x, y in self.obstacles)
            print(f"[A*] 障碍物范围: X[{min_x}, {max_x}], Y[{min_y}, {max_y}]")
            print(f"[A*] 起点在范围内: X({min_x <= start_node.x <= max_x}), Y({min_y <= start_node.y <= max_y})")
            print(f"[A*] 终点在范围内: X({min_x <= goal_node.x <= max_x}), Y({min_y <= goal_node.y <= max_y})")
        
        # 检查起点和终点周围是否被障碍物包围
        def count_obstacles_around(x, y, radius=2):
            count = 0
            total = 0
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    total += 1
                    if (x + dx, y + dy) in self.obstacles:
                        count += 1
            return count, total
        
        start_obs, start_total = count_obstacles_around(start_node.x, start_node.y)
        goal_obs, goal_total = count_obstacles_around(goal_node.x, goal_node.y)
        print(f"[A*] 起点周围障碍物: {start_obs}/{start_total} ({start_obs/start_total*100:.1f}%)")
        print(f"[A*] 终点周围障碍物: {goal_obs}/{goal_total} ({goal_obs/goal_total*100:.1f}%)")
        
        iteration = 0
        max_iterations = 9000  # 最大迭代次数限制
        
        while open_set and iteration < max_iterations:
            current = heapq.heappop(open_set)
            iteration += 1
            
            # 从字典中移除
            if (current.x, current.y) in open_set_dict:
                del open_set_dict[(current.x, current.y)]
            
            # 每 500 次迭代打印进度
            if iteration % 500 == 0:
                print(f"[A*] 规划中... 迭代 {iteration}, 开放集 {len(open_set)}, 已探索 {len(closed_set)}")
            
            # 到达目标
            if current == goal_node:
                path = self._reconstruct_path(current, cache_key)
                self.stats['planning_success'] += 1
                elapsed_time = time.time() - start_time
                print(f"[A*] 路径规划成功! 迭代 {iteration} 次, 路径长度 {len(path)} 点, 耗时 {elapsed_time:.3f}s")
                print(f"[A*] 路径: {path[:5]}{'...' if len(path) > 5 else ''}")
                return path
            
            closed_set.add((current.x, current.y))
            
            # 扩展邻居
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), 
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor_x = current.x + dx
                neighbor_y = current.y + dy
                
                # 检查是否已访问或是障碍物
                if (neighbor_x, neighbor_y) in closed_set:
                    continue
                if (neighbor_x, neighbor_y) in self.obstacles:
                    continue
                
                # 计算代价
                move_cost = math.sqrt(dx * dx + dy * dy) * self.resolution
                g = current.g + move_cost
                h = self._heuristic(neighbor_x, neighbor_y, goal_node)
                
                # 检查是否已经在开放集中且代价更高
                neighbor_key = (neighbor_x, neighbor_y)
                if neighbor_key in open_set_dict:
                    existing_node = open_set_dict[neighbor_key]
                    if g >= existing_node.g:
                        continue  # 已有更优路径
                
                neighbor = Node(neighbor_x, neighbor_y, g, h)
                neighbor.parent = current
                
                heapq.heappush(open_set, neighbor)
                open_set_dict[neighbor_key] = neighbor
        
        if iteration >= max_iterations:
            print(f"[A*] 规划超时! 达到最大迭代次数 {max_iterations}")
        else:
            print(f"[A*] 无法找到路径! 迭代 {iteration} 次, 开放集为空")
        print(f"[A*] 已探索节点数: {len(closed_set)}")
        
        # 更新统计
        self.stats['planning_failures'] += 1
        elapsed_time = time.time() - start_time
        print(f"[A*] 规划失败，耗时 {elapsed_time:.3f}s")
        
        # 尝试扩大搜索范围：清除起点和终点周围的障碍物后重试
        print(f"[A*] 尝试扩大搜索范围，清除起点和终点周围障碍物...")
        self._clear_nearby_obstacles(start_node.x, start_node.y, radius=3)
        self._clear_nearby_obstacles(goal_node.x, goal_node.y, radius=3)
        
        # 重新规划
        print(f"[A*] 重新规划路径...")
        return self._plan_with_current_obstacles(start_node, goal_node, cache_key)
    
    def _convert_invalid_point(self, x: int, y: int, point_type: str = 'point') -> Optional[Tuple[int, int]]:
        """
        将不合理的点（在障碍物中）转换为合理的点
        
        策略：
        1. 优先在原地周围寻找最近的空闲点
        2. 使用螺旋搜索模式，从内到外
        3. 对于起点：优先寻找机器人当前朝向的方向
        4. 对于终点：优先寻找从起点到终点的延长线方向
        
        Args:
            x: 原始栅格X坐标
            y: 原始栅格Y坐标
            point_type: 'start' 或 'goal'
        
        Returns:
            转换后的 (x, y) 坐标，如果找不到则返回 None
        """
        print(f"[A*] 开始点转换: ({x}, {y}) 类型={point_type}")
        
        # 搜索半径逐步扩大
        for radius in range(1, 10):
            candidates = []
            
            # 在当前半径上搜索所有点
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # 只检查半径边界上的点（避免重复检查）
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    
                    nx, ny = x + dx, y + dy
                    
                    # 检查是否在障碍物外
                    if (nx, ny) not in self.obstacles:
                        # 计算距离和方向得分
                        distance = math.sqrt(dx * dx + dy * dy)
                        
                        # 检查周围是否有足够的自由空间（避免狭窄区域）
                        free_space = self._count_free_space(nx, ny, radius=2)
                        
                        if free_space >= 4:  # 至少4个方向是自由的
                            candidates.append((nx, ny, distance, free_space))
            
            if candidates:
                # 按距离优先，其次按自由空间排序
                candidates.sort(key=lambda c: (c[2], -c[3]))
                best = candidates[0]
                print(f"[A*] 点转换成功: ({x}, {y}) -> ({best[0]}, {best[1]}), "
                      f"距离={best[2]:.1f}, 自由空间={best[3]}")
                return (best[0], best[1])
        
        print(f"[A*] 点转换失败: 在 ({x}, {y}) 周围找不到合理的点")
        return None
    
    def _count_free_space(self, x: int, y: int, radius: int = 2) -> int:
        """计算某点周围自由空间的数量（8方向）"""
        free_count = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in self.obstacles:
                free_count += 1
        
        return free_count
    
    def _clear_nearby_obstacles(self, x: int, y: int, radius: int = 2):
        """清除指定位置附近的障碍物"""
        cleared_count = 0
        cleared_positions = []
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    pos = (x + dx, y + dy)
                    if pos in self.obstacles:
                        self.obstacles.discard(pos)
                        self.static_obstacles.discard(pos)  # 同时清除静态障碍物
                        cleared_count += 1
                        cleared_positions.append(pos)
        
        if cleared_count > 0:
            print(f"[A*] 清除障碍物: 位置 ({x}, {y}) 周围 {radius} 格，共清除 {cleared_count} 个障碍物")
        else:
            print(f"[A*] 位置 ({x}, {y}) 周围没有需要清除的障碍物")
    
    def _plan_with_current_obstacles(self, start_node: Node, goal_node: Node, cache_key: str = None) -> Optional[List[List[float]]]:
        """使用当前的障碍物地图重新规划路径（使用更宽容的启发式，允许绕路）"""
        import time
        retry_start_time = time.time()
        
        open_set = []
        heapq.heappush(open_set, start_node)
        closed_set: Set[Tuple[int, int]] = set()
        open_set_dict: Dict[Tuple[int, int], Node] = {(start_node.x, start_node.y): start_node}
        
        iteration = 0
        max_iterations = 8000  # 增加迭代次数，允许更多探索
        
        # 计算起点到终点的直线距离
        direct_distance = math.sqrt((start_node.x - goal_node.x)**2 + (start_node.y - goal_node.y)**2)
        
        # 动态调整启发式权重：距离越远，权重越低（更愿意绕路）
        heuristic_weight = max(0.01, min(1.0, 3.0 / direct_distance)) if direct_distance > 0 else 0.5
        
        print(f"[A*] 使用宽容启发式，权重={heuristic_weight:.2f}，允许绕路探索")
        
        while open_set and iteration < max_iterations:
            current = heapq.heappop(open_set)
            iteration += 1
            
            if (current.x, current.y) in open_set_dict:
                del open_set_dict[(current.x, current.y)]
            
            if current == goal_node:
                path = self._reconstruct_path(current, cache_key)
                self.stats['planning_success'] += 1
                elapsed_time = time.time() - retry_start_time
                print(f"[A*] 重新规划成功! 迭代 {iteration} 次, 路径长度 {len(path)} 点, 耗时 {elapsed_time:.3f}s")
                return path
            
            closed_set.add((current.x, current.y))
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), 
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor_x = current.x + dx
                neighbor_y = current.y + dy
                
                if (neighbor_x, neighbor_y) in closed_set:
                    continue
                if (neighbor_x, neighbor_y) in self.obstacles:
                    continue
                
                move_cost = math.sqrt(dx * dx + dy * dy) * self.resolution
                g = current.g + move_cost
                
                # 使用加权启发式，降低对目标的"执念"，允许更多绕路
                h = self._heuristic(neighbor_x, neighbor_y, goal_node) * heuristic_weight
                
                neighbor_key = (neighbor_x, neighbor_y)
                if neighbor_key in open_set_dict:
                    existing = open_set_dict[neighbor_key]
                    if g < existing.g:
                        existing.g = g
                        existing.f = g + h
                        existing.parent = current
                else:
                    neighbor = Node(neighbor_x, neighbor_y, g, h)
                    neighbor.parent = current
                    heapq.heappush(open_set, neighbor)
                    open_set_dict[neighbor_key] = neighbor
        
        if iteration >= max_iterations:
            print(f"[A*] 重新规划超时! 已探索 {len(closed_set)} 个节点")
        else:
            print(f"[A*] 重新规划仍无法找到路径! 已探索 {len(closed_set)} 个节点")
        
        self.stats['planning_failures'] += 1
        return None
    
    def _heuristic(self, x: int, y: int, goal: Node) -> float:
        """启发式函数（欧几里得距离）"""
        return math.sqrt((x - goal.x) ** 2 + (y - goal.y) ** 2) * self.resolution
    
    def _reconstruct_path(self, node: Node, cache_key: str = None) -> List[List[float]]:
        """重建路径并可选地缓存结果"""
        import time
        path = []
        current = node
        while current:
            path.append([
                current.x * self.resolution,
                current.y * self.resolution
            ])
            current = current.parent
        path = path[::-1]
        
        # 检查路径是否穿过障碍物
        collision_points = []
        for i, point in enumerate(path):
            grid_x = int(point[0] / self.resolution)
            grid_y = int(point[1] / self.resolution)
            if (grid_x, grid_y) in self.obstacles:
                collision_points.append((i, point, (grid_x, grid_y)))
        
        if collision_points:
            print(f"[A*] ⚠️ 警告: 路径穿过障碍物! 碰撞点数量: {len(collision_points)}")
            for idx, point, grid in collision_points[:5]:  # 只显示前5个
                print(f"[A*]   点 {idx}: 实际位置 [{point[0]:.2f}, {point[1]:.2f}], 栅格 {grid}")
        else:
            print(f"[A*] ✓ 路径检查通过，无碰撞")
        
        # 缓存路径
        if cache_key:
            self.path_cache[cache_key] = path.copy()
            self.cache_timestamp[cache_key] = time.time()
            print(f"[A*] 路径已缓存: {cache_key}")
        
        return path
    
    def get_stats(self) -> Dict:
        """获取规划统计信息"""
        return self.stats.copy()
    
    def clear_cache(self):
        """清除路径缓存"""
        self.path_cache.clear()
        self.cache_timestamp.clear()
        print("[A*] 路径缓存已清除")
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'planning_attempts': 0,
            'planning_success': 0,
            'planning_failures': 0,
            'point_conversions': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }


class DWAPlanner:
    """DWA (Dynamic Window Approach) 局部避障规划器"""
    
    def __init__(self, 
                 max_speed: float = 0.5,
                 max_yaw_rate: float = 1.0,
                 max_accel: float = 0.5,
                 max_yaw_accel: float = 1.0,
                 dt: float = 0.1,
                 predict_time: float = 3.0,
                 robot_radius: float = 0.3):
        """
        初始化 DWA 规划器
        
        Args:
            max_speed: 最大线速度 (m/s)
            max_yaw_rate: 最大角速度 (rad/s)
            max_accel: 最大线加速度 (m/s^2)
            max_yaw_accel: 最大角加速度 (rad/s^2)
            dt: 时间步长 (s)
            predict_time: 预测时间 (s)
            robot_radius: 机器人半径 (m)
        """
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.max_accel = max_accel
        self.max_yaw_accel = max_yaw_accel
        self.dt = dt
        self.predict_time = predict_time
        self.robot_radius = robot_radius
        
        # 速度采样参数
        self.v_resolution = 0.05
        self.yaw_rate_resolution = 0.1
        
        # 代价权重
        self.weight_goal = 1.0
        self.weight_speed = 0.1
        self.weight_obstacle = 1.0
    
    def plan(self, 
             current_pos: List[float],
             current_yaw: float,
             current_v: float,
             current_yaw_rate: float,
             goal: List[float],
             obstacles: List[List[float]]) -> Tuple[float, float]:
        """
        计算最优速度指令
        
        Args:
            current_pos: 当前位置 [x, y]
            current_yaw: 当前朝向 (rad)
            current_v: 当前线速度
            current_yaw_rate: 当前角速度
            goal: 目标位置 [x, y]
            obstacles: 障碍物位置列表 [[x, y], ...]
        
        Returns:
            (v, yaw_rate) 最优速度指令
        """
        # 计算动态窗口
        dw = self._calc_dynamic_window(current_v, current_yaw_rate)
        
        best_v = 0.0
        best_yaw_rate = 0.0
        min_cost = float('inf')
        
        # 采样速度空间 - 使用 linspace 确保至少有一定数量的采样点
        v_samples = max(3, int((dw[1] - dw[0]) / self.v_resolution) + 1)
        yaw_rate_samples = max(3, int((dw[3] - dw[2]) / self.yaw_rate_resolution) + 1)
        
        for v in np.linspace(dw[0], dw[1], v_samples):
            for yaw_rate in np.linspace(dw[2], dw[3], yaw_rate_samples):
                # 预测轨迹
                trajectory = self._predict_trajectory(
                    current_pos, current_yaw, v, yaw_rate
                )
                
                # 计算代价
                goal_cost = self._calc_goal_cost(trajectory, goal)
                speed_cost = self._calc_speed_cost(v)
                obstacle_cost = self._calc_obstacle_cost(trajectory, obstacles)
                
                total_cost = (self.weight_goal * goal_cost +
                             self.weight_speed * speed_cost +
                             self.weight_obstacle * obstacle_cost)
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_v = v
                    best_yaw_rate = yaw_rate
        
        return best_v, best_yaw_rate
    
    def _calc_dynamic_window(self, v: float, yaw_rate: float) -> List[float]:
        """计算动态窗口"""
        # 速度限制
        vs = [0.0, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]
        
        # 加速度限制
        vd = [
            v - self.max_accel * self.dt,
            v + self.max_accel * self.dt,
            yaw_rate - self.max_yaw_accel * self.dt,
            yaw_rate + self.max_yaw_accel * self.dt
        ]
        
        # 动态窗口是两者的交集
        return [
            max(vs[0], vd[0]),
            min(vs[1], vd[1]),
            max(vs[2], vd[2]),
            min(vs[3], vd[3])
        ]
    
    def _predict_trajectory(self, 
                           pos: List[float], 
                           yaw: float, 
                           v: float, 
                           yaw_rate: float) -> List[List[float]]:
        """预测轨迹"""
        trajectory = [[pos[0], pos[1]]]
        x, y = pos[0], pos[1]
        
        time = 0.0
        while time <= self.predict_time:
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
            yaw += yaw_rate * self.dt
            trajectory.append([x, y])
            time += self.dt
        
        return trajectory
    
    def _calc_goal_cost(self, trajectory: List[List[float]], goal: List[float]) -> float:
        """计算目标代价（距离目标的距离）"""
        final_pos = trajectory[-1]
        return math.sqrt((final_pos[0] - goal[0]) ** 2 + (final_pos[1] - goal[1]) ** 2)
    
    def _calc_speed_cost(self, v: float) -> float:
        """计算速度代价（鼓励高速）"""
        return self.max_speed - v
    
    def _calc_obstacle_cost(self, trajectory: List[List[float]], 
                           obstacles: List[List[float]]) -> float:
        """计算障碍物代价"""
        min_distance = float('inf')
        
        for point in trajectory:
            for obs in obstacles:
                distance = math.sqrt((point[0] - obs[0]) ** 2 + (point[1] - obs[1]) ** 2)
                if distance < min_distance:
                    min_distance = distance
        
        # 如果距离小于机器人半径，代价很大
        if min_distance < self.robot_radius:
            return float('inf')
        
        return 1.0 / min_distance


class PathPlanner:
    """组合路径规划器（A* + DWA）"""
    
    def __init__(self, client_id: int = 0):
        """
        初始化路径规划器
        
        Args:
            client_id: PyBullet 客户端 ID
        """
        self.client_id = client_id
        self.astar = AStarPlanner(resolution=0.1, robot_radius=0.3)
        self.dwa = DWAPlanner(
            max_speed=0.3,
            max_yaw_rate=1.0,
            max_accel=0.5,
            max_yaw_accel=1.0,
            dt=0.1,
            predict_time=2.0,
            robot_radius=0.3
        )
        self.global_path: Optional[List[List[float]]] = None
        self.current_path_index = 0
    
    def update_obstacles_from_scene(self, scene_objects: Dict[str, Dict], min_obstacle_radius: float = 0.2):
        """从场景图中更新障碍物
        
        Args:
            scene_objects: 场景物体字典
            min_obstacle_radius: 最小障碍物半径阈值，小于此值的物体不被视为障碍物（默认0.2m）
        """
        obstacle_positions = []
        obstacle_sizes = []
        
        print(f"\n[PathPlanner] ====== 更新障碍物 ======")
        print(f"[PathPlanner] 场景物体总数: {len(scene_objects)}")
        print(f"[PathPlanner] 最小障碍物半径阈值: {min_obstacle_radius}m")
        
        for obj_name, obj_info in scene_objects.items():
            obj_type = obj_info.get('type', 'unknown')
            obj_id = obj_info.get('id', 'N/A')
            
            print(f"\n[PathPlanner] 检查物体: '{obj_name}' (ID: {obj_id}, type: {obj_type})")
            
            # 不将可抓取物体视为障碍物
            if obj_type == 'graspable':
                print(f"[PathPlanner]   -> 跳过: type='graspable' 不被视为障碍物")
                continue
            
            # 获取物体半径（优先使用 radius 字段，否则从 size 计算）
            radius = obj_info.get('radius')
            if radius is None:
                # 从 size 字段计算
                size = obj_info.get('size', [0.1, 0.1, 0.1])
                radius = max(size[0], size[1]) / 2.0 if len(size) >= 2 else 0.1
                print(f"[PathPlanner]   半径从 size 计算: {radius:.2f}m")
            else:
                print(f"[PathPlanner]   半径从 radius 字段: {radius:.2f}m")
            
            # 跳过小物品（半径小于阈值）
            if radius < min_obstacle_radius:
                print(f"[PathPlanner]   -> 跳过: 半径 {radius:.2f}m < 阈值 {min_obstacle_radius}m")
                continue
            
            pos = obj_info.get('position', [0, 0, 0])
            obstacle_positions.append(pos)
            obstacle_sizes.append(radius)
            print(f"[PathPlanner]   -> 添加为障碍物: '{obj_name}' 位置 [{pos[0]:.2f}, {pos[1]:.2f}], 半径 {radius:.2f}m")
        
        print(f"\n[PathPlanner] ====== 障碍物统计 ======")
        print(f"[PathPlanner] 障碍物总数: {len(obstacle_positions)}")
        if obstacle_positions:
            obstacle_list = []
            for name, info in scene_objects.items():
                info_radius = info.get('radius', 0)
                info_type = info.get('type', '')
                if info_radius >= min_obstacle_radius and info_type != 'graspable':
                    obstacle_list.append(f"{name}({info_radius:.2f}m)")
            print(f"[PathPlanner] 障碍物列表: {obstacle_list}")
        print(f"[PathPlanner] =================================\n")
        
        self.astar.update_obstacles(obstacle_positions, obstacle_sizes, self.client_id)
    
    def plan_global_path(
        self, 
        start: List[float], 
        goal: List[float],
        scene_objects: Optional[Dict[str, Dict]] = None,
        max_search_radius: float = 5.0,
        radius_step: float = 0.5
    ) -> Optional[List[List[float]]]:
        """
        规划全局路径，支持递归寻找替代目标
        
        如果目标不可达，会向外递归搜索可达的替代目标。
        如果替代目标距离原目标太远，会分多段导航。
        
        Args:
            start: 起点 [x, y]
            goal: 目标点 [x, y]
            scene_objects: 场景物体信息，用于寻找替代目标
            max_search_radius: 最大搜索半径（米）
            radius_step: 搜索半径步长（米）
        
        Returns:
            路径点列表（包含可能的中间目标点），如果无法到达返回 None
        """
        print(f"[PathPlanner] 开始规划路径: {start} -> {goal}")
        
        # 首先尝试直接规划到目标
        direct_path = self.astar.plan(start, goal)
        if direct_path is not None:
            print(f"[PathPlanner] 直接路径规划成功，路径长度: {len(direct_path)} 点")
            self.global_path = direct_path
            self.current_path_index = 0
            return direct_path
        
        print(f"[PathPlanner] 直接路径规划失败，开始递归搜索替代目标...")
        
        # 递归寻找可达的替代目标（最大深度 2，最多 2 个中间点）
        waypoints = self._find_reachable_waypoints_recursive(
            start, goal, scene_objects or {}, 
            max_search_radius=max_search_radius,
            radius_step=radius_step,
            max_depth=2
        )
        
        if waypoints is None or len(waypoints) == 0:
            print(f"[PathPlanner] 无法找到任何可达路径")
            self.global_path = None
            self.current_path_index = 0
            return None
        
        # 连接所有路径段
        full_path = []
        current_pos = start
        
        for i, waypoint in enumerate(waypoints):
            print(f"[PathPlanner] 规划第 {i+1}/{len(waypoints)} 段路径: {current_pos} -> {waypoint}")
            segment_path = self.astar.plan(current_pos, waypoint)
            
            if segment_path is None:
                print(f"[PathPlanner] 第 {i+1} 段路径规划失败")
                return None
            
            # 添加路径段（避免重复添加起点）
            if i == 0:
                full_path.extend(segment_path)
            else:
                full_path.extend(segment_path[1:])  # 跳过重复的起点
            
            current_pos = waypoint
        
        print(f"[PathPlanner] 多段路径规划成功，共 {len(waypoints)} 段，总路径长度: {len(full_path)} 点")
        print(f"[PathPlanner] 中间目标点: {waypoints}")
        
        self.global_path = full_path
        self.current_path_index = 0
        return full_path
    
    def _find_reachable_waypoints_recursive(
        self,
        start: List[float],
        goal: List[float],
        scene_objects: Dict[str, Dict],
        max_search_radius: float,
        radius_step: float,
        current_radius: float = 0.5,
        max_depth: int = 3,
        current_depth: int = 0,
        visited_goals: Optional[Set[Tuple[float, float]]] = None
    ) -> Optional[List[List[float]]]:
        """
        递归寻找可达的中间目标点（优化版本）
        
        策略：
        1. 在当前半径范围内搜索可达的候选点
        2. 只对最有希望的候选点用 A* 验证
        3. 限制递归深度，避免过深搜索
        
        Args:
            start: 起点 [x, y]
            goal: 最终目标 [x, y]
            scene_objects: 场景物体信息
            max_search_radius: 最大搜索半径
            radius_step: 半径步长
            current_radius: 当前搜索半径
            max_depth: 最大递归深度
            current_depth: 当前递归深度
            visited_goals: 已访问的目标点（避免循环）
        
        Returns:
            中间目标点列表（包含最终目标），如果无法到达返回 None
        """
        if visited_goals is None:
            visited_goals = set()
        
        # 检查递归深度
        if current_depth >= max_depth:
            print(f"[PathPlanner] 达到最大递归深度 {max_depth}，停止搜索")
            return None
        
        # 检查是否超过最大搜索半径
        if current_radius > max_search_radius:
            print(f"[PathPlanner] 超过最大搜索半径 {max_search_radius}m，停止搜索")
            return None
        
        print(f"[PathPlanner] 递归搜索: 起点 {start}, 目标 {goal}, 半径 {current_radius:.1f}m, 深度 {current_depth}")
        
        # 检查起点是否在障碍物中
        start_grid_x = int(start[0] / self.astar.resolution)
        start_grid_y = int(start[1] / self.astar.resolution)
        if (start_grid_x, start_grid_y) in self.astar.obstacles:
            print(f"[PathPlanner] 警告: 起点 ({start[0]:.2f}, {start[1]:.2f}) 在障碍物中！尝试清除附近障碍物...")
            # 清除起点周围的障碍物
            self.astar._clear_nearby_obstacles(start_grid_x, start_grid_y, radius=3)
        
        # 检查目标点是否在障碍物中
        goal_grid_x = int(goal[0] / self.astar.resolution)
        goal_grid_y = int(goal[1] / self.astar.resolution)
        if (goal_grid_x, goal_grid_y) in self.astar.obstacles:
            print(f"[PathPlanner] 警告: 目标点 ({goal[0]:.2f}, {goal[1]:.2f}) 在障碍物中！")
        
        # 在当前半径范围内生成候选点
        candidates = self._generate_candidates_at_radius(start, goal, current_radius, num_samples=16)
        
        # 快速筛选：排除在障碍物中的点，按距离目标远近排序
        valid_candidates = []
        for candidate in candidates:
            candidate_node_x = int(candidate[0] / self.astar.resolution)
            candidate_node_y = int(candidate[1] / self.astar.resolution)
            
            if (candidate_node_x, candidate_node_y) not in self.astar.obstacles:
                dist_to_goal = math.sqrt((candidate[0] - goal[0])**2 + (candidate[1] - goal[1])**2)
                valid_candidates.append((dist_to_goal, candidate))
        
        if not valid_candidates:
            # 当前半径没有有效候选点，扩大半径
            print(f"[PathPlanner] 半径 {current_radius:.1f}m 范围内无有效候选点，扩大搜索半径...")
            print(f"[PathPlanner] 当前障碍物数量: {len(self.astar.obstacles)} 个栅格")
            return self._find_reachable_waypoints_recursive(
                start, goal, scene_objects,
                max_search_radius, radius_step,
                current_radius + radius_step, max_depth, current_depth, visited_goals
            )
        
        # 按距离目标远近排序
        valid_candidates.sort(key=lambda x: x[0])
        
        print(f"[PathPlanner] 找到 {len(valid_candidates)} 个有效候选点，开始验证...")
        
        # 只验证前 3 个最有希望的候选点
        for i, (dist_to_goal, candidate) in enumerate(valid_candidates[:3]):
            candidate_tuple = (round(candidate[0], 2), round(candidate[1], 2))
            
            # 避免重复访问
            if candidate_tuple in visited_goals:
                continue
            
            visited_goals.add(candidate_tuple)
            
            print(f"[PathPlanner] 验证候选点 {i+1}/3: {candidate}, 距离目标: {dist_to_goal:.3f}m")
            
            # 用 A* 验证从起点到该候选点的路径
            test_path = self.astar.plan(start, candidate)
            
            if test_path is None:
                continue
            
            print(f"[PathPlanner] 候选点可达，检查是否能到达最终目标...")
            
            # 尝试从该候选点直接到达最终目标
            final_path = self.astar.plan(candidate, goal)
            
            if final_path is not None:
                # 可以直接到达最终目标
                print(f"[PathPlanner] 找到直达路径！")
                return [candidate, goal]
            
            # 不能直接到达，递归搜索（但限制深度）
            if current_depth < max_depth - 1:
                print(f"[PathPlanner] 需要更多中间点，继续递归...")
                sub_waypoints = self._find_reachable_waypoints_recursive(
                    candidate, goal, scene_objects,
                    max_search_radius, radius_step,
                    radius_step, max_depth, current_depth + 1, visited_goals
                )
                
                if sub_waypoints is not None:
                    # 找到完整路径
                    return [candidate] + sub_waypoints
        
        # 当前半径范围内没有找到可行路径，扩大半径继续搜索
        print(f"[PathPlanner] 半径 {current_radius:.1f}m 范围内未找到可行路径，扩大搜索半径...")
        return self._find_reachable_waypoints_recursive(
            start, goal, scene_objects,
            max_search_radius, radius_step,
            current_radius + radius_step, max_depth, current_depth, visited_goals
        )
    
    def _generate_candidates_at_radius(
        self,
        start: List[float],
        goal: List[float],
        radius: float,
        num_samples: int = 16
    ) -> List[List[float]]:
        """
        在从起点到目标的方向上，在指定半径处生成候选点
        
        Args:
            start: 起点
            goal: 目标
            radius: 搜索半径
            num_samples: 采样点数
        
        Returns:
            候选点列表
        """
        candidates = []
        
        # 计算从起点到目标的方向
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        dist_to_goal = math.sqrt(dx**2 + dy**2)
        
        if dist_to_goal > 0.01:
            # 有明确方向，优先在该方向附近采样
            base_angle = math.atan2(dy, dx)
            
            # 在目标方向周围采样（扇形区域）
            angle_range = math.pi / 2  # 90度扇形
            for i in range(num_samples):
                angle_offset = (i / (num_samples - 1) - 0.5) * angle_range
                angle = base_angle + angle_offset
                
                candidate_x = start[0] + radius * math.cos(angle)
                candidate_y = start[1] + radius * math.sin(angle)
                candidates.append([candidate_x, candidate_y])
        else:
            # 起点接近目标，全方向采样
            for i in range(num_samples):
                angle = 2 * math.pi * i / num_samples
                candidate_x = start[0] + radius * math.cos(angle)
                candidate_y = start[1] + radius * math.sin(angle)
                candidates.append([candidate_x, candidate_y])
        
        return candidates
    
    def get_local_goal(self, current_pos: List[float], lookahead_distance: float = 1.0) -> List[float]:
        """获取局部目标点（前瞻）"""
        if not self.global_path:
            return current_pos
        
        # 找到路径上最近的点
        min_dist = float('inf')
        closest_idx = 0
        for i, point in enumerate(self.global_path):
            dist = math.sqrt((point[0] - current_pos[0]) ** 2 + 
                           (point[1] - current_pos[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # 前瞻
        target_idx = min(closest_idx + int(lookahead_distance / self.astar.resolution), 
                        len(self.global_path) - 1)
        
        return self.global_path[target_idx]
    
    def compute_velocity(self, 
                        current_pos: List[float],
                        current_yaw: float,
                        current_v: float,
                        current_yaw_rate: float,
                        scene_objects: Dict[str, Dict]) -> Tuple[float, float]:
        """
        计算速度指令
        
        Returns:
            (v, yaw_rate) 速度指令
        """
        # 获取局部目标
        local_goal = self.get_local_goal(current_pos)
        
        # 提取障碍物位置
        obstacles = []
        for obj_name, obj_info in scene_objects.items():
            if obj_info.get('type') != 'graspable':
                pos = obj_info.get('position', [0, 0, 0])
                obstacles.append([pos[0], pos[1]])
        
        # 使用 DWA 计算速度
        return self.dwa.plan(
            current_pos, current_yaw, current_v, current_yaw_rate,
            local_goal, obstacles
        )
