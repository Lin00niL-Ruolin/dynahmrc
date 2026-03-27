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
                                 client_id: int = 0,
                                 silent: bool = True):  # 默认静默
        """
        更新静态障碍物地图
        
        Args:
            obstacle_positions: 障碍物位置列表 [[x, y, z], ...]
            obstacle_sizes: 障碍物尺寸列表（半径），可选
            client_id: PyBullet 客户端 ID
            silent: 是否静默模式
        """
        self.static_obstacles.clear()
        
        for i, pos in enumerate(obstacle_positions):
            try:
                # 将障碍物位置转换为栅格坐标
                grid_x = int(pos[0] / self.resolution)
                grid_y = int(pos[1] / self.resolution)
                
                # 获取障碍物尺寸（如果有）
                obj_radius = obstacle_sizes[i] if obstacle_sizes and i < len(obstacle_sizes) else 0.0
                
                # 添加障碍物及其周围区域（考虑机器人半径 + 障碍物尺寸）
                total_radius = self.robot_radius + obj_radius * 0.5 + 0.05
                radius_cells = int(total_radius / self.resolution) + 1
                
                for dx in range(-radius_cells, radius_cells + 1):
                    for dy in range(-radius_cells, radius_cells + 1):
                        if dx * dx + dy * dy <= radius_cells * radius_cells:
                            self.static_obstacles.add((grid_x + dx, grid_y + dy))
            except Exception:
                continue
        
        self._merge_obstacles()
    
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
            except Exception:
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
                        client_id: int = 0,
                        silent: bool = True):
        """兼容旧接口，更新静态障碍物"""
        self.update_static_obstacles(obstacle_positions, obstacle_sizes, client_id, silent)
    
    def plan(self, start: List[float], goal: List[float], 
             robot_id: str = None,
             use_cache: bool = True,
             max_retries: int = 3,
             silent: bool = True) -> Optional[List[List[float]]]:  # 默认静默
        """
        规划路径 - 支持动态障碍物、缓存和智能点转换
        
        Args:
            start: 起点 [x, y]
            goal: 终点 [x, y]
            robot_id: 机器人ID（用于动态障碍物排除和缓存）
            use_cache: 是否使用缓存
            max_retries: 最大重试次数
            silent: 是否静默模式（不打印日志）
        
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
        if (start_node.x, start_node.y) in self.obstacles:
            converted = self._convert_invalid_point(start_node.x, start_node.y, 'start', silent)
            if converted:
                start_node.x, start_node.y = converted
                self.stats['point_conversions'] += 1
            else:
                # 无法转换，清除附近障碍物
                self._clear_nearby_obstacles(start_node.x, start_node.y, radius=2)
        
        if (goal_node.x, goal_node.y) in self.obstacles:
            converted = self._convert_invalid_point(goal_node.x, goal_node.y, 'goal', silent)
            if converted:
                goal_node.x, goal_node.y = converted
                self.stats['point_conversions'] += 1
            else:
                # 无法转换，清除附近障碍物
                self._clear_nearby_obstacles(goal_node.x, goal_node.y, radius=2)
        
        # A* 算法
        open_set = []
        heapq.heappush(open_set, start_node)
        closed_set: Set[Tuple[int, int]] = set()
        open_set_dict: Dict[Tuple[int, int], Node] = {(start_node.x, start_node.y): start_node}
        
        iteration = 0
        max_iterations = 9000  # 最大迭代次数限制
        
        while open_set and iteration < max_iterations:
            current = heapq.heappop(open_set)
            iteration += 1
            
            # 从字典中移除
            if (current.x, current.y) in open_set_dict:
                del open_set_dict[(current.x, current.y)]
            
            # 到达目标
            if current == goal_node:
                path = self._reconstruct_path(current, cache_key)
                self.stats['planning_success'] += 1
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
        
        # 更新统计
        self.stats['planning_failures'] += 1
        
        # 尝试扩大搜索范围：清除起点和终点周围的障碍物后重试
        self._clear_nearby_obstacles(start_node.x, start_node.y, radius=3)
        self._clear_nearby_obstacles(goal_node.x, goal_node.y, radius=3)
        
        # 重新规划
        return self._plan_with_current_obstacles(start_node, goal_node, cache_key, silent)
    
    def _convert_invalid_point(self, x: int, y: int, point_type: str = 'point', silent: bool = True) -> Optional[Tuple[int, int]]:
        """
        将不合理的点（在障碍物中）转换为合理的点
        
        Args:
            x: 原始栅格X坐标
            y: 原始栅格Y坐标
            point_type: 'start' 或 'goal'
            silent: 是否静默模式
        
        Returns:
            转换后的 (x, y) 坐标，如果找不到则返回 None
        """
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
                return (best[0], best[1])
        
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
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    pos = (x + dx, y + dy)
                    if pos in self.obstacles:
                        self.obstacles.discard(pos)
                        self.static_obstacles.discard(pos)
    
    def _plan_with_current_obstacles(self, start_node: Node, goal_node: Node, 
                                     cache_key: str = None,
                                     silent: bool = True) -> Optional[List[List[float]]]:
        """
        使用当前障碍物地图重新规划（使用更宽松的启发式）
        
        Args:
            start_node: 起点节点
            goal_node: 终点节点
            cache_key: 缓存键
            silent: 是否静默模式
        
        Returns:
            路径点列表，如果无法到达返回 None
        """
        import time
        start_time = time.time()
        
        open_set = []
        heapq.heappush(open_set, start_node)
        closed_set: Set[Tuple[int, int]] = set()
        open_set_dict: Dict[Tuple[int, int], Node] = {(start_node.x, start_node.y): start_node}
        
        iteration = 0
        max_iterations = 12000  # 增加最大迭代次数
        heuristic_weight = 1.5  # 使用次优启发式，允许绕路
        
        while open_set and iteration < max_iterations:
            current = heapq.heappop(open_set)
            iteration += 1
            
            if (current.x, current.y) in open_set_dict:
                del open_set_dict[(current.x, current.y)]
            
            if current == goal_node:
                path = self._reconstruct_path(current, cache_key)
                self.stats['planning_success'] += 1
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
                # 使用加权启发式，允许次优路径
                h = self._heuristic(neighbor_x, neighbor_y, goal_node) * heuristic_weight
                
                neighbor_key = (neighbor_x, neighbor_y)
                if neighbor_key in open_set_dict:
                    existing_node = open_set_dict[neighbor_key]
                    if g >= existing_node.g:
                        continue
                
                neighbor = Node(neighbor_x, neighbor_y, g, h)
                neighbor.parent = current
                
                heapq.heappush(open_set, neighbor)
                open_set_dict[neighbor_key] = neighbor
        
        self.stats['planning_failures'] += 1
        return None
    
    def _heuristic(self, x: int, y: int, goal: Node) -> float:
        """
        启发式函数 - 使用对角线距离（考虑8方向移动）
        
        Args:
            x: 当前X坐标
            y: 当前Y坐标
            goal: 目标节点
        
        Returns:
            估计代价
        """
        dx = abs(x - goal.x)
        dy = abs(y - goal.y)
        
        # 对角线距离
        diagonal = min(dx, dy)
        straight = abs(dx - dy)
        
        return diagonal * math.sqrt(2) * self.resolution + straight * self.resolution
    
    def _reconstruct_path(self, node: Node, cache_key: str = None) -> List[List[float]]:
        """
        重建路径
        
        Args:
            node: 终点节点
            cache_key: 缓存键
        
        Returns:
            路径点列表 [[x, y], ...]
        """
        path = []
        current = node
        
        while current:
            # 将栅格坐标转换回世界坐标
            world_x = current.x * self.resolution
            world_y = current.y * self.resolution
            path.append([world_x, world_y])
            current = current.parent
        
        path.reverse()
        
        # 缓存路径
        if cache_key:
            self.path_cache[cache_key] = path.copy()
            self.cache_timestamp[cache_key] = __import__('time').time()
        
        return path
    
    def update_obstacles_from_scene(self, scene_graph: Dict, robot_id: str = None, silent: bool = True):
        """
        从场景图更新障碍物
        
        Args:
            scene_graph: 场景图字典
            robot_id: 当前机器人ID（排除自身）
            silent: 是否静默模式
        """
        obstacle_positions = []
        obstacle_sizes = []
        
        for name, info in scene_graph.items():
            # 排除机器人和地面
            if name.startswith('robot_') or name == 'ground':
                continue
            
            # 排除当前机器人自身
            if robot_id and name == robot_id:
                continue
            
            pos = info.get('position', [0, 0, 0])
            size = info.get('size', [0.1, 0.1, 0.1])
            
            # 只考虑地面上的物体（z < 1.0）
            if pos[2] < 1.0:
                obstacle_positions.append(pos)
                # 使用物体尺寸的最大值作为半径
                radius = max(size[0], size[1]) / 2 if len(size) >= 2 else 0.1
                obstacle_sizes.append(radius)
        
        self.update_static_obstacles(obstacle_positions, obstacle_sizes, silent=silent)
    
    def plan_global_path(self, start: List[float], goal: List[float],
                        scene_objects: Dict = None,
                        max_search_radius: float = 5.0,
                        radius_step: float = 0.5,
                        robot_id: str = None,
                        silent: bool = True) -> Optional[List[List[float]]]:
        """
        全局路径规划（带智能点转换）
        
        Args:
            start: 起点 [x, y]
            goal: 终点 [x, y]
            scene_objects: 场景物体信息
            max_search_radius: 最大搜索半径
            radius_step: 半径步长
            robot_id: 机器人ID
            silent: 是否静默模式
        
        Returns:
            路径点列表，如果无法到达返回 None
        """
        # 更新障碍物
        if scene_objects:
            self.update_obstacles_from_scene(scene_objects, robot_id, silent=silent)
        
        # 尝试直接规划
        path = self.plan(start, goal, robot_id=robot_id, silent=silent)
        if path:
            return path
        
        # 如果直接规划失败，尝试在目标周围搜索
        import math
        
        # 策略1: 在目标周围环形搜索
        for radius in np.arange(radius_step, max_search_radius + radius_step, radius_step):
            for angle in np.arange(0, 2 * math.pi, math.pi / 4):
                new_goal = [
                    goal[0] + radius * math.cos(angle),
                    goal[1] + radius * math.sin(angle)
                ]
                path = self.plan(start, new_goal, robot_id=robot_id, silent=silent)
                if path:
                    # 在路径末尾添加原始目标点
                    path.append(goal)
                    return path
        
        # 策略2: 尝试向目标方向移动一段距离（部分路径）
        # 计算从起点到目标的方向
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        dist_to_goal = math.sqrt(dx * dx + dy * dy)
        
        if dist_to_goal > 0.5:  # 如果距离足够远
            # 尝试走到距离起点一定比例的点上
            for ratio in [0.7, 0.5, 0.3]:  # 尝试走到70%、50%、30%的位置
                partial_goal = [
                    start[0] + dx * ratio,
                    start[1] + dy * ratio
                ]
                path = self.plan(start, partial_goal, robot_id=robot_id, silent=silent)
                if path:
                    # 添加原始目标作为最终点（机器人会尽可能接近）
                    path.append(goal)
                    return path
        
        # 策略3: 尝试短距离移动（可能目标就在附近但被小障碍物阻挡）
        for dist in [0.5, 1.0, 1.5]:
            for angle in np.arange(0, 2 * math.pi, math.pi / 6):  # 更细的角度步长
                test_goal = [
                    start[0] + dist * math.cos(angle),
                    start[1] + dist * math.sin(angle)
                ]
                path = self.plan(start, test_goal, robot_id=robot_id, silent=silent)
                if path:
                    path.append(goal)
                    return path
        
        return None


class DWAPlanner:
    """DWA (Dynamic Window Approach) 局部路径规划器"""
    
    def __init__(self, dt: float = 0.1, predict_time: float = 3.0,
                 max_speed: float = 0.5, min_speed: float = 0.0,
                 max_yaw_rate: float = 1.0, max_accel: float = 0.5,
                 max_dyaw_rate: float = 1.0, v_resolution: float = 0.05,
                 yaw_rate_resolution: float = 0.1, robot_radius: float = 0.3,
                 goal_cost_gain: float = 1.0, speed_cost_gain: float = 1.0,
                 obstacle_cost_gain: float = 2.0):
        """
        初始化 DWA 规划器
        
        Args:
            dt: 时间步长
            predict_time: 预测时间
            max_speed: 最大线速度
            min_speed: 最小线速度
            max_yaw_rate: 最大角速度
            max_accel: 最大加速度
            max_dyaw_rate: 最大角加速度
            v_resolution: 速度分辨率
            yaw_rate_resolution: 角速度分辨率
            robot_radius: 机器人半径
            goal_cost_gain: 目标代价权重
            speed_cost_gain: 速度代价权重
            obstacle_cost_gain: 障碍物代价权重
        """
        self.dt = dt
        self.predict_time = predict_time
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_yaw_rate = max_yaw_rate
        self.max_accel = max_accel
        self.max_dyaw_rate = max_dyaw_rate
        self.v_resolution = v_resolution
        self.yaw_rate_resolution = yaw_rate_resolution
        self.robot_radius = robot_radius
        self.goal_cost_gain = goal_cost_gain
        self.speed_cost_gain = speed_cost_gain
        self.obstacle_cost_gain = obstacle_cost_gain
    
    def plan(self, state: List[float], goal: List[float], 
             obstacles: List[List[float]], 
             silent: bool = True) -> Tuple[float, float]:
        """
        规划下一步动作
        
        Args:
            state: 当前状态 [x, y, yaw, v, w]
            goal: 目标位置 [x, y]
            obstacles: 障碍物位置列表 [[x, y], ...]
            silent: 是否静默模式
        
        Returns:
            (线速度, 角速度)
        """
        # 计算动态窗口
        dw = self._calc_dynamic_window(state)
        
        # 评估所有可能的轨迹
        best_u = [0.0, 0.0]
        min_cost = float('inf')
        
        for v in np.arange(dw[0], dw[1], self.v_resolution):
            for w in np.arange(dw[2], dw[3], self.yaw_rate_resolution):
                trajectory = self._predict_trajectory(state, v, w)
                cost = self._calc_cost(trajectory, goal, obstacles)
                
                if cost < min_cost:
                    min_cost = cost
                    best_u = [v, w]
        
        return best_u[0], best_u[1]
    
    def _calc_dynamic_window(self, state: List[float]) -> List[float]:
        """计算动态窗口"""
        v, w = state[3], state[4]
        
        # 速度限制
        vs = [self.min_speed, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]
        
        # 加速度限制
        vd = [
            v - self.max_accel * self.dt,
            v + self.max_accel * self.dt,
            w - self.max_dyaw_rate * self.dt,
            w + self.max_dyaw_rate * self.dt
        ]
        
        # 动态窗口是两者的交集
        return [
            max(vs[0], vd[0]),  # min_v
            min(vs[1], vd[1]),  # max_v
            max(vs[2], vd[2]),  # min_w
            min(vs[3], vd[3])   # max_w
        ]
    
    def _predict_trajectory(self, state: List[float], v: float, w: float) -> List[List[float]]:
        """预测轨迹"""
        trajectory = [state[:2]]  # 只记录位置
        x, y, yaw = state[0], state[1], state[2]
        
        for _ in range(int(self.predict_time / self.dt)):
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
            yaw += w * self.dt
            trajectory.append([x, y])
        
        return trajectory
    
    def _calc_cost(self, trajectory: List[List[float]], goal: List[float], 
                   obstacles: List[List[float]]) -> float:
        """计算轨迹代价"""
        # 目标代价（距离目标的距离）
        goal_cost = math.sqrt((trajectory[-1][0] - goal[0])**2 + 
                             (trajectory[-1][1] - goal[1])**2)
        
        # 速度代价（鼓励高速）
        speed_cost = self.max_speed - trajectory[-1][0]
        
        # 障碍物代价
        obstacle_cost = 0.0
        for point in trajectory:
            for obs in obstacles:
                dist = math.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
                if dist < self.robot_radius:
                    obstacle_cost += float('inf')
                else:
                    obstacle_cost += 1.0 / dist
        
        return (self.goal_cost_gain * goal_cost + 
                self.speed_cost_gain * speed_cost + 
                self.obstacle_cost_gain * obstacle_cost)


# 向后兼容的别名
PathPlanner = AStarPlanner
