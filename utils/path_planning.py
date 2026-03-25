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
    """A* 全局路径规划器"""
    
    def __init__(self, resolution: float = 0.1, robot_radius: float = 0.3):
        """
        初始化 A* 规划器
        
        Args:
            resolution: 栅格地图分辨率（米）
            robot_radius: 机器人半径（米）
        """
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.obstacles: Set[Tuple[int, int]] = set()
    
    def update_obstacles(self, obstacle_positions: List[List[float]], obstacle_sizes: Optional[List[float]] = None, client_id: int = 0):
        """
        更新障碍物地图
        
        Args:
            obstacle_positions: 障碍物位置列表 [[x, y, z], ...]
            obstacle_sizes: 障碍物尺寸列表（半径），可选
            client_id: PyBullet 客户端 ID
        """
        self.obstacles.clear()
        
        print(f"[A*] 更新障碍物地图: {len(obstacle_positions)} 个障碍物")
        
        for i, pos in enumerate(obstacle_positions):
            # 将障碍物位置转换为栅格坐标
            grid_x = int(pos[0] / self.resolution)
            grid_y = int(pos[1] / self.resolution)
            
            # 获取障碍物尺寸（如果有）
            obj_radius = obstacle_sizes[i] if obstacle_sizes and i < len(obstacle_sizes) else 0.0
            
            # 添加障碍物及其周围区域（考虑机器人半径 + 障碍物尺寸）
            # 使用较小的膨胀半径，避免过度膨胀
            total_radius = self.robot_radius + obj_radius + 0.05  # 额外 5cm 安全距离
            radius_cells = int(total_radius / self.resolution) + 1
            
            print(f"[A*] 障碍物 {i}: 位置 ({pos[0]:.2f}, {pos[1]:.2f}) -> 栅格 ({grid_x}, {grid_y}), "
                  f"物体半径 {obj_radius:.2f}m, 膨胀半径 {total_radius:.2f}m ({radius_cells} 格)")
            
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy <= radius_cells * radius_cells:
                        self.obstacles.add((grid_x + dx, grid_y + dy))
        
        print(f"[A*] 障碍物地图更新完成: 共 {len(self.obstacles)} 个栅格被标记为障碍物")
    
    def plan(self, start: List[float], goal: List[float]) -> Optional[List[List[float]]]:
        """
        规划路径
        
        Args:
            start: 起点 [x, y]
            goal: 终点 [x, y]
        
        Returns:
            路径点列表 [[x, y], ...]，如果无法到达返回 None
        """
        # 转换为栅格坐标
        start_node = Node(int(start[0] / self.resolution), int(start[1] / self.resolution))
        goal_node = Node(int(goal[0] / self.resolution), int(goal[1] / self.resolution))
        
        # 检查起点和终点是否在障碍物中
        if (start_node.x, start_node.y) in self.obstacles:
            print(f"[A*] 起点在障碍物中: ({start_node.x}, {start_node.y})")
            # 尝试清除起点附近的障碍物
            self._clear_nearby_obstacles(start_node.x, start_node.y, radius=2)
        if (goal_node.x, goal_node.y) in self.obstacles:
            print(f"[A*] 终点在障碍物中: ({goal_node.x}, {goal_node.y})，尝试清除附近障碍物")
            # 清除终点附近的障碍物（允许终点靠近障碍物）
            self._clear_nearby_obstacles(goal_node.x, goal_node.y, radius=2)
        
        # A* 算法
        open_set = []
        heapq.heappush(open_set, start_node)
        closed_set: Set[Tuple[int, int]] = set()
        
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
        
        iteration = 0
        max_iterations = 5000  # 最大迭代次数限制
        
        while open_set and iteration < max_iterations:
            current = heapq.heappop(open_set)
            iteration += 1
            
            # 每 500 次迭代打印进度
            if iteration % 500 == 0:
                print(f"[A*] 规划中... 迭代 {iteration}, 开放集 {len(open_set)}, 已探索 {len(closed_set)}")
            
            # 到达目标
            if current == goal_node:
                path = self._reconstruct_path(current)
                print(f"[A*] 路径规划成功! 迭代 {iteration} 次, 路径长度 {len(path)} 点")
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
                
                neighbor = Node(neighbor_x, neighbor_y, g, h)
                neighbor.parent = current
                
                heapq.heappush(open_set, neighbor)
        
        if iteration >= max_iterations:
            print(f"[A*] 规划超时! 达到最大迭代次数 {max_iterations}")
        else:
            print(f"[A*] 无法找到路径! 迭代 {iteration} 次, 开放集为空")
        print(f"[A*] 已探索节点数: {len(closed_set)}")
        return None
    
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
                        cleared_count += 1
                        cleared_positions.append(pos)
        
        if cleared_count > 0:
            print(f"[A*] 清除障碍物: 位置 ({x}, {y}) 周围 {radius} 格，共清除 {cleared_count} 个障碍物")
            print(f"[A*] 清除的障碍物位置: {cleared_positions}")
        else:
            print(f"[A*] 位置 ({x}, {y}) 周围没有需要清除的障碍物")
    
    def _heuristic(self, x: int, y: int, goal: Node) -> float:
        """启发式函数（欧几里得距离）"""
        return math.sqrt((x - goal.x) ** 2 + (y - goal.y) ** 2) * self.resolution
    
    def _reconstruct_path(self, node: Node) -> List[List[float]]:
        """重建路径"""
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
        
        return path


class DWAPlanner:
    """DWA (Dynamic Window Approach) 局部避障规划器"""
    
    def __init__(self, 
                 max_speed: float = 0.5,
                 max_yaw_rate: float = 1.0,
                 max_accel: float = 0.2,
                 max_yaw_accel: float = 0.5,
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
        
        # 采样速度空间
        sample_count = 0
        for v in np.arange(dw[0], dw[1], self.v_resolution):
            for yaw_rate in np.arange(dw[2], dw[3], self.yaw_rate_resolution):
                sample_count += 1
                
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
        
        # 调试信息
        dist_to_goal = math.sqrt((current_pos[0] - goal[0])**2 + (current_pos[1] - goal[1])**2)
        print(f"[DWA] 位置: [{current_pos[0]:.2f}, {current_pos[1]:.2f}], 目标距离: {dist_to_goal:.2f}m")
        print(f"[DWA] 动态窗口: v=[{dw[0]:.2f}, {dw[1]:.2f}], yaw_rate=[{dw[2]:.2f}, {dw[3]:.2f}]")
        print(f"[DWA] 采样 {sample_count} 个速度，最优: v={best_v:.3f}m/s, yaw_rate={best_yaw_rate:.3f}rad/s, 代价={min_cost:.3f}")
        
        # 如果最优速度会碰撞，打印警告
        if best_v > 0:
            test_traj = self._predict_trajectory(current_pos, current_yaw, best_v, best_yaw_rate)
            obs_cost = self._calc_obstacle_cost(test_traj, obstacles)
            if obs_cost == float('inf'):
                print(f"[DWA] ⚠️ 警告: 最优速度会导致碰撞!")
        
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
            max_accel=0.2,
            max_yaw_accel=0.5,
            dt=0.1,
            predict_time=2.0,
            robot_radius=0.3
        )
        self.global_path: Optional[List[List[float]]] = None
        self.current_path_index = 0
    
    def update_obstacles_from_scene(self, scene_objects: Dict[str, Dict]):
        """从场景图中更新障碍物"""
        obstacle_positions = []
        obstacle_sizes = []
        
        for obj_name, obj_info in scene_objects.items():
            obj_type = obj_info.get('type', 'unknown')
            if obj_type != 'graspable':  # 不将可抓取物体视为障碍物
                pos = obj_info.get('position', [0, 0, 0])
                obstacle_positions.append(pos)
                
                # 获取物体尺寸（如果有）
                size = obj_info.get('size', [0.1, 0.1, 0.1])
                # 使用 x-y 平面的最大半径
                raw_radius = max(size[0], size[1]) / 2.0 if len(size) >= 2 else 0.1
                
                # 对家具类障碍物使用更小的膨胀半径（避免过度膨胀）
                if obj_type == 'furniture':
                    # 家具只膨胀其实际尺寸的一半，因为 AABB 往往比实际占用空间大
                    radius = raw_radius * 0.5
                    print(f"[PathPlanner] 家具 '{obj_name}': 原始半径 {raw_radius:.2f}m -> 使用半径 {radius:.2f}m")
                else:
                    radius = raw_radius
                
                obstacle_sizes.append(radius)
        
        self.astar.update_obstacles(obstacle_positions, obstacle_sizes, self.client_id)
    
    def plan_global_path(self, start: List[float], goal: List[float]) -> Optional[List[List[float]]]:
        """规划全局路径"""
        self.global_path = self.astar.plan(start, goal)
        self.current_path_index = 0
        return self.global_path
    
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
