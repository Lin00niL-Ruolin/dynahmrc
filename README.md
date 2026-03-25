# DynaHMRC - 异构多机器人动态协作框架

基于 BestMan 仿真平台实现论文《DynaHMRC: Dynamic Heterogeneous Multi-Robot Collaboration via Large Language Models》的复现代码。

## 项目结构

```
dynahmrc/
├── __init__.py              # 包入口，导出 DynaHMRCSystem
├── system.py                # 主系统集成类
├── run_demo.py              # 一键运行入口
├── robots/                  # 异构机器人类
│   ├── arm_robot.py         # 固定机械臂（仅操作）
│   ├── mobile_base.py       # 移动基座（仅导航）
│   └── mobile_manipulator.py # 移动操作复合（导航+操作）
├── integration/             # 集成层
│   ├── bestman_adapter.py   # BestMan API 适配器
│   └── robot_factory.py     # 机器人动态工厂
└── scenarios/               # 测试场景
    ├── warehouse_task.py    # 仓储协作场景
    └── assembly_task.py     # 装配任务场景
```

## 快速开始

### 1. 运行仓储场景

```bash
python -m dynahmrc.run_demo --scenario warehouse
```

### 2. 运行装配场景

```bash
python -m dynahmrc.run_demo --scenario assembly
```

### 3. 运行自定义任务

```python
from dynahmrc import DynaHMRCSystem

# 方式1: 使用JSON场景文件（推荐）
scene_config = {
    "config_path": "Config/default.yaml",
    "gui": True,
    "scene_path": "Asset/Scene/your_scene.json"  # JSON场景文件路径
}

# 方式2: 使用objects列表逐个定义
scene_config = {
    "config_path": "Config/default.yaml",
    "gui": True,
    "objects": [
        {
            "name": "box",
            "type": "part",
            "model_path": "Asset/Scene/Object/Box/mobility.urdf",
            "position": [1.0, 1.0, 0.5],
            "orientation": [0, 0, 0, 1],
            "scale": 1.0  # 可选，默认值为1
        }
    ]
}

robot_configs = [
    {
        "robot_id": "robot_1",
        "robot_type": "mobile_manipulator",
        "robot_model": "panda_on_segbot",
        "init_position": [0, 0, 0]
    }
]

llm_config = {
    "provider": "kimi",  # 或 "mock", "openai"
    "api_key": "your-api-key",
    "model": "kimi-k2.5"
}

# 创建系统并执行任务
system = DynaHMRCSystem(scene_config, robot_configs, llm_config)
result = system.execute_task("把箱子搬到桌子上")
print(f"任务完成: {result.success}")
```

## 场景配置

支持两种方式配置场景：

### 方式1: JSON场景文件（推荐）

使用BestMan的`create_scene`函数加载完整的场景文件：

```python
scene_config = {
    "config_path": "Config/default.yaml",
    "gui": True,
    "scene_path": "Asset/Scene/your_scene.json"
}
```

JSON场景文件格式示例：
```json
[
    {
        "obj_name": "table",
        "model_path": "Asset/Scene/Object/Table/mobility.urdf",
        "object_position": [0, 0, 0],
        "object_orientation": [0, 0, 0, 1],
        "scale": 1.0,
        "fixed_base": true
    },
    {
        "obj_name": "box",
        "model_path": "Asset/Scene/Object/Box/mobility.urdf",
        "object_position": [1.0, 0, 0.5],
        "object_orientation": [0, 0, 0, 1],
        "scale": 0.5,
        "fixed_base": false
    }
]
```

### 方式2: Objects列表

在代码中逐个定义场景物体：

```python
scene_config = {
    "config_path": "Config/default.yaml",
    "gui": True,
    "objects": [
        {
            "name": "box",
            "type": "part",
            "model_path": "Asset/Scene/Object/Box/mobility.urdf",
            "position": [1.0, 1.0, 0.5],
            "orientation": [0, 0, 0, 1],
            "scale": 1.0  # 可选，默认值为1
        }
    ]
}
```

**物体配置参数说明：**
- `name`: 物体名称
- `type`: 物体类型（如"part", "platform"等）
- `model_path`: 模型文件路径（支持.urdf, .obj, .stl格式）
- `position`: 位置坐标 `[x, y, z]`
- `orientation`: 四元数 `[qx, qy, qz, qw]`
- `scale`: 缩放比例（可选，默认值为1）

## 核心组件

### DynaHMRCSystem

主系统集成类，封装整个协作流程：

- `__init__()`: 初始化配置
- `initialize()`: 初始化 BestMan 场景和机器人
- `execute_task()`: 执行自然语言任务（主入口）
- `emergency_stop()`: 紧急停止
- `shutdown()`: 关闭系统
- `set_metrics_collector()`: 设置评估指标收集器

### BestManAdapter

桥接 LLM 逻辑与 BestMan API：

- `execute_action()`: 执行动作
- `get_robot_states()`: 获取机器人状态
- `get_robot_capabilities()`: 获取机器人能力

### RobotFactory

动态创建机器人：

- `create_robot()`: 根据类型创建机器人
- `get_robot()`: 获取机器人实例
- `get_all_robots()`: 获取所有机器人

## 三类异构机器人

| 机器人类型 | 能力 | 适用场景 |
|-----------|------|---------|
| ArmRobot (Ma) | manipulation, perception | 固定位置操作 |
| MobileBase (Mo) | navigation, transport | 大范围运输 |
| MobileManipulator (Ma-Mo) | navigation, manipulation, transport, perception | 完整任务 |
| Drone (UAV) | navigation, pick, place (aerial) | 高空/难到达区域 |

## 三类任务

| 任务类型 | 描述 | 评估能力 |
|---------|------|---------|
| Pack Objects | 将物体装入容器 | 基本操作能力 |
| Make Sandwich | 按顺序堆叠食材 | 顺序约束处理 |
| Sort Solids | 按颜色匹配放置 | 颜色识别和精确放置 |

## 动态变化类型 (Variations)

论文中评估的五种动态变化场景：

| 变化类型 | 描述 | 测试能力 |
|---------|------|---------|
| Static | 无变化 | 基础协作能力 |
| CTO | Change Task Objective | 目标变化适应能力 |
| IRZ | Inaccessible Region Zone | 不可达区域处理 |
| ANC | Add New Collaborator | 新成员加入适应 |
| REC | Remove Existing Collaborator | 成员离开适应 |

## 实现的核心算法

1. **Algorithm 1: Dynamic Replanning** - 动态重规划
   - 在 `system.py` 的 `_execute_with_monitoring()` 中实现
   - 监控执行状态，失败时触发重规划

2. **Algorithm 2: Task Allocation** - 任务分配
   - 在 `coordinator.py` 中实现
   - 基于 LLM 的任务分解和机器人分配

3. **Algorithm 3: Conflict Resolution** - 冲突解决
   - 在 `_replan_after_failure()` 中实现
   - 任务重新分配和路径调整

## 四阶段协作流程 (Four-Stage Collaboration)

论文中提出的核心协作框架，实现去中心化的异构多机器人协作：

### Stage 1: Self-Description (自我介绍)
- 每个机器人基于自身能力和任务理解生成自我介绍
- 存储在记忆模块中供后续阶段使用
- 实现：`RobotAgent.self_describe()`

### Stage 2: Task Allocation (任务分配)
- 每个机器人提出任务分配方案和竞选演讲
- 分析任务并分配给最适合的机器人
- 识别需要协作同步的关键点
- 实现：`RobotAgent.propose_allocation()`

### Stage 3: Leader Election (领导者选举)
- 基于竞选演讲和任务分配方案投票
- 得票最多的机器人成为领导者
- 领导者负责任务协调和冲突解决
- 实现：`RobotAgent.vote_leader()`

### Stage 4: Closed-Loop Execution (闭环执行)
- 基于观察、计划和历史执行动作
- 持续反馈循环，根据环境变化调整
- 支持动作失败后的重规划
- 实现：`RobotAgent.execute_step()`

### 使用方法
```python
from dynahmrc import DynaHMRCSystem

# 创建系统
system = DynaHMRCSystem(scene_config, robot_configs, llm_config)

# 使用四阶段协作执行任务
result = system.execute_task_with_four_stages(
    "把箱子搬到桌子上",
    max_steps=100
)

print(f"任务完成: {result['success']}")
print(f"领导者: {result['leader']}")
print(f"任务分配: {result['robot_assignments']}")
```

### 核心组件
- `FourStageCollaboration`: 四阶段协作框架主类
- `RobotAgent`: 去中心化机器人Agent
- `MemoryModule`: 记忆模块，管理历史上下文
- `CollaborationManager`: 协作状态管理

## 评估指标 (Metrics)

论文中使用的评估指标：

| 指标 | 全称 | 描述 |
|-----|------|------|
| SUCC | Success Rate | 任务成功率 |
| PS | Partial Success | 部分成功率 |
| TS | Task Steps | 任务总步数 |
| AS | Action Steps | 每个机器人的动作步数 |
| CC | Communication Count | 通信次数 |

## 运行实验

### 快速测试
```bash
python -m dynahmrc.run_experiments --quick
```

### 运行所有论文实验
```bash
python -m dynahmrc.run_experiments --llm kimi --trials 8
```

### 运行特定任务和变化类型
```bash
python -m dynahmrc.run_experiments \
    --tasks pack_objects make_sandwich \
    --variations static cto irz \
    --teams Ma-MoMa-UAV Ma-MoMa
```

### 使用 Mock LLM 进行测试
```bash
python -m dynahmrc.run_experiments --llm mock --trials 2
```

## 实验结果

实验结果将保存在 `experiments/results/` 目录下：
- `metrics_*.json`: 详细的指标数据
- `results_*.csv`: CSV 格式的结果表格
- `table_*.txt`: 可读的表格格式
- `latex_table_*.txt`: LaTeX 表格格式

## 依赖关系

- **BestMan**: 底层仿真平台（PyBullet）
- **dyna_hmrc_web**: LLM 逻辑层（通过 import 引用）
- **PyBullet**: 物理仿真引擎
- **OpenAI**: LLM API 客户端

## 注意事项

1. 不要修改 BestMan 源码，通过继承和适配使用
2. 机器人类型必须正确对应 asset/robot/ 中的实际文件
3. 使用绝对导入路径（假设工作目录是项目根目录）
4. 异常处理：捕获 BestMan 的 PyBullet 错误并转换为 feedback 给 LLM
5. 实验运行需要设置 LLM API Key（或使用 mock 模式）

## 引用

```bibtex
@article{dynahmrc2024,
  title={DynaHMRC: Dynamic Heterogeneous Multi-Robot Collaboration via Large Language Models},
  journal={IEEE Transactions on Robotics},
  year={2024}
}
```
