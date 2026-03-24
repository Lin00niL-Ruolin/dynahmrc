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

# 配置
scene_config = {
    "config_path": "Config/default.yaml",
    "gui": True,
    "objects": [...]
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

## 核心组件

### DynaHMRCSystem

主系统集成类，封装整个协作流程：

- `__init__()`: 初始化配置
- `initialize()`: 初始化 BestMan 场景和机器人
- `execute_task()`: 执行自然语言任务（主入口）
- `emergency_stop()`: 紧急停止
- `shutdown()`: 关闭系统

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
| ArmRobot | manipulation, perception | 固定位置操作 |
| MobileBase | navigation, transport | 大范围运输 |
| MobileManipulator | navigation, manipulation, transport, perception | 完整任务 |

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

## 依赖关系

- **BestMan**: 底层仿真平台（PyBullet）
- **dyna_hmrc_web**: LLM 逻辑层（通过 import 引用）
- **PyBullet**: 物理仿真引擎

## 注意事项

1. 不要修改 BestMan 源码，通过继承和适配使用
2. 三类机器人类型必须正确对应 asset/robot/ 中的实际文件
3. 使用绝对导入路径（假设工作目录是项目根目录）
4. 异常处理：捕获 BestMan 的 PyBullet 错误并转换为 feedback 给 LLM

## 引用

```bibtex
@article{dynahmrc2024,
  title={DynaHMRC: Dynamic Heterogeneous Multi-Robot Collaboration via Large Language Models},
  journal={},
  year={2024}
}
```
