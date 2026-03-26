"""
第四阶段（执行阶段）测试 - 直接测试任务执行而无需经过前三个阶段
测试内容：
1. 初始化 BestMan 仿真环境
2. 创建机器人和场景
3. 直接设置任务计划和 Leader
4. 运行第四阶段执行循环
5. 观察机器人执行动作
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BestMan 导入
from Env.Client import Client
from Visualization.Visualizer import Visualizer

# DynaHMRC 导入
from dynahmrc.integration.bestman_adapter import BestManAdapter
from dynahmrc.integration.robot_factory import RobotFactory
from dynahmrc.core.collaboration import FourStageCollaboration, CollaborationResult
from dynahmrc.core.robot_agent import RobotAgent

# Import MockLLMClient
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Dyna_hmrc_web'))
from dynahmrc_web.dynahmrc.utils.llm_api import MockLLMClient


def load_config(config_path: str = "Config/default.yaml"):
    """加载配置文件"""
    from yacs.config import CfgNode as CN
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    with open(config_path, "r") as f:
        cfg = CN.load_cfg(f)
    
    return cfg


class Stage4Tester:
    """第四阶段测试器 - 直接测试执行阶段"""
    
    def __init__(self):
        self.client = None
        self.visualizer = None
        self.robot_factory = None
        self.adapter = None
        self.collaboration = None
        self.robots = {}
        
    def setup_environment(self):
        """设置仿真环境"""
        print("\n" + "="*60)
        print("1. 初始化 BestMan 仿真环境")
        print("="*60)
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "Config", "default.yaml")
        
        try:
            cfg = load_config(config_path)
            cfg.Client.enable_GUI = True
            cfg.Client.enable_Debug = True
            
            self.client = Client(cfg.Client)
            print("   ✓ BestMan 客户端初始化成功")
            
            self.visualizer = Visualizer(self.client, cfg.Visualizer)
            print("   ✓ Visualizer 初始化成功")
            
        except Exception as e:
            print(f"   [ERROR] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def setup_scene(self):
        """设置测试场景"""
        print("\n" + "="*60)
        print("2. 创建测试场景")
        print("="*60)
        
        try:
            # 创建桌子
            table_id = self.client.load_object(
                obj_name="table",
                model_path="Asset/Scene/Object/URDF_models/furniture_table_rectangle_high/table.urdf",
                object_position=[0, 0, 0],
                object_orientation=[0, 0, 0, 1],
                scale=1.0,
                fixed_base=True
            )
            print(f"   ✓ 创建桌子 (ID: {table_id})")
            
            # 创建箱子
            box_id = self.client.load_object(
                obj_name="box",
                model_path="Asset/Scene/Object/URDF_models/cracker_box/model.urdf",
                object_position=[0.5, 0, 0.5],
                object_orientation=[0, 0, 0, 1],
                scale=1.0,
                fixed_base=False
            )
            print(f"   ✓ 创建箱子 (ID: {box_id})")
            
            # 创建目标区域
            target_id = self.client.load_object(
                obj_name="target_zone",
                model_path="Asset/Scene/Object/URDF_models/clear_box/model.urdf",
                object_position=[2.0, 0, 0.1],
                object_orientation=[0, 0, 0, 1],
                scale=1.0,
                fixed_base=True
            )
            print(f"   ✓ 创建目标区域 (ID: {target_id})")
            
            # 创建准备区域
            prep_id = self.client.load_object(
                obj_name="prep_station",
                model_path="Asset/Scene/Object/URDF_models/clear_box/model.urdf",
                object_position=[0.0, 2.0, 0.1],
                object_orientation=[0, 0, 0, 1],
                scale=1.0,
                fixed_base=True
            )
            print(f"   ✓ 创建准备区域 (ID: {prep_id})")
            
            return {
                "table": table_id,
                "box": box_id,
                "target_zone": target_id,
                "prep_station": prep_id
            }
            
        except Exception as e:
            print(f"   [ERROR] 创建场景失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def setup_robots(self):
        """设置机器人"""
        print("\n" + "="*60)
        print("3. 创建机器人")
        print("="*60)
        
        try:
            self.robot_factory = RobotFactory(self.client, self.visualizer)
            
            # 创建移动操作机器人 (helper)
            robot1 = self.robot_factory.create_robot(
                robot_id="helper_mobile_manipulator",
                robot_type="mobile_manipulator",
                robot_model="panda_on_segbot",
                init_position=[-1.0, 0, 0]
            )
            print(f"   ✓ 创建机器人 helper_mobile_manipulator")
            
            # 创建移动基座机器人 (logistics)
            robot2 = self.robot_factory.create_robot(
                robot_id="logistics_mobile_base",
                robot_type="mobile_base",
                robot_model="segbot",
                init_position=[-1.0, 1.0, 0]
            )
            print(f"   ✓ 创建机器人 logistics_mobile_base")
            
            # 创建机械臂机器人 (precision_arm_1)
            robot3 = self.robot_factory.create_robot(
                robot_id="precision_arm_1",
                robot_type="arm",
                robot_model="panda",
                init_position=[1.0, -1.0, 0]
            )
            print(f"   ✓ 创建机器人 precision_arm_1")
            
            # 创建第二个机械臂机器人 (precision_arm_2)
            robot4 = self.robot_factory.create_robot(
                robot_id="precision_arm_2",
                robot_type="arm",
                robot_model="panda",
                init_position=[2.0, -1.0, 0]
            )
            print(f"   ✓ 创建机器人 precision_arm_2")
            
            return self.robot_factory.get_all_robots()
            
        except Exception as e:
            print(f"   [ERROR] 创建机器人失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def setup_adapter(self, scene_objects):
        """设置 BestManAdapter"""
        print("\n" + "="*60)
        print("4. 创建 BestManAdapter")
        print("="*60)
        
        try:
            all_robots = self.robot_factory.get_all_robots()
            self.adapter = BestManAdapter(all_robots, self.client)
            
            # 注册场景物体
            for obj_name, obj_id in scene_objects.items():
                obj_type = "furniture" if obj_name in ["table", "target_zone", "prep_station"] else "graspable"
                self.adapter.register_scene_object(obj_name, obj_id, obj_type)
            
            print(f"   ✓ Adapter 创建成功，管理 {len(all_robots)} 个机器人")
            print(f"   ✓ 注册 {len(self.adapter.scene_objects)} 个场景物体")
            
            return True
            
        except Exception as e:
            print(f"   [ERROR] 创建 Adapter 失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_mock_llm_client(self, predefined_responses=None):
        """创建 Mock LLM 客户端，可以预定义响应"""
        
        class Stage4MockLLMClient:
            """专门用于测试第四阶段的 Mock LLM"""
            
            def __init__(self, responses=None):
                self.responses = responses or {}
                self.call_count = 0
                
            def generate(self, prompt, temperature=0.7):
                """生成响应"""
                self.call_count += 1
                
                # 根据 prompt 内容判断当前阶段
                if "Execution" in prompt or "action" in prompt.lower():
                    return self._generate_execution_response(prompt)
                elif "Task Allocation" in prompt:
                    return self._generate_allocation_response(prompt)
                elif "Leader Election" in prompt:
                    return self._generate_leader_response(prompt)
                else:
                    return self._generate_self_description_response(prompt)
            
            def _generate_execution_response(self, prompt):
                """生成执行阶段响应 - 返回具体动作"""
                import random
                
                # 根据机器人名称决定动作
                if "helper_mobile_manipulator" in prompt:
                    # helper 机器人执行导航到准备区域
                    actions = [
                        {
                            "action": "navigate",
                            "params": {"target": [0.0, 1.5, 0.0]},
                            "reasoning": "Moving to prep station to assist with preparation"
                        },
                        {
                            "action": "navigate", 
                            "params": {"x": 0.5, "y": 0.5, "z": 0.0},
                            "reasoning": "Moving closer to work area"
                        },
                        {
                            "action": "wait",
                            "params": {"duration": 1.0},
                            "reasoning": "Waiting for other robots to complete their tasks"
                        }
                    ]
                elif "logistics_mobile_base" in prompt:
                    # logistics 机器人执行运输任务
                    actions = [
                        {
                            "action": "navigate",
                            "params": {"target": [1.5, 0.0, 0.0]},
                            "reasoning": "Transporting items to target zone"
                        },
                        {
                            "action": "navigate",
                            "params": {"x": -0.5, "y": 1.0, "z": 0.0},
                            "reasoning": "Moving to pickup location"
                        }
                    ]
                elif "precision_arm_1" in prompt:
                    # precision_arm_1 执行抓取
                    actions = [
                        {
                            "action": "pick",
                            "params": {"object_id": "box", "object_name": "box"},
                            "reasoning": "Picking up the box for processing"
                        },
                        {
                            "action": "communicate",
                            "params": {
                                "to": "helper_mobile_manipulator",
                                "message": "precision_arm_1 ready. Awaiting confirmation.",
                                "broadcast": False
                            },
                            "reasoning": "Communicating with helper robot"
                        }
                    ]
                elif "precision_arm_2" in prompt:
                    # precision_arm_2 执行放置
                    actions = [
                        {
                            "action": "place",
                            "params": {"location": [2.0, 0.0, 0.5]},
                            "reasoning": "Placing item at target zone"
                        },
                        {
                            "action": "communicate",
                            "params": {
                                "to": "precision_arm_1",
                                "message": "precision_arm_2 ready for coordination.",
                                "broadcast": False
                            },
                            "reasoning": "Coordinating with precision_arm_1"
                        }
                    ]
                else:
                    actions = [
                        {
                            "action": "wait",
                            "params": {"duration": 1.0},
                            "reasoning": "Default wait action"
                        }
                    ]
                
                # 随机选择一个动作
                action = random.choice(actions)
                
                return f"""```json
{str(action).replace("'", '"')}
```"""
            
            def _generate_allocation_response(self, prompt):
                """生成任务分配响应"""
                return """```json
{
    "subtasks": [
        {"id": "subtask_1", "description": "Navigate to prep station", "assigned_to": "helper_mobile_manipulator"},
        {"id": "subtask_2", "description": "Transport materials", "assigned_to": "logistics_mobile_base"},
        {"id": "subtask_3", "description": "Pick and place objects", "assigned_to": "precision_arm_1"},
        {"id": "subtask_4", "description": "Coordinate with team", "assigned_to": "precision_arm_2"}
    ],
    "reasoning": "Task allocation based on robot capabilities"
}
```"""
            
            def _generate_leader_response(self, prompt):
                """生成 Leader 选举响应"""
                return """```json
{
    "vote": "helper_mobile_manipulator",
    "reasoning": "Mobile manipulator has the most capabilities for coordination"
}
```"""
            
            def _generate_self_description_response(self, prompt):
                """生成自我描述响应"""
                return """```json
{
    "capabilities": ["navigation", "manipulation", "transport"],
    "strengths": ["mobile", "versatile"],
    "limitations": ["limited payload"]
}
```"""
        
        return Stage4MockLLMClient(predefined_responses)
    
    def setup_collaboration(self, robots):
        """设置协作系统 - 直接准备第四阶段"""
        print("\n" + "="*60)
        print("5. 设置协作系统（直接准备第四阶段）")
        print("="*60)
        
        try:
            # 创建 Mock LLM
            mock_llm = self.create_mock_llm_client()
            
            # 创建机器人代理
            robot_agents = []
            for robot_id, robot in robots.items():
                agent = RobotAgent(
                    name=robot_id,
                    robot_type=robot.robot_type,
                    capabilities=robot.capabilities,
                    llm_client=mock_llm
                )
                robot_agents.append(agent)
                print(f"   ✓ 创建机器人代理: {robot_id}")
            
            # 创建协作系统，传入机器人列表
            self.collaboration = FourStageCollaboration(
                robots=robot_agents,
                enable_reflection=False,  # 禁用反射以简化测试
                max_execution_steps=20   # 限制执行步数
            )
            
            # 设置 Adapter
            self.collaboration.set_adapter(self.adapter)
            
            print(f"   ✓ 已注册 {len(robot_agents)} 个机器人代理")
            
            # 直接设置 Leader 和任务计划（跳过前三个阶段）
            self.collaboration.leader_name = "helper_mobile_manipulator"
            self.collaboration.task_plan = {
                "task": "Test Stage 4 Execution",
                "subtasks": [
                    {"id": "subtask_1", "description": "Navigate to prep station", "assigned_to": "helper_mobile_manipulator"},
                    {"id": "subtask_2", "description": "Transport to target", "assigned_to": "logistics_mobile_base"},
                    {"id": "subtask_3", "description": "Pick object", "assigned_to": "precision_arm_1"},
                    {"id": "subtask_4", "description": "Place object", "assigned_to": "precision_arm_2"}
                ]
            }
            
            print(f"   ✓ 设置 Leader: {self.collaboration.leader_name}")
            print(f"   ✓ 设置任务计划: {len(self.collaboration.task_plan['subtasks'])} 个子任务")
            
            return True
            
        except Exception as e:
            print(f"   [ERROR] 设置协作系统失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_stage4_test(self):
        """运行第四阶段测试"""
        print("\n" + "="*60)
        print("6. 运行第四阶段（执行阶段）")
        print("="*60)
        
        try:
            # 直接调用第四阶段执行
            from dynahmrc.core.collaboration import CollaborationPhase
            
            # 设置阶段为执行
            self.collaboration.manager.current_phase = CollaborationPhase.EXECUTION
            
            # 运行执行阶段
            task = "Test Stage 4 Execution"
            success = self.collaboration._run_execution(task)
            
            print(f"\n   ✓ 第四阶段执行完成: {'成功' if success else '失败'}")
            
            # 显示执行历史
            print(f"\n   执行历史 ({len(self.collaboration.execution_history)} 步):")
            for i, record in enumerate(self.collaboration.execution_history[:10]):  # 只显示前10步
                robot = record.get('robot', 'unknown')
                action = record.get('action', {})
                feedback = record.get('feedback', {})
                action_type = action.get('action', 'unknown')
                success_flag = feedback.get('success', False)
                print(f"     Step {i+1}: {robot} -> {action_type} ({'✓' if success_flag else '✗'})")
            
            if len(self.collaboration.execution_history) > 10:
                print(f"     ... 还有 {len(self.collaboration.execution_history) - 10} 步")
            
            return success
            
        except Exception as e:
            print(f"   [ERROR] 第四阶段执行失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """运行完整测试"""
        print("\n" + "="*70)
        print("第四阶段（执行阶段）独立测试")
        print("="*70)
        
        # 1. 设置环境
        if not self.setup_environment():
            print("\n[FAILED] 环境设置失败")
            return False
        
        # 2. 设置场景
        scene_objects = self.setup_scene()
        if scene_objects is None:
            print("\n[FAILED] 场景设置失败")
            return False
        
        # 3. 设置机器人
        robots = self.setup_robots()
        if robots is None:
            print("\n[FAILED] 机器人设置失败")
            return False
        
        # 4. 设置 Adapter
        if not self.setup_adapter(scene_objects):
            print("\n[FAILED] Adapter 设置失败")
            return False
        
        # 5. 设置协作系统
        if not self.setup_collaboration(robots):
            print("\n[FAILED] 协作系统设置失败")
            return False
        
        # 6. 运行第四阶段测试
        success = self.run_stage4_test()
        
        # 7. 清理
        print("\n" + "="*60)
        print("7. 测试完成，清理资源")
        print("="*60)
        
        # 保持仿真运行一段时间以便观察
        print("   保持仿真运行 5 秒以便观察...")
        import time
        time.sleep(5)
        
        # 断开连接
        if self.client:
            self.client.disconnect()
            print("   ✓ 断开 BestMan 连接")
        
        print("\n" + "="*70)
        if success:
            print("[SUCCESS] 第四阶段测试完成！")
        else:
            print("[FAILED] 第四阶段测试失败")
        print("="*70)
        
        return success


def main():
    """主函数"""
    tester = Stage4Tester()
    success = tester.run()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
