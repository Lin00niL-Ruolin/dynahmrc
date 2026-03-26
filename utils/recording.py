"""
录像和日志记录管理模块
提供仿真录像和LLM交互日志记录功能
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import pickle


class SimulationRecorder:
    """
    仿真录像管理器
    
    功能：
    1. 在第四阶段开始时启动录像
    2. 在第四阶段结束或用户停止时保存录像
    3. 支持PyBullet的Blender格式录像
    """
    
    def __init__(self, client, enable_recording: bool = True, output_dir: str = "recordings"):
        """
        初始化录像管理器
        
        Args:
            client: PyBullet Client 实例
            enable_recording: 是否启用录像
            output_dir: 录像输出目录
        """
        self.client = client
        self.enable_recording = enable_recording
        self.output_dir = output_dir
        self.is_recording = False
        self.recorder = None
        self.mtl_recorder = {}
        self.start_time = None
        self.frame_count = 0
        
        # 创建输出目录
        if self.enable_recording:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"[SimulationRecorder] 录像功能已启用，输出目录: {self.output_dir}")
        else:
            print(f"[SimulationRecorder] 录像功能已禁用")
    
    def start_recording(self):
        """开始录像"""
        if not self.enable_recording:
            print("[SimulationRecorder] 录像功能未启用，跳过")
            return

        if self.is_recording:
            print("[SimulationRecorder] 已经在录像中")
            return

        try:
            # 尝试导入PyBulletRecorder
            from Visualization.blender_render import PyBulletRecorder
            
            self.recorder = PyBulletRecorder()
            self.mtl_recorder = {}
            self.is_recording = True
            self.start_time = time.time()
            self.frame_count = 0
            
            # 注册当前场景中的所有物体
            self._register_scene_objects()
            
            print(f"[SimulationRecorder] 开始录像")
            
        except ImportError:
            print("[SimulationRecorder] 警告: PyBulletRecorder 未找到，录像功能不可用")
            self.enable_recording = False
    
    def _register_scene_objects(self):
        """注册场景中的所有物体到录像器"""
        if self.recorder is None:
            return
        
        try:
            import pybullet as p
            
            # 获取所有物体
            num_bodies = p.getNumBodies(physicsClientId=self.client.client_id)
            
            for i in range(num_bodies):
                body_id = p.getBodyUniqueId(i, physicsClientId=self.client.client_id)
                
                # 获取物体信息
                try:
                    body_info = p.getBodyInfo(body_id, physicsClientId=self.client.client_id)
                    model_path = body_info[1].decode('utf-8') if isinstance(body_info[1], bytes) else str(body_info[1])
                    
                    # 注册物体
                    self.recorder.register_object(body_id, model_path, scale=1.0)
                    
                except Exception as e:
                    print(f"[SimulationRecorder] 注册物体 {body_id} 失败: {e}")
            
            print(f"[SimulationRecorder] 已注册 {num_bodies} 个物体")
            
        except Exception as e:
            print(f"[SimulationRecorder] 注册场景物体失败: {e}")
    
    def add_keyframe(self):
        """添加关键帧"""
        if not self.is_recording or self.recorder is None:
            return
        
        try:
            self.recorder.add_keyframe()
            self.frame_count += 1
        except Exception as e:
            print(f"[SimulationRecorder] 添加关键帧失败: {e}")
    
    def stop_recording(self, task_name: str = "simulation"):
        """
        停止录像并保存
        
        Args:
            task_name: 任务名称，用于文件名
        """
        if not self.is_recording or self.recorder is None:
            return
        
        try:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{task_name}_{timestamp}.pkl"
            filepath = os.path.join(self.output_dir, filename)
            
            # 保存录像
            self.recorder.save(filepath, self.mtl_recorder)
            
            duration = time.time() - self.start_time if self.start_time else 0
            
            print(f"[SimulationRecorder] 录像已保存: {filepath}")
            print(f"[SimulationRecorder] 录像时长: {duration:.1f}s, 帧数: {self.frame_count}")
            
        except Exception as e:
            print(f"[SimulationRecorder] 保存录像失败: {e}")
        finally:
            self.is_recording = False
            self.recorder = None
            self.frame_count = 0


class LLMLogger:
    """
    LLM交互日志记录器
    
    功能：
    1. 记录所有提示词（Prompt）
    2. 记录所有LLM响应（Response）
    3. 保存为易读的文本文件
    """
    
    def __init__(self, enable_logging: bool = True, output_dir: str = "logs"):
        """
        初始化日志记录器
        
        Args:
            enable_logging: 是否启用日志记录
            output_dir: 日志输出目录
        """
        self.enable_logging = enable_logging
        self.output_dir = output_dir
        self.log_file = None
        self.session_start_time = datetime.now()
        
        if self.enable_logging:
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 创建日志文件
            timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
            log_filename = f"llm_interaction_{timestamp}.txt"
            log_filepath = os.path.join(self.output_dir, log_filename)
            
            self.log_file = open(log_filepath, 'w', encoding='utf-8')
            
            # 写入文件头
            self._write_header()
            
            print(f"[LLMLogger] 日志记录已启用: {log_filepath}")
        else:
            print(f"[LLMLogger] 日志记录已禁用")
    
    def _write_header(self):
        """写入日志文件头"""
        if self.log_file is None:
            return
        
        header = f"""{'='*80}
LLM Interaction Log
Session Start: {self.session_start_time.strftime("%Y-%m-%d %H:%M:%S")}
{'='*80}

"""
        self.log_file.write(header)
        self.log_file.flush()
    
    def log_interaction(self, 
                       robot_name: str,
                       stage: str,
                       prompt: str, 
                       response: str,
                       metadata: Optional[Dict] = None):
        """
        记录一次LLM交互
        
        Args:
            robot_name: 机器人名称
            stage: 协作阶段（如 "Execution", "TaskAllocation" 等）
            prompt: 提示词
            response: LLM响应
            metadata: 额外元数据
        """
        if not self.enable_logging or self.log_file is None:
            return
        
        try:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            # 构建日志条目
            log_entry = f"""
{'='*80}
[{timestamp}] Robot: {robot_name} | Stage: {stage}
{'='*80}

--- PROMPT ---
{prompt}

--- RESPONSE ---
{response}
"""
            
            # 添加元数据
            if metadata:
                log_entry += f"\n--- METADATA ---\n"
                for key, value in metadata.items():
                    log_entry += f"{key}: {value}\n"
            
            log_entry += "\n"
            
            # 写入文件
            self.log_file.write(log_entry)
            self.log_file.flush()
            
        except Exception as e:
            print(f"[LLMLogger] 记录日志失败: {e}")
    
    def log_message(self, message: str, level: str = "INFO"):
        """
        记录普通消息
        
        Args:
            message: 消息内容
            level: 日志级别（INFO, WARNING, ERROR）
        """
        if not self.enable_logging or self.log_file is None:
            return
        
        try:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            log_entry = f"[{timestamp}] [{level}] {message}\n"
            self.log_file.write(log_entry)
            self.log_file.flush()
        except Exception as e:
            print(f"[LLMLogger] 记录消息失败: {e}")
    
    def close(self):
        """关闭日志文件"""
        if self.log_file is not None:
            try:
                # 写入文件尾
                end_time = datetime.now()
                footer = f"""
{'='*80}
Session End: {end_time.strftime("%Y-%m-%d %H:%M:%S")}
Duration: {(end_time - self.session_start_time).total_seconds():.1f}s
{'='*80}
"""
                self.log_file.write(footer)
                self.log_file.close()
                self.log_file = None
                print(f"[LLMLogger] 日志文件已关闭")
            except Exception as e:
                print(f"[LLMLogger] 关闭日志文件失败: {e}")
    
    def __del__(self):
        """析构时确保文件关闭"""
        self.close()


class RecordingManager:
    """
    录像和日志管理器（组合类）
    统一管理录像和日志功能
    """
    
    def __init__(self, 
                 client,
                 enable_recording: bool = True,
                 enable_logging: bool = True,
                 output_dir: str = "outputs"):
        """
        初始化管理器
        
        Args:
            client: PyBullet Client 实例
            enable_recording: 是否启用录像
            enable_logging: 是否启用日志记录
            output_dir: 输出根目录
        """
        self.enable_recording = enable_recording
        self.enable_logging = enable_logging
        
        # 创建子目录
        recording_dir = os.path.join(output_dir, "recordings")
        logging_dir = os.path.join(output_dir, "logs")
        
        # 初始化录像器
        self.recorder = SimulationRecorder(
            client=client,
            enable_recording=enable_recording,
            output_dir=recording_dir
        )
        
        # 初始化日志记录器
        self.logger = LLMLogger(
            enable_logging=enable_logging,
            output_dir=logging_dir
        )
        
        print(f"[RecordingManager] 初始化完成")
        print(f"  - 录像: {'启用' if enable_recording else '禁用'}")
        print(f"  - 日志: {'启用' if enable_logging else '禁用'}")
    
    def start_recording(self):
        """开始录像"""
        self.recorder.start_recording()
    
    def stop_recording(self, task_name: str = "simulation"):
        """停止录像"""
        self.recorder.stop_recording(task_name)
    
    def add_keyframe(self):
        """添加关键帧（应在仿真步进时调用）"""
        self.recorder.add_keyframe()
    
    def log_llm_interaction(self, 
                           robot_name: str,
                           stage: str,
                           prompt: str, 
                           response: str,
                           metadata: Optional[Dict] = None):
        """记录LLM交互"""
        self.logger.log_interaction(robot_name, stage, prompt, response, metadata)
    
    def log_message(self, message: str, level: str = "INFO"):
        """记录消息"""
        self.logger.log_message(message, level)
    
    def close(self):
        """关闭所有资源"""
        self.stop_recording()
        self.logger.close()
    
    def __del__(self):
        """析构时确保资源释放"""
        self.close()
