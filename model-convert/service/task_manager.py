#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务管理器模块

此模块提供了任务的创建、查询、更新等功能，是任务管理系统的核心。
"""
import uuid
import json
import logging
import os
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

# 导入数据库管理器
from .database import get_db_manager, DatabaseManager

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 任务状态常量
TASK_STATUS = {
    'PENDING': 'pending',      # 等待执行
    'RUNNING': 'running',      # 正在执行
    'PAUSED': 'paused',        # 已暂停
    'COMPLETED': 'completed',  # 执行成功
    'FAILED': 'failed'         # 执行失败
}

# 支持的平台
SUPPORTED_PLATFORMS = {
    'HUAWEI': 'huawei',        # 华为昇腾
    'ROCKCHIP': 'rockchip',    # 瑞芯微
    'CAMBRICON': 'cambricon',  # 寒武纪
    'ONNX': 'onnx'             # ONNX平台
}

# 支持的任务类型
TASK_TYPES = {
    'ONNX_TO_OM': 'onnx_to_om',           # ONNX转OM（华为）
    'ONNX_TO_RKNN': 'onnx_to_rknn',       # ONNX转RKNN（瑞芯微）
    'ONNX_TO_CAMB': 'onnx_to_cambricon',  # ONNX转寒武纪
    'YOLO_TO_ONNX': 'yolo_to_onnx'        # YOLO转ONNX
}

class TaskManager:
    """
    任务管理器，提供任务相关的核心功能
    """
    
    def __init__(self):
        """
        初始化任务管理器
        """
        self.db_manager = get_db_manager()
    
    def create_task(self, task_type: str, platform: str, input_path: str, 
                   output_path: str, parameters: Optional[Dict[str, Any]] = None, 
                   task_id: Optional[str] = None) -> str:
        """
        创建新的模型转换任务
        
        Args:
            task_type: 任务类型（如 'onnx_to_om'）
            platform: 目标平台（如 'huawei'）
            input_path: 输入模型路径
            output_path: 输出模型路径
            parameters: 转换参数
            task_id: 可选的任务ID，如果不提供则自动生成
        
        Returns:
            task_id: 创建的任务ID
        
        Raises:
            ValueError: 参数验证失败时抛出
        """
        # 创建任务
        try:
            # 验证必要参数
            if not all([task_type, platform, input_path, output_path]):
                raise ValueError("缺少必要的任务参数")
                
            # 验证平台和任务类型的支持情况
            if platform not in SUPPORTED_PLATFORMS.values():
                raise ValueError(f"不支持的平台: {platform}")
                
            if task_type not in TASK_TYPES.values():
                raise ValueError(f"不支持的任务类型: {task_type}")
            
            # 确保任务ID唯一
            if not task_id:
                import time
                import random
                task_id = f"task_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
            
            # 准备任务数据
            task_data = {
                'id': task_id,
                'task_type': task_type,
                'platform': platform,
                'status': TASK_STATUS['PENDING'],
                'input_path': input_path,
                'output_path': output_path,
                'parameters': json.dumps(parameters) if parameters else None
            }
            
            created_task_id = self.db_manager.create_task(task_data)
            logger.info(f"创建任务成功: ID={created_task_id}, 类型={task_type}, 平台={platform}")
            return created_task_id
        except Exception as e:
            logger.error(f"创建任务失败: {str(e)}")
            raise
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务信息
        
        Args:
            task_id: 任务ID
        
        Returns:
            任务信息字典，如果任务不存在则返回None
        """
        try:
            task = self.db_manager.get_task(task_id)
            if task:
                # 将参数字符串转换回字典
                if task.get('parameters'):
                    task['parameters'] = json.loads(task['parameters'])
                logger.info(f"获取任务成功: {task_id}")
            else:
                logger.warning(f"任务不存在: {task_id}")
            return task
        except Exception as e:
            logger.error(f"获取任务失败: {str(e)}")
            raise
    
    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """
        获取等待执行的任务列表
        
        Returns:
            待执行任务列表
        """
        try:
            tasks = self.db_manager.get_tasks_by_status(TASK_STATUS['PENDING'])
            # 解析参数字符串
            for task in tasks:
                if task.get('parameters'):
                    task['parameters'] = json.loads(task['parameters'])
            logger.info(f"获取待执行任务数量: {len(tasks)}")
            return tasks
        except Exception as e:
            logger.error(f"获取待执行任务失败: {str(e)}")
            raise
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        获取所有任务
        
        Returns:
            所有任务列表
        """
        try:
            tasks = self.db_manager.get_all_tasks()
            # 解析参数字符串
            for task in tasks:
                if task.get('parameters'):
                    task['parameters'] = json.loads(task['parameters'])
            logger.info(f"获取所有任务数量: {len(tasks)}")
            return tasks
        except Exception as e:
            logger.error(f"获取所有任务失败: {str(e)}")
            raise
    
    def start_task(self, task_id: str) -> bool:
        """
        开始执行任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            是否成功开始执行
        """
        try:
            task = self.get_task(task_id)
            if not task:
                logger.warning(f"任务不存在: {task_id}")
                return False
            
            if task['status'] != TASK_STATUS['PENDING']:
                logger.warning(f"任务状态不允许开始执行: {task_id}, 当前状态: {task['status']}")
                return False
            
            # 更新任务状态为运行中
            updates = {
                'status': TASK_STATUS['RUNNING'],
                'started_at': datetime.now().isoformat()
            }
            
            self.db_manager.update_task(task_id, updates)
            logger.info(f"任务开始执行: {task_id}")
            return True
        except Exception as e:
            logger.error(f"开始任务失败: {str(e)}")
            raise
    
    def complete_task(self, task_id: str, error_message: Optional[str] = None, 
                     log_path: Optional[str] = None) -> bool:
        """
        完成任务
        
        Args:
            task_id: 任务ID
            error_message: 错误信息，如果有
            log_path: 日志文件路径
        
        Returns:
            是否成功完成
        """
        try:
            task = self.get_task(task_id)
            if not task:
                logger.warning(f"任务不存在: {task_id}")
                return False
            
            if task['status'] != TASK_STATUS['RUNNING']:
                logger.warning(f"任务状态不允许完成: {task_id}, 当前状态: {task['status']}")
                return False
            
            # 准备更新数据
            updates = {
                'completed_at': datetime.now().isoformat(),
                'status': TASK_STATUS['FAILED'] if error_message else TASK_STATUS['COMPLETED'],
                'error_message': error_message,
                'log_path': log_path
            }
            
            # 移除None值
            updates = {k: v for k, v in updates.items() if v is not None}
            
            self.db_manager.update_task(task_id, updates)
            
            if error_message:
                logger.warning(f"任务执行失败: {task_id}, 错误: {error_message}")
            else:
                logger.info(f"任务执行成功: {task_id}")
            
            return True
        except Exception as e:
            logger.error(f"完成任务失败: {str(e)}")
            raise
    
    def update_task_status(self, task_id: str, status: str, 
                          error_message: Optional[str] = None) -> bool:
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
            error_message: 错误信息（如果状态为失败）
        
        Returns:
            是否更新成功
        """
        try:
            if status not in TASK_STATUS.values():
                raise ValueError(f"不支持的任务状态: {status}")
            
            updates = {'status': status}
            if status == TASK_STATUS['FAILED'] and error_message:
                updates['error_message'] = error_message
            
            self.db_manager.update_task(task_id, updates)
            logger.info(f"任务状态更新成功: {task_id}, 新状态: {status}")
            return True
        except Exception as e:
            logger.error(f"更新任务状态失败: {str(e)}")
            raise
    
    def pause_task(self, task_id: str) -> bool:
        """
        暂停任务（仅适用于正在执行的任务）
        
        Args:
            task_id: 任务ID
        
        Returns:
            是否暂停成功
        """
        try:
            task = self.get_task(task_id)
            if not task:
                logger.warning(f"任务不存在: {task_id}")
                return False
            
            if task['status'] != TASK_STATUS['RUNNING']:
                logger.warning(f"只有正在执行的任务才能被暂停: {task_id}, 当前状态: {task['status']}")
                return False
            
            # 更新任务状态为已暂停
            success = self.update_task_status(
                task_id=task_id,
                status=TASK_STATUS['PAUSED']
            )
            
            if success:
                logger.info(f"任务已暂停: {task_id}")
            
            return success
        except Exception as e:
            logger.error(f"暂停任务失败: {str(e)}")
            raise
    
    def resume_task(self, task_id: str) -> bool:
        """
        恢复任务（仅适用于已暂停的任务）
        
        Args:
            task_id: 任务ID
        
        Returns:
            是否恢复成功
        """
        try:
            task = self.get_task(task_id)
            if not task:
                logger.warning(f"任务不存在: {task_id}")
                return False
            
            if task['status'] != TASK_STATUS['PAUSED']:
                logger.warning(f"只有已暂停的任务才能被恢复: {task_id}, 当前状态: {task['status']}")
                return False
            
            # 更新任务状态为运行中
            success = self.update_task_status(
                task_id=task_id,
                status=TASK_STATUS['RUNNING']
            )
            
            if success:
                logger.info(f"任务已恢复: {task_id}")
            
            return success
        except Exception as e:
            logger.error(f"恢复任务失败: {str(e)}")
            raise
    
    def delete_task(self, task_id: str) -> bool:
        """
        删除任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            是否删除成功
        """
        try:
            result = self.db_manager.delete_task(task_id)
            return result
        except Exception as e:
            logger.error(f"删除任务失败: {str(e)}")
            raise

# 创建全局任务管理器实例
task_manager = None

def get_task_manager() -> TaskManager:
    """
    获取任务管理器实例
    
    Returns:
        TaskManager实例
    """
    global task_manager
    if task_manager is None:
        task_manager = TaskManager()
    return task_manager

if __name__ == "__main__":
    # 测试任务管理器功能
    manager = TaskManager()
    task_id = None
    
    # 创建测试任务
    try:
        task_id = manager.create_task(
            task_type='onnx_to_om',
            platform='huawei',
            input_path='/test/input.onnx',
            output_path='/test/output.om',
            parameters={'input_shape': '1,3,640,640'}
        )
        print(f"创建任务成功: {task_id}")
        
        # 获取任务
        task = manager.get_task(task_id)
        print(f"任务信息: {task}")
        
        # 开始任务
        manager.start_task(task_id)
        
        # 完成任务
        manager.complete_task(task_id)
        
        # 获取所有任务
        tasks = manager.get_all_tasks()
        print(f"所有任务: {tasks}")
        
    finally:
        # 清理测试数据
        if task_id is not None:
            manager.delete_task(task_id)