#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务管理API接口

此模块提供了简单的函数接口，方便其他模块调用任务管理功能。
"""
import logging
from typing import Optional, Dict, Any, List, Union

# 导入任务管理组件
from .task_manager import get_task_manager, TASK_STATUS, SUPPORTED_PLATFORMS, TASK_TYPES
from .task_scheduler import get_task_scheduler

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_conversion_task(task_type: str, platform: str, input_path: str, 
                          output_path: str, parameters: Optional[Dict[str, Any]] = None, 
                          task_id: Optional[str] = None) -> str:
    """
    创建模型转换任务
    
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
        Exception: 创建任务失败时抛出
    """
    try:
        task_manager = get_task_manager()
        task_id = task_manager.create_task(
            task_type=task_type,
            platform=platform,
            input_path=input_path,
            output_path=output_path,
            parameters=parameters,
            task_id=task_id
        )
        logger.info(f"创建转换任务成功: {task_id}")
        return task_id
    except Exception as e:
        logger.error(f"创建转换任务失败: {str(e)}")
        raise

def get_task_info(task_id: str) -> Optional[Dict[str, Any]]:
    """
    获取任务信息
    
    Args:
        task_id: 任务ID
    
    Returns:
        任务信息字典，如果任务不存在则返回None
    """
    try:
        task_manager = get_task_manager()
        task = task_manager.get_task(task_id)
        return task
    except Exception as e:
        logger.error(f"获取任务信息失败: {str(e)}")
        raise

def get_all_tasks() -> List[Dict[str, Any]]:
    """
    获取所有任务列表
    
    Returns:
        所有任务列表
    """
    try:
        task_manager = get_task_manager()
        tasks = task_manager.get_all_tasks()
        return tasks
    except Exception as e:
        logger.error(f"获取任务列表失败: {str(e)}")
        raise

def get_tasks_by_status(status: str) -> List[Dict[str, Any]]:
    """
    根据状态获取任务列表
    
    Args:
        status: 任务状态（如 'pending', 'running', 'completed', 'failed'）
    
    Returns:
        指定状态的任务列表
    """
    try:
        task_manager = get_task_manager()
        # 先验证状态是否有效
        if status not in TASK_STATUS.values():
            raise ValueError(f"无效的任务状态: {status}")
        
        # 如果是待执行状态，使用专门的方法
        if status == TASK_STATUS['PENDING']:
            tasks = task_manager.get_pending_tasks()
        else:
            # 其他状态需要从数据库管理器直接获取
            from .database import get_db_manager
            db_manager = get_db_manager()
            tasks = db_manager.get_tasks_by_status(status)
            # 解析参数字符串
            for task in tasks:
                if task.get('parameters'):
                    import json
                    task['parameters'] = json.loads(task['parameters'])
        
        return tasks
    except Exception as e:
        logger.error(f"获取任务列表失败: {str(e)}")
        raise

def execute_task_immediately(task_id: str) -> bool:
    """
    立即执行指定的任务
    
    Args:
        task_id: 任务ID
    
    Returns:
        是否执行成功
    """
    try:
        scheduler = get_task_scheduler()
        success = scheduler.execute_task_immediately(task_id)
        return success
    except Exception as e:
        logger.error(f"立即执行任务失败: {str(e)}")
        raise

def cancel_task(task_id: str) -> bool:
    """
    取消任务（仅适用于等待执行的任务）
    
    Args:
        task_id: 任务ID
    
    Returns:
        是否取消成功
    """
    try:
        task_manager = get_task_manager()
        task = task_manager.get_task(task_id)
        
        if not task:
            logger.warning(f"任务不存在: {task_id}")
            return False
        
        if task['status'] != TASK_STATUS['PENDING']:
            logger.warning(f"只有等待执行的任务才能被取消: {task_id}, 当前状态: {task['status']}")
            return False
        
        # 更新任务状态为失败
        success = task_manager.update_task_status(
            task_id=task_id,
            status=TASK_STATUS['FAILED'],
            error_message="任务被用户取消"
        )
        
        if success:
            logger.info(f"任务已取消: {task_id}")
        
        return success
    except Exception as e:
        logger.error(f"取消任务失败: {str(e)}")
        raise

def delete_task(task_id: str) -> bool:
    """
    删除任务
    
    Args:
        task_id: 任务ID
    
    Returns:
        是否删除成功
    """
    try:
        task_manager = get_task_manager()
        success = task_manager.delete_task(task_id)
        return success
    except Exception as e:
        logger.error(f"删除任务失败: {str(e)}")
        raise

def create_huawei_onnx_to_om_task(input_path: str, output_path: str, 
                                 parameters: Optional[Dict[str, Any]] = None, 
                                 task_id: Optional[str] = None) -> str:
    """
    创建华为平台ONNX转OM的任务（便捷函数）
    
    Args:
        input_path: 输入ONNX模型路径
        output_path: 输出OM模型路径
        parameters: 转换参数
        task_id: 可选的任务ID
    
    Returns:
        创建的任务ID
    """
    return create_conversion_task(
        task_type=TASK_TYPES['ONNX_TO_OM'],
        platform=SUPPORTED_PLATFORMS['HUAWEI'],
        input_path=input_path,
        output_path=output_path,
        parameters=parameters,
        task_id=task_id
    )

def get_task_status(task_id: str) -> Optional[str]:
    """
    获取任务状态
    
    Args:
        task_id: 任务ID
    
    Returns:
        任务状态字符串，如果任务不存在则返回None
    """
    try:
        task = get_task_info(task_id)
        return task.get('status') if task else None
    except Exception:
        return None

def is_task_completed(task_id: str) -> bool:
    """
    检查任务是否已完成
    
    Args:
        task_id: 任务ID
    
    Returns:
        如果任务已成功完成则返回True
    """
    try:
        status = get_task_status(task_id)
        return status == TASK_STATUS['COMPLETED']
    except Exception:
        return False

def is_task_failed(task_id: str) -> bool:
    """
    检查任务是否失败
    
    Args:
        task_id: 任务ID
    
    Returns:
        如果任务失败则返回True
    """
    try:
        status = get_task_status(task_id)
        return status == TASK_STATUS['FAILED']
    except Exception:
        return False

def get_tasks_paginated(page: int = 1, page_size: int = 10) -> Dict[str, Any]:
    """
    分页获取任务列表
    
    Args:
        page: 页码，从1开始
        page_size: 每页任务数量
    
    Returns:
        包含任务列表、总数、页码等信息的字典
    """
    try:
        from .database import get_db_manager
        db_manager = get_db_manager()
        result = db_manager.get_tasks_paginated(page, page_size)
        
        # 解析参数字符串
        for task in result['tasks']:
            if task.get('parameters'):
                import json
                task['parameters'] = json.loads(task['parameters'])
        
        return result
    except Exception as e:
        logger.error(f"分页获取任务列表失败: {str(e)}")
        raise

def pause_task(task_id: str) -> bool:
    """
    暂停任务（仅适用于正在执行的任务）
    
    Args:
        task_id: 任务ID
    
    Returns:
        是否暂停成功
    """
    try:
        task_manager = get_task_manager()
        success = task_manager.pause_task(task_id)
        return success
    except Exception as e:
        logger.error(f"暂停任务失败: {str(e)}")
        raise

def resume_task(task_id: str) -> bool:
    """
    恢复任务（仅适用于已暂停的任务）
    
    Args:
        task_id: 任务ID
    
    Returns:
        是否恢复成功
    """
    try:
        task_manager = get_task_manager()
        success = task_manager.resume_task(task_id)
        return success
    except Exception as e:
        logger.error(f"恢复任务失败: {str(e)}")
        raise

def delete_tasks_batch(task_ids: List[str]) -> Dict[str, bool]:
    """
    批量删除任务
    
    Args:
        task_ids: 任务ID列表
    
    Returns:
        每个任务ID对应的删除结果字典
    """
    try:
        from .database import get_db_manager
        db_manager = get_db_manager()
        results = db_manager.delete_tasks_batch(task_ids)
        return results
    except Exception as e:
        logger.error(f"批量删除任务失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 示例用法
    print("任务管理API接口示例")
    print(f"支持的任务状态: {list(TASK_STATUS.values())}")
    print(f"支持的平台: {list(SUPPORTED_PLATFORMS.values())}")
    print(f"支持的任务类型: {list(TASK_TYPES.values())}")
    
    # 这里可以添加更多示例代码