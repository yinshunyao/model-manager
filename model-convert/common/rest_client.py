#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务状态更新客户端
用于通过REST API方式更新任务状态
"""

import requests
from typing import Optional, Dict, Any, List

# 导入配置函数
try:
    # 尝试相对导入
    from config.config_loader import config_loader
except (ImportError, ModuleNotFoundError):
    # 如果相对导入失败，尝试绝对导入
    try:
        import sys
        import os
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from config.config_loader import config_loader
    except (ImportError, ModuleNotFoundError):
        # 如果仍然失败，抛出异常
        raise ImportError("无法导入config_loader，请检查config模块是否存在")


class TaskStatusClient:
    """任务状态更新客户端"""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        初始化任务状态客户端
        
        Args:
            base_url: 任务管理中心的基础URL，如果未提供则从配置文件读取
        """
        if base_url:
            self.base_url = base_url.rstrip('/')
        else:
            config = config_loader.get_task_center_config()
            self.base_url = config.get('url', 'http://localhost:8000').rstrip('/')
    
    def _make_request(self, method: str, uri: str, **kwargs) -> Dict[str, Any]:
        """
        发送HTTP请求的通用方法
        
        Args:
            method: HTTP方法 ('GET', 'POST', 'PUT', 'DELETE')
            uri: URI路径
            **kwargs: 其他传递给requests的参数
            
        Returns:
            Dict[str, Any]: API响应数据
        """
        url = f"{self.base_url}{uri}"
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[str]: 任务状态，如果任务不存在则返回None
        """
        try:
            uri = f"/tasks/{task_id}/status"
            result = self._make_request('GET', uri)
            if result.get('success'):
                return result['data']['status']
            return None
        except requests.RequestException:
            return None
    
    def pause_task(self, task_id: str) -> bool:
        """
        暂停任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 操作是否成功
        """
        try:
            uri = "/tasks/pause"
            data = {"task_id": task_id}
            result = self._make_request('POST', uri, json=data)
            return result.get('success', False)
        except requests.RequestException:
            return False
    
    def resume_task(self, task_id: str) -> bool:
        """
        恢复任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 操作是否成功
        """
        try:
            uri = "/tasks/resume"
            data = {"task_id": task_id}
            result = self._make_request('POST', uri, json=data)
            return result.get('success', False)
        except requests.RequestException:
            return False
    
    def delete_task(self, task_id: str) -> bool:
        """
        删除任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 操作是否成功
        """
        try:
            uri = "/tasks/batch-delete"
            data = {"task_ids": [task_id]}
            result = self._make_request('POST', uri, json=data)
            return result.get('success', False)
        except requests.RequestException:
            return False
    
    def batch_delete_tasks(self, task_ids: List[str]) -> Dict[str, Any]:
        """
        批量删除任务
        
        Args:
            task_ids: 任务ID列表
            
        Returns:
            Dict[str, Any]: 删除结果，包含成功/失败信息
        """
        try:
            uri = "/tasks/batch-delete"
            data = {"task_ids": task_ids}
            result = self._make_request('POST', uri, json=data)
            return result if result.get('success') else {}
        except requests.RequestException:
            return {}
    
    def complete_task(self, task_id: str, error_message: Optional[str] = None, 
                     log_path: Optional[str] = None) -> bool:
        """
        完成任务（标记为成功或失败）
        
        Args:
            task_id: 任务ID
            error_message: 错误信息（可选，如果提供则标记为失败）
            log_path: 日志路径（可选）
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 注意：此功能需要在服务端实现对应的API端点
            # 这里假设有一个/tasks/complete端点
            uri = "/tasks/complete"
            data = {
                "task_id": task_id,
                "error_message": error_message,
                "log_path": log_path
            }
            # 移除None值
            data = {k: v for k, v in data.items() if v is not None}
            result = self._make_request('POST', uri, json=data)
            return result.get('success', False)
        except requests.RequestException:
            return False
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 注意：此功能需要在服务端实现对应的API端点
            # 这里假设有一个/tasks/cancel端点
            uri = "/tasks/cancel"
            data = {"task_id": task_id}
            result = self._make_request('POST', uri, json=data)
            return result.get('success', False)
        except requests.RequestException:
            return False


# 创建全局客户端实例
task_status_client = TaskStatusClient()