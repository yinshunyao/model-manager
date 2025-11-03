#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务管理服务包初始化文件
"""

# 导出主要组件
from .database import get_db_manager
from .task_manager import get_task_manager, TASK_STATUS, SUPPORTED_PLATFORMS, TASK_TYPES
from .task_executor import get_task_executor
from .task_scheduler import get_task_scheduler

__all__ = [
    'get_db_manager',
    'get_task_manager',
    'TASK_STATUS',
    'SUPPORTED_PLATFORMS',
    'TASK_TYPES',
    'get_task_executor',
    'get_task_scheduler'
]

# 包版本
__version__ = '1.0.0'