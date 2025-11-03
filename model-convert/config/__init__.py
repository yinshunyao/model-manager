#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置模块初始化文件
提供配置相关的便捷导入和配置获取函数

本模块导出ConfigLoader类和config_loader实例，
以及一系列便捷函数用于获取特定配置项。
"""

from typing import Any, Dict, Optional
from .config_loader import ConfigLoader, config_loader

# 定义便捷函数，直接从config_loader实例导出
def get_minio_config() -> Dict[str, Any]:
    """获取MinIO配置"""
    return config_loader.get_minio_config()

def get_mysql_config() -> Dict[str, Any]:
    """获取MySQL配置"""
    return config_loader.get_mysql_config()

def get_server_config() -> Dict[str, Any]:
    """获取服务器配置"""
    return config_loader.get_server_config()

def get_ascend_config() -> Dict[str, Any]:
    """获取华为昇腾配置"""
    return config_loader.get_ascend_config()

def get_rockchip_config() -> Dict[str, Any]:
    """获取瑞芯微配置"""
    return config_loader.get_rockchip_config()

def get_cambricon_config() -> Dict[str, Any]:
    """获取寒武纪配置"""
    return config_loader.get_cambricon_config()

def get_conversion_config() -> Dict[str, Any]:
    """获取模型转换配置"""
    return config_loader.get_conversion_config()

def get_logging_config() -> Dict[str, Any]:
    """获取日志配置"""
    return config_loader.get_logging_config()

def get_config_value(key_path: str, default: Optional[Any] = None) -> Any:
    """获取指定路径的配置值"""
    return config_loader.get_config_value(key_path, default)

def reload_config() -> None:
    """重新加载所有配置"""
    return config_loader.reload_config()

__all__ = [
    'ConfigLoader',
    'config_loader',
    'get_minio_config',
    'get_mysql_config',
    'get_server_config',
    'get_ascend_config',
    'get_rockchip_config',
    'get_cambricon_config',
    'get_conversion_config',
    'get_logging_config',
    'get_config_value',
    'reload_config'
]