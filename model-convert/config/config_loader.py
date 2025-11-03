#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置加载器模块
用于加载和管理所有配置文件
"""

import os
import yaml
from typing import Dict, Any, Optional

class ConfigLoader:
    """
    配置加载器类，负责加载和管理所有配置
    """
    
    _instance: Optional['ConfigLoader'] = None
    _config_cache: Dict[str, Any] = {}
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化配置加载器"""
        self.config_dir = os.path.dirname(os.path.abspath(__file__))
        self.main_config_path = os.path.join(self.config_dir, 'config.yaml')
        
    def load_config(self) -> Dict[str, Any]:
        """
        加载主配置文件
        
        Returns:
            Dict[str, Any]: 配置字典
        
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
        """
        if 'main' not in self._config_cache:
            if not os.path.exists(self.main_config_path):
                raise FileNotFoundError(f"配置文件不存在: {self.main_config_path}")
            
            with open(self.main_config_path, 'r', encoding='utf-8') as f:
                self._config_cache['main'] = yaml.safe_load(f)
        
        return self._config_cache['main']
    
    def get_minio_config(self) -> Dict[str, Any]:
        """
        获取MinIO配置
        
        Returns:
            Dict[str, Any]: MinIO配置字典
        """
        config = self.load_config()
        return config.get('minio', {})
    
    def get_mysql_config(self) -> Dict[str, Any]:
        """
        获取MySQL配置
        
        Returns:
            Dict[str, Any]: MySQL配置字典
        """
        config = self.load_config()
        return config.get('mysql', {})
    
    def get_server_config(self) -> Dict[str, Any]:
        """
        获取服务器配置
        
        Returns:
            Dict[str, Any]: 服务器配置字典
        """
        config = self.load_config()
        return config.get('server', {})
    
    def get_ascend_config(self) -> Dict[str, Any]:
        """
        获取华为昇腾配置
        
        Returns:
            Dict[str, Any]: 华为昇腾配置字典
        """
        config = self.load_config()
        return config.get('ascend', {})
    
    def get_rockchip_config(self) -> Dict[str, Any]:
        """
        获取瑞芯微配置
        
        Returns:
            Dict[str, Any]: 瑞芯微配置字典
        """
        config = self.load_config()
        return config.get('rockchip', {})
    
    def get_cambricon_config(self) -> Dict[str, Any]:
        """
        获取寒武纪配置
        
        Returns:
            Dict[str, Any]: 寒武纪配置字典
        """
        config = self.load_config()
        return config.get('cambricon', {})
    
    def get_conversion_config(self) -> Dict[str, Any]:
        """
        获取模型转换配置
        
        Returns:
            Dict[str, Any]: 模型转换配置字典
        """
        config = self.load_config()
        return config.get('conversion', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        获取日志配置
        
        Returns:
            Dict[str, Any]: 日志配置字典
        """
        config = self.load_config()
        return config.get('logging', {})
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        获取指定路径的配置值
        
        Args:
            key_path: 配置键路径，使用点分隔，例如 "minio.endpoint"
            default: 默认值
        
        Returns:
            Any: 配置值或默认值
        """
        config = self.load_config()
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def reload_config(self) -> None:
        """
        重新加载所有配置
        """
        self._config_cache.clear()
        self.load_config()

# 创建全局配置加载器实例
config_loader = ConfigLoader()