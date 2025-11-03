#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据库操作模块

此模块提供了SQLite数据库的初始化和基本操作功能，用于任务信息的持久化存储。
"""
import sqlite3
import os
import logging
from typing import Optional, Dict, Any, List

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    数据库管理器，负责数据库的初始化和任务相关的CRUD操作
    """
    
    def __init__(self, db_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_manager.db')):
        """
        初始化数据库管理器
        
        Args:
            db_path: SQLite数据库文件路径
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._init_database()
    
    def _init_database(self):
        """
        初始化数据库，创建必要的表
        """
        try:
            # 确保数据库目录存在
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            # 连接数据库
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # 创建任务表
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    parameters TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    log_path TEXT
                )
            ''')
            
            self.conn.commit()
            logger.info(f"数据库初始化成功: {self.db_path}")
        except Exception as e:
            logger.error(f"数据库初始化失败: {str(e)}")
            if self.conn:
                self.conn.rollback()
            raise
    
    def close(self):
        """
        关闭数据库连接
        """
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            logger.info("数据库连接已关闭")
    
    def _ensure_connection(self):
        """
        确保数据库连接处于活动状态
        """
        if self.conn is None or self.cursor is None:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            logger.info("数据库连接已重新建立")
    
    def create_task(self, task_data: Dict[str, Any]) -> str:
        """
        创建新任务
        
        Args:
            task_data: 任务数据字典
        
        Returns:
            task_id: 创建的任务ID
        """
        try:
            self._ensure_connection()
            task_id = task_data.pop('id') if 'id' in task_data else None
            if task_id is None:
                raise ValueError("任务数据必须包含id字段")
            
            # 准备SQL插入语句
            keys = ['id'] + list(task_data.keys())
            placeholders = ['?' for _ in keys]
            values = [task_id] + list(task_data.values())
            
            sql = f"INSERT INTO tasks ({', '.join(keys)}) VALUES ({', '.join(placeholders)})"
            self.cursor.execute(sql, values)
            self.conn.commit()
            
            logger.info(f"任务创建成功: {task_id}")
            return task_id
        except Exception as e:
            logger.error(f"任务创建失败: {str(e)}")
            if self.conn:
                self.conn.rollback()
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
            self._ensure_connection()
            self.cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = self.cursor.fetchone()
            
            if row:
                # 获取列名
                columns = [desc[0] for desc in self.cursor.description]
                return dict(zip(columns, row))
            return None
        except Exception as e:
            logger.error(f"获取任务失败: {str(e)}")
            raise
    
    def get_tasks_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        根据状态获取任务列表
        
        Args:
            status: 任务状态
        
        Returns:
            任务列表
        """
        try:
            self._ensure_connection()
            self.cursor.execute("SELECT * FROM tasks WHERE status = ? ORDER BY created_at ASC", (status,))
            rows = self.cursor.fetchall()
            
            # 获取列名
            columns = [desc[0] for desc in self.cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"获取任务列表失败: {str(e)}")
            raise
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        获取所有任务
        
        Returns:
            任务列表
        """
        try:
            self._ensure_connection()
            self.cursor.execute("SELECT * FROM tasks ORDER BY created_at DESC")
            rows = self.cursor.fetchall()
            
            # 获取列名
            columns = [desc[0] for desc in self.cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"获取所有任务失败: {str(e)}")
            raise
    
    def get_tasks_paginated(self, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """
        分页获取任务列表
        
        Args:
            page: 页码，从1开始
            page_size: 每页任务数量
        
        Returns:
            包含任务列表、总数、页码等信息的字典
        """
        try:
            self._ensure_connection()
            
            # 计算偏移量
            offset = (page - 1) * page_size
            
            # 获取总数
            self.cursor.execute("SELECT COUNT(*) FROM tasks")
            total = self.cursor.fetchone()[0]
            
            # 获取分页数据
            self.cursor.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?", 
                (page_size, offset)
            )
            rows = self.cursor.fetchall()
            
            # 获取列名
            columns = [desc[0] for desc in self.cursor.description]
            tasks = [dict(zip(columns, row)) for row in rows]
            
            # 计算总页数
            total_pages = (total + page_size - 1) // page_size if total > 0 else 0
            
            return {
                'tasks': tasks,
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': total_pages
            }
        except Exception as e:
            logger.error(f"分页获取任务列表失败: {str(e)}")
            raise
    
    def delete_tasks_batch(self, task_ids: List[str]) -> Dict[str, bool]:
        """
        批量删除任务
        
        Args:
            task_ids: 任务ID列表
        
        Returns:
            每个任务ID对应的删除结果字典
        """
        try:
            self._ensure_connection()
            results = {}
            
            # 逐个删除任务
            for task_id in task_ids:
                try:
                    self.cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                    results[task_id] = self.cursor.rowcount > 0
                    if results[task_id]:
                        logger.info(f"任务删除成功: {task_id}")
                    else:
                        logger.warning(f"任务不存在: {task_id}")
                except Exception as e:
                    logger.error(f"删除任务失败 {task_id}: {str(e)}")
                    results[task_id] = False
            
            # 提交所有删除操作
            self.conn.commit()
            
            return results
        except Exception as e:
            logger.error(f"批量删除任务失败: {str(e)}")
            if self.conn:
                self.conn.rollback()
            raise
    
    def update_task(self, task_id: str, updates: Dict[str, Any]):
        """
        更新任务信息
        
        Args:
            task_id: 任务ID
            updates: 更新的字段和值
        """
        try:
            self._ensure_connection()
            # 准备SQL更新语句
            set_clause = ', '.join([f"{key} = ?" for key in updates.keys()])
            values = list(updates.values()) + [task_id]
            
            sql = f"UPDATE tasks SET {set_clause} WHERE id = ?"
            self.cursor.execute(sql, values)
            self.conn.commit()
            
            logger.info(f"任务更新成功: {task_id}, 更新字段: {list(updates.keys())}")
        except Exception as e:
            logger.error(f"任务更新失败: {str(e)}")
            if self.conn:
                self.conn.rollback()
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
            self._ensure_connection()
            self.cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            affected_rows = self.cursor.rowcount
            self.conn.commit()
            
            if affected_rows > 0:
                logger.info(f"任务删除成功: {task_id}")
                return True
            else:
                logger.warning(f"任务不存在: {task_id}")
                return False
        except Exception as e:
            logger.error(f"任务删除失败: {str(e)}")
            if self.conn:
                self.conn.rollback()
            raise

# 创建全局数据库管理器实例
db_manager = None

def get_db_manager() -> DatabaseManager:
    """
    获取数据库管理器实例
    
    Returns:
        DatabaseManager实例
    """
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager

if __name__ == "__main__":
    # 测试数据库操作
    db = DatabaseManager()
    
    # 测试创建任务
    task_data = {
        'id': 'test_task_001',
        'task_type': 'onnx_to_om',
        'platform': 'huawei',
        'status': 'pending',
        'input_path': '/path/to/input.onnx',
        'output_path': '/path/to/output.om',
        'parameters': '{"input_shape": "1,3,640,640"}'
    }
    
    try:
        task_id = db.create_task(task_data)
        print(f"创建任务成功: {task_id}")
        
        # 测试获取任务
        task = db.get_task(task_id)
        print(f"获取任务: {task}")
        
        # 测试更新任务
        db.update_task(task_id, {'status': 'running', 'started_at': '2024-01-01 12:00:00'})
        
        # 测试获取所有任务
        tasks = db.get_all_tasks()
        print(f"所有任务数量: {len(tasks)}")
        
    finally:
        # 清理测试数据
        db.delete_task(task_id)
        db.close()