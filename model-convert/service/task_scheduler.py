#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务调度器模块

此模块负责从数据库中获取待执行的任务，并调度任务执行器执行它们。
"""
import time
import logging
import threading
import signal
import sys
from typing import Optional

# 导入任务管理器和执行器
from .task_manager import get_task_manager, TaskManager
from .task_executor import get_task_executor, TaskExecutor

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskScheduler:
    """
    任务调度器，负责任务的调度和执行
    """
    
    def __init__(self, check_interval: int = 10, max_concurrent_tasks: int = 1):
        """
        初始化任务调度器
        
        Args:
            check_interval: 检查新任务的间隔时间（秒）
            max_concurrent_tasks: 最大并发任务数
        """
        self.check_interval = check_interval
        self.max_concurrent_tasks = max_concurrent_tasks
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.running_tasks = 0
        
        # 获取任务管理器和执行器实例
        self.task_manager = get_task_manager()
        self.task_executor = get_task_executor()
        
        # 设置信号处理
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """
        设置信号处理程序，用于优雅停止
        """
        def signal_handler(sig, frame):
            logger.info(f"接收到信号 {sig}，正在停止调度器...")
            self.stop()
            sys.exit(0)
        
        # 捕获 SIGINT (Ctrl+C) 和 SIGTERM (kill)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start(self):
        """
        启动任务调度器
        """
        with self.lock:
            if self.running:
                logger.warning("调度器已经在运行")
                return
            
            self.running = True
            self.thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.thread.start()
            
            logger.info(f"任务调度器已启动，检查间隔: {self.check_interval}秒")
    
    def stop(self):
        """
        停止任务调度器
        """
        with self.lock:
            if not self.running:
                logger.warning("调度器没有在运行")
                return
            
            self.running = False
            logger.info("正在停止任务调度器...")
        
        # 等待调度线程结束
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=30)
            
        logger.info("任务调度器已停止")
    
    def _scheduler_loop(self):
        """
        调度器主循环
        """
        while self.running:
            try:
                self._process_pending_tasks()
            except Exception as e:
                logger.error(f"调度器循环异常: {str(e)}")
            
            # 等待下一次检查
            wait_time = self.check_interval
            while self.running and wait_time > 0:
                time.sleep(1)
                wait_time -= 1
    
    def _process_pending_tasks(self):
        """
        处理待执行的任务
        """
        # 检查当前运行中的任务数
        if self.running_tasks >= self.max_concurrent_tasks:
            logger.info(f"已达到最大并发任务数: {self.max_concurrent_tasks}，跳过任务检查")
            return
        
        try:
            # 获取待执行任务
            pending_tasks = self.task_manager.get_pending_tasks()
            
            if not pending_tasks:
                logger.info("没有待执行的任务")
                return
            
            logger.info(f"发现 {len(pending_tasks)} 个待执行任务")
            
            # 逐个处理任务
            for task in pending_tasks:
                # 再次检查并发任务数
                if self.running_tasks >= self.max_concurrent_tasks:
                    logger.info(f"已达到最大并发任务数: {self.max_concurrent_tasks}")
                    break
                
                # 启动任务执行线程
                task_thread = threading.Thread(
                    target=self._execute_task_wrapper, 
                    args=(task,)
                )
                task_thread.daemon = True
                
                # 增加运行中任务计数
                with self.lock:
                    self.running_tasks += 1
                
                # 启动任务线程
                task_thread.start()
                
                # 为了避免短时间内启动太多任务，这里可以添加一个小延迟
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"处理待执行任务时发生异常: {str(e)}")
    
    def _execute_task_wrapper(self, task: dict):
        """
        任务执行包装器，处理任务执行的生命周期
        
        Args:
            task: 任务信息字典
        """
        task_id = task.get('id')
        if not task_id:
            logger.error("任务缺少ID，跳过执行")
            return
        
        try:
            # 标记任务开始执行
            if self.task_manager.start_task(task_id):
                logger.info(f"任务开始执行: {task_id}")
                
                # 执行任务
                result = self.task_executor.execute_task(task)
                
                # 完成任务
                self.task_manager.complete_task(
                    task_id=task_id,
                    error_message=result.get('error_message'),
                    log_path=result.get('log_path')
                )
            else:
                logger.warning(f"无法开始任务: {task_id}")
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"任务执行线程异常: {task_id}, 错误: {error_msg}")
            
            # 确保任务被标记为失败
            try:
                self.task_manager.complete_task(task_id, error_message=error_msg)
            except Exception as inner_e:
                logger.error(f"更新任务状态失败: {task_id}, 错误: {str(inner_e)}")
        finally:
            # 减少运行中任务计数
            with self.lock:
                self.running_tasks -= 1
            
            logger.info(f"任务执行线程结束: {task_id}")
    
    def execute_task_immediately(self, task_id: str) -> bool:
        """
        立即执行指定的任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            是否成功启动执行
        """
        try:
            # 获取任务信息
            task = self.task_manager.get_task(task_id)
            if not task:
                logger.warning(f"任务不存在: {task_id}")
                return False
            
            if task.get('status') != 'pending':
                logger.warning(f"任务状态不允许立即执行: {task_id}, 当前状态: {task.get('status')}")
                return False
            
            # 直接在当前线程执行任务
            if self.task_manager.start_task(task_id):
                logger.info(f"立即执行任务: {task_id}")
                
                # 执行任务
                result = self.task_executor.execute_task(task)
                
                # 完成任务
                self.task_manager.complete_task(
                    task_id=task_id,
                    error_message=result.get('error_message'),
                    log_path=result.get('log_path')
                )
                
                return result.get('success', False)
            else:
                return False
                
        except Exception as e:
            logger.error(f"立即执行任务失败: {task_id}, 错误: {str(e)}")
            
            # 确保任务被标记为失败
            try:
                self.task_manager.complete_task(task_id, error_message=str(e))
            except Exception:
                pass
            
            return False

# 创建全局任务调度器实例
task_scheduler = None

def get_task_scheduler() -> TaskScheduler:
    """
    获取任务调度器实例
    
    Returns:
        TaskScheduler实例
    """
    global task_scheduler
    if task_scheduler is None:
        task_scheduler = TaskScheduler()
    return task_scheduler

if __name__ == "__main__":
    # 测试任务调度器
    scheduler = TaskScheduler(check_interval=5)
    
    try:
        print("启动任务调度器...")
        scheduler.start()
        
        # 保持主程序运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在停止...")
    finally:
        scheduler.stop()
        print("任务调度器已停止")