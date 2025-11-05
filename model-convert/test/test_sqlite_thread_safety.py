#!/usr/bin/env python3
"""
测试SQLite线程安全问题

此脚本用于测试多个线程同时访问SQLite数据库时的线程安全问题。
"""
import logging
import os
import sys
import threading
import time
import uuid
import json

# 将根目录添加到Python路径
root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
logging.warning(f"根目录: {root_path}")
sys.path.insert(0, root_path)

from service.task_manager import get_task_manager

# 测试任务数量
NUM_TEST_TASKS = 10

# 测试函数：创建任务
def create_task(task_manager, task_id):
    """创建一个测试任务"""
    try:
        task_data = {
            'task_type': 'yolo_to_onnx',
            'platform': 'onnx',
            'input_path': f'/test/input_{task_id}.pt',
            'output_path': f'/test/output_{task_id}.onnx',
            'parameters': {
                'imgsz': 640,
                'simplify': True,
                'opset_version': 12
            }
        }
        
        created_id = task_manager.create_task(**task_data, task_id=task_id)
        print(f"线程 {threading.current_thread().name} 创建任务成功: {created_id}")
        return True
    except Exception as e:
        print(f"线程 {threading.current_thread().name} 创建任务失败: {e}")
        return False

# 测试函数：获取任务
def get_task(task_manager, task_id):
    """获取一个测试任务"""
    try:
        task = task_manager.get_task(task_id)
        if task:
            print(f"线程 {threading.current_thread().name} 获取任务成功: {task_id}")
        else:
            print(f"线程 {threading.current_thread().name} 获取任务失败: 任务不存在 {task_id}")
        return True
    except Exception as e:
        print(f"线程 {threading.current_thread().name} 获取任务失败: {e}")
        return False

# 测试函数：更新任务
def update_task(task_manager, task_id):
    """更新一个测试任务"""
    try:
        success = task_manager.start_task(task_id)
        if success:
            print(f"线程 {threading.current_thread().name} 更新任务成功: {task_id}")
        else:
            print(f"线程 {threading.current_thread().name} 更新任务失败: {task_id}")
        return success
    except Exception as e:
        print(f"线程 {threading.current_thread().name} 更新任务失败: {e}")
        return False

# 测试函数：删除任务
def delete_task(task_manager, task_id):
    """删除一个测试任务"""
    try:
        success = task_manager.delete_task(task_id)
        if success:
            print(f"线程 {threading.current_thread().name} 删除任务成功: {task_id}")
        else:
            print(f"线程 {threading.current_thread().name} 删除任务失败: {task_id}")
        return success
    except Exception as e:
        print(f"线程 {threading.current_thread().name} 删除任务失败: {e}")
        return False

# 测试函数：完整任务流程
def test_task_flow(task_manager, task_id):
    """测试完整的任务流程"""
    try:
        # 创建任务
        create_task(task_manager, task_id)
        
        # 获取任务
        get_task(task_manager, task_id)
        
        # 更新任务
        update_task(task_manager, task_id)
        
        # 再次获取任务
        get_task(task_manager, task_id)
        
        # 删除任务
        delete_task(task_manager, task_id)
        
        # 再次获取任务（应该不存在）
        get_task(task_manager, task_id)
        
        return True
    except Exception as e:
        print(f"线程 {threading.current_thread().name} 任务流程失败: {e}")
        return False

# 主测试函数
def main():
    """主测试函数"""
    print("开始测试SQLite线程安全问题...")
    
    # 获取任务管理器实例
    task_manager = get_task_manager()
    
    # 创建测试任务ID列表
    task_ids = [f"test_task_{i}_{uuid.uuid4().hex[:8]}" for i in range(NUM_TEST_TASKS)]
    
    # 创建线程列表
    threads = []
    
    # 创建多个线程同时执行任务流程
    for i in range(NUM_TEST_TASKS):
        thread = threading.Thread(target=test_task_flow, args=(task_manager, task_ids[i]), name=f"Thread-{i+1}")
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    print("\n所有测试线程已完成")
    
    # 清理：确保所有测试任务都被删除
    print("\n开始清理测试任务...")
    for task_id in task_ids:
        try:
            task_manager.delete_task(task_id)
        except:
            pass
    print("测试任务清理完成")
    
    print("\nSQLite线程安全测试完成")

if __name__ == "__main__":
    main()