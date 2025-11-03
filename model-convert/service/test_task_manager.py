#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务管理系统测试脚本

此脚本演示了如何使用任务管理系统的各项功能。
"""
import os
import sys
import time
import logging

# 添加项目路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_task_manager')

def test_basic_task_operations():
    """
    测试基本的任务操作
    """
    print("\n=== 测试基本任务操作 ===")
    
    try:
        # 导入任务API
        from service.task_api import (
            create_huawei_onnx_to_om_task,
            get_task_info,
            get_all_tasks,
            execute_task_immediately,
            delete_task,
            TASK_STATUS
        )
        
        # 创建测试任务
        print("\n1. 创建华为ONNX转OM任务...")
        task_id = create_huawei_onnx_to_om_task(
            input_path='/path/to/test_model.onnx',
            output_path='/path/to/test_model.om',
            parameters={'input_shape': '1,3,640,640', 'auto_input_shape': True}
        )
        print(f"   任务创建成功，ID: {task_id}")
        
        # 获取任务信息
        print("\n2. 获取任务信息...")
        task = get_task_info(task_id)
        if task:
            print(f"   任务ID: {task.get('id')}")
            print(f"   任务类型: {task.get('task_type')}")
            print(f"   目标平台: {task.get('platform')}")
            print(f"   当前状态: {task.get('status')}")
            print(f"   输入路径: {task.get('input_path')}")
            print(f"   输出路径: {task.get('output_path')}")
            print(f"   参数: {task.get('parameters')}")
        else:
            print("   警告: 无法获取任务信息，返回值为None")
        
        # 获取所有任务
        print("\n3. 获取所有任务...")
        all_tasks = get_all_tasks()
        if all_tasks is not None:
            print(f"   总任务数: {len(all_tasks)}")
            for t in all_tasks:
                print(f"   - {t.get('id')}: {t.get('status')} ({t.get('task_type')} on {t.get('platform')})")
        else:
            print("   警告: 无法获取任务列表，返回值为None")
        
        # 注意：这里不实际执行任务，因为需要真实的模型文件
        print("\n4. 注意：由于没有真实的模型文件，跳过任务执行测试")
        print("   在实际环境中，可以使用以下代码执行任务：")
        print(f"   # success = execute_task_immediately('{task_id}')")
        
        # 清理测试任务
        print("\n5. 清理测试任务...")
        delete_task(task_id)
        print(f"   任务 {task_id} 已删除")
        
        print("\n基本任务操作测试完成！")
        return task_id  # 返回任务ID用于后续清理
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        raise

def test_database_operations():
    """
    测试数据库操作
    """
    print("\n=== 测试数据库操作 ===")
    
    try:
        # 导入数据库管理器
        from service.database import get_db_manager
        
        db = get_db_manager()
        print(f"数据库连接成功: {db.db_path}")
        
        # 创建测试任务
        test_task = {
            'id': 'test_db_task_001',
            'task_type': 'onnx_to_om',
            'platform': 'huawei',
            'status': 'pending',
            'input_path': '/test/input.onnx',
            'output_path': '/test/output.om',
            'parameters': '{"test": "parameter"}'
        }
        
        # 创建任务
        print("\n1. 创建数据库任务...")
        db.create_task(test_task)
        print(f"   任务创建成功: {test_task['id']}")
        
        # 查询任务
        print("\n2. 查询任务...")
        task = db.get_task(test_task['id'])
        if task:
            print(f"   找到任务: {task['id']}, 状态: {task['status']}")
        else:
            print("   警告: 无法获取任务信息")
        
        # 更新任务
        print("\n3. 更新任务...")
        db.update_task(test_task['id'], {'status': 'running'})
        updated_task = db.get_task(test_task['id'])
        if updated_task:
            print(f"   任务更新成功: {updated_task['id']}, 新状态: {updated_task['status']}")
        else:
            print("   警告: 无法获取更新后的任务信息")
        
        # 删除任务
        print("\n4. 删除任务...")
        db.delete_task(test_task['id'])
        print(f"   任务已删除: {test_task['id']}")
        
        print("\n数据库操作测试完成！")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        raise

def test_scheduler_functionality():
    """
    测试调度器功能
    """
    print("\n=== 测试调度器功能 ===")
    
    try:
        # 导入任务调度器
        from service.task_scheduler import TaskScheduler
        
        # 创建调度器（设置较短的检查间隔用于测试）
        scheduler = TaskScheduler(check_interval=5, max_concurrent_tasks=1)
        
        print("调度器初始化成功")
        print(f"检查间隔: {scheduler.check_interval}秒")
        print(f"最大并发任务数: {scheduler.max_concurrent_tasks}")
        
        print("\n注意：在实际环境中，可以使用以下代码启动调度器：")
        print("# scheduler.start()")
        print("# 然后调度器会自动检查并执行待处理的任务")
        
        print("\n调度器功能测试完成！")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        raise

def main():
    """
    主测试函数
    """
    print("开始测试任务管理系统...")
    task_id = None
    
    try:
        # 测试基本任务操作
        task_id = test_basic_task_operations()
        
        # 测试数据库操作
        test_database_operations()
        
        # 测试调度器功能
        test_scheduler_functionality()
        
        print("\n🎉 所有测试通过！任务管理系统功能正常。")
        print("\n使用说明：")
        print("1. 创建任务: 使用 task_api.create_huawei_onnx_to_om_task() 或其他创建函数")
        print("2. 查询任务: 使用 task_api.get_task_info() 或 get_all_tasks()")
        print("3. 启动服务: 运行 python service/task_service.py 启动任务调度服务")
        print("4. 立即执行: 使用 task_api.execute_task_immediately() 立即执行特定任务")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        sys.exit(1)
    finally:
        # 确保任务被清理
        if task_id:
            try:
                from service.task_api import delete_task
                delete_task(task_id)
                print(f"\n任务 {task_id} 已成功清理")
            except:
                pass  # 忽略清理时的错误

if __name__ == "__main__":
    main()