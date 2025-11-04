#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
REST API ONNX 转 OM 模型转换接口单元测试

此测试文件使用 pytest 框架测试 convert_onnx_to_om 功能，包括：
1. 测试ONNX文件转换任务创建
2. 测试任务查询
3. 测试任务删除
4. 按顺序执行：创建 -> 查询 -> 删除
"""

# 首先修改sys.path，确保能正确导入模块
import sys
import os

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取test目录的父目录，即model-convert目录
model_convert_path = os.path.dirname(os.path.dirname(current_file_path))
# 将model-convert目录添加到sys.path的最前面
if model_convert_path not in sys.path:
    sys.path.insert(0, model_convert_path)

# 调试信息
print(f"当前工作目录: {os.getcwd()}")
print(f"当前文件路径: {current_file_path}")
print(f"model-convert目录路径: {model_convert_path}")
print(f"sys.path: {sys.path}")

# 导入其他模块
import logging
import requests
import time
import pytest
from pathlib import Path

# 导入配置加载器
from config.config_loader import config_loader
# 导入 MinIO 工具
from tools.handle_file_minio import minio_handler, init_minio_handler


@pytest.fixture(scope="module")
def test_setup():
    """测试前的准备工作和测试后的清理工作"""
    # 从配置文件获取服务器配置
    try:
        server_config = config_loader.get_server_config()
        host = server_config.get('host', '0.0.0.0')
        port = server_config.get('port', 8000)
        
        # 判断是本地还是远程
        if host == '0.0.0.0' or host == '127.0.0.1' or host == 'localhost':
            base_url = f"http://127.0.0.1:{port}"
        else:
            base_url = f"http://{host}:{port}"
    except Exception as e:
        logging.warning(f"配置加载失败，使用默认URL: {str(e)}")
        base_url = "http://127.0.0.1:8000"
    
    logging.info(f"API基础URL: {base_url}")
    
    # 初始化 MinIO 处理器
    minio_handler_instance = init_minio_handler()
    use_minio = minio_handler_instance is not None
    
    # 创建临时目录用于测试文件
    current_dir = Path(__file__).parent
    test_dir = current_dir.parent / "model_demo"
    input_onnx_path = str(test_dir / "out" / "yolo11n.onnx")
    output_om_path = str(os.path.join(test_dir, "test_model.om"))
    
    # MinIO 配置
    minio_bucket = "test-bucket"
    minio_onnx_object = f"test_model_{int(time.time())}.onnx"
    minio_om_object = f"test_model_{int(time.time())}.om"
    
    # 存储任务ID，用于清理
    task_ids_to_cleanup = []
    
    # 如果使用 MinIO，创建测试存储桶并上传文件
    if use_minio and minio_handler_instance:
        try:
            logging.info(f"创建 MinIO 存储桶: {minio_bucket}")
            if hasattr(minio_handler_instance, 'create_bucket'):
                try:
                    minio_handler_instance.create_bucket(minio_bucket)
                except:
                    pass  # 存储桶可能已存在
                
            # 检查本地ONNX文件是否存在
            if os.path.exists(input_onnx_path):
                logging.info(f"上传测试文件: {minio_bucket}/{minio_onnx_object}")
                if hasattr(minio_handler_instance, 'upload_file'):
                    minio_handler_instance.upload_file(
                        minio_bucket, 
                        minio_onnx_object, 
                        input_onnx_path
                    )
                    logging.info("ONNX文件上传成功")
            else:
                logging.warning(f"本地ONNX文件不存在: {input_onnx_path}")
                use_minio = False
        except Exception as e:
            logging.error(f"MinIO初始化失败: {str(e)}")
            use_minio = False
    
    # 提供测试数据给测试函数
    yield {
        "base_url": base_url,
        "minio_handler": minio_handler_instance,
        "use_minio": use_minio,
        "input_onnx_path": input_onnx_path,
        "output_om_path": output_om_path,
        "minio_bucket": minio_bucket,
        "minio_onnx_object": minio_onnx_object,
        "minio_om_object": minio_om_object,
        "task_ids_to_cleanup": task_ids_to_cleanup
    }
    
    # 测试后的清理工作
    # 清理创建的任务
    if task_ids_to_cleanup:
        try:
            url = f"{base_url}/tasks/batch-delete"
            data = {"task_ids": task_ids_to_cleanup}
            response = requests.post(url, json=data)
            logging.info(f"清理任务 - 响应状态码: {response.status_code}")
            if response.status_code == 200:
                logging.info("测试任务清理成功")
        except Exception as e:
            logging.warning(f"清理任务失败: {str(e)}")
    
    # 如果使用 MinIO，清理测试对象
    if use_minio and minio_handler_instance:
        try:
            if hasattr(minio_handler_instance, 'client') and minio_handler_instance.client:
                # 清理输出文件
                try:
                    minio_handler_instance.client.remove_object(minio_bucket, minio_om_object)
                    logging.info(f"已清理 MinIO 输出: {minio_bucket}/{minio_om_object}")
                except:
                    pass
                
                # 清理输入文件
                try:
                    minio_handler_instance.client.remove_object(minio_bucket, minio_onnx_object)
                    logging.info(f"已清理 MinIO 输入: {minio_bucket}/{minio_onnx_object}")
                except:
                    pass
        except:
            pass


def test_complete_task_workflow(test_setup):
    """
    完整测试：创建任务 -> 查询任务 -> 删除任务
    """
    try:
        # 步骤1: 创建转换任务
        logging.info("=" * 60)
        logging.info("步骤1: 创建ONNX转OM任务")
        logging.info("=" * 60)
        
        url = f"{test_setup['base_url']}/convert/onnx-to-om"
        
        # 根据是否使用MinIO选择输入路径
        if test_setup['use_minio'] and os.path.exists(test_setup['input_onnx_path']):
            input_path = f"{test_setup['minio_bucket']}/{test_setup['minio_onnx_object']}"
            output_path = f"{test_setup['minio_bucket']}/{test_setup['minio_om_object']}"
            logging.info(f"使用MinIO路径: {input_path} -> {output_path}")
        else:
            # 使用本地路径（如果文件存在）
            if not os.path.exists(test_setup['input_onnx_path']):
                logging.warning("跳过测试：ONNX文件不存在")
                return
            input_path = test_setup['input_onnx_path']
            output_path = test_setup['output_om_path']
            logging.info(f"使用本地路径: {input_path} -> {output_path}")
        
        data = {
            "input_model_path": input_path,
            "output_model_path": output_path,
            "soc_version": "Ascend910B",
            "auto_input_shape": True
        }
        
        response = requests.post(url, json=data, timeout=10)
        
        # 检查响应状态码
        assert response.status_code in [200, 201], f"创建任务失败，状态码: {response.status_code}, 响应: {response.text}"
        
        logging.info(f"创建任务 - 响应状态码: {response.status_code}")
        
        # 解析响应获取任务ID
        response_data = response.json()
        assert response_data.get('success', False), f"创建任务返回失败: {response_data}"
        
        task_id = response_data['data']['task_id']
        logging.info(f"任务创建成功，任务ID: {task_id}")
        
        # 保存任务ID用于清理
        test_setup['task_ids_to_cleanup'].append(task_id)
        
        # 步骤2: 查询任务信息
        logging.info("=" * 60)
        logging.info("步骤2: 查询任务信息")
        logging.info("=" * 60)
        
        url = f"{test_setup['base_url']}/tasks/query"
        data = {"task_id": task_id}
        
        response = requests.post(url, json=data, timeout=10)
        
        assert response.status_code == 200, f"查询任务失败，状态码: {response.status_code}, 响应: {response.text}"
        
        logging.info(f"查询任务 - 响应状态码: {response.status_code}")
        
        response_data = response.json()
        assert response_data.get('success', False), f"查询任务返回失败: {response_data}"
        
        task_info = response_data['data']['task_info']
        logging.info(f"任务信息: ID={task_info['id']}, 状态={task_info['status']}")
        logging.info(f"输入路径: {task_info['input_path']}")
        logging.info(f"输出路径: {task_info['output_path']}")
        
        # 验证任务信息
        assert task_info['id'] == task_id
        assert task_info['task_type'] == 'onnx_to_om'
        assert task_info['platform'] == 'huawei'
        assert task_info['status'] in ['pending', 'running']
        
        # 步骤3: 获取任务状态
        logging.info("=" * 60)
        logging.info("步骤3: 获取任务状态")
        logging.info("=" * 60)
        
        url = f"{test_setup['base_url']}/tasks/{task_id}/status"
        
        response = requests.get(url, timeout=10)
        
        assert response.status_code == 200, f"获取状态失败，状态码: {response.status_code}, 响应: {response.text}"
        
        logging.info(f"获取状态 - 响应状态码: {response.status_code}")
        
        response_data = response.json()
        assert response_data.get('success', False), f"获取状态返回失败: {response_data}"
        
        status_data = response_data['data']
        logging.info(f"任务状态: {status_data['status']}")
        logging.info(f"是否完成: {status_data['is_completed']}")
        logging.info(f"是否失败: {status_data['is_failed']}")
        
        # 步骤4: 分页查询任务列表（验证任务存在于列表中）
        logging.info("=" * 60)
        logging.info("步骤4: 分页查询任务列表")
        logging.info("=" * 60)
        
        url = f"{test_setup['base_url']}/tasks/list"
        data = {"page": 1, "page_size": 10}
        
        response = requests.post(url, json=data, timeout=10)
        
        assert response.status_code == 200, f"分页查询失败，状态码: {response.status_code}, 响应: {response.text}"
        
        logging.info(f"分页查询 - 响应状态码: {response.status_code}")
        
        response_data = response.json()
        assert response_data.get('success', False), f"分页查询返回失败: {response_data}"
        
        list_data = response_data['data']
        logging.info(f"任务总数: {list_data['total']}")
        logging.info(f"当前页: {list_data['page']}")
        logging.info(f"每页数量: {list_data['page_size']}")
        logging.info(f"总页数: {list_data['total_pages']}")
        
        # 验证任务是否在列表中
        task_ids_in_list = [t['id'] for t in list_data['tasks']]
        assert task_id in task_ids_in_list, "任务未出现在任务列表中"
        logging.info(f"任务 {task_id} 已在任务列表中")
        
        # 步骤5: 删除任务
        logging.info("=" * 60)
        logging.info("步骤5: 删除任务")
        logging.info("=" * 60)
        
        url = f"{test_setup['base_url']}/tasks/batch-delete"
        data = {"task_ids": [task_id]}
        
        response = requests.post(url, json=data, timeout=10)
        
        assert response.status_code == 200, f"删除任务失败，状态码: {response.status_code}, 响应: {response.text}"
        
        logging.info(f"删除任务 - 响应状态码: {response.status_code}")
        
        response_data = response.json()
        assert response_data.get('success', False), f"删除任务返回失败: {response_data}"
        
        delete_data = response_data['data']
        logging.info(f"删除成功: {delete_data['success_count']}/{delete_data['total_count']}")
        
        assert delete_data['success_count'] == 1, "任务删除失败"
        
        # 从清理列表中移除（已删除）
        test_setup['task_ids_to_cleanup'].remove(task_id)
        
        logging.info("=" * 60)
        logging.info("所有测试步骤完成！")
        logging.info("=" * 60)
        
    except AssertionError:
        raise
    except Exception as e:
        logging.error(f"测试过程中发生异常: {str(e)}")
        pytest.fail(f"测试失败: {str(e)}")


# 设置日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
