#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
华为910b平台模型转换任务测试

此测试文件专门测试 rest_predict_910b.py 提供的模型转换功能，包括：
1. 模型转换任务测试
2. 回调功能测试

测试流程：
1. 启动回调服务器
2. 上传测试文件到MinIO
3. 发送模型转换任务请求
4. 验证回调结果
5. 清理测试资源
"""

import sys
import os
import logging
import requests
import time
import threading
import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取test目录的父目录，即model-convert目录
model_convert_path = os.path.dirname(os.path.dirname(current_file_path))
# 将model-convert目录添加到sys.path的最前面
if model_convert_path not in sys.path:
    sys.path.insert(0, model_convert_path)

# 导入配置加载器
from config.config_loader import config_loader
# 导入 MinIO 工具
from tools.handle_file_minio import minio_handler, init_minio_handler
from rest_simple.utils import *

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CallbackHandler(BaseHTTPRequestHandler):
    """回调服务器处理类"""
    
    # 存储接收到的回调数据
    callbacks = {}
    
    def do_POST(self):
        """处理POST请求"""
        if self.path == '/api/predict/callback':
            # 读取请求体
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # 解析JSON数据
            try:
                callback_data = json.loads(post_data.decode('utf-8'))
                task_id = callback_data.get('task_id')
                
                # 存储回调数据
                CallbackHandler.callbacks[task_id] = callback_data
                
                logger.info(f"收到回调数据 - 任务ID: {task_id}")
                logger.info(f"回调内容: {callback_data}")
                
                # 发送响应
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                response = {
                    "code": 200,
                    "message": f"infer_task {task_id} received"
                }
                self.wfile.write(json.dumps(response).encode('utf-8'))
            except Exception as e:
                logger.error(f"处理回调数据失败: {str(e)}")
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def start_callback_server(port=38000):
    """启动回调服务器"""
    try:
        server = HTTPServer(('localhost', port), CallbackHandler)
        logger.info(f"回调服务器启动在端口 {port}")
        
        # 在单独的线程中运行服务器
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        
        return server, server_thread
    except Exception as e:
        logger.error(f"启动回调服务器失败: {str(e)}")
        return None, None


def stop_callback_server(server):
    """停止回调服务器"""
    if server:
        try:
            server.shutdown()
            server.server_close()
            logger.info("回调服务器已停止")
        except Exception as e:
            logger.error(f"停止回调服务器失败: {str(e)}")


def upload_test_files(minio_handler_instance, bucket_name, test_files):
    """
    上传测试文件到MinIO
    
    Args:
        minio_handler_instance: MinIO处理器实例
        bucket_name: 存储桶名称
        test_files: 测试文件列表 [(local_path, minio_object_name), ...]
        
    Returns:
        bool: 上传是否成功
    """
    try:
        # 创建存储桶
        if hasattr(minio_handler_instance, 'create_bucket'):
            try:
                minio_handler_instance.create_bucket(bucket_name)
            except:
                pass  # 存储桶可能已存在
        
        # 上传文件
        for local_path, minio_object_name in test_files:
            if os.path.exists(local_path):
                logger.info(f"上传测试文件: {local_path} -> {bucket_name}/{minio_object_name}")
                if hasattr(minio_handler_instance, 'upload_file'):
                    success = minio_handler_instance.upload_file(
                        bucket_name, 
                        minio_object_name, 
                        local_path
                    )
                    if not success:
                        logger.error(f"文件上传失败: {local_path}")
                        return False
                else:
                    logger.error("MinIO处理器缺少upload_file方法")
                    return False
            else:
                logger.error(f"本地文件不存在: {local_path}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"上传测试文件失败: {str(e)}")
        return False


def cleanup_minio_files(minio_handler_instance, bucket_name, object_names):
    """
    清理MinIO中的测试文件
    
    Args:
        minio_handler_instance: MinIO处理器实例
        object_names: 对象名称列表
    """
    # 暂时不清楚
    return
    # try:
    #     if hasattr(minio_handler_instance, 'client') and minio_handler_instance.client:
    #         for object_name in object_names:
    #             try:
    #                 minio_handler_instance.client.remove_object(bucket_name, object_name)
    #                 logger.info(f"已清理 MinIO 文件: {bucket_name}/{object_name}")
    #             except Exception as e:
    #                 logger.warning(f"清理 MinIO 文件失败: {bucket_name}/{object_name}, 错误: {str(e)}")
    # except Exception as e:
    #     logger.warning(f"清理 MinIO 文件时出错: {str(e)}")


def send_conversion_task(api_base_url, task_data):
    """
    发送模型转换任务
    
    Args:
        api_base_url: API基础URL
        task_data: 任务数据
        
    Returns:
        str: 任务ID，如果失败返回None
    """
    try:
        url = f"{api_base_url}/api/v1/predict"
        logger.info(f"发送模型转换任务到: {url}")
        logger.info(f"任务数据: {task_data}")
        
        response = requests.post(url, json=task_data, timeout=30)
        
        logger.info(f"响应状态码: {response.status_code}")
        logger.info(f"响应内容: {response.text}")
        
        if response.status_code == 200:
            response_data = response.json()
            task_id = response_data.get('data', {}).get('task_id')
            return task_id
        else:
            logger.error(f"发送任务失败，状态码: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"发送模型转换任务失败: {str(e)}")
        return None


def wait_for_callback(task_id, timeout=600):
    """
    等待回调结果
    
    Args:
        task_id: 任务ID
        timeout: 超时时间（秒）
        
    Returns:
        dict: 回调数据，如果超时返回None
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if task_id in CallbackHandler.callbacks:
            return CallbackHandler.callbacks[task_id]
        time.sleep(1)
    
    logger.warning(f"等待回调超时，任务ID: {task_id}")
    return None


def test_conversion_task(api_base_url, minio_handler_instance):
    """
    测试模型转换任务
    
    Args:
        api_base_url: API基础URL
        minio_handler_instance: MinIO处理器实例
    """
    logger.info("=" * 60)
    logger.info("开始测试模型转换任务")
    logger.info("=" * 60)

    onnx_file_path = str(test_dir / "yolo11n.onnx")
    
    # 检查文件是否存在
    if not os.path.exists(onnx_file_path):
        logger.error(f"测试ONNX文件不存在: {onnx_file_path}")
        return False
    
    # 上传ONNX文件到MinIO
    onnx_object_name = f"test_{int(time.time())}_yolo11n.onnx"
    test_files = [(onnx_file_path, onnx_object_name)]
    
    if not upload_test_files(minio_handler_instance, BUCKET_ONNX, test_files):
        logger.error("上传测试文件失败")
        return False
    
    # 准备任务数据
    task_data = {
        "task_id": int(time.time()),
        "model_id": 1001,
        "model_type": "YOLO",
        "source_file": "",  # 转换任务不需要源文件
        "model_file": f"{onnx_object_name}",
        "onnx_file": f"{onnx_object_name}",
        # 不提供engine_file，触发转换任务
        "platform": "Huawei",
    }
    
    # 发送转换任务
    task_id = send_conversion_task(api_base_url, task_data)
    if not task_id:
        logger.error("发送模型转换任务失败")
        return False
    
    logger.info(f"模型转换任务已发送，任务ID: {task_id}")
    
    # 等待回调结果
    callback_data = wait_for_callback(str(task_data["task_id"]))
    if not callback_data:
        logger.error("等待模型转换任务回调超时")
        return False
    
    # 验证回调数据
    result = callback_data.get("result", "")
    engine_file = callback_data.get("engine_file", "")
    
    if "转换成功" in result and engine_file:
        logger.info("模型转换任务测试成功")
        logger.info(f"回调结果: {callback_data}")
        return True
    else:
        logger.error("模型转换任务测试失败")
        logger.error(f"回调结果: {callback_data}")
        return False


def main():
    """主函数"""
    logger.info("开始执行华为910b平台模型转换任务测试")
    
    # 启动回调服务器
    callback_server, server_thread = start_callback_server(task_server_port)
    if not callback_server:
        logger.error("无法启动回调服务器")
        return False
    
    try:
        
        logger.info(f"API基础URL: {api_base_url}")
        
        # 初始化 MinIO 处理器
        minio_handler_instance = init_minio_handler()
        if not minio_handler_instance:
            logger.error("MinIO处理器初始化失败")
            return False
        
        # 测试模型转换任务
        conversion_success = test_conversion_task(api_base_url, minio_handler_instance)
        
        # 输出测试结果
        logger.info("=" * 60)
        logger.info("测试结果汇总")
        logger.info("=" * 60)
        logger.info(f"模型转换任务: {'通过' if conversion_success else '失败'}")
        
        if conversion_success:
            logger.info("模型转换任务测试通过!")
            return True
        else:
            logger.error("模型转换任务测试失败!")
            return False
            
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        return False
    finally:
        # 停止回调服务器
        stop_callback_server(callback_server)


if __name__ == "__main__":
    # task server
    task_server_port = 38000
    # 910b 接口地址
    api_base_url = "http://127.0.0.1:39000"
    # 测试文件目录
    test_dir = Path(model_convert_path) / "model_demo"

    success = main()
    sys.exit(0 if success else 1)