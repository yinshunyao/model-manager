#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
REST API ONNX 转 OM 模型转换接口单元测试

此测试文件使用 unittest 框架测试 convert_onnx_to_om 功能，包括：
1. 测试ONNX文件转换任务创建
2. 测试任务查询
3. 测试任务删除
4. 按顺序执行：创建 -> 查询 -> 删除
"""
import logging
import unittest
import os
import sys
import requests
import time
from pathlib import Path

# 添加项目路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置加载器
from config.config_loader import config_loader
# 导入 MinIO 工具
from tools.handle_file_minio import minio_handler, init_minio_handler


class TestRestONNXToOM(unittest.TestCase):
    """ONNX 转 OM REST API 测试类"""

    def setUp(self):
        """测试前的准备工作"""
        # 从配置文件获取服务器配置
        try:
            server_config = config_loader.get_server_config()
            host = server_config.get('host', '0.0.0.0')
            port = server_config.get('port', 8000)
            
            # 判断是本地还是远程
            if host == '0.0.0.0' or host == '127.0.0.1' or host == 'localhost':
                self.base_url = f"http://127.0.0.1:{port}"
            else:
                self.base_url = f"http://{host}:{port}"
        except Exception as e:
            logging.warning(f"配置加载失败，使用默认URL: {str(e)}")
            self.base_url = "http://127.0.0.1:8000"
        
        logging.info(f"API基础URL: {self.base_url}")
        
        # 初始化 MinIO 处理器
        self.minio_handler = init_minio_handler()
        self.use_minio = self.minio_handler is not None
        
        # 创建临时目录用于测试文件
        self.current_dir = Path(__file__).parent
        self.test_dir = self.current_dir.parent / "model_demo"
        self.input_onnx_path = str(self.test_dir / "out" / "yolo11n.onnx")
        self.output_om_path = str(os.path.join(self.test_dir, "test_model.om"))
        
        # MinIO 配置
        self.minio_bucket = "test-bucket"
        self.minio_onnx_object = f"test_model_{int(time.time())}.onnx"
        self.minio_om_object = f"test_model_{int(time.time())}.om"
        
        # 存储任务ID，用于清理
        self.task_ids_to_cleanup = []
        
        # 如果使用 MinIO，创建测试存储桶并上传文件
        if self.use_minio and self.minio_handler:
            try:
                logging.info(f"创建 MinIO 存储桶: {self.minio_bucket}")
                if hasattr(self.minio_handler, 'create_bucket'):
                    try:
                        self.minio_handler.create_bucket(self.minio_bucket)
                    except:
                        pass  # 存储桶可能已存在
                
                # 检查本地ONNX文件是否存在
                if os.path.exists(self.input_onnx_path):
                    logging.info(f"上传测试文件: {self.minio_bucket}/{self.minio_onnx_object}")
                    if hasattr(self.minio_handler, 'upload_file'):
                        self.minio_handler.upload_file(
                            self.minio_bucket, 
                            self.minio_onnx_object, 
                            self.input_onnx_path
                        )
                        logging.info("ONNX文件上传成功")
                else:
                    logging.warning(f"本地ONNX文件不存在: {self.input_onnx_path}")
                    self.use_minio = False
            except Exception as e:
                logging.error(f"MinIO初始化失败: {str(e)}")
                self.use_minio = False

    def tearDown(self):
        """测试后的清理工作"""
        # 清理创建的任务
        if self.task_ids_to_cleanup:
            try:
                url = f"{self.base_url}/tasks/batch-delete"
                data = {"task_ids": self.task_ids_to_cleanup}
                response = requests.post(url, json=data)
                logging.info(f"清理任务 - 响应状态码: {response.status_code}")
                if response.status_code == 200:
                    logging.info("测试任务清理成功")
            except Exception as e:
                logging.warning(f"清理任务失败: {str(e)}")
        
        # 如果使用 MinIO，清理测试对象
        if self.use_minio and self.minio_handler:
            try:
                if hasattr(self.minio_handler, 'client') and self.minio_handler.client:
                    # 清理输出文件
                    try:
                        self.minio_handler.client.remove_object(self.minio_bucket, self.minio_om_object)
                        logging.info(f"已清理 MinIO 输出: {self.minio_bucket}/{self.minio_om_object}")
                    except:
                        pass
                    
                    # 清理输入文件
                    try:
                        self.minio_handler.client.remove_object(self.minio_bucket, self.minio_onnx_object)
                        logging.info(f"已清理 MinIO 输入: {self.minio_bucket}/{self.minio_onnx_object}")
                    except:
                        pass
            except:
                pass

    def test_complete_task_workflow(self):
        """
        完整测试：创建任务 -> 查询任务 -> 删除任务
        """
        try:
            # 步骤1: 创建转换任务
            logging.info("=" * 60)
            logging.info("步骤1: 创建ONNX转OM任务")
            logging.info("=" * 60)
            
            url = f"{self.base_url}/convert/onnx-to-om"
            
            # 根据是否使用MinIO选择输入路径
            if self.use_minio and os.path.exists(self.input_onnx_path):
                input_path = f"{self.minio_bucket}/{self.minio_onnx_object}"
                output_path = f"{self.minio_bucket}/{self.minio_om_object}"
                logging.info(f"使用MinIO路径: {input_path} -> {output_path}")
            else:
                # 使用本地路径（如果文件存在）
                if not os.path.exists(self.input_onnx_path):
                    logging.warning("跳过测试：ONNX文件不存在")
                    return
                input_path = self.input_onnx_path
                output_path = self.output_om_path
                logging.info(f"使用本地路径: {input_path} -> {output_path}")
            
            data = {
                "input_model_path": input_path,
                "output_model_path": output_path,
                "soc_version": "Ascend910B",
                "auto_input_shape": True
            }
            
            response = requests.post(url, json=data, timeout=10)
            
            # 检查响应状态码
            self.assertIn(response.status_code, [200, 201], 
                         f"创建任务失败，状态码: {response.status_code}, 响应: {response.text}")
            
            logging.info(f"创建任务 - 响应状态码: {response.status_code}")
            
            # 解析响应获取任务ID
            response_data = response.json()
            self.assertTrue(response_data.get('success', False), 
                          f"创建任务返回失败: {response_data}")
            
            task_id = response_data['data']['task_id']
            logging.info(f"任务创建成功，任务ID: {task_id}")
            
            # 保存任务ID用于清理
            self.task_ids_to_cleanup.append(task_id)
            
            # 步骤2: 查询任务信息
            logging.info("=" * 60)
            logging.info("步骤2: 查询任务信息")
            logging.info("=" * 60)
            
            url = f"{self.base_url}/tasks/query"
            data = {"task_id": task_id}
            
            response = requests.post(url, json=data, timeout=10)
            
            self.assertEqual(response.status_code, 200, 
                           f"查询任务失败，状态码: {response.status_code}, 响应: {response.text}")
            
            logging.info(f"查询任务 - 响应状态码: {response.status_code}")
            
            response_data = response.json()
            self.assertTrue(response_data.get('success', False), 
                          f"查询任务返回失败: {response_data}")
            
            task_info = response_data['data']['task_info']
            logging.info(f"任务信息: ID={task_info['id']}, 状态={task_info['status']}")
            logging.info(f"输入路径: {task_info['input_path']}")
            logging.info(f"输出路径: {task_info['output_path']}")
            
            # 验证任务信息
            self.assertEqual(task_info['id'], task_id)
            self.assertEqual(task_info['task_type'], 'onnx_to_om')
            self.assertEqual(task_info['platform'], 'huawei')
            self.assertIn(task_info['status'], ['pending', 'running'])
            
            # 步骤3: 获取任务状态
            logging.info("=" * 60)
            logging.info("步骤3: 获取任务状态")
            logging.info("=" * 60)
            
            url = f"{self.base_url}/tasks/{task_id}/status"
            
            response = requests.get(url, timeout=10)
            
            self.assertEqual(response.status_code, 200, 
                           f"获取状态失败，状态码: {response.status_code}, 响应: {response.text}")
            
            logging.info(f"获取状态 - 响应状态码: {response.status_code}")
            
            response_data = response.json()
            self.assertTrue(response_data.get('success', False), 
                          f"获取状态返回失败: {response_data}")
            
            status_data = response_data['data']
            logging.info(f"任务状态: {status_data['status']}")
            logging.info(f"是否完成: {status_data['is_completed']}")
            logging.info(f"是否失败: {status_data['is_failed']}")
            
            # 步骤4: 分页查询任务列表（验证任务存在于列表中）
            logging.info("=" * 60)
            logging.info("步骤4: 分页查询任务列表")
            logging.info("=" * 60)
            
            url = f"{self.base_url}/tasks/list"
            data = {"page": 1, "page_size": 10}
            
            response = requests.post(url, json=data, timeout=10)
            
            self.assertEqual(response.status_code, 200, 
                           f"分页查询失败，状态码: {response.status_code}, 响应: {response.text}")
            
            logging.info(f"分页查询 - 响应状态码: {response.status_code}")
            
            response_data = response.json()
            self.assertTrue(response_data.get('success', False), 
                          f"分页查询返回失败: {response_data}")
            
            list_data = response_data['data']
            logging.info(f"任务总数: {list_data['total']}")
            logging.info(f"当前页: {list_data['page']}")
            logging.info(f"每页数量: {list_data['page_size']}")
            logging.info(f"总页数: {list_data['total_pages']}")
            
            # 验证任务是否在列表中
            task_ids_in_list = [t['id'] for t in list_data['tasks']]
            self.assertIn(task_id, task_ids_in_list, "任务未出现在任务列表中")
            logging.info(f"任务 {task_id} 已在任务列表中")
            
            # 步骤5: 删除任务
            logging.info("=" * 60)
            logging.info("步骤5: 删除任务")
            logging.info("=" * 60)
            
            url = f"{self.base_url}/tasks/batch-delete"
            data = {"task_ids": [task_id]}
            
            response = requests.post(url, json=data, timeout=10)
            
            self.assertEqual(response.status_code, 200, 
                           f"删除任务失败，状态码: {response.status_code}, 响应: {response.text}")
            
            logging.info(f"删除任务 - 响应状态码: {response.status_code}")
            
            response_data = response.json()
            self.assertTrue(response_data.get('success', False), 
                          f"删除任务返回失败: {response_data}")
            
            delete_data = response_data['data']
            logging.info(f"删除成功: {delete_data['success_count']}/{delete_data['total_count']}")
            
            self.assertEqual(delete_data['success_count'], 1, "任务删除失败")
            
            # 从清理列表中移除（已删除）
            self.task_ids_to_cleanup.remove(task_id)
            
            logging.info("=" * 60)
            logging.info("所有测试步骤完成！")
            logging.info("=" * 60)
            
        except AssertionError:
            raise
        except Exception as e:
            logging.error(f"测试过程中发生异常: {str(e)}")
            self.fail(f"测试失败: {str(e)}")


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    unittest.main(verbosity=2)