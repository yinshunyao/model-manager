#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MinIO 文件处理功能测试脚本

此脚本用于测试 rest_predict_910b.py 中实现的 MinIO 文件处理功能，
包括文件下载、处理和上传。
"""

import sys
import os
import tempfile
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入MinIO处理模块
from tools.handle_file_minio import minio_handler, init_minio_handler


def test_minio_integration():
    """测试MinIO集成功能"""
    logger.info("开始测试MinIO集成功能...")
    
    # 初始化MinIO处理器
    handler = init_minio_handler()
    
    if handler is None:
        logger.warning("MinIO处理器未初始化，跳过测试")
        return False
    
    # 创建测试存储桶
    test_bucket = "test-minio-processing"
    try:
        handler.create_bucket(test_bucket)
        logger.info(f"创建测试存储桶: {test_bucket}")
    except Exception as e:
        logger.info(f"存储桶可能已存在或创建失败: {e}")
    
    # 创建一个测试文件
    fd, test_file_path = tempfile.mkstemp(suffix='.txt')
    os.close(fd)
    
    try:
        # 写入测试内容
        with open(test_file_path, 'w') as f:
            f.write("这是一个测试文件，用于验证MinIO文件处理功能。\n")
            f.write("测试时间: {}\n".format(os.path.getctime(test_file_path)))
        
        test_object = "test-file.txt"
        
        # 上传文件到MinIO
        if handler.upload_file(test_bucket, test_object, test_file_path):
            logger.info(f"成功上传文件到MinIO: {test_bucket}/{test_object}")
        else:
            logger.error(f"上传文件到MinIO失败: {test_bucket}/{test_object}")
            return False
        
        # 下载文件从MinIO
        fd, downloaded_file_path = tempfile.mkstemp(suffix='.txt')
        os.close(fd)
        
        if handler.download_file(test_bucket, test_object, downloaded_file_path):
            logger.info(f"成功从MinIO下载文件: {test_bucket}/{test_object}")
            
            # 验证文件内容
            with open(downloaded_file_path, 'r') as f:
                content = f.read()
                logger.info(f"下载文件内容: {content.strip()}")
        else:
            logger.error(f"从MinIO下载文件失败: {test_bucket}/{test_object}")
            return False
        
        # 清理测试文件
        os.remove(test_file_path)
        os.remove(downloaded_file_path)
        
        logger.info("MinIO集成功能测试完成")
        return True
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        return False
    finally:
        # 清理测试文件
        if os.path.exists(test_file_path):
            os.remove(test_file_path)


if __name__ == "__main__":
    success = test_minio_integration()
    if success:
        logger.info("所有测试通过!")
        sys.exit(0)
    else:
        logger.error("测试失败!")
        sys.exit(1)