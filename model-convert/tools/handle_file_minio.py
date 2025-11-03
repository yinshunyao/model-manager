#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MinIO 文件操作工具

此模块提供了与 MinIO 对象存储交互的核心功能，主要包括：
1. 文件上传：将本地文件上传到MinIO服务器的指定存储桶
2. 文件下载：从MinIO服务器下载文件到本地

模块特性：
- 自动从config目录加载MinIO配置信息
- 提供全局minio_handler对象，其他模块可直接导入使用
- 包含完整的错误处理和日志记录
- 支持自动创建存储桶

使用示例：
    from tools.handle_file_minio import minio_handler
    minio_handler.upload_file('my-bucket', 'remote-file.txt', '/path/to/local/file.txt')
    minio_handler.download_file('my-bucket', 'remote-file.txt', '/path/to/download.txt')
"""

import os
import logging
from typing import Dict, Optional

# 导入MinIO客户端和异常处理
from minio import Minio
from minio.error import S3Error

# 导入配置加载器
from config import config_loader

# 设置模块导出内容
__all__ = ['MinioHandler', 'minio_handler', 'init_minio_handler']

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MinioHandler:
    """
    MinIO 操作处理器类
    提供与 MinIO 对象存储交互的文件上传和下载功能
    """
    
    def __init__(self, endpoint: str, access_key: str, secret_key: str, 
                 secure: bool = False, region: Optional[str] = None):
        """
        初始化 MinIO 客户端
        
        Args:
            endpoint (str): MinIO 服务器地址和端口
            access_key (str): 访问密钥
            secret_key (str): 密钥
            secure (bool): 是否使用 HTTPS
            region (Optional[str]): 区域（如果适用）
        """
        try:
            self.client = Minio(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
                region=region
            )
            logger.info(f"成功连接到 MinIO 服务器: {endpoint}")
        except Exception as e:
            logger.error(f"连接 MinIO 服务器失败: {str(e)}")
            raise
    
    def create_bucket(self, bucket_name: str, location: str = "us-east-1") -> bool:
        """
        创建存储桶
        
        Args:
            bucket_name (str): 存储桶名称
            location (str): 存储桶位置，默认为 "us-east-1"
            
        Returns:
            bool: 创建是否成功
        """
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name, location=location)
                logger.info(f"成功创建存储桶: {bucket_name}")
                return True

            else:
                logger.info(f"存储桶已存在: {bucket_name}")
                return True
        except Exception as e:
            logger.error(f"创建存储桶失败: {str(e)}")
            return False
    
    def upload_file(self, bucket_name: str, object_name: str, 
                   file_path: str, content_type: str = "application/octet-stream",
                   metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        上传文件到 MinIO
        
        Args:
            bucket_name (str): 存储桶名称
            object_name (str): 对象名称（MinIO 中的文件名）
            file_path (str): 本地文件路径
            content_type (str): 内容类型
            metadata (Optional[Dict[str, str]]): 元数据
            
        Returns:
            bool: 上传是否成功
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"本地文件不存在: {file_path}")
                return False

            logger.warning(f"准备上传文件: {file_path}")
            
            # 检查存储桶是否存在
            if not self.client.bucket_exists(bucket_name):
                logger.info(f"存储桶不存在，尝试创建: {bucket_name}")
                if not self.create_bucket(bucket_name):
                    return False
            
            # 上传文件
            result = self.client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=file_path,
                content_type=content_type,
                metadata=metadata
            )
            
            logger.info(f"文件上传成功: {file_path} -> {bucket_name}/{object_name}, "
                      f"ETag: {result.etag}")
            return True
        except Exception as e:
            logger.error(f"文件上传失败: {str(e)}")
            return False
    
    def download_file(self, bucket_name: str, object_name: str, 
                     file_path: str) -> bool:
        """
        从 MinIO 下载文件
        
        Args:
            bucket_name (str): 存储桶名称
            object_name (str): 对象名称
            file_path (str): 本地文件路径
            
        Returns:
            bool: 下载是否成功
        """
        try:
            # 确保目标目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 下载文件
            self.client.fget_object(bucket_name, object_name, file_path)
            logger.info(f"文件下载成功: {bucket_name}/{object_name} -> {file_path}")
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.error(f"文件不存在: {bucket_name}/{object_name}")
            else:
                logger.error(f"文件下载失败: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"文件下载失败: {str(e)}")
            return False


# 创建全局 MinIO 处理器实例
minio_handler = None

def init_minio_handler() -> Optional[MinioHandler]:
    """
    初始化全局 MinIO 处理器实例
    使用配置文件中的 MinIO 配置
    
    Returns:
        Optional[MinioHandler]: MinIO 处理器实例，如果初始化失败返回 None
    """
    global minio_handler
    
    try:
        # 从配置文件获取 MinIO 配置
        minio_config = config_loader.get_minio_config()
        
        # 提取必要的配置项，设置默认值
        endpoint = minio_config.get('endpoint', 'localhost:9000')
        access_key = minio_config.get('access_key', 'minioadmin')
        secret_key = minio_config.get('secret_key', 'minioadmin')
        secure = minio_config.get('secure', False)
        region = minio_config.get('region', None)
        
        # 创建 MinIO 处理器实例
        minio_handler = MinioHandler(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region
        )
        
        logger.info(f"全局 MinIO 处理器实例初始化成功，连接到 {endpoint}")
        
        # 自动创建默认存储桶
        default_bucket = minio_config.get('bucket_name', 'model-storage')
        minio_handler.create_bucket(default_bucket)
        
        return minio_handler
    except Exception as e:
        logger.error(f"初始化 MinIO 处理器实例失败: {str(e)}")
        return None


def main():
    """
    主函数，用于测试 MinIO 操作功能
    包括初始化、创建存储桶、文件上传和下载测试
    """
    try:
        # 初始化 MinIO 处理器
        handler = init_minio_handler()
        if not handler:
            raise Exception("MinIO 处理器初始化失败")
        
        # 获取默认存储桶名称
        minio_config = get_minio_config()
        bucket_name = minio_config.get('bucket_name', 'model-storage')
        
        # 测试文件上传
        print("测试文件上传...")
        # 创建一个临时测试文件
        test_file_path = "/tmp/test_minio_upload.txt"
        with open(test_file_path, "w") as f:
            f.write("这是一个测试文件，用于测试MinIO文件上传功能。")
        
        # 上传文件到MinIO
        object_name = "test_upload.txt"
        success = handler.upload_file(bucket_name, object_name, test_file_path)
        print(f"文件上传 {'成功' if success else '失败'}")
        
        # 测试文件下载
        print("测试文件下载...")
        download_path = "/tmp/test_minio_download.txt"
        success = handler.download_file(bucket_name, object_name, download_path)
        print(f"文件下载 {'成功' if success else '失败'}")
        
        # 清理临时文件
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        
        print("测试完成！")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# 当模块被导入时，自动初始化 MinIO 处理器
try:
    # 尝试初始化，但如果失败不阻止模块导入
    init_minio_handler()
except Exception as e:
    logger.warning(f"模块导入时自动初始化 MinIO 处理器失败: {str(e)}")
    logger.warning("可以稍后手动调用 init_minio_handler() 进行初始化")