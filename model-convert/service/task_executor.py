#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务执行器模块

此模块负责实际执行模型转换任务，包括设置环境变量、调用转换工具等。
"""
import os
import sys
import logging
import tempfile
import shutil
from typing import Optional, Dict, Any
from datetime import datetime

# 添加项目路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置华为昇腾环境变量
def setup_huawei_environment():
    """
    设置华为昇腾环境变量
    """
    try:
        os.environ["ASCEND_HOME"] = "/usr/local/Ascend/ascend-toolkit/8.0.0"
        os.environ["LD_LIBRARY_PATH"] = (
            f"{os.environ['ASCEND_HOME']}/lib64:"
            "/usr/local/Ascend/driver/lib64:"
            f"{os.environ.get('LD_LIBRARY_PATH', '')}"
        )
        os.environ["PATH"] = f"{os.environ['ASCEND_HOME']}/bin:{os.environ.get('PATH', '')}"
        os.environ["PYTHONPATH"] = (
            f"{os.environ['ASCEND_HOME']}/python/site-packages:"
            f"{os.environ.get('PYTHONPATH', '')}"
        )
        logging.info("华为昇腾环境变量设置成功")
    except Exception as e:
        logging.error(f"设置华为昇腾环境变量失败: {str(e)}")
        raise

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 预先设置华为环境变量
setup_huawei_environment()

# 导入转换模块
onnx_to_om = None
yolo_to_onnx = None
try:
    from convert.onnx_to_om import onnx_to_om
    logger.info("成功导入华为平台转换模块")
except ImportError as e:
    logger.warning(f"导入华为平台转换模块失败: {str(e)}，某些功能可能不可用")
try:
    from convert.yolo_to_onnx import convert_yolo11_to_onnx
    yolo_to_onnx = convert_yolo11_to_onnx
    logger.info("成功导入YOLO到ONNX转换模块")
except ImportError as e:
    logger.warning(f"导入YOLO到ONNX转换模块失败: {str(e)}，某些功能可能不可用")

class TaskExecutor:
    """
    任务执行器，负责执行模型转换任务
    """
    
    def __init__(self):
        """
        初始化任务执行器
        """
        self.temp_dirs = []
    
    def __del__(self):
        """
        清理临时文件
        """
        self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """
        清理所有创建的临时目录
        """
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"清理临时目录: {temp_dir}")
            except Exception as e:
                logger.error(f"清理临时目录失败 {temp_dir}: {str(e)}")
        self.temp_dirs = []
    
    def _create_temp_dir(self) -> str:
        """
        创建临时目录
        
        Returns:
            临时目录路径
        """
        temp_dir = tempfile.mkdtemp(prefix="model_convert_")
        self.temp_dirs.append(temp_dir)
        logger.info(f"创建临时目录: {temp_dir}")
        return temp_dir
    
    def execute_huawei_conversion(self, input_path: str, output_path: str, 
                                 parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        执行华为平台的模型转换（ONNX转OM）
        
        Args:
            input_path: 输入ONNX模型路径
            output_path: 输出OM模型路径
            parameters: 转换参数
        
        Returns:
            是否转换成功
        """
        try:
            # 验证输入文件存在
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"输入文件不存在: {input_path}")
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 准备参数
            params = parameters.copy() if parameters else {}
            
            # 处理特殊情况：如果输出路径是MinIO路径，需要先输出到本地
            is_minio_output = output_path.startswith('minio://') or 'minio' in output_path.lower()
            local_output_path = None
            
            if is_minio_output:
                # 创建临时本地输出路径
                temp_dir = self._create_temp_dir()
                # 移除可能的.om后缀，因为onnx_to_om函数会自动添加
                output_filename = os.path.basename(output_path).replace('.om', '')
                local_output_path = os.path.join(temp_dir, output_filename)
                actual_output_path = local_output_path
            else:
                # 本地输出，直接使用
                # 移除可能的.om后缀
                actual_output_path = output_path.replace('.om', '')
            
            logger.info(f"开始华为模型转换: {input_path} -> {actual_output_path}.om")
            logger.info(f"转换参数: {params}")
            
            # 执行转换
            if onnx_to_om is None:
                raise ImportError("华为平台转换模块未成功导入，无法执行转换")
            success = onnx_to_om(input_path, actual_output_path, **params)
            
            if success:
                logger.info(f"华为模型转换成功: {actual_output_path}.om")
                
                # 如果需要上传到MinIO
                if is_minio_output:
                    try:
                        # 这里应该添加MinIO上传逻辑
                        # 暂时只是打印信息
                        logger.info(f"需要上传到MinIO: {actual_output_path}.om -> {output_path}")
                        # 实际项目中应该调用MinIO上传函数
                        # from tools.handle_file_minio import minio_handler
                        # minio_handler.upload_file(actual_output_path + '.om', output_path)
                    except Exception as e:
                        logger.error(f"上传到MinIO失败: {str(e)}")
                        raise
            
            return success
            
        except Exception as e:
            logger.error(f"华为模型转换失败: {str(e)}")
            raise
    
    def execute_rockchip_conversion(self, input_path: str, output_path: str, 
                                   parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        执行瑞芯微平台的模型转换（暂未实现）
        
        Args:
            input_path: 输入模型路径
            output_path: 输出模型路径
            parameters: 转换参数
        
        Returns:
            是否转换成功
        """
        logger.warning("瑞芯微平台模型转换功能暂未实现")
        raise NotImplementedError("瑞芯微平台模型转换功能暂未实现")
    
    def execute_cambricon_conversion(self, input_path: str, output_path: str, 
                                    parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        执行寒武纪平台的模型转换（暂未实现）
        
        Args:
            input_path: 输入模型路径
            output_path: 输出模型路径
            parameters: 转换参数
        
        Returns:
            是否转换成功
        """
        logger.warning("寒武纪平台模型转换功能暂未实现")
        raise NotImplementedError("寒武纪平台模型转换功能暂未实现")
    
    def execute_yolo_to_onnx_conversion(self, input_path: str, output_path: str, 
                                        parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        执行YOLO到ONNX的转换
        
        Args:
            input_path: 输入YOLO模型路径
            output_path: 输出ONNX模型路径
            parameters: 转换参数
        
        Returns:
            是否转换成功
        """
        try:
            # 验证输入文件存在
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"输入文件不存在: {input_path}")
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 准备参数
            params = parameters.copy() if parameters else {}
            
            # 处理特殊情况：如果输出路径是MinIO路径，需要先输出到本地
            is_minio_output = output_path.startswith('minio://') or 'minio' in output_path.lower()
            local_output_path = None
            
            if is_minio_output:
                # 创建临时本地输出路径
                temp_dir = self._create_temp_dir()
                # 移除可能的.onnx后缀，因为convert_yolo11_to_onnx函数会自动添加
                output_filename = os.path.basename(output_path).replace('.onnx', '')
                local_output_path = os.path.join(temp_dir, output_filename)
                actual_output_path = local_output_path
            else:
                # 本地输出，直接使用
                # 移除可能的.onnx后缀
                actual_output_path = output_path.replace('.onnx', '')
            
            logger.info(f"开始YOLO到ONNX转换: {input_path} -> {actual_output_path}.onnx")
            logger.info(f"转换参数: {params}")
            
            # 执行转换
            if yolo_to_onnx is None:
                raise ImportError("YOLO到ONNX转换模块未成功导入，无法执行转换")
            
            # 获取转换参数
            img_size = params.get('img_size', 640)
            simplify = params.get('simplify', True)
            opset_version = params.get('opset_version', 13)
            
            # 执行转换
            success = yolo_to_onnx(
                model_path=input_path,
                output_path=actual_output_path,
                imgsz=img_size,
                simplify=simplify,
                opset_version=opset_version
            )
            
            if success:
                logger.info(f"YOLO到ONNX转换成功: {actual_output_path}.onnx")
                
                # 如果需要上传到MinIO
                if is_minio_output:
                    try:
                        # 这里应该添加MinIO上传逻辑
                        # 暂时只是打印信息
                        logger.info(f"需要上传到MinIO: {actual_output_path}.onnx -> {output_path}")
                        # 实际项目中应该调用MinIO上传函数
                        # from tools.handle_file_minio import minio_handler
                        # minio_handler.upload_file(actual_output_path + '.onnx', output_path)
                    except Exception as e:
                        logger.error(f"上传到MinIO失败: {str(e)}")
                        raise
            
            return success
            
        except Exception as e:
            logger.error(f"YOLO到ONNX转换失败: {str(e)}")
            raise
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单个任务
        
        Args:
            task: 任务信息字典
        
        Returns:
            执行结果，包含状态和可能的错误信息
        """
        task_id = task.get('id')
        if not task_id:
            raise ValueError("任务缺少ID")
            
        task_type = task.get('task_type')
        platform = task.get('platform')
        input_path = task.get('input_path')
        output_path = task.get('output_path')
        
        # 验证必要参数
        if not all([task_type, platform, input_path, output_path]):
            raise ValueError(f"任务参数不完整: 任务ID={task_id}, 缺少必要字段")
            
        # 确保参数类型正确
        input_path = str(input_path)
        output_path = str(output_path)
        
        parameters = task.get('parameters', {})
        
        result = {
            'success': False,
            'error_message': None,
            'log_path': None
        }
        
        try:
            logger.info(f"开始执行任务: {task_id}, 类型: {task_type}, 平台: {platform}")
            
            # 根据平台选择转换方法
            if platform == 'huawei' and task_type == 'onnx_to_om':
                success = self.execute_huawei_conversion(input_path, output_path, parameters)
            elif platform == 'rockchip':
                success = self.execute_rockchip_conversion(input_path, output_path, parameters)
            elif platform == 'cambricon':
                success = self.execute_cambricon_conversion(input_path, output_path, parameters)
            elif platform == 'onnx' and task_type == 'yolo_to_onnx':
                success = self.execute_yolo_to_onnx_conversion(input_path, output_path, parameters)
            else:
                raise ValueError(f"不支持的任务类型和平台组合: {task_type} on {platform}")
            
            result['success'] = success
            
            if success:
                logger.info(f"任务执行成功: {task_id}")
            else:
                logger.error(f"任务执行失败: {task_id}")
                result['error_message'] = "转换过程返回失败状态"
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"任务执行异常: {task_id}, 错误: {error_msg}")
            result['error_message'] = error_msg
        
        return result

# 创建全局任务执行器实例
task_executor = None

def get_task_executor() -> TaskExecutor:
    """
    获取任务执行器实例
    
    Returns:
        TaskExecutor实例
    """
    global task_executor
    if task_executor is None:
        task_executor = TaskExecutor()
    return task_executor

if __name__ == "__main__":
    # 测试任务执行器
    executor = TaskExecutor()
    
    # 这里可以添加测试代码
    print("任务执行器初始化成功")