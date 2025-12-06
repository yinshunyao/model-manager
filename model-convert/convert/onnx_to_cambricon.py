#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX 模型转换为寒武纪 MagicMind 模型

此模块提供了将 ONNX 模型转换为寒武纪 MLU 处理器支持的 MagicMind 格式的功能。
转换过程依赖寒武纪 MagicMind 工具链。
"""

import logging
import os
import subprocess
import argparse
from typing import Dict, Optional, Union, Tuple, Any

try:
    # 尝试导入 MagicMind 相关模块
    import magicmind.python.runtime as mm
    import magicmind.python.builder as mb
    import magicmind.python.parser as mp
    magicmind_available = True
except ImportError:
    magicmind_available = False
    logging.warning("MagicMind 未安装，无法执行寒武纪平台模型转换")

logger = logging.getLogger(__name__)


def onnx_to_magicmind(
    onnx_model_path: str,
    output_mm_path: str,
    input_shape: Optional[Union[str, Tuple[int, ...]]] = None,
    precision_mode: str = "force_float32",
    **kwargs
) -> bool:
    """
    将 ONNX 模型转换为寒武纪 MagicMind 模型
    
    Args:
        onnx_model_path: ONNX 模型文件路径
        output_mm_path: 输出 MagicMind 模型文件路径
        input_shape: 模型输入形状，可选
        precision_mode: 精度模式，支持 force_float32, force_float16 等
        **kwargs: 其他参数
        
    Returns:
        bool: 转换是否成功
    """
    if not magicmind_available:
        raise ImportError("MagicMind 未安装，无法执行寒武纪平台模型转换")
    
    try:
        # 验证输入文件存在
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX 模型文件不存在: {onnx_model_path}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_mm_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"开始转换 ONNX 模型到 MagicMind 格式: {onnx_model_path} -> {output_mm_path}")
        
        # 使用 MagicMind Python API 进行转换
        # 创建 MagicMind 构建器
        builder = mb.IBuilder()
        
        # 创建网络
        network = mb.INetwork()
        
        # 创建 ONNX 解析器
        parser = mp.IParser()
        
        # 解析 ONNX 模型
        logger.info("正在解析 ONNX 模型...")
        parser.parse_from_file(network, onnx_model_path)
        
        # 设置构建配置
        config = builder.create_builder_config()
        
        # 设置精度模式
        if precision_mode == "force_float32":
            config.parse_from_string("{\"precision_config\": {\"precision_mode\": \"force_float32\"}}")
        elif precision_mode == "force_float16":
            config.parse_from_string("{\"precision_config\": {\"precision_mode\": \"force_float16\"}}")
        else:
            # 默认使用 force_float32
            config.parse_from_string("{\"precision_config\": {\"precision_mode\": \"force_float32\"}}")
        
        # 如果指定了输入形状，则设置
        if input_shape is not None:
            logger.info(f"设置输入形状: {input_shape}")
            # 这里需要根据具体的模型来设置输入形状
            # 示例代码，实际应用中需要根据模型结构调整
            inputs = network.get_input(0)
            if isinstance(input_shape, str):
                # 解析字符串形式的形状，例如 "1,3,224,224"
                shape_list = [int(x.strip()) for x in input_shape.split(',')]
                inputs.set_dimension(shape_list)
            elif isinstance(input_shape, (tuple, list)):
                inputs.set_dimension(list(input_shape))
        
        # 构建模型
        logger.info("正在构建 MagicMind 模型...")
        model = builder.build_model("magicmind_model", network, config)
        
        if model is None:
            raise RuntimeError("MagicMind 模型构建失败")
        
        # 保存模型
        logger.info(f"正在保存 MagicMind 模型到: {output_mm_path}")
        assert model.serialize_to_file(output_mm_path).ok()
        
        logger.info(f"成功转换 ONNX 模型到 MagicMind 格式: {output_mm_path}")
        return True
        
    except Exception as e:
        logger.error(f"寒武纪模型转换失败: {str(e)}")
        raise RuntimeError(f"寒武纪模型转换失败: {str(e)}") from e


def onnx_to_magicmind_cli(
    onnx_model_path: str,
    output_mm_path: str,
    input_shape: Optional[str] = None,
    precision_mode: str = "force_float32",
    **kwargs
) -> bool:
    """
    使用命令行方式将 ONNX 模型转换为寒武纪 MagicMind 模型
    
    Args:
        onnx_model_path: ONNX 模型文件路径
        output_mm_path: 输出 MagicMind 模型文件路径
        input_shape: 模型输入形状，例如 "1,3,224,224"
        precision_mode: 精度模式
        **kwargs: 其他参数
        
    Returns:
        bool: 转换是否成功
    """
    try:
        # 验证输入文件存在
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX 模型文件不存在: {onnx_model_path}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_mm_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"开始使用命令行方式转换 ONNX 模型到 MagicMind 格式: {onnx_model_path} -> {output_mm_path}")
        
        # 构建命令行参数
        cmd = [
            "mmconvert",
            "--framework", "onnx",
            "--model", onnx_model_path,
            "--output", output_mm_path,
            "--precision", precision_mode
        ]
        
        # 如果指定了输入形状
        if input_shape is not None:
            cmd.extend(["--input_shape", input_shape])
        
        # 添加其他参数
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])
        
        # 执行转换命令
        logger.info(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"命令执行成功，输出: {result.stdout}")
        logger.info(f"成功转换 ONNX 模型到 MagicMind 格式: {output_mm_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {e.stderr}")
        raise RuntimeError(f"寒武纪模型转换命令执行失败: {e.stderr}") from e
    except Exception as e:
        logger.error(f"寒武纪模型转换失败: {str(e)}")
        raise RuntimeError(f"寒武纪模型转换失败: {str(e)}") from e


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="ONNX 模型转换为寒武纪 MagicMind 模型")
    parser.add_argument("--onnx_model", "-i", required=True, help="输入 ONNX 模型文件路径")
    parser.add_argument("--output_model", "-o", required=True, help="输出 MagicMind 模型文件路径")
    parser.add_argument("--input_shape", help="输入形状，例如 '1,3,224,224'")
    parser.add_argument("--precision", default="force_float32", 
                       choices=["force_float32", "force_float16"],
                       help="精度模式")
    
    args = parser.parse_args()
    
    try:
        # 尝试使用 CLI 方式转换
        success = onnx_to_magicmind_cli(
            onnx_model_path=args.onnx_model,
            output_mm_path=args.output_model,
            input_shape=args.input_shape,
            precision_mode=args.precision
        )
        
        if success:
            print(f"模型转换成功: {args.output_model}")
        else:
            print("模型转换失败")
    except Exception as e:
        print(f"模型转换失败: {str(e)}")