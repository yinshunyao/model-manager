#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX 模型转换为瑞芯微 RK3588 处理器支持的 RKNN 格式模型

此模块提供了将 ONNX 模型转换为瑞芯微 RK3588 处理器支持的 RKNN 格式的功能。
转换过程依赖瑞芯微 RKNN Toolkit 2 工具链。
"""
import logging
import os
import argparse
from typing import Dict, Optional, Union, Tuple, Any
import onnx

# 导入RKNN Toolkit 2
try:
    from rknn.api import RKNN
    rknn_available = True
except ImportError:
    RKNN = None
    rknn_available = False
    logger.warning("警告：RKNN Toolkit 2 未安装，无法执行ONNX到RKNN的转换。")
    logger.warning("请参考瑞芯微官方文档安装RKNN Toolkit 2：https://www.rock-chips.com/a/cn/downloadCenter/index.html")


def get_input_shape_from_onnx(onnx_model_path: str) -> Dict[str, Tuple[int, ...]]:
    """
    从 ONNX 模型文件中自动获取输入形状
    
    Args:
        onnx_model_path (str): ONNX 模型文件路径
    
    Returns:
        Dict[str, Tuple[int, ...]]: 输入名称到形状的映射字典
    
    Raises:
        ValueError: 当无法从模型中获取输入形状时
    """
    try:
        # 加载 ONNX 模型
        model = onnx.load(onnx_model_path)
        
        # 获取模型的输入
        inputs = model.graph.input
        
        if not inputs:
            raise ValueError("无法从 ONNX 模型中获取输入信息")
        
        input_shapes = {}
        
        for input_node in inputs:
            # 获取输入名称
            input_name = input_node.name
            
            # 获取输入形状
            shape = []
            for dim in input_node.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    # 固定维度
                    shape.append(int(dim.dim_value))
                elif dim.dim_param:
                    # 参数化维度，这里我们使用默认值 1
                    shape.append(1)
                else:
                    # 未知维度，使用默认值 1
                    shape.append(1)
            
            input_shapes[input_name] = tuple(shape)
        
        if not input_shapes:
            raise ValueError("无法从 ONNX 模型中解析出有效的输入形状")
        
        return input_shapes
    
    except (FileNotFoundError, IOError):
        raise FileNotFoundError(f"ONNX 模型文件不存在或无法读取: {onnx_model_path}")
    except Exception as e:
        raise ValueError(f"从 ONNX 模型获取输入形状时出错: {str(e)}")


def onnx_to_rknn(
    onnx_model_path: str,
    output_rknn_path: str,
    input_shape: Optional[Union[str, Tuple[int, ...]]] = None,
    target_platform: str = "rk3588",
    precision_mode: str = "float32",
    auto_input_shape: bool = True,
    **kwargs
) -> bool:
    """
    将 ONNX 模型转换为瑞芯微 RK3588 处理器支持的 RKNN 格式模型

    Args:
        onnx_model_path (str): ONNX 模型文件路径
        output_rknn_path (str): 输出 RKNN 模型文件路径
        input_shape (Optional[Union[str, Tuple[int, ...]]]): 模型输入形状
            - 如果为 None 且 auto_input_shape=True，则自动从模型中获取完整的输入名称和形状
            - 如果是字符串且不包含输入名称（格式为 "N,C,H,W"），则自动从模型中获取输入名称
            - 如果是字符串且包含输入名称（格式为 "input_name:N,C,H,W"），则直接使用指定的输入名称
            - 如果是元组（格式为 (N,C,H,W)），则自动从模型中获取输入名称
        target_platform (str): 目标瑞芯微处理器版本，默认为 "rk3588"
        precision_mode (str): 精度模式，默认为 "float32"
        auto_input_shape (bool): 是否自动从模型中获取输入形状，默认为 True
        **kwargs: 其他 RKNN 工具参数

    Returns:
        bool: 转换是否成功
    """
    try:
        # 检查RKNN Toolkit 2是否安装
        if not rknn_available:
            logging.error("RKNN Toolkit 2 未安装，无法执行ONNX到RKNN的转换。")
            return False
        
        # 验证输入文件存在
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX 模型文件不存在: {onnx_model_path}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_rknn_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 处理输入形状
        if input_shape is None:
            # 自动获取输入形状
            input_shapes = get_input_shape_from_onnx(onnx_model_path)
            if not input_shapes:
                raise ValueError("无法自动获取输入形状，请手动指定")
        elif isinstance(input_shape, tuple):
            # 从模型中获取输入名称
            input_shapes = get_input_shape_from_onnx(onnx_model_path)
            if not input_shapes:
                raise ValueError("无法自动获取输入名称，请手动指定")
            # 使用指定的形状替换自动获取的形状
            input_name = list(input_shapes.keys())[0]
            input_shapes = {input_name: input_shape}
        elif isinstance(input_shape, str):
            # 解析输入形状字符串
            if ":" in input_shape:
                # 格式为 "input_name:N,C,H,W"
                input_name, shape_str = input_shape.split(":")
                shape = tuple(map(int, shape_str.split(",")))
                input_shapes = {input_name: shape}
            else:
                # 格式为 "N,C,H,W"，从模型中获取输入名称
                shape = tuple(map(int, input_shape.split(",")))
                input_shapes = get_input_shape_from_onnx(onnx_model_path)
                if not input_shapes:
                    raise ValueError("无法自动获取输入名称，请手动指定")
                input_name = list(input_shapes.keys())[0]
                input_shapes = {input_name: shape}
        else:
            raise ValueError(f"不支持的输入形状类型: {type(input_shape)}")
        
        # 初始化RKNN对象
        rknn = RKNN()
        
        # 配置RKNN参数
        rknn_config = {
            'target_platform': target_platform,
            'precision': precision_mode,
            'optimizer': kwargs.get('optimizer', 'level3'),
            'quantized_dtype': kwargs.get('quantized_dtype', 'asymmetric_affine-u8'),
            'batch_size': kwargs.get('batch_size', 1),
        }
        
        # 加载ONNX模型
        logging.info(f"加载ONNX模型: {onnx_model_path}")
        logging.info(f"输入形状: {input_shapes}")
        
        # 处理RKNN Toolkit 2的输入格式
        input_name = list(input_shapes.keys())[0]
        input_shape = list(input_shapes.values())[0]
        
        # 配置输入
        rknn.config(**rknn_config)
        
        # 加载模型
        ret = rknn.load_onnx(
            model=onnx_model_path,
            inputs=[input_name],
            input_size_list=[list(input_shape[1:])],  # RKNN Toolkit 2需要去掉batch维度
            **kwargs
        )
        
        if ret != 0:
            raise RuntimeError(f"加载ONNX模型失败: {ret}")
        
        # 构建RKNN模型
        logging.info("构建RKNN模型...")
        ret = rknn.build(do_quantization=False, **kwargs)
        
        if ret != 0:
            raise RuntimeError(f"构建RKNN模型失败: {ret}")
        
        # 导出RKNN模型
        logging.info(f"导出RKNN模型到: {output_rknn_path}")
        ret = rknn.export_rknn(output_rknn_path)
        
        if ret != 0:
            raise RuntimeError(f"导出RKNN模型失败: {ret}")
        
        # 释放资源
        rknn.release()
        
        logging.info(f"ONNX到RKNN转换成功: {output_rknn_path}")
        return True
        
    except Exception as e:
        logging.error(f"ONNX到RKNN转换失败: {str(e)}")
        return False


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="ONNX 模型转换为 RKNN 格式模型")
    parser.add_argument("--input", type=str, required=True, help="输入 ONNX 模型文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出 RKNN 模型文件路径")
    parser.add_argument("--input_shape", type=str, help="输入形状，格式为 N,C,H,W 或 input_name:N,C,H,W")
    parser.add_argument("--target_platform", type=str, default="rk3588", help="目标瑞芯微处理器版本")
    parser.add_argument("--precision_mode", type=str, default="float32", help="精度模式")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 执行转换
    success = onnx_to_rknn(
        onnx_model_path=args.input,
        output_rknn_path=args.output,
        input_shape=args.input_shape,
        target_platform=args.target_platform,
        precision_mode=args.precision_mode
    )
    
    if success:
        logging.info("转换完成")
        exit(0)
    else:
        logging.error("转换失败")
        exit(1)