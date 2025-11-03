#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX 模型转换为华为昇腾 910B 处理器支持的 OM 格式模型

此模块提供了将 ONNX 模型转换为华为昇腾 910B 处理器支持的 OM 格式的功能。
转换过程依赖华为昇腾 AI 工具链中的 ATC（Ascend Tensor Compiler）工具。
"""
import logging
import os
import subprocess
import argparse
from typing import Dict, Optional, Union, Tuple, Any, List
import onnx


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
                    # 注意：这可能不适用于所有模型，特别是对于动态形状
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

def onnx_to_om(
    onnx_model_path: str,
    output_om_path: str,
    input_shape: Optional[Union[str, Tuple[int, ...]]] = None,
    soc_version: str = "Ascend910B",
    precision_mode: str = "allow_fp32_to_fp16",
    log_level: str = "error",
    auto_input_shape: bool = True,
    **kwargs
) -> bool:
    """
    将 ONNX 模型转换为华为昇腾 910B 处理器支持的 OM 格式模型

    Args:
        onnx_model_path (str): ONNX 模型文件路径
        output_om_path (str): 输出 OM 模型文件路径
        input_shape (Optional[Union[str, Tuple[int, ...]]]): 模型输入形状
            - 如果为 None 且 auto_input_shape=True，则自动从模型中获取完整的输入名称和形状
            - 如果是字符串且不包含输入名称（格式为 "N,C,H,W"），则自动从模型中获取输入名称
            - 如果是字符串且包含输入名称（格式为 "input_name:N,C,H,W"），则直接使用指定的输入名称
            - 如果是元组（格式为 (N,C,H,W)），则自动从模型中获取输入名称
        soc_version (str): 目标昇腾处理器版本，默认为 "Ascend910B"
        precision_mode (str): 精度模式，默认为 "allow_fp32_to_fp16"
        log_level (str): 日志级别，默认为 "error"
        auto_input_shape (bool): 是否自动从模型中获取输入形状，默认为 True
        **kwargs: 其他 ATC 工具参数

    Returns:
        bool: 转换是否成功

    Raises:
        FileNotFoundError: 当 ONNX 模型文件不存在时
        RuntimeError: 当 ATC 工具调用失败时
    """
    # 检查 ONNX 模型文件是否存在
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX 模型文件不存在: {onnx_model_path}")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_om_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 自动从 ONNX 模型获取输入形状（如果未提供且启用了自动获取）
    if input_shape is None and auto_input_shape:
        # logging.warning("未提供输入形状，尝试从 ONNX 模型中自动获取...")
        try:
            input_shapes = get_input_shape_from_onnx(onnx_model_path)
            logging.warning(f"成功从模型中获取输入形状: {input_shapes}")
            
            # 构建 input_shape_str，格式为 "input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2"
            input_shape_parts = []
            for name, shape in input_shapes.items():
                shape_str = ",".join(map(str, shape))
                input_shape_parts.append(f"{name}:{shape_str}")
            input_shape_str = ";".join(input_shape_parts)
        except Exception as e:
            logging.error(f"自动获取输入形状失败: {str(e)}")
            raise ValueError("无法获取输入形状，请手动提供 input_shape 参数") from e
    else:
        # 处理用户提供的输入形状
        # 为了兼容 ATC 工具的要求，我们需要确保格式为 "input_name:n,c,h,w"
        if isinstance(input_shape, dict):
            # 处理字典格式，支持多个输入
            input_parts = []
            for input_name, shape in input_shape.items():
                if isinstance(shape, tuple):
                    shape_str = ",".join(map(str, shape))
                else:
                    shape_str = str(shape)
                input_parts.append(f"{input_name}:{shape_str}")
            input_shape_str = ";".join(input_parts)
        elif isinstance(input_shape, tuple):
            # 处理元组格式，尝试从ONNX模型获取输入名称
            shape_str = ",".join(map(str, input_shape))
            try:
                # 尝试从模型中获取输入名称
                input_names = list(get_input_shape_from_onnx(onnx_model_path).keys())
                if input_names:
                    # 使用模型中的第一个输入名称
                    input_name = input_names[0]
                    print(f"自动获取输入名称: {input_name}")
                else:
                    # 如果无法获取，使用默认名称
                    input_name = "images"
                    print(f"无法获取输入名称，使用默认名称: {input_name}")
            except Exception as e:
                # 如果获取失败，使用默认名称
                print(f"获取输入名称时出错: {str(e)}，使用默认名称 'images'")
                input_name = "images"
            
            input_shape_str = f"{input_name}:{shape_str}"
        elif isinstance(input_shape, str):
            # 如果已经是字符串，检查是否包含输入名称
            if ":" not in input_shape:
                # 如果不包含输入名称，尝试从模型获取
                try:
                    # 尝试从模型中获取输入名称
                    input_names = list(get_input_shape_from_onnx(onnx_model_path).keys())
                    if input_names:
                        # 使用模型中的第一个输入名称
                        input_name = input_names[0]
                        print(f"自动获取输入名称: {input_name}")
                    else:
                        # 如果无法获取，使用默认名称
                        input_name = "images"
                        print(f"无法获取输入名称，使用默认名称: {input_name}")
                except Exception as e:
                    # 如果获取失败，使用默认名称
                    print(f"获取输入名称时出错: {str(e)}，使用默认名称 'images'")
                    input_name = "images"
                
                input_shape_str = f"{input_name}:{input_shape}"
            else:
                input_shape_str = input_shape
        else:
            raise ValueError(f"不支持的输入形状类型: {type(input_shape)}")

    # 构建 ATC 命令
    cmd = [
        "atc",
        "--model=" + onnx_model_path,
        "--framework=5",  # 5 表示 ONNX 框架
        "--output=" + output_om_path,
        "--input_shape=" + input_shape_str,
        "--soc_version=" + soc_version,
        # --precision_mode=force_fp32
        # "--precision_mode=" + precision_mode,
        "--log=" + log_level,
        # "--output_type=FP32"
    ]

    # 添加额外的参数
    for key, value in kwargs.items():
        cmd.append(f"--{key}={value}")

    try:
        # logging.warning(f"开始转换模型: {onnx_model_path}")
        # logging.warning(f"目标输出: {output_om_path}")
        cmd = ' '.join(cmd)
        logging.warning(f"执行命令: {cmd}")

        # 执行 ATC 命令
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
        )

        # logging.warning("模型转换成功!")
        logging.warning(f"模型转换成功!输出 OM 模型路径: {output_om_path}")
        return True

    except subprocess.CalledProcessError as e:
        logging.warning(f"模型转换失败!错误信息: {e.stderr}")
        raise RuntimeError(f"ATC 工具调用失败: {e.stderr}") from e
    except Exception as e:
        logging.warning(f"转换过程中发生未知错误: {str(e)}")
        raise RuntimeError(f"模型转换失败: {str(e)}") from e


if __name__ == "__main__":
    # 单个模型转换示例
    print("示例1: 自动获取输入形状")
    onnx_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11n.onnx")
    om_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo11n")
    # onnx_model = "/Users/shunyaoyin/Documents/code/other/model-manager/model-convert/model_demo/out/yolo11n.onnx"
    # om_model = "/Users/shunyaoyin/Documents/code/other/model-manager/model-convert/model_demo/out/yolo11n.om"
    try:
        # 自动获取输入形状
        success = onnx_to_om(onnx_model, om_model, auto_input_shape=True)
        print(f"转换 {'成功' if success else '失败'}")
    except Exception as e:
        print(f"示例1失败: {str(e)}")
    
    # print("\n示例2: 手动指定输入形状（字符串格式）")
    # try:
    #     # 手动指定输入形状
    #     success = onnx_to_om(onnx_model, om_model, "1,3,640,640")
    #     print(f"转换 {'成功' if success else '失败'}")
    # except Exception as e:
    #     print(f"示例2失败: {str(e)}")
    
    # print("\n示例3: 禁用自动获取输入形状")
    # try:
    #     # 当禁用自动获取时，必须提供 input_shape 参数
    #     onnx_to_om(onnx_model, om_model, auto_input_shape=False)
    # except ValueError as e:
    #     print(f"预期的错误: {str(e)}")
    
    # print("\n示例4: 手动指定输入形状（元组格式）")
    # try:
    #     # 手动指定单输入形状（元组格式）
    #     # 系统会自动从ONNX模型获取输入名称
    #     input_shape_tuple = (1, 3, 640, 640)
    #     success = onnx_to_om(onnx_model, om_model, input_shape_tuple)
    #     print(f"转换 {'成功' if success else '失败'}")
    # except Exception as e:
    #     print(f"示例4失败: {str(e)}")
    
    # print("\n示例5: 显式指定输入名称和形状")
    # try:
    #     # 显式指定输入名称和形状，适用于需要精确控制的场景
    #     # 当您确切知道模型的输入名称时可以使用此方式
    #     success = onnx_to_om(onnx_model, om_model, "input_name:1,3,640,640")
    #     print(f"转换 {'成功' if success else '失败'}")
    # except Exception as e:
    #     print(f"示例5失败: {str(e)}")
    
    # 注意: 实际使用时，请确保替换为您自己的模型路径和正确的输入形状