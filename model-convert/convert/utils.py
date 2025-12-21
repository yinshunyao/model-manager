#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/14
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 
# @Software: PyCharm
import logging
from pydantic import BaseModel
from typing import Dict, Optional, Union, Tuple, Any, List
import onnx
import json



class ParamsOnnx(BaseModel):
    task: Optional[str] = "detect"
    version: Optional[str] = ""
    batch: Optional[int] = 1
    channels: Optional[int] = 0
    imgsz: Optional[List[int]] = [640, 640]
    names: Optional[Dict[int, str]] = None
    class_names: Optional[Dict[str, Any]] = None
    stride: Optional[int] = 32
    input: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None

    @property
    def atc_params(self):
        # --input_shape="input_ids:1,128;attention_mask:1,128;token_type_ids:1,128"
        # --input_shape="images:1,3,640,640;scale_factor:1,2"
        input_params =";".join([f"{k}:{','.join(str(i) for i in v)}"for k, v in self.input.items()])
        if self.task in ("detect", "segment"):
            return [
                "--input_format=NCHW",
                "--output_type=FP32",
                f'--input_shape={input_params}',
                # "--output=output.1",
            ]
        elif self.task == "classify":
            return [
                "--input_format=NCHW",
                "--output_type=FP32",
                f'--input_shape="input:{input_params}"',
                # "--output=output.1",
            ]

        raise NotImplementedError(f"{self.task} not supported")


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

        ## key: "imgsz"  value: "[640, 640]"
        imgsz = None
        for i, item in enumerate(model.metadata_props):
            # print(i, item)
            if item.key == "imgsz":
                imgsz = item.value
                imgsz = json.loads(imgsz)
                break

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
                    # shape.append(1)
                    continue
                else:
                    # 未知维度，使用默认值 1
                    # shape.append(1)
                    continue

            if len(shape) < 3:
                if imgsz:
                    shape.append(int(imgsz[0]))
                    shape.append(int(imgsz[1]))
                else:
                    logging.error(f"get shape error")
                    raise ValueError("无法从 ONNX 模型中解析出有效的输入形状")

            input_shapes[input_name] = tuple(shape)

        if not input_shapes:
            raise ValueError("无法从 ONNX 模型中解析出有效的输入形状")

        return input_shapes

    except (FileNotFoundError, IOError):
        raise FileNotFoundError(f"ONNX 模型文件不存在或无法读取: {onnx_model_path}")
    except Exception as e:
        raise ValueError(f"从 ONNX 模型获取输入形状时出错: {str(e)}")


def get_params_onnx(onnx_model_path: str) -> ParamsOnnx:
    """

    :param onnx_model_path:
    :return:
    """
    try:
        # 加载 ONNX 模型
        model = onnx.load(onnx_model_path)
        params = ParamsOnnx()
        for i, item in enumerate(model.metadata_props):
            # print(i, item)
            if item.key == "imgsz":
                params.imgsz = json.loads(item.value)
            elif item.key == "version":
                params.version = item.value
                logging.warning(f"模型版本: {item.value}")
            elif item.key == "task":
                params.task  = item.value
            elif item.key == "batch":
                params.batch = item.value
            elif item.key == "names":
                params.names = item.value # json.loads(item.value)
            elif item.key == "channels":
                params.channels = int(item.value)
            elif item.key == "stride":
                params.stride = int(item.value)
            elif item.key == "class_names":
                params.class_names = item.value

        input_shapes = { }
        # v7 float32[batch,3,height,width]
        # v8 float32[1,3,640,640]

        # if params.channels == 0:
        for input_node in model.graph.input:
            # 获取输入名称
            input_name = input_node.name

            # 获取输入形状
            shape = []
            for dim in input_node.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    # 固定维度
                    shape.append(int(dim.dim_value))
                elif dim.dim_param:
                    if dim.dim_param in params:
                        shape.append(params[dim.dim_param])
                    elif dim.dim_param == "height":
                        shape.append(params.imgsz[0])
                    elif dim.dim_param == "width":
                        shape.append(params.imgsz[1])
                    else:
                        logging.error(f"get shape error:{dim.dim_param}")
                        shape.append(1)
                else:
                    # 未知维度，使用默认值 1
                    shape.append(1)
                    # 参数化维度，这里我们使用默认值 1
                    # 注意：这可能不适用于所有模型，特别是对于动态形状

            # v8 float32[1,3,640,640]
            if input_name == "images":
                # v8 float32[1,3,640,640]
                params.channels = shape[1]
            input_shapes[input_name] = tuple(shape)
        # model.graph.input
        logging.warning(f"input_shapes:{input_shapes}")
        params.input = input_shapes

        output_params = {}
        for output_node in model.graph.output:
            # 获取输入名称
            output_name = output_node.name

            # 获取输入形状
            shape = []
            for dim in output_node.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    # 固定维度
                    shape.append(int(dim.dim_value))
                elif dim.dim_param:
                    if dim.dim_param in params:
                        shape.append(params[dim.dim_param])
                    else:
                        logging.error(f"get shape error:{dim.dim_param}")
                        shape.append(1)
                else:
                    # 未知维度，使用默认值 1
                    shape.append(1)
                    # 参数化维度，这里我们使用默认值 1
                    # 注意：这可能不适用于所有模型，特别是对于动态形状

            # v7 float32[batch,Concatoutput0_dim_1,anchors]
            # v8  float32[1,84,8400]
            # if input_name == "images":
            #     # v8 float32[1,3,640,640]
            #     params.channels = shape[1]
            output_params[output_name] = tuple(shape)

        params.output = output_params
        logging.warning(f"输出信息:{output_params}")
    except (FileNotFoundError, IOError):
        raise FileNotFoundError(f"ONNX 模型文件不存在或无法读取: {onnx_model_path}")
    except Exception as e:
        logging.error(f"从 ONNX 模型获取输入形状时出错: {str(e)}", exc_info=True)
        raise ValueError(f"从 ONNX 模型获取输入形状时出错: {str(e)}")

    return params

if __name__ == '__main__':
    dy = "/Users/shunyaoyin/Documents/code/other/model-manager/model-convert-data/model_demo/dy-9-yolo11n.onnx"
    ysy = "/Users/shunyaoyin/Documents/code/other/model-manager/model-convert-data/model_demo/yolo11n.onnx"
    # shapes = get_input_shape_from_onnx(dy)
    # print(shapes)
    # shapes = get_input_shape_from_onnx("/Users/shunyaoyin/Documents/code/other/model-manager/model-convert-data/model_demo/yolo11n.onnx")
    # print(shapes)
    params = get_params_onnx(dy)
    print(params)
    print(params.atc_params)
    params = get_params_onnx(ysy)
    print(params)
    print(params.atc_params)