#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/08/11
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 
# @Software: PyCharm
import magicmind.python.api as mm
import cv2
import numpy as np
import os


# 1. 加载MagicMind引擎配置
def create_config():
    config = mm.Config()
    # 设置推理精度（FP16/INT8）
    config.parse_from_string("precision_mode=force_float16")
    # 启用动态形状（适配不同输入尺寸）
    config.parse_from_string("dynamic_shape=true")
    return config


# 2. 转换ONNX模型为MagicMind模型
def build_model(onnx_model_path, mm_model_path, config):
    if not os.path.exists(mm_model_path):
        # 创建builder
        builder = mm.Builder()
        # 解析ONNX模型
        network = mm.Network()
        network.parse(mm.ModelFormat.ONNX, onnx_model_path)
        # 构建MagicMind模型
        model = builder.build_model("yolov5", network, config)
        # 保存模型
        model.save(mm_model_path)
        print(f"MagicMind model saved to {mm_model_path}")
    else:
        print(f"MagicMind model already exists: {mm_model_path}")


# 3. 图像预处理
def preprocess_image(image_path, input_size=(640, 640)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 缩放至输入尺寸
    img = cv2.resize(img, input_size)
    # 归一化处理
    img = img / 255.0
    # 调整维度为[batch, channel, height, width]
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img


# 4. 执行推理
def infer(mm_model_path, image_path):
    # 加载模型
    model = mm.Model()
    model.load(mm_model_path)
    # 创建推理引擎
    engine = model.create_engine()
    # 创建上下文
    context = engine.create_context()

    # 预处理图像
    input_data = preprocess_image(image_path)
    # 设置输入
    context.set_input(0, input_data)
    # 执行推理
    context.enqueue()
    # 获取输出
    output = context.get_output(0)
    return output


# 5. 后处理（解析检测结果）
def postprocess(output, origin_image_path, input_size=(640, 640), conf_threshold=0.5):
    origin_img = cv2.imread(origin_image_path)
    h, w = origin_img.shape[:2]
    # 解析输出（YOLOv5输出格式：[x1, y1, x2, y2, conf, class_id]）
    detections = output[0]
    for det in detections:
        if det[4] < conf_threshold:
            continue
        # 坐标映射回原图
        x1, y1, x2, y2 = det[:4]
        x1 = int(x1 * w / input_size[0])
        y1 = int(y1 * h / input_size[1])
        x2 = int(x2 * w / input_size[0])
        y2 = int(y2 * h / input_size[1])
        # 绘制检测框
        cv2.rectangle(origin_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(origin_img, f"Class: {int(det[5])} Conf: {det[4]:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite("result.jpg", origin_img)
    print("Detection result saved as result.jpg")


# 主函数
if __name__ == "__main__":
    onnx_model = "yolov5s.onnx"
    mm_model = "yolov5s_cambricon.mm"
    test_image = "test.jpg"

    # 构建模型
    config = create_config()
    build_model(onnx_model, mm_model, config)

    # 执行推理与后处理
    output = infer(mm_model, test_image)
    postprocess(output, test_image)