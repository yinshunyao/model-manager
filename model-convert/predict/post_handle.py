#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/14
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 
# @Software: PyCharm
import logging

import numpy as np


def is_yolov8_format(output_data):
    """
    检测输出数据是否是YOLOv8格式
    
    YOLOv8输出格式特征：
    - 展平后的形状应该是 (num_boxes * (4 + num_classes),)
    - 常见的YOLOv8输出：8400个检测框，每个84维（4个bbox坐标 + 80个类别）
    - 所以 705600 = 8400 * 84
    
    Args:
        output_data: np.ndarray，输出数据
        
    Returns:
        tuple: (is_yolov8, num_boxes, num_classes) 或 (False, None, None)
    """
    if not isinstance(output_data, np.ndarray):
        output_data = np.array(output_data)
    
    # 确保是一维数组
    if len(output_data.shape) > 1:
        output_data = output_data.flatten()
    
    total_size = output_data.size
    
    # 尝试常见的YOLOv8配置
    # 常见的检测框数量：8400 (80*80 + 40*40 + 20*20), 25200 (更密集的检测)
    # 常见的类别数：80 (COCO), 20 (VOC), 或其他
    common_num_boxes = [8400, 25200, 16800, 12600, 6300]
    common_num_classes = [80, 20, 1, 10, 100]
    
    for num_boxes in common_num_boxes:
        for num_classes in common_num_classes:
            expected_size = num_boxes * (4 + num_classes)
            if total_size == expected_size:
                logging.info(f"检测到YOLOv8格式: {total_size} = {num_boxes} * {4 + num_classes}")
                return True, num_boxes, num_classes
    
    # 如果无法匹配常见配置，尝试自动推断
    # 假设至少有4个bbox坐标，然后推断类别数
    if total_size >= 4:
        # 尝试推断：假设是 (num_boxes, 4 + num_classes) 格式
        # 从4开始尝试，因为至少需要4个坐标
        for num_classes in range(1, 1000):
            num_boxes = total_size // (4 + num_classes)
            if num_boxes * (4 + num_classes) == total_size and num_boxes > 0:
                # 检查是否合理的检测框数量（通常在1000-50000之间）
                if 100 <= num_boxes <= 50000:
                    logging.info(f"自动推断YOLOv8格式: {total_size} = {num_boxes} * {4 + num_classes}")
                    return True, num_boxes, num_classes
    
    return False, None, None


def postprocess_yolov8(output_data, img_shape, conf_thres=0.25, iou_thres=0.7, num_boxes=None, num_classes=None, model_input_shape=(640, 640)):
    """
    处理YOLOv8格式的输出数据
    
    Args:
        output_data: np.ndarray，输出数据，可以是任意形状，会自动展平
        img_shape: (H, W) of original image，原始图像尺寸
        conf_thres: 置信度阈值
        iou_thres: IoU阈值
        num_boxes: 检测框数量，如果为None则自动推断
        num_classes: 类别数量，如果为None则自动推断
        model_input_shape: (W, H) 模型输入尺寸，默认(640, 640)，用于坐标转换
    
    Returns:
        list: 检测结果列表，每个元素为 [x1, y1, x2, y2, confidence, class_id]
    """
    if not isinstance(output_data, np.ndarray):
        output_data = np.array(output_data)
    
    # 确保是一维数组
    original_shape = output_data.shape
    if len(output_data.shape) > 1:
        output_data = output_data.flatten()
        logging.info(f"输出数据从 {original_shape} 展平为 {output_data.shape}")
    
    total_size = output_data.size
    
    # 自动推断YOLOv8格式参数
    if num_boxes is None or num_classes is None:
        is_yolov8, inferred_num_boxes, inferred_num_classes = is_yolov8_format(output_data)
        if is_yolov8:
            num_boxes = inferred_num_boxes if num_boxes is None else num_boxes
            num_classes = inferred_num_classes if num_classes is None else num_classes
        else:
            # 如果无法推断，尝试使用默认值
            logging.warning(f"无法自动推断YOLOv8格式，使用默认值: num_boxes=8400, num_classes=80")
            num_boxes = num_boxes if num_boxes is not None else 8400
            num_classes = num_classes if num_classes is not None else 80
    
    # 验证数据大小
    expected_size = num_boxes * (4 + num_classes)
    if total_size != expected_size:
        raise ValueError(
            f"输出数据大小不匹配: 期望 {expected_size} (num_boxes={num_boxes}, num_classes={num_classes}), "
            f"实际 {total_size}"
        )
    
    logging.info(f"处理YOLOv8输出: num_boxes={num_boxes}, num_classes={num_classes}, img_shape={img_shape}, model_input_shape={model_input_shape}")
    
    # Step 1: reshape to [4+num_classes, num_boxes]
    output = output_data.reshape(4 + num_classes, num_boxes)
    
    # Step 2: transpose to [num_boxes, 4+num_classes]
    output = output.T  # now (num_boxes, 4+num_classes)
    
    # Step 3: split box and class confs
    boxes = output[:, :4]  # [x, y, w, h] (center + wh, normalized)
    class_confs = output[:, 4:]  # (num_boxes, num_classes)
    
    # 调试：检查坐标范围
    if len(boxes) > 0:
        sample_box = boxes[0]
        logging.debug(f"样本坐标值: x={sample_box[0]:.6f}, y={sample_box[1]:.6f}, w={sample_box[2]:.6f}, h={sample_box[3]:.6f}")
    
    # Step 4: get max class conf and id
    max_conf = np.max(class_confs, axis=1)  # (num_boxes,)
    class_ids = np.argmax(class_confs, axis=1)  # (num_boxes,)
    
    # Step 5: filter by confidence
    valid_mask = max_conf >= conf_thres
    boxes = boxes[valid_mask]
    scores = max_conf[valid_mask]
    class_ids = class_ids[valid_mask]
    
    if len(boxes) == 0:
        return []
    
    # Step 6: convert [x, y, w, h] (normalized center+wh) to [x1, y1, x2, y2] (pixel)
    # YOLOv8输出的坐标是归一化的（0-1之间），基于模型输入尺寸
    # 需要先转换为模型输入尺寸的像素坐标，然后缩放到原始图像尺寸
    
    # 原始图像尺寸 (H, W)
    orig_h, orig_w = img_shape
    # 模型输入尺寸 (W, H) - cv2.resize使用的是(width, height)格式
    model_w, model_h = model_input_shape
    
    # 归一化坐标（0-1之间）
    x_center_norm = boxes[:, 0]  # 归一化的中心x坐标
    y_center_norm = boxes[:, 1]  # 归一化的中心y坐标
    w_norm = boxes[:, 2]          # 归一化的宽度
    h_norm = boxes[:, 3]          # 归一化的高度
    
    # 检查坐标范围，判断是否是归一化坐标
    max_val = max(np.max(x_center_norm), np.max(y_center_norm), np.max(w_norm), np.max(h_norm))
    min_val = min(np.min(x_center_norm), np.min(y_center_norm), np.min(w_norm), np.min(h_norm))
    
    # 如果坐标值在0-1之间，认为是归一化坐标；否则认为是像素坐标
    if max_val <= 1.0 and min_val >= 0.0:
        # 归一化坐标，转换为模型输入尺寸的像素坐标
        logging.debug("检测到归一化坐标，进行转换")
        x_center = x_center_norm * model_w
        y_center = y_center_norm * model_h
        w = w_norm * model_w
        h = h_norm * model_h
    else:
        # 已经是像素坐标，直接使用（可能是基于模型输入尺寸的像素坐标）
        logging.warning(f"检测到非归一化坐标，最大值: {max_val:.2f}，最小值: {min_val:.2f}，假设已经是模型输入尺寸的像素坐标")
        x_center = x_center_norm
        y_center = y_center_norm
        w = w_norm
        h = h_norm
    
    # 转换为边界框坐标 (x1, y1, x2, y2)
    x1_model = x_center - w / 2
    y1_model = y_center - h / 2
    x2_model = x_center + w / 2
    y2_model = y_center + h / 2
    
    # 计算缩放比例（从模型输入尺寸缩放到原始图像尺寸）
    scale_x = orig_w / model_w
    scale_y = orig_h / model_h
    
    # 缩放到原始图像尺寸
    x1 = x1_model * scale_x
    y1 = y1_model * scale_y
    x2 = x2_model * scale_x
    y2 = y2_model * scale_y
    
    # 裁剪坐标到图像范围内
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)
    
    # Step 7: apply NMS (you need torchvision or custom NMS)
    try:
        import torch
        import torchvision
        boxes_t = torch.tensor(np.stack([x1, y1, x2, y2], axis=1), dtype=torch.float32)
        scores_t = torch.tensor(scores, dtype=torch.float32)
        keep = torchvision.ops.nms(boxes_t, scores_t, iou_thres)
        keep = keep.numpy()
    except Exception as e:
        logging.warning(f"NMS failed, skipping: {e}")
        keep = np.arange(len(x1))
    
    # Step 8: build results
    results = []
    for i in keep:
        results.append([
            int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]),
            float(scores[i]),
            int(class_ids[i])
        ])
    
    return results