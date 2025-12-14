#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/14
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 
# @Software: PyCharm
import logging

import numpy as np


def postprocess_yolov8(output_data, img_shape, conf_thres=0.25, iou_thres=0.7):
    """
    output_data: np.ndarray, shape (705600,)
    img_shape: (H, W) of original image
    """
    logging.warning(f"img_shape:{img_shape}")
    # Step 1: reshape to [84, 8400]
    output = output_data.reshape(84, 8400)

    # Step 2: transpose to [8400, 84]
    output = output.T  # now (8400, 84)

    # Step 3: split box and class confs
    boxes = output[:, :4]  # [x, y, w, h] (center + wh, normalized)
    class_confs = output[:, 4:]  # (8400, 80)

    # Step 4: get max class conf and id
    max_conf = np.max(class_confs, axis=1)  # (8400,)
    class_ids = np.argmax(class_confs, axis=1)  # (8400,)

    # Step 5: filter by confidence
    valid_mask = max_conf >= conf_thres
    boxes = boxes[valid_mask]
    scores = max_conf[valid_mask]
    class_ids = class_ids[valid_mask]

    if len(boxes) == 0:
        return []

    # Step 6: convert [x, y, w, h] (normalized center+wh) to [x1, y1, x2, y2] (pixel)
    img_h, img_w = img_shape
    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2

    # Step 7: apply NMS (you need torchvision or custom NMS)
    try:
        import torch
        import torchvision
        boxes_t = torch.tensor(np.stack([x1, y1, x2, y2], axis=1), dtype=torch.float32)
        scores_t = torch.tensor(scores, dtype=torch.float32)
        keep = torchvision.ops.nms(boxes_t, scores_t, iou_thres)
        keep = keep.numpy()
    except Exception as e:
        print("Warning: NMS failed, skipping:", e)
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