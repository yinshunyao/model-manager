import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

BUCKET_ONNX = "onnx-file"
BUCKET_MODEL = "model-file"
BUCKET_SOURCE = "source-file"
BUCKET_TARGET = "target-file"
BUCKET_ENGINE = "engine-file"

# COCO类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def draw_detection_results(
    image_path: str, 
    results: List[List], 
    output_path: Optional[str] = None,
    conf_threshold: float = 0.5,
    font_scale: float = 0.6,
    thickness: int = 2
) -> Optional[str]:
    """
    在图片上绘制检测结果
    
    Args:
        image_path (str): 原始图片路径
        results (list): 检测结果列表，每个元素为 [x1, y1, x2, y2, confidence, class_id, class_name] 或 [x1, y1, x2, y2, confidence, class_id]
        output_path (str): 输出图片路径，如果不指定则自动生成
        conf_threshold (float): 置信度阈值
        font_scale (float): 字体大小
        thickness (int): 线条粗细
        
    Returns:
        str: 输出图片路径，如果出错则返回None
    """
    try:
        # 读取原始图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 获取图片尺寸
        img_height, img_width = image.shape[:2]
        
        # 定义颜色列表（BGR格式）
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 洋红
            (0, 255, 255),  # 黄色
            (128, 0, 128),  # 紫色
            (255, 165, 0),  # 橙色
            (0, 128, 128),  # 橄榄色
            (128, 128, 0),  # 青绿色
        ]
        
        # 绘制检测框和标签
        for i, result in enumerate(results):
            # 解析检测结果
            if len(result) >= 6:
                x1, y1, x2, y2 = map(int, result[:4])
                conf = float(result[4])
                class_id = int(result[5])
                # 如果有class_name则使用，否则使用class_id作为名称
                class_name = result[6] if len(result) > 6 else f"class_{class_id}"
            else:
                continue  # 跳过无效的结果
            
            # 过滤低置信度检测
            if conf < conf_threshold:
                continue
            
            # 确保坐标在图片范围内
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            
            # 选择颜色
            color = colors[class_id % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # 准备标签文本
            label = f"{class_name}: {conf:.2f}"
            
            # 获取文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # 绘制标签背景
            cv2.rectangle(
                image, 
                (x1, y1 - text_height - baseline), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            
            # 绘制标签文本
            cv2.putText(
                image, 
                label, 
                (x1, y1 - baseline), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (255, 255, 255), 
                thickness
            )
        
        # 生成输出路径
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.dirname(image_path)
            output_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
        
        # 保存结果图片
        cv2.imwrite(output_path, image)
        
        print(f"检测结果图片已保存: {output_path}")
        print(f"绘制了 {len([r for r in results if len(r) >= 5 and float(r[4]) >= conf_threshold])} 个检测框")
        
        return output_path
        
    except ImportError:
        print("错误: 需要安装opencv-python来绘制检测结果")
        print("请运行: pip install opencv-python")
        return None
    except Exception as e:
        print(f"绘制检测结果时出错: {e}")
        return None