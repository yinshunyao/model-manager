import os
import sys
import json
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入要测试的模块
from predict_onnx import ONNXPredictor

# 确保核心依赖已正确导入
try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("错误: 未安装torch或ultralytics。请运行 'pip install -r requirements.txt' 安装所需依赖。")
    sys.exit(1)

try:
    from sklearn.metrics import average_precision_score
except ImportError:
    print("警告: 未安装scikit-learn，将使用简化的mAP计算")
    average_precision_score = None


class YOLOEvaluator:
    """
    YOLO模型评估器，支持PT模型和ONNX模型的精度和mAP计算
    """
    
    def __init__(self, dataset_path: str, dataset_config: str):
        """
        初始化评估器
        
        Args:
            dataset_path (str): 数据集根目录路径
            dataset_config (str): 数据集配置文件路径
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_config = dataset_config
        self.class_names = {}
        self.val_images = []
        self.val_labels = []
        
        # 加载数据集配置
        self._load_dataset_config()
        # 加载验证数据
        self._load_validation_data()
    
    def _load_dataset_config(self):
        """加载数据集配置文件"""
        try:
            with open(self.dataset_config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.class_names = config.get('names', {})
            print(f"数据集类别数量: {len(self.class_names)}")
            print(f"类别名称: {list(self.class_names.values())[:10]}...")  # 显示前10个类别
            
        except Exception as e:
            raise RuntimeError(f"加载数据集配置失败: {e}")
    
    def _load_validation_data(self):
        """加载验证数据"""
        try:
            # 构建验证集路径
            val_images_dir = self.dataset_path / "images" / "val"
            val_labels_dir = self.dataset_path / "labels" / "val"
            
            if not val_images_dir.exists():
                raise FileNotFoundError(f"验证图片目录不存在: {val_images_dir}")
            if not val_labels_dir.exists():
                raise FileNotFoundError(f"验证标签目录不存在: {val_labels_dir}")
            
            # 获取所有图片文件
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = []
            for ext in image_extensions:
                image_files.extend(val_images_dir.glob(f"*{ext}"))
                image_files.extend(val_images_dir.glob(f"*{ext.upper()}"))
            
            self.val_images = sorted(image_files)
            
            # 加载对应的标签文件
            self.val_labels = []
            for img_path in self.val_images:
                label_path = val_labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    self.val_labels.append(label_path)
                else:
                    self.val_labels.append(None)
            
            print(f"加载验证数据: {len(self.val_images)} 张图片")
            
        except Exception as e:
            raise RuntimeError(f"加载验证数据失败: {e}")
    
    def _parse_yolo_label(self, label_path: Path) -> List[Dict]:
        """
        解析YOLO格式的标签文件
        
        Args:
            label_path (Path): 标签文件路径
            
        Returns:
            List[Dict]: 标签列表，每个元素包含 {'class': class_id, 'bbox': [x_center, y_center, width, height]}
        """
        if label_path is None or not label_path.exists():
            return []
        
        labels = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        labels.append({
                            'class': class_id,
                            'bbox': [x_center, y_center, width, height]
                        })
        except Exception as e:
            print(f"解析标签文件失败 {label_path}: {e}")
        
        return labels
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        计算两个边界框的IoU
        
        Args:
            box1, box2: 边界框 [x1, y1, x2, y2]
            
        Returns:
            float: IoU值
        """
        # 计算交集区域
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # 计算并集区域
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _convert_yolo_to_xyxy(self, yolo_bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        将YOLO格式的边界框转换为xyxy格式
        
        Args:
            yolo_bbox: [x_center, y_center, width, height] (归一化坐标)
            img_width, img_height: 图片尺寸
            
        Returns:
            List[float]: [x1, y1, x2, y2] (像素坐标)
        """
        x_center, y_center, width, height = yolo_bbox
        
        # 转换为像素坐标
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # 计算xyxy坐标
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        return [x1, y1, x2, y2]
    
    def _save_visualization_result(self, image_path: str, predictions: list, gt_boxes: list, 
                                 output_dir: str, image_idx: int):
        """
        保存可视化结果图片
        
        Args:
            image_path (str): 原始图片路径
            predictions (list): 预测结果
            gt_boxes (list): 真实标签框
            output_dir (str): 输出目录
            image_idx (int): 图片索引
        """
        try:
            import cv2
            import numpy as np
            
            # 读取原始图片
            image = cv2.imread(image_path)
            if image is None:
                return
            
            img_height, img_width = image.shape[:2]
            
            # 定义颜色
            pred_color = (0, 255, 0)  # 绿色 - 预测框
            gt_color = (0, 0, 255)    # 红色 - 真实框
            
            # 绘制预测框
            for i, pred in enumerate(predictions):
                if isinstance(pred, dict) and 'bbox' in pred:
                    bbox = pred['bbox']
                    conf = pred.get('conf', 0)
                    class_name = pred.get('class_name', f"class_{pred.get('class', 0)}")
                    
                    # 转换坐标（归一化 -> 像素坐标）
                    x1 = int(bbox[0] * img_width)
                    y1 = int(bbox[1] * img_height)
                    x2 = int(bbox[2] * img_width)
                    y2 = int(bbox[3] * img_height)
                    
                    # 确保坐标在图片范围内
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(0, min(x2, img_width))
                    y2 = max(0, min(y2, img_height))
                    
                    # 绘制边界框
                    cv2.rectangle(image, (x1, y1), (x2, y2), pred_color, 2)
                    
                    # 绘制标签
                    label = f"{class_name}: {conf:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # 绘制标签背景
                    cv2.rectangle(
                        image, 
                        (x1, y1 - text_height - baseline), 
                        (x1 + text_width, y1), 
                        pred_color, 
                        -1
                    )
                    
                    # 绘制标签文本
                    cv2.putText(
                        image, 
                        label, 
                        (x1, y1 - baseline), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (255, 255, 255), 
                        2
                    )
            
            # 绘制真实框
            for i, gt_box in enumerate(gt_boxes):
                bbox = gt_box['bbox']
                class_id = gt_box['class']
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                
                # 绘制边界框（虚线效果）
                x1, y1, x2, y2 = bbox
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                
                # 绘制虚线框
                self._draw_dashed_rectangle(image, (x1, y1), (x2, y2), gt_color, 2)
                
                # 绘制标签
                label = f"GT: {class_name}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # 绘制标签背景
                cv2.rectangle(
                    image, 
                    (x1, y2), 
                    (x1 + text_width, y2 + text_height + baseline), 
                    gt_color, 
                    -1
                )
                
                # 绘制标签文本
                cv2.putText(
                    image, 
                    label, 
                    (x1, y2 + text_height), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1
                )
            
            # 添加图例
            legend_y = 30
            cv2.putText(image, "Green: Predictions", (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
            cv2.putText(image, "Red: Ground Truth", (10, legend_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, gt_color, 2)
            
            # 添加统计信息
            stats_text = f"Predictions: {len(predictions)}, GT: {len(gt_boxes)}"
            cv2.putText(image, stats_text, (10, legend_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 保存图片
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"debug_{image_idx:03d}_{base_name}.jpg")
            cv2.imwrite(output_path, image)
            
            if image_idx < 5:  # 只打印前几张的详细信息
                print(f"  保存可视化结果: {output_path}")
                
        except Exception as e:
            logging.error(f"保存可视化结果时出错: {e}")
    
    def _draw_dashed_rectangle(self, image, pt1, pt2, color, thickness):
        """
        绘制虚线矩形
        """
        x1, y1 = pt1
        x2, y2 = pt2
        
        # 绘制虚线效果
        dash_length = 10
        gap_length = 5
        
        # 上边
        for x in range(x1, x2, dash_length + gap_length):
            cv2.line(image, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        
        # 下边
        for x in range(x1, x2, dash_length + gap_length):
            cv2.line(image, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        # 左边
        for y in range(y1, y2, dash_length + gap_length):
            cv2.line(image, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        
        # 右边
        for y in range(y1, y2, dash_length + gap_length):
            cv2.line(image, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def evaluate_yolo_pt_model(self, model_path: str, conf_threshold: float = 0.5, 
                              iou_threshold: float = 0.45) -> Dict:
        """
        评估YOLO PT模型
        
        Args:
            model_path (str): PT模型路径
            conf_threshold (float): 置信度阈值
            iou_threshold (float): IoU阈值
            
        Returns:
            Dict: 评估结果
        """
        print(f"\n=== 开始评估YOLO PT模型 ===")
        print(f"模型路径: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PT模型文件不存在: {model_path}")
        
        # 加载模型
        model = YOLO(model_path)
        
        # 初始化评估指标
        total_predictions = 0
        total_ground_truth = 0
        correct_predictions = 0
        
        class_precisions = {}
        class_recalls = {}
        class_aps = {}
        
        # 对每张图片进行推理和评估
        for i, (img_path, label_path) in enumerate(zip(self.val_images, self.val_labels)):
            if i % 10 == 0:
                print(f"处理进度: {i}/{len(self.val_images)}")
            
            # 读取图片
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            img_height, img_width = image.shape[:2]
            
            # 进行推理
            results = model(img_path, conf=conf_threshold, iou=iou_threshold, verbose=False)
            
            # 解析预测结果
            predictions = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes
                    for j in range(len(boxes)):
                        conf = float(boxes.conf[j])
                        if conf >= conf_threshold:
                            # 获取边界框坐标 (xyxy格式)
                            x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy()
                            class_id = int(boxes.cls[j])
                            
                            predictions.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'conf': conf,
                                'class': class_id
                            })
            
            # 解析真实标签
            gt_labels = self._parse_yolo_label(label_path)
            gt_boxes = []
            for gt_label in gt_labels:
                bbox_xyxy = self._convert_yolo_to_xyxy(gt_label['bbox'], img_width, img_height)
                gt_boxes.append({
                    'bbox': bbox_xyxy,
                    'class': gt_label['class']
                })
            
            # 计算匹配
            matched_gt = set()
            for pred in predictions:
                pred_bbox = pred['bbox']
                pred_class = pred['class']
                
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    if gt_box['class'] != pred_class:
                        continue
                    
                    iou = self._calculate_iou(pred_bbox, gt_box['bbox'])
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx >= 0:
                    matched_gt.add(best_gt_idx)
                    correct_predictions += 1
            
            total_predictions += len(predictions)
            total_ground_truth += len(gt_boxes)
        
        # 计算整体指标
        precision = correct_predictions / total_predictions if total_predictions > 0 else 0
        recall = correct_predictions / total_ground_truth if total_ground_truth > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'model_type': 'YOLO_PT',
            'model_path': model_path,
            'total_predictions': total_predictions,
            'total_ground_truth': total_ground_truth,
            'correct_predictions': correct_predictions,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold
        }
        
        print(f"PT模型评估完成:")
        print(f"  总预测数: {total_predictions}")
        print(f"  总真实标签数: {total_ground_truth}")
        print(f"  正确预测数: {correct_predictions}")
        print(f"  精度: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1_score:.4f}")
        
        return results
    
    def evaluate_onnx_model(self, model_path: str, conf_threshold: float = 0.5, 
                           iou_threshold: float = 0.45, save_visualizations: bool = False,
                           output_dir: str = "onnx_debug_output") -> Dict:
        """
        评估ONNX模型
        
        Args:
            model_path (str): ONNX模型路径
            conf_threshold (float): 置信度阈值
            iou_threshold (float): IoU阈值
            save_visualizations (bool): 是否保存可视化图片
            output_dir (str): 可视化输出目录
            
        Returns:
            Dict: 评估结果
        """
        print(f"\n=== 开始评估ONNX模型 ===")
        print(f"模型路径: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX模型文件不存在: {model_path}")
        
        # 创建ONNX推理器
        predictor = ONNXPredictor(model_path)
        
        # 检查是否成功加载了类别名称
        if predictor.class_names:
            print(f"ONNX模型类别名称已加载: {len(predictor.class_names)} 个类别")
        else:
            print("警告: ONNX模型未包含类别名称信息")
        
        # 创建输出目录（如果需要保存可视化）
        if save_visualizations:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print(f"可视化结果将保存到: {output_dir}")
        
        # 初始化评估指标
        total_predictions = 0
        total_ground_truth = 0
        correct_predictions = 0
        
        # 对每张图片进行推理和评估
        for i, (img_path, label_path) in enumerate(zip(self.val_images, self.val_labels)):
            if i % 10 == 0:
                print(f"处理进度: {i}/{len(self.val_images)}")
            
            try:
                # 进行推理
                predictions = predictor.predict(
                    str(img_path), 
                    conf_threshold=conf_threshold, 
                    iou_threshold=iou_threshold
                )
                
                # 确保predictions是列表格式
                if not isinstance(predictions, list):
                    predictions = []
                
                # 读取图片获取尺寸
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                img_height, img_width = image.shape[:2]
                
                # 解析真实标签
                gt_labels = self._parse_yolo_label(label_path)
                gt_boxes = []
                for gt_label in gt_labels:
                    bbox_xyxy = self._convert_yolo_to_xyxy(gt_label['bbox'], img_width, img_height)
                    gt_boxes.append({
                        'bbox': bbox_xyxy,
                        'class': gt_label['class']
                    })
                
                # 保存可视化结果（如果启用）
                if save_visualizations and len(predictions) > 0:
                    self._save_visualization_result(
                        str(img_path), predictions, gt_boxes, output_dir, i
                    )
                
                # 计算匹配
                matched_gt = set()
                for pred in predictions:
                    if isinstance(pred, dict) and 'bbox' in pred and 'class' in pred:
                        pred_bbox = pred['bbox']
                        pred_class = pred['class']
                        pred_class_name = pred.get('class_name', f'class_{pred_class}')
                        
                        best_iou = 0
                        best_gt_idx = -1
                        
                        for gt_idx, gt_box in enumerate(gt_boxes):
                            if gt_idx in matched_gt:
                                continue
                            if gt_box['class'] != pred_class:
                                continue
                            
                            iou = self._calculate_iou(pred_bbox, gt_box['bbox'])
                            if iou > best_iou and iou >= iou_threshold:
                                best_iou = iou
                                best_gt_idx = gt_idx
                        
                        if best_gt_idx >= 0:
                            matched_gt.add(best_gt_idx)
                            correct_predictions += 1
                
                total_predictions += len(predictions)
                total_ground_truth += len(gt_boxes)
                
            except Exception as e:
                logging.error(f"处理图片 {img_path} 时出错: {e}", exc_info=True)
                continue
        
        # 计算整体指标
        precision = correct_predictions / total_predictions if total_predictions > 0 else 0
        recall = correct_predictions / total_ground_truth if total_ground_truth > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'model_type': 'ONNX',
            'model_path': model_path,
            'total_predictions': total_predictions,
            'total_ground_truth': total_ground_truth,
            'correct_predictions': correct_predictions,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold
        }
        
        print(f"ONNX模型评估完成:")
        print(f"  总预测数: {total_predictions}")
        print(f"  总真实标签数: {total_ground_truth}")
        print(f"  正确预测数: {correct_predictions}")
        print(f"  精度: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1_score:.4f}")
        
        return results
    
    def compare_models(self, pt_results: Dict, onnx_results: Dict) -> Dict:
        """
        比较PT模型和ONNX模型的性能
        
        Args:
            pt_results (Dict): PT模型评估结果
            onnx_results (Dict): ONNX模型评估结果
            
        Returns:
            Dict: 比较结果
        """
        comparison = {
            'precision_diff': pt_results['precision'] - onnx_results['precision'],
            'recall_diff': pt_results['recall'] - onnx_results['recall'],
            'f1_score_diff': pt_results['f1_score'] - onnx_results['f1_score'],
            'predictions_diff': pt_results['total_predictions'] - onnx_results['total_predictions'],
            'pt_model': pt_results,
            'onnx_model': onnx_results
        }
        
        print(f"\n=== 模型性能比较 ===")
        print(f"精度差异 (PT - ONNX): {comparison['precision_diff']:.4f}")
        print(f"召回率差异 (PT - ONNX): {comparison['recall_diff']:.4f}")
        print(f"F1分数差异 (PT - ONNX): {comparison['f1_score_diff']:.4f}")
        print(f"预测数量差异 (PT - ONNX): {comparison['predictions_diff']}")
        
        return comparison


def main():
    """
    主函数：运行PT模型和ONNX模型的评估测试
    """
    import argparse
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='YOLO模型评估测试')
    parser.add_argument('--enable-visualization', action='store_true', 
                       help='启用可视化调试，生成带标注的图片')
    parser.add_argument('--output-dir', default='onnx_debug_output',
                       help='可视化输出目录 (默认: onnx_debug_output)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='置信度阈值 (默认: 0.5)')
    parser.add_argument('--test-onnx-only', action='store_true',
                       help='只测试ONNX模型，跳过PT模型')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 设置路径
    current_dir = Path(__file__).parent.parent
    dataset_path = current_dir / "dataset" / "coco8" / "datasets" / "coco8"
    dataset_config = current_dir / "dataset" / "coco8" / "coco8.yaml"
    
    pt_model_path = current_dir / "model_demo" / "yolo11n.pt"
    onnx_model_path = current_dir / "model_demo" / "out" / "yolo11n.onnx"
    
    print("=== YOLO模型评估测试 ===")
    print(f"数据集路径: {dataset_path}")
    print(f"数据集配置: {dataset_config}")
    print(f"PT模型路径: {pt_model_path}")
    print(f"ONNX模型路径: {onnx_model_path}")
    print(f"可视化调试: {'启用' if args.enable_visualization else '禁用'}")
    if args.enable_visualization:
        print(f"输出目录: {args.output_dir}")
    
    # 检查文件是否存在
    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return
    
    if not dataset_config.exists():
        print(f"错误: 数据集配置文件不存在: {dataset_config}")
        return
    
    if not args.test_onnx_only and not pt_model_path.exists():
        print(f"错误: PT模型文件不存在: {pt_model_path}")
        return
    
    if not onnx_model_path.exists():
        print(f"错误: ONNX模型文件不存在: {onnx_model_path}")
        return
    
    try:
        # 创建评估器
        evaluator = YOLOEvaluator(str(dataset_path), str(dataset_config))
        
        pt_results = None
        onnx_results = None
        
        # 评估PT模型（除非只测试ONNX）
        if not args.test_onnx_only:
            print("\n开始评估PT模型...")
            pt_results = evaluator.evaluate_yolo_pt_model(str(pt_model_path))
        
        # 评估ONNX模型
        print(f"\n开始评估ONNX模型...")
        onnx_results = evaluator.evaluate_onnx_model(
            str(onnx_model_path),
            conf_threshold=args.conf_threshold,
            save_visualizations=args.enable_visualization,
            output_dir=args.output_dir
        )
        
        # 比较模型性能（如果有PT结果）
        if pt_results is not None:
            comparison = evaluator.compare_models(pt_results, onnx_results)
        else:
            comparison = None
            print("\n跳过模型性能比较（仅测试ONNX模型）")
        
        # 保存结果到文件
        results_file = current_dir / "test" / "evaluation_results.json"
        results_data = {
            'onnx_results': onnx_results,
            'comparison': comparison
        }
        
        if pt_results is not None:
            results_data['pt_results'] = pt_results
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估结果已保存到: {results_file}")
        
        if args.enable_visualization:
            print(f"\n可视化结果已保存到: {args.output_dir}")
            print("查看生成的图片文件来调试ONNX推理结果")
        
    except Exception as e:
        print(f"评估过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
