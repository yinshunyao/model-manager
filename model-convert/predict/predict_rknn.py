import logging
import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from predict.post_handle import postprocess_yolov8

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入RKNN Toolkit 2
try:
    from rknn.api import RKNN
    rknn_available = True
except ImportError:
    RKNN = None
    rknn_available = False
    logging.warning("警告：RKNN Toolkit 2 未安装，无法执行RKNN推理。")


class RK3588_Predictor:
    """
    瑞芯微RK3588设备上的模型推理类，支持目标检测和图像分类任务
    """
    
    def __init__(self, rknn_model_path, debug=False):
        """
        初始化函数，加载RKNN模型到内存
        
        Args:
            rknn_model_path (str): RKNN模型文件路径
            debug (bool): 是否开启调试模式，默认False
        """
        self.rknn_model_path = rknn_model_path
        self.rknn = None
        self.debug = debug
        
        # 验证模型文件存在
        if not os.path.exists(rknn_model_path):
            raise FileNotFoundError(f"RKNN模型文件不存在: {rknn_model_path}")
        
        # 加载模型
        self._load_model()
        
    def _load_model(self):
        """
        加载RKNN模型到内存（内部方法）
        """
        # 检查RKNN Toolkit 2是否安装
        if not rknn_available:
            raise ImportError("RKNN Toolkit 2 未安装，无法执行RKNN推理。")
        
        try:
            # 初始化RKNN对象
            self.rknn = RKNN(verbose=self.debug)
            
            # 加载RKNN模型
            logger.info(f"加载RKNN模型: {self.rknn_model_path}")
            ret = self.rknn.load_rknn(self.rknn_model_path)
            if ret != 0:
                raise RuntimeError(f"加载RKNN模型失败: {ret}")
            
            # 初始化运行时环境
            # 注意：在RK3588设备上运行时，target参数应为'rk3588'
            # 在PC上模拟运行时，target参数应为None或'rv1126'等
            ret = self.rknn.init_runtime(target='rk3588')
            if ret != 0:
                # 如果设备初始化失败，尝试使用模拟模式
                logger.warning(f"RK3588设备初始化失败，尝试使用模拟模式: {ret}")
                ret = self.rknn.init_runtime()
                if ret != 0:
                    raise RuntimeError(f"初始化RKNN运行时环境失败: {ret}")
            
            logger.info(f"RKNN模型加载成功: {self.rknn_model_path}")
            
        except Exception as e:
            logger.error(f"加载RKNN模型失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"加载RKNN模型失败: {str(e)}")
    
    def _preprocess_image(self, image, input_shape=(640, 640)):
        """
        图像预处理（内部方法）
        
        Args:
            image: 输入图像（cv2格式）
            input_shape: 模型输入尺寸
        
        Returns:
            np.ndarray: 预处理后的图像数据
        """
        # 复制图像以避免修改原图
        img = image.copy()
        
        # 调整图像大小
        img = cv2.resize(img, input_shape)
        
        # 转换为RGB（如果需要）
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        
        # 转换为模型输入格式 (N, C, H, W)
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        img = np.expand_dims(img, axis=0)   # (C, H, W) -> (N, C, H, W)
        
        return img
    
    def predict(self, image):
        """
        目标检测推理
        
        Args:
            image: 输入图像（cv2格式或文件路径）
        
        Returns:
            list: 检测结果列表，每个元素为 [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        # 检查输入类型
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise FileNotFoundError(f"无法读取图像文件: {image}")
        
        # 预处理图像
        input_data = self._preprocess_image(image)
        
        # 执行推理
        try:
            if self.rknn is None:
                raise RuntimeError("RKNN模型未正确加载")
            
            # 执行推理
            outputs = self.rknn.inference(inputs=[input_data])
            
            if not outputs or len(outputs) == 0:
                raise RuntimeError("推理输出为空")
            
            # 获取第一个输出
            output_data = outputs[0]
            
            # 将输出转换为numpy数组
            if not isinstance(output_data, np.ndarray):
                output_data = np.array(output_data)
            
            logger.info(f"推理完成，输出shape: {output_data.shape}")
            
            # 解析输出结果
            results = []
            
            # 展平输出数据以便统一处理
            original_shape = output_data.shape
            if len(output_data.shape) > 1:
                output_data_flat = output_data.flatten()
                logger.info(f"输出数据从 {original_shape} 展平为 {output_data_flat.shape}")
            else:
                output_data_flat = output_data
            
            # 尝试检测是否是YOLOv8格式
            from predict.post_handle import is_yolov8_format
            is_yolov8, num_boxes, num_classes = is_yolov8_format(output_data_flat)
            
            if is_yolov8:
                # 使用YOLOv8后处理
                logger.info(f"检测到YOLOv8格式，使用YOLOv8后处理: num_boxes={num_boxes}, num_classes={num_classes}")
                results = postprocess_yolov8(output_data_flat, image.shape[:2])
            else:
                # 尝试其他格式：假设是 [x1, y1, x2, y2, score, class_id, ...] 格式
                logger.info(f"未检测到YOLOv8格式，尝试通用格式解析")
                conf_threshold = 0.3  # 置信度阈值
                
                # 尝试每6个或7个值一组
                for stride in [6, 7]:
                    if len(output_data_flat) % stride == 0:
                        num_detections = len(output_data_flat) // stride
                        logger.info(f"尝试stride={stride}，检测框数量={num_detections}")
                        
                        for i in range(0, len(output_data_flat), stride):
                            if i + stride - 1 < len(output_data_flat):
                                if stride == 6:
                                    x1, y1, x2, y2, score, class_id = output_data_flat[i:i+6]
                                else:  # stride == 7
                                    x1, y1, x2, y2, score, class_id, _ = output_data_flat[i:i+7]
                                
                                # 过滤低置信度的检测结果
                                if score >= conf_threshold:
                                    results.append([float(x1), float(y1), float(x2), float(y2), float(score), int(class_id)])
                        
                        if len(results) > 0:
                            logger.info(f"使用stride={stride}解析成功，检测数={len(results)}")
                            break
                
                if len(results) == 0:
                    logger.warning(f"无法解析输出数据，shape={original_shape}, size={output_data_flat.size}")
                    # 如果无法解析，返回空结果
                    return []
            
            # 应用NMS处理
            results = self._apply_nms(results)
            
            # 添加类别名称
            for result in results:
                if len(result) < 7:  # 如果还没有添加类别名称
                    class_id = int(result[5])
                    class_names = ['person', 'car']  # 示例类别名称，实际使用时应该从配置文件或模型元数据中获取
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                    result.append(class_name)
            
            return results
            
        except Exception as e:
            logger.error(f"推理过程中出错: {str(e)}", exc_info=True)
            raise RuntimeError(f"推理过程中出错: {str(e)}")
    
    def predict_cls(self, image):
        """
        图像分类推理
        
        Args:
            image: 输入图像（cv2格式或文件路径）
        
        Returns:
            list: 分类结果列表，每个元素为 [class_name, confidence]
        """
        # 检查输入类型
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise FileNotFoundError(f"无法读取图像文件: {image}")
        
        # 预处理图像
        input_data = self._preprocess_image(image)
        
        # 执行推理
        try:
            if self.rknn is None:
                raise RuntimeError("RKNN模型未正确加载")
            
            # 执行推理
            outputs = self.rknn.inference(inputs=[input_data])
            
            if not outputs or len(outputs) == 0:
                raise RuntimeError("推理输出为空")
            
            # 获取第一个输出
            output_data = outputs[0]
            
            # 将输出转换为numpy数组
            if not isinstance(output_data, np.ndarray):
                output_data = np.array(output_data)
            
            logger.info(f"分类推理完成，输出shape: {output_data.shape}")
            
            # 解析分类结果
            results = []
            if len(output_data) > 0:
                # 如果输出是多维数组，展平
                if len(output_data.shape) > 1:
                    output_data = output_data.flatten()
                
                # 获取概率最高的两个类别索引
                top_indices = output_data.argsort()[-2:][::-1]
                
                # 这里使用简单的类别名称映射，实际应用中应替换为模型的真实类别列表
                class_names = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 
                              'class_5', 'class_6', 'class_7', 'class_8', 'class_9']
                
                for idx in top_indices:
                    if idx < len(class_names):
                        class_name = class_names[idx]
                    else:
                        class_name = f'class_{idx}'
                    confidence = float(output_data[idx])
                    results.append([class_name, confidence])
            
            return results[:20]  # 返回前20个结果
            
        except Exception as e:
            logger.error(f"分类推理过程中出错: {str(e)}", exc_info=True)
            raise RuntimeError(f"分类推理过程中出错: {str(e)}")
    
    def __del__(self):
        """
        析构函数，释放模型资源
        """
        if self.rknn is not None:
            try:
                self.rknn.release()
                logger.info("RKNN模型资源已释放")
            except Exception as e:
                logger.warning(f"释放RKNN模型资源时出错: {str(e)}")
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        计算两个边界框之间的IoU (Intersection over Union)
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            float: IoU值
        """
        # 计算交集坐标
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 计算交集面积
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 计算两个框的面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 计算并集面积
        union_area = box1_area + box2_area - intersection_area
        
        # 避免除零错误
        if union_area == 0:
            return 0
            
        # 返回IoU
        return intersection_area / union_area
    
    def _apply_nms(self, detections: List[List[float]], iou_threshold: float = 0.45) -> List[List[float]]:
        """
        对检测结果应用非极大值抑制(NMS)
        
        Args:
            detections: 检测结果列表，每个元素为 [x1, y1, x2, y2, confidence, class_id]
            iou_threshold: IoU阈值
            
        Returns:
            List[List[float]]: NMS后的检测结果
        """
        if len(detections) == 0:
            return detections
            
        # 按置信度降序排序
        sorted_indices = sorted(range(len(detections)), key=lambda i: detections[i][4], reverse=True)
        keep_indices = []
        
        while sorted_indices:
            # 保留置信度最高的检测框
            current_idx = sorted_indices.pop(0)
            keep_indices.append(current_idx)
            
            # 如果没有剩余的检测框，则退出循环
            if not sorted_indices:
                break
                
            # 计算当前检测框与剩余所有检测框的IoU
            current_box = detections[current_idx]
            remaining_indices = []
            
            for idx in sorted_indices:
                box = detections[idx]
                # 只对相同类别的检测框进行NMS
                if len(current_box) > 5 and len(box) > 5 and current_box[5] == box[5]:  # class_id相同
                    iou = self._calculate_iou(current_box[:4], box[:4])
                    # 如果IoU小于阈值，则保留该检测框
                    if iou < iou_threshold:
                        remaining_indices.append(idx)
                else:
                    # 不同类别的检测框直接保留
                    remaining_indices.append(idx)
            
            # 更新剩余检测框索引
            sorted_indices = remaining_indices
        
        # 返回NMS后的检测结果
        return [detections[i] for i in keep_indices]


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
    
    # 当前代码文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 示例：使用对象进行推理
    # 注意：这里使用示例模型路径，实际使用时请替换为真实的RKNN模型路径
    model_name = "yolo11n.rknn"
    rknn_model_path = os.path.join(current_dir, "..", "model_demo", model_name)
    
    # 图片路径
    test_image_path = os.path.join(current_dir, "..", "000000000009.jpg")
    
    try:
        # 创建预测器实例
        predictor = RK3588_Predictor(rknn_model_path, debug=True)
        
        # 从文件读取图像
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            raise FileNotFoundError(f"无法读取测试图像: {test_image_path}")
        
        # 执行目标检测推理
        print("\n执行目标检测推理...")
        detection_results = predictor.predict(test_image)
        print(f"检测结果: {detection_results}")
        
        # 执行分类推理
        print("\n执行图像分类推理...")
        classification_results = predictor.predict_cls(test_image)
        print(f"分类结果: {classification_results}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

