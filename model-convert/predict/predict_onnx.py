import logging
import os
import sys
import numpy as np
import cv2
from typing import Union, Tuple, List, Optional

# 确保核心依赖已正确导入
try:
    import onnxruntime as ort
except ImportError:
    print("错误: 未安装onnxruntime。请运行 'pip install onnxruntime' 安装所需依赖。")
    sys.exit(1)


class ONNXPredictor:
    """
    ONNX模型推理类，支持加载ONNX模型并对图片进行推理
    """
    
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        """
        初始化ONNX推理器
        
        Args:
            model_path (str): ONNX模型文件路径
            providers (List[str], optional): 推理提供者列表，默认为['CPUExecutionProvider']
        """
        self.model_path = model_path
        self.providers = providers or ['CPUExecutionProvider']
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None
        self.class_names = {}
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """
        加载ONNX模型并初始化推理会话
        """
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        try:
            print(f"正在加载ONNX模型: {self.model_path}")
            
            # 创建推理会话
            self.session = ort.InferenceSession(
                self.model_path,
                providers=self.providers
            )
            
            # 获取输入输出信息
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.input_shape = self.session.get_inputs()[0].shape
            
            # 从 json 文件中获取 names
            self._load_class_names()
            
            print(f"模型加载成功!")
            print(f"输入名称: {self.input_name}")
            print(f"输入形状: {self.input_shape}")
            print(f"输出名称: {self.output_names}")
            print(f"可用提供者: {self.session.get_providers()}")
            
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    def _load_class_names(self):
        """
        从ONNX模型元数据中加载类别名称
        """
        # 优先尝试从同名JSON侧车文件读取（model.onnx -> model.names.json）
        try:
            import json
            base, _ = os.path.splitext(self.model_path)
            names_path = f"{base}.names.json"
            if os.path.exists(names_path):
                with open(names_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                names = data.get("names", {})
                # JSON键为字符串，转换为整型键
                self.class_names = {int(k): v for k, v in names.items()}
                print(f"从文件加载类别名称: {len(self.class_names)} 个类别")
                print(f"类别示例: {list(self.class_names.values())[:10]}")
                return
        except Exception as e:
            print(f"从侧车文件读取类别名称失败，尝试从元数据读取: {e}")

        # 回退：从ONNX元数据读取
        try:
            import onnx
            import json
            model = onnx.load(self.model_path)
            if model.metadata_props:
                for prop in model.metadata_props:
                    if prop.key == "class_names":
                        names_data = json.loads(prop.value)
                        if "names" in names_data:
                            names = names_data["names"]
                            self.class_names = {int(k): v for k, v in names.items()}
                            print(f"从模型元数据加载类别名称: {len(self.class_names)} 个类别")
                            print(f"类别示例: {list(self.class_names.values())[:10]}")
                            return
            print("模型元数据中未找到类别名称信息")
        except ImportError:
            print("警告: 未安装onnx库，无法读取类别名称")
        except Exception as e:
            print(f"读取类别名称时出错: {e}")
    
    def preprocess_image(self, image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        预处理图片，将其转换为模型输入格式
        
        Args:
            image_path (str): 图片路径
            target_size (Tuple[int, int], optional): 目标尺寸，如果不指定则使用模型默认尺寸
            
        Returns:
            np.ndarray: 预处理后的图片数组
        """
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 获取目标尺寸
        if target_size is None:
            # 从模型输入形状推断尺寸 (通常是 [batch, channels, height, width])
            if self.input_shape is not None and len(self.input_shape) >= 4:
                target_size = (self.input_shape[2], self.input_shape[3])  # (height, width)
            else:
                target_size = (640, 640)  # 默认尺寸
        
        # 调整图片尺寸
        resized_image = cv2.resize(image, target_size)
        
        # 转换颜色空间 (BGR -> RGB)
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # 归一化到 [0, 1]
        normalized_image = rgb_image.astype(np.float32) / 255.0
        
        # 转换为模型输入格式 (H, W, C) -> (C, H, W)
        transposed_image = np.transpose(normalized_image, (2, 0, 1))
        
        # 添加batch维度 (C, H, W) -> (1, C, H, W)
        input_array = np.expand_dims(transposed_image, axis=0)
        
        return input_array
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        计算两个边界框的交并比(IoU)
        
        Args:
            bbox1: 第一个边界框 [x1, y1, x2, y2]
            bbox2: 第二个边界框 [x1, y1, x2, y2]
            
        Returns:
            float: IoU值
        """
        # 计算交集区域坐标
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # 计算交集面积
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 计算每个边界框的面积
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # 计算并集面积
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # 计算IoU
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _nms_by_class(self, detections: List[dict], iou_threshold: float = 0.45, nms_merge: bool = False) -> List[dict]:
        """
        按类别进行非极大值抑制(NMS)
        
        Args:
            detections: 检测结果列表
            iou_threshold: IoU阈值
            nms_merge: 是否合并所有类别进行统一NMS，默认为False（按类别分组）
            
        Returns:
            List[dict]: NMS后的检测结果
        """
        if nms_merge:
            # 合并所有类别进行统一NMS
            print("执行合并类别NMS")
            # 按置信度降序排序所有检测框
            sorted_detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
            
            # 应用NMS
            keep = []
            for i, det_i in enumerate(sorted_detections):
                # 假设当前检测框是保留的
                should_keep = True
                
                # 与已保留的检测框比较IoU
                for j in keep:
                    det_j = sorted_detections[j]
                    iou = self._calculate_iou(det_i['bbox'], det_j['bbox'])
                    
                    # 如果IoU超过阈值，丢弃当前检测框
                    if iou > iou_threshold:
                        should_keep = False
                        break
                
                if should_keep:
                    keep.append(i)
            
            # 将保留的检测框添加到结果中
            nms_results = [sorted_detections[idx] for idx in keep]
        else:
            # 按类别分组
            class_groups = {}
            for detection in detections:
                class_id = detection['class']
                if class_id not in class_groups:
                    class_groups[class_id] = []
                class_groups[class_id].append(detection)
            
            # 对每个类别单独进行NMS
            nms_results = []
            for class_id, class_detections in class_groups.items():
                # 按置信度降序排序
                sorted_detections = sorted(class_detections, key=lambda x: x['conf'], reverse=True)
                
                # 应用NMS
                keep = []
                for i, det_i in enumerate(sorted_detections):
                    # 假设当前检测框是保留的
                    should_keep = True
                    
                    # 与已保留的检测框比较IoU
                    for j in keep:
                        det_j = sorted_detections[j]
                        iou = self._calculate_iou(det_i['bbox'], det_j['bbox'])
                        
                        # 如果IoU超过阈值，丢弃当前检测框
                        if iou > iou_threshold:
                            should_keep = False
                            break
                    
                    if should_keep:
                        keep.append(i)
                
                # 将保留的检测框添加到结果中
                for idx in keep:
                    nms_results.append(sorted_detections[idx])
        
        print(f"NMS前检测数量: {len(detections)}, NMS后检测数量: {len(nms_results)}, 合并模式: {nms_merge}")
        return nms_results
    
    def postprocess_yolo(self, outputs: List[np.ndarray], conf_threshold: float = 0.5, 
                        iou_threshold: float = 0.45, nms_merge: bool = False) -> List[dict]:
        """
        YOLO模型后处理，解析检测结果并应用NMS抑制
        
        Args:
            outputs (List[np.ndarray]): 模型输出
            conf_threshold (float): 置信度阈值
            iou_threshold (float): IoU阈值
            nms_merge (bool): 是否合并所有类别进行统一NMS，默认为False（按类别分组）
            
        Returns:
            List[dict]: 检测结果列表，每个元素包含 {'bbox': [x1, y1, x2, y2], 'conf': conf, 'class': class_id}
        """
        if not outputs:
            return []
        
        # 取第一个输出 (通常是检测结果)
        predictions = outputs[0]
        print(f"原始输出形状: {predictions.shape}")
        
        # YOLO输出格式通常是 [batch, features, num_detections] 
        # 其中features = 4(bbox) + 1(objectness) + num_classes
        if len(predictions.shape) == 3:
            # 检查是否需要转置
            batch_size, features, num_detections = predictions.shape
            print(f"3D输出: batch={batch_size}, features={features}, detections={num_detections}")
            
            # 移除batch维度并转置为 [num_detections, features]
            predictions = predictions[0].T  # [features, num_detections] -> [num_detections, features]
            print(f"转置后形状: {predictions.shape}")
        elif len(predictions.shape) == 2:
            print(f"2D输出形状: {predictions.shape}")
            # 已经是 [num_detections, features] 格式
        else:
            print(f"警告: 意外的输出形状 {predictions.shape}")
            return []
        
        # 检查输出格式
        if len(predictions.shape) != 2:
            print(f"错误: 输出形状不正确 {predictions.shape}")
            return []
        
        num_detections, num_features = predictions.shape
        print(f"最终格式: 检测数量={num_detections}, 特征维度={num_features}")
        
        # 验证特征维度并自动适应
        expected_classes = len(self.class_names) if self.class_names else 80
        expected_features = 4 + expected_classes  # bbox + objectness + classes
        print(f"期望特征维度: {expected_features} (4+{expected_classes})")
        
        if num_features < 5:
            print(f"错误: 输出特征维度不足 ({num_features} < 5)")
            return []
        
        # 自动适应不同的输出格式
        if num_features != expected_features:
            print(f"特征维度不匹配: 期望{expected_features}，实际{num_features}")
            
            # 尝试推断实际的类别数量
            actual_classes = num_features - 5  # 减去4(bbox) + 1(objectness)
            print(f"推断的实际类别数量: {actual_classes}")
            
            if actual_classes > 0:
                # 更新期望类别数量
                expected_classes = actual_classes
                print(f"自动调整为实际类别数量: {expected_classes}")
            else:
                print(f"警告: 无法推断类别数量，使用默认值{expected_classes}")
        
        # 最终确认的特征分解
        print(f"最终特征分解:")
        print(f"  - 边界框坐标: 索引0-3")
        print(f"  - objectness: 索引4") 
        print(f"  - 类别概率: 索引5-{4+expected_classes} (共{expected_classes}个类别)")
            
        # 解析结果
        results = []
        for i, pred in enumerate(predictions):
            try:
                # 边界框坐标 (center_x, center_y, width, height)
                cx, cy, w, h = pred[:4]
                
                # 转换为 (x1, y1, x2, y2) 格式
                x1 = float(cx - w / 2)
                y1 = float(cy - h / 2)
                x2 = float(cx + w / 2)
                y2 = float(cy + h / 2)
                
                # 获取类别ID
                if num_features >  4:
                    # 获取类别概率，只取前expected_classes个
                    class_probs = pred[4:4+expected_classes]
                    class_id = int(np.argmax(class_probs))
                    class_conf = float(class_probs[class_id])
                    
                    # 验证类别ID范围
                    if class_id >= expected_classes:
                        # 强制限制在有效范围内
                        class_id = min(class_id, expected_classes - 1)
                        class_conf = float(class_probs[class_id])
                    
                    # 最终置信度
                    final_conf = class_conf
                else:
                    # 如果没有类别概率，使用objectness作为置信度
                    class_id = 0
                    final_conf = 1
                    print(f"警告: 特征维度不足，无法解析类别信息")

                if final_conf < conf_threshold:
                    continue
                # 获取类别名称（如果可用）
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                class_conf = final_conf
                # 调试信息（仅前几个检测）
                if i < 3:
                    print(f"检测 {i} 详细信息:")
                    print(f"  特征维度: {num_features}")
                    print(f"  类别概率范围: [5:{5+expected_classes}]")
                    print(f"  类别ID: {class_id} (范围: 0-{expected_classes-1})")
                    print(f"  类别置信度: {class_conf:.4f}")
                    print(f"  类别名称: {class_name}")
                
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': final_conf,
                    'class': class_id,
                    'class_name': class_name,
                    'objectness': class_conf  # 保留原始objectness分数
                })
                
            except Exception as e:
                print(f"解析第{i}个检测结果时出错: {e}")
                continue
        
        print(f"成功解析 {len(results)} 个检测结果")
        
        # 应用NMS抑制
        if results and nms_merge:
            results = self._nms_by_class(results, iou_threshold, nms_merge)
        
        return results
    
    def predict(self, image_path: str, conf_threshold: float = 0.5, 
                iou_threshold: float = 0.45, nms_merge: bool = False) -> Union[np.ndarray, List[dict]]:
        """
        对图片进行推理
        
        Args:
            image_path (str): 图片路径
            conf_threshold (float): 置信度阈值 (仅用于YOLO模型)
            iou_threshold (float): IoU阈值 (仅用于YOLO模型)
            nms_merge (bool): 是否合并所有类别进行统一NMS，默认为False（按类别分组）
            
        Returns:
            Union[np.ndarray, List[dict]]: 推理结果
                - 如果是YOLO模型，返回检测结果列表
                - 如果是其他模型，返回原始输出数组
        """
        if self.session is None:
            raise RuntimeError("模型未加载，请先初始化ONNXPredictor")
        
        try:
            # 预处理图片
            input_array = self.preprocess_image(image_path)
            
            print(f"开始推理图片: {image_path}")
            print(f"输入形状: {input_array.shape}")
            
            # 执行推理
            outputs = self.session.run(self.output_names, {self.input_name: input_array})
            
            print(f"推理完成，输出数量: {len(outputs)}")
            
            # 根据输出形状判断是否为YOLO模型并进行相应后处理
            if len(outputs) == 1:
                output = outputs[0]
                print(f"单输出模型，输出形状: {output.shape}")

                # [x, y, w, h, objectness, class_0, class_1, ..., class_79]
                # 推测该 ONNX 模型在导出时可能将置信度与类别概率合并
                #  # 1， 84， 8400，支持 80类， 可能该objectness已经删除

                # 检查是否为YOLO格式的输出
                # YOLO通常输出形状为 [1, 84, 8400] 或 [1, num_features, num_detections]
                if len(output.shape) == 3 and output.shape[0] == 1:
                    # 可能是YOLO模型，尝试后处理
                    try:
                        results = self.postprocess_yolo(outputs, conf_threshold, iou_threshold, nms_merge)
                        print(f"YOLO后处理完成，检测到 {len(results)} 个目标")
                        return results
                    except Exception as e:
                        print(f"YOLO后处理失败: {e}")
                        print("返回原始输出供调试")
                        return outputs
                elif len(output.shape) == 2:
                    # 2D输出，可能是已经处理过的结果
                    try:
                        results = self.postprocess_yolo(outputs, conf_threshold, iou_threshold, nms_merge)
                        print(f"2D输出后处理完成，检测到 {len(results)} 个目标")
                        return results
                    except Exception as e:
                        print(f"2D输出后处理失败: {e}")
                        return outputs
                else:
                    print(f"未知的输出格式: {output.shape}")
                    return outputs
            else:
                print(f"多输出模型，输出数量: {len(outputs)}")
                return outputs
            
        except Exception as e:
            raise RuntimeError(f"推理失败: {e}")
    
    def predict_raw(self, image_path: str) -> List[np.ndarray]:
        """
        对图片进行推理，返回原始输出（不进行后处理）
        
        Args:
            image_path (str): 图片路径
            
        Returns:
            List[np.ndarray]: 原始推理输出
        """
        if self.session is None:
            raise RuntimeError("模型未加载，请先初始化ONNXPredictor")
        
        # 预处理图片
        input_array = self.preprocess_image(image_path)
        
        # 执行推理
        outputs = self.session.run(self.output_names, {self.input_name: input_array})
        
        return outputs
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            dict: 模型信息字典
        """
        if self.session is None:
            return {"error": "模型未加载"}
        
        info = {
            "model_path": self.model_path,
            "input_name": self.input_name,
            "input_shape": self.input_shape,
            "output_names": self.output_names,
            "providers": self.session.get_providers(),
            # "model_meta": self.session.get_modelmeta().__dict__ if hasattr(self.session, 'get_modelmeta') else None,
            "model_meta": self.session.get_modelmeta(),
            "class_names": self.class_names,
            "num_classes": len(self.class_names)
        }
        
        return info


def draw_detection_results(image_path: str, results: list, output_path: Optional[str] = None, 
                          conf_threshold: float = 0.5, font_scale: float = 0.6, 
                          thickness: int = 2):
    """
    在图片上绘制检测结果
    
    Args:
        image_path (str): 原始图片路径
        results (list): 检测结果列表
        output_path (str): 输出图片路径，如果不指定则自动生成
        conf_threshold (float): 置信度阈值
        font_scale (float): 字体大小
        thickness (int): 线条粗细
        
    Returns:
        str: 输出图片路径
    """
    try:
        import cv2
        import numpy as np
        
        # 读取原始图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 获取图片尺寸
        img_height, img_width = image.shape[:2]
        
        # 过滤低置信度检测
        filtered_results = [r for r in results if r['conf'] >= conf_threshold]
        
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
        for i, result in enumerate(filtered_results):
            bbox = result['bbox']
            conf = result['conf']
            class_id = result['class']
            class_name = result.get('class_name', f'class_{class_id}')
            
            # 转换边界框坐标（归一化 -> 像素坐标）
            x1 = int(bbox[0] * img_width)
            y1 = int(bbox[1] * img_height)
            x2 = int(bbox[2] * img_width)
            y2 = int(bbox[3] * img_height)
            
            # 确保坐标在图片范围内
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            
            # 选择颜色
            color = colors[i % len(colors)]
            
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
        print(f"绘制了 {len(filtered_results)} 个检测框")
        
        return output_path
        
    except ImportError:
        print("错误: 需要安装opencv-python来绘制检测结果")
        print("请运行: pip install opencv-python")
        return None
    except Exception as e:
        print(f"绘制检测结果时出错: {e}")
        return None

def test_with_visualization(model_path: str, image_path: str, output_dir: str = "output"):
    """
    测试推理并生成可视化结果
    
    Args:
        model_path (str): ONNX模型路径
        image_path (str): 测试图片路径
        output_dir (str): 输出目录
    """
    print("=== ONNX模型推理测试（带可视化） ===")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"图片文件不存在: {image_path}")
        return
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 创建推理器
        print(f"加载模型: {model_path}")
        predictor = ONNXPredictor(model_path)
        
        # 获取模型信息
        model_info = predictor.get_model_info()
        print(f"\n模型类别数量: {model_info.get('num_classes', '未知')}")
        
        # 进行推理
        print(f"\n开始推理图片: {image_path}")
        results = predictor.predict(image_path, conf_threshold=0.5)
        
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
            # YOLO检测结果
            print(f"\n检测到 {len(results)} 个目标:")
            
            # 按置信度排序
            results_sorted = sorted(results, key=lambda x: x['conf'], reverse=True)
            
            for i, result in enumerate(results_sorted[:10]):  # 显示前10个
                print(f"  {i+1}. {result['class_name']} (类别{result['class']}) "
                      f"置信度={result['conf']:.3f}")
            
            # 绘制检测结果
            print(f"\n生成可视化结果...")
            output_path = draw_detection_results(
                image_path, 
                results, 
                output_path=os.path.join(output_dir, "detection_result.jpg"),
                conf_threshold=0.3  # 使用较低的阈值显示更多检测
            )
            
            if output_path:
                print(f"✅ 可视化完成: {output_path}")
                
                # 生成检测结果报告
                report_path = os.path.join(output_dir, "detection_report.txt")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(f"ONNX模型检测报告\n")
                    f.write(f"模型路径: {model_path}\n")
                    f.write(f"图片路径: {image_path}\n")
                    f.write(f"检测时间: {__import__('datetime').datetime.now()}\n")
                    f.write(f"总检测数: {len(results)}\n\n")
                    
                    for i, result in enumerate(results_sorted):
                        f.write(f"{i+1}. {result['class_name']} (类别{result['class']})\n")
                        f.write(f"   置信度: {result['conf']:.4f}\n")
                        f.write(f"   边界框: {result['bbox']}\n")
                        f.write(f"   目标检测分数: {result.get('objectness', 'N/A'):.4f}\n\n")
                
                print(f"检测报告已保存: {report_path}")
        else:
            print("未检测到目标或输出格式不正确")
            
    except Exception as e:
        logging.warning(f"测试过程出错: {e}", exc_info=True)
        import traceback
        traceback.print_exc()

def main():
    """
    示例用法
    """
    # 示例：使用YOLO ONNX模型进行推理
    model_path = "../model_demo/out/yolo11n.onnx"  # 替换为实际的模型路径
    image_path = "../dataset/coco8/datasets/coco8/images/val/000000000049.jpg"  # 替换为实际的图片路径
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请确保模型文件存在，或修改model_path变量")
        
        # 尝试使用数据集中的图片进行测试
        dataset_images = [
            "dataset/coco8/datasets/coco8/images/val/000000000036.jpg",
            "dataset/coco8/datasets/coco8/images/val/000000000042.jpg"
        ]
        
        for img_path in dataset_images:
            if os.path.exists(img_path):
                print(f"使用数据集图片进行测试: {img_path}")
                test_with_visualization(model_path, img_path)
                break
        else:
            print("未找到可用的测试图片")
        return
    
    if not os.path.exists(image_path):
        print(f"图片文件不存在: {image_path}")
        print("请确保图片文件存在，或修改image_path变量")
        return
    
    # 运行测试
    test_with_visualization(model_path, image_path)


if __name__ == "__main__":
    main()
