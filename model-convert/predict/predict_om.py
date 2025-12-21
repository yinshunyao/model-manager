import logging
import onnx
import os
# 增加父目录到python环境变量中
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
import subprocess
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import ctypes
import threading
from predict.post_handle import postprocess_yolov8

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 配置华为环境
# 手动设置昇腾环境变量（路径替换为实际值，参考终端中env的输出）
os.environ["ASCEND_HOME"] = "/usr/local/Ascend/ascend-toolkit/8.0.0"
os.environ["LD_LIBRARY_PATH"] = (
    f"{os.environ['ASCEND_HOME']}/lib64:"
    "/usr/local/Ascend/driver/lib64:"
    f"{os.environ.get('LD_LIBRARY_PATH', '')}"
)
os.environ["PATH"] = f"{os.environ['ASCEND_HOME']}/bin:{os.environ.get('PATH', '')}"
os.environ["PYTHONPATH"] = (
    f"{os.environ['ASCEND_HOME']}/python/site-packages:"
    f"{os.environ.get('PYTHONPATH', '')}"
)

try:
    import acl  # type: ignore
    # 初始化 ACL
    acl.init()
except ImportError:
    logging.warning("未安装 ACL，请检查环境变量")

class HUAWEI_910B_Predictor:
    """
    华为910B设备上的模型推理类，支持目标检测和图像分类任务
    """
    
    def __init__(self, om_model_path, debug=False, save=False):
        """
        初始化函数，加载OM模型到内存
        
        Args:
            om_model_path (str): OM模型文件路径
            debug (bool): 是否开启调试模式，默认False（用于设置昇腾环境变量）
            save (bool): 是否保存检测结果图片，默认False
            output_dir (str): 保存检测结果图片的目录，如果为None则使用当前工作目录下的output目录
        """
        self.om_model_path = om_model_path
        self.model = None
        self.debug = debug
        self.save = save
        self._released = False  # 标记资源是否已释放

        self.output_dir = os.path.join(current_dir, "output")
        
        # 确保输出目录存在（如果启用了save功能）
        if self.save:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Save模式: 检测结果将保存到 {self.output_dir}")
        
        # 设置调试环境变量（仅用于debug模式）
        if self.debug:
            os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
            os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '1'
        
        # 验证模型文件存在
        if not os.path.exists(om_model_path):
            raise FileNotFoundError(f"OM模型文件不存在: {om_model_path}")
        
        # 加载模型（使用锁保护）
        self._load_model()
        
    def _load_model(self, device_id=0):
        """
        加载OM模型到内存（内部方法）
        """
        # 初始化为模拟模式作为默认值
        self.model = {'simulation_mode': True}
        
        # 将华为设备的Python依赖导入放在函数内部
        try:
            # 确保昇腾Python路径在sys.path中
            ascend_home = os.environ.get("ASCEND_HOME", "/usr/local/Ascend/ascend-toolkit/8.0.0")
            ascend_python_path = os.path.join(ascend_home, "python", "site-packages")
            
            if os.path.exists(ascend_python_path):
                if ascend_python_path not in sys.path:
                    sys.path.insert(0, ascend_python_path)
                    logger.info(f"已将昇腾Python路径添加到sys.path: {ascend_python_path}")
            else:
                logger.warning(f"昇腾Python路径不存在: {ascend_python_path}")
                logger.warning(f"ASCEND_HOME: {ascend_home}")
            
            # 尝试导入华为AscendCL API
            try:
                import acl  # type: ignore
                acl_available = True
                logger.info("成功导入华为AscendCL API")
            except ImportError as e:
                acl_available = False
                logger.warning(f"导入华为AscendCL API失败: {str(e)}")
                logger.warning(f"ASCEND_HOME: {os.environ.get('ASCEND_HOME', '未设置')}")
                logger.warning(f"昇腾Python路径: {ascend_python_path}, 存在: {os.path.exists(ascend_python_path)}")
                # 显示sys.path中与昇腾相关的路径
                ascend_paths = [p for p in sys.path if 'ascend' in p.lower() or 'Ascend' in p]
                if ascend_paths:
                    logger.warning(f"sys.path中的昇腾相关路径: {ascend_paths}")
                else:
                    logger.warning(f"sys.path中未找到昇腾相关路径，前5个路径: {sys.path[:5]}")
                
            if not acl_available:
                raise ImportError("华为AscendCL API不可用")
            
            # 设置运行模式
            ret = acl.rt.set_device(device_id)  # type: ignore
            if ret != 0:
                raise RuntimeError(f"设置设备失败: {ret}")
            
            # 创建上下文
            context, ret = acl.rt.create_context(0)  # type: ignore
            if ret != 0:
                raise RuntimeError(f"创建上下文失败: {ret}")
            
            # 加载模型
            model_id, ret = acl.mdl.load_from_file(self.om_model_path)  # type: ignore
            if ret != 0:
                raise RuntimeError(f"加载模型失败: {ret}")
            
            # 获取模型描述信息
            model_desc = acl.mdl.create_desc()  # type: ignore
            ret = acl.mdl.get_desc(model_desc, model_id)  # type: ignore
            if ret != 0:
                raise RuntimeError(f"获取模型描述失败: {ret}")
            
            # 更新模型实例，覆盖模拟模式
            self.model = {
                'model_id': model_id,
                'model_desc': model_desc,
                'context': context,
                'device_id': device_id,
                'simulation_mode': False
            }
            
            print(f"OM模型加载成功: {self.om_model_path}, model_id:{model_id}")
        except Exception as e:
            # 如果是在非华为设备上运行，保持模拟模式
            logging.error(f"无法加载华为ACL，使用模拟模式: {e}", exc_info=True)
            self.model = {'simulation_mode': True}
            logging.warning(f"模拟模式加载模型: {self.om_model_path}")
    
    def _preprocess_image(self, image, input_shape=(640, 640)):
        """
        图像预处理（内部方法）
        
        Args:
            image: 输入图像（cv2格式）
            input_shape: 模型输入尺寸
        
        Returns:
            tuple: (预处理后的图像数据, 原始图像尺寸, 模型输入尺寸)
        """
        # 保存原始图像尺寸
        original_h, original_w = image.shape[:2]
        original_shape = (original_h, original_w)
        
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
        
        return img, original_shape, input_shape
    
    def predict(self, image, device_id=0, image_path=None):
        """
        目标检测推理
        
        Args:
            image: 输入图像（cv2格式或文件路径）
            device_id: 设备ID，默认为0
            image_path: 原始图像路径（用于debug模式下生成输出文件名），如果为None则自动推断
        
        Returns:
            list: 检测结果列表，每个元素为 [x1, y1, x2, y2, confidence, class_id, class_name]
        """
        if self._released:
            raise RuntimeError("Predictor资源已释放，无法执行推理")
        
        # 检查输入类型
        original_image_path = image_path
        if isinstance(image, str):
            original_image_path = image
            image = cv2.imread(image)
            if image is None:
                raise FileNotFoundError(f"无法读取图像文件: {image}")
        
        # 保存原始图像尺寸
        original_image_shape = image.shape[:2]  # (H, W)
        
        # 预处理图像
        input_data, original_shape, model_input_shape = self._preprocess_image(image)

        # 检查模型是否存在且是字典类型
        if isinstance(self.model, dict):
            # 模拟模式下的推理
            if self.model.get('simulation_mode', True):
                # 模拟检测结果
                results = [
                    [100, 100, 200, 200, 0.9, 0],
                    [300, 300, 400, 400, 0.85, 1]
                ]
                # 应用NMS处理
                results = self._apply_nms(results)
                # 添加类别名称
                for result in results:
                    if len(result) < 7:  # 如果还没有添加类别名称
                        class_id = int(result[5])
                        class_names = ['person', 'car']  # 示例类别名称
                        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                        result.append(class_name)

                # 保存检测结果图片
                if self.save:
                    self._draw_detections(image, results, original_image_path)

                print("模拟推理完成（目标检测）")
                return results

            # 华为设备上的实际推理
            try:
                import acl  # type: ignore

                # 获取模型描述
                model_desc = self.model.get('model_desc')
                model_id = self.model.get('model_id')

                # 获取输入数量
                num_inputs = acl.mdl.get_num_inputs(model_desc)  # type: ignore
                if num_inputs < 1:
                    raise RuntimeError("模型没有输入")

                # 获取输入大小
                input_size = acl.mdl.get_input_size_by_index(model_desc, 0)  # type: ignore

                # 分配输入缓冲区
                input_buffer, ret = acl.rt.malloc(input_size, 0)  # type: ignore
                if ret != 0:
                    raise RuntimeError(f"分配输入缓冲区失败: {ret}")

                # 复制输入数据到设备
                ret = acl.rt.memcpy(input_buffer, input_size,  # type: ignore
                                   acl.util.bytes_to_ptr(input_data.tobytes()),  # type: ignore
                                   input_data.nbytes, 1)  # type: ignore
                if ret != 0:
                    raise RuntimeError(f"复制输入数据失败: {ret}")

                # === 创建输入 Dataset ===
                input_dataset = acl.mdl.create_dataset()
                input_data_buffer = acl.create_data_buffer(input_buffer, input_size)
                acl.mdl.add_dataset_buffer(input_dataset, input_data_buffer)

                # === 创建输出 Dataset ===
                num_outputs = acl.mdl.get_num_outputs(model_desc)
                output_dataset = acl.mdl.create_dataset()
                output_buffers = []
                for i in range(num_outputs):
                    size = acl.mdl.get_output_size_by_index(model_desc, i)
                    buf, ret = acl.rt.malloc(size, 0)
                    if ret != 0:
                        raise RuntimeError(f"输出 buffer {i} 分配失败")
                    dbuf = acl.create_data_buffer(buf, size)
                    acl.mdl.add_dataset_buffer(output_dataset, dbuf)
                    output_buffers.append((buf, size))

                # 执行模型推理
                ret = acl.mdl.execute(model_id, input_dataset, output_dataset)  # type: ignore
                if ret != 0:
                    raise RuntimeError(f"执行推理失败: {ret}")

                # 获取输出大小
                output_buffer, output_size = output_buffers[0]
                # output_size = acl.mdl.get_output_size_by_index(model_desc, 0)  # type: ignore
                #
                # # 分配输出缓冲区
                # output_buffer, ret = acl.rt.malloc(output_size, 0)  # type: ignore
                # if ret != 0:
                #     raise RuntimeError(f"分配输出缓冲区失败: {ret}")

                # 从设备获取输出
                output_data = np.zeros(output_size // 4, dtype=np.float32)
                # ret = acl.rt.memcpy(acl.util.bytes_to_ptr(output_data.tobytes()),  # type: ignore
                #                    output_data.nbytes, output_buffer,
                #                    output_size, 2)  # type: ignore acl.MEMCPY_DEVICE_TO_HOST
                dst_ptr = output_data.ctypes.data
                # output_data.ctypes.data_as(ctypes.c_void_p)
                ret = acl.rt.memcpy(
                    dst_ptr,  # ← 正确获取数组内存地址
                    output_data.nbytes,
                    output_buffer,
                    output_size,
                    2  # 或直接用 2
                )
                if ret != 0:
                    raise RuntimeError(f"复制输出数据失败: {ret}")

                # 释放缓冲区
                acl.rt.free(input_buffer)  # type: ignore
                # acl.rt.free(output_buffer)  # type: ignore
                for buf, size in output_buffers:
                    acl.rt.free(buf)

                # 解析输出结果（这里需要根据实际模型输出格式进行调整）
                # 自动识别YOLOv8格式或其他格式
                results = []
                logger.info(f"实际推理完成（目标检测）shape:{output_data.shape}, model id:{model_id}, num_outputs:{num_outputs}, num_inputs:{num_inputs}")

                # 确保output_data是numpy数组
                if not isinstance(output_data, np.ndarray):
                    output_data = np.array(output_data)

                # 展平输出数据以便统一处理
                output_shape = output_data.shape
                if len(output_data.shape) > 1:
                    output_data_flat = output_data.flatten()
                    logger.info(f"输出数据从 {output_shape} 展平为 {output_data_flat.shape}")
                else:
                    output_data_flat = output_data

                # 尝试检测是否是YOLOv8格式
                from predict.post_handle import is_yolov8_format
                is_yolov8, num_boxes, num_classes = is_yolov8_format(output_data_flat)

                if is_yolov8:
                    # 使用YOLOv8后处理
                    # 注意：YOLOv8输出的坐标是归一化的（基于模型输入尺寸），需要转换到原始图像尺寸
                    logger.info(f"检测到YOLOv8格式，使用YOLOv8后处理: num_boxes={num_boxes}, num_classes={num_classes}")
                    logger.info(f"原始图像尺寸: {original_shape}, 模型输入尺寸: {model_input_shape}")
                    results = postprocess_yolov8(
                        output_data_flat,
                        original_shape,  # 原始图像尺寸 (H, W)
                        model_input_shape=model_input_shape  # 模型输入尺寸 (W, H)，用于坐标转换
                    )
                else:
                    # 尝试其他格式：假设是 [x1, y1, x2, y2, score, class_id, ...] 格式
                    logger.info(f"未检测到YOLOv8格式，尝试通用格式解析")
                    conf_threshold = 0.3  # 置信度阈值
                    valid_detections = 0
                    low_conf_detections = 0

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
                                        valid_detections += 1
                                    else:
                                        low_conf_detections += 1

                            if valid_detections > 0:
                                logger.info(f"使用stride={stride}解析成功，有效检测数={valid_detections}")
                                break

                    if len(results) == 0:
                        logger.warning(f"无法解析输出数据，shape={output_shape}, size={output_data_flat.size}")
                        # 如果无法解析，返回空结果
                        return []

                # 如果检测结果过多，进一步限制数量
                if len(results) > 1000:
                    # 按置信度排序，只保留前1000个
                    results.sort(key=lambda x: x[4], reverse=True)
                    results = results[:1000]
                    logger.info(f"检测结果过多，已限制到前1000个")

                # 应用NMS处理
                results = self._apply_nms(results)

                # 添加类别名称（如果还没有）
                for result in results:
                    if len(result) < 7:  # 如果还没有添加类别名称
                        class_id = int(result[5])
                        class_names = ['person', 'car']  # 示例类别名称
                        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                        result.append(class_name)

                # 保存检测结果图片
                if self.save:
                    self._draw_detections(image, results, original_image_path)

                return results

            except ImportError:
                # 如果导入失败，回退到模拟模式
                print("华为AscendCL API不可用，使用模拟模式")
                results = [
                    [100, 100, 200, 200, 0.9, 0],
                    [300, 300, 400, 400, 0.85, 1]
                ]
                # 应用NMS处理
                results = self._apply_nms(results)
                # 添加类别名称
                for result in results:
                    if len(result) < 7:  # 如果还没有添加类别名称
                        class_id = int(result[5])
                        class_names = ['person', 'car']  # 示例类别名称
                        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                        result.append(class_name)

                # 保存检测结果图片
                if self.save:
                    self._draw_detections(image, results, original_image_path)

                return results
    
    def predict_cls(self, image, device_id=0):
        """
        图像分类推理
        
        Args:
            image: 输入图像（cv2格式或文件路径）
            device_id: 设备ID，默认为0
        
        Returns:
            list: 分类结果列表，每个元素为 [class_name, confidence]
        """
        if self._released:
            raise RuntimeError("Predictor资源已释放，无法执行推理")
        
        # 检查输入类型
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise FileNotFoundError(f"无法读取图像文件: {image}")
        
        # 预处理图像（分类任务不需要坐标转换，只需要图像数据）
        input_data, _, _ = self._preprocess_image(image)
        # 检查模型是否存在且是字典类型
        if isinstance(self.model, dict):
            # 模拟模式下的推理
            if self.model.get('simulation_mode', True):
                # 模拟分类结果
                results = [
                    ['cat', 0.75],
                    ['dog', 0.15]
                ]
                print("模拟推理完成（图像分类）")
                return results

            # 华为设备上的实际推理
            try:
                import acl  # type: ignore

                # 获取模型描述
                model_desc = self.model.get('model_desc')
                model_id = self.model.get('model_id')

                # 获取输入数量
                num_inputs = acl.mdl.get_num_inputs(model_desc)  # type: ignore
                if num_inputs < 1:
                    raise RuntimeError("模型没有输入")

                # 获取输入大小
                input_size = acl.mdl.get_input_size_by_index(model_desc, 0)  # type: ignore

                # 分配输入缓冲区
                input_buffer, ret = acl.rt.malloc(input_size, 0)  # type: ignore
                if ret != 0:
                    raise RuntimeError(f"分配输入缓冲区失败: {ret}")

                # 复制输入数据到设备
                ret = acl.rt.memcpy(input_buffer, input_size,  # type: ignore
                                   acl.util.bytes_to_ptr(input_data.tobytes()),  # type: ignore
                                   input_data.nbytes, 1)  # type: ignore
                if ret != 0:
                    raise RuntimeError(f"复制输入数据失败: {ret}")

                # === 创建输入 Dataset ===
                input_dataset = acl.mdl.create_dataset()
                input_data_buffer = acl.create_data_buffer(input_buffer, input_size)
                acl.mdl.add_dataset_buffer(input_dataset, input_data_buffer)

                # === 创建输出 Dataset ===
                num_outputs = acl.mdl.get_num_outputs(model_desc)
                output_dataset = acl.mdl.create_dataset()
                output_buffers = []
                for i in range(num_outputs):
                    size = acl.mdl.get_output_size_by_index(model_desc, i)
                    buf, ret = acl.rt.malloc(size, 0)
                    if ret != 0:
                        raise RuntimeError(f"输出 buffer {i} 分配失败")
                    dbuf = acl.create_data_buffer(buf, size)
                    acl.mdl.add_dataset_buffer(output_dataset, dbuf)
                    output_buffers.append((buf, size))

                # 执行模型推理
                ret = acl.mdl.execute(model_id, input_dataset, output_dataset)  # type: ignore
                if ret != 0:
                    raise RuntimeError(f"执行推理失败: {ret}")

                # 获取输出数据
                output_buffer, output_size = output_buffers[0]

                # 从设备获取输出
                output_data = np.zeros(output_size // 4, dtype=np.float32)
                ret = acl.rt.memcpy(acl.util.bytes_to_ptr(output_data.tobytes()),  # type: ignore
                                   output_data.nbytes, output_buffer,
                                   output_size, 2)  # type: ignore acl.MEMCPY_DEVICE_TO_HOST
                if ret != 0:
                    raise RuntimeError(f"复制输出数据失败: {ret}")

                # 释放缓冲区
                acl.rt.free(input_buffer)  # type: ignore
                acl.rt.free(output_buffer)  # type: ignore

                # 解析分类结果
                # 假设输出是分类概率数组
                print(f"实际推理完成（图像分类）:{output_data.tolist()[:10]}")

                # 这里可以根据实际模型的类别列表进行映射
                # 以下是一个示例实现，将概率最高的两个类别作为结果返回
                results = []
                if len(output_data) > 0:
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

                return results[:20]

            except ImportError:
                # 如果导入失败，回退到模拟模式
                print("华为AscendCL API不可用，使用模拟模式")
                results = [
                    ['cat', 0.75],
                    ['dog', 0.15]
                ]
                return results
    
    def release(self):
        """
        显式释放模型资源（推荐使用此方法而不是依赖析构函数）
        """
        if self._released:
            logger.warning("资源已经释放，跳过重复释放")
            return

        # 检查模型是否存在且不是模拟模式
        if isinstance(self.model, dict) and not self.model.get('simulation_mode', False):
            try:
                # 尝试导入华为AscendCL API
                try:
                    import acl  # type: ignore
                    acl_available = True
                except ImportError:
                    acl_available = False
                    logger.warning("无法导入acl模块，跳过资源释放")

                if acl_available:
                    try:
                        # 按正确顺序释放资源
                        model_desc = self.model.get('model_desc')
                        model_id = self.model.get('model_id')
                        context = self.model.get('context')
                        device_id = self.model.get('device_id', 0)

                        # 1. 销毁模型描述
                        if model_desc is not None:
                            try:
                                acl.mdl.destroy_desc(model_desc)  # type: ignore
                                logger.debug("模型描述已销毁")
                            except Exception as e:
                                logger.warning(f"销毁模型描述失败: {e}")

                        # 2. 卸载模型
                        if model_id is not None:
                            try:
                                ret = acl.mdl.unload(model_id)  # type: ignore
                                if ret != 0:
                                    logger.warning(f"卸载模型失败，返回码: {ret}")
                                else:
                                    logger.debug("模型已卸载")
                            except Exception as e:
                                logger.warning(f"卸载模型失败: {e}")

                        # 3. 销毁上下文
                        if context is not None:
                            try:
                                ret = acl.rt.destroy_context(context)  # type: ignore
                                if ret != 0:
                                    logger.warning(f"销毁上下文失败，返回码: {ret}")
                                else:
                                    logger.debug("上下文已销毁")
                            except Exception as e:
                                logger.warning(f"销毁上下文失败: {e}")

                        # 4. 重置设备
                        try:
                            ret = acl.rt.reset_device(device_id)  # type: ignore
                            if ret != 0:
                                logger.warning(f"重置设备失败，返回码: {ret}")
                            else:
                                logger.debug(f"设备 {device_id} 已重置")
                        except Exception as e:
                            logger.warning(f"重置设备失败: {e}")

                        # 5. 最终化ACL（注意：这会影响所有使用ACL的实例）
                        # 如果多个predictor实例共享ACL，不应该在这里finalize
                        # 只有在确定没有其他实例使用时才finalize
                        # 暂时注释掉，避免影响其他实例
                        # try:
                        #     acl.finalize()  # type: ignore
                        #     logger.debug("ACL已最终化")
                        # except Exception as e:
                        #     logger.warning(f"ACL最终化失败: {e}")

                        logger.warning("OM模型资源已释放")
                    except Exception as e:
                        logger.error(f"释放资源时出错: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"释放资源过程中出现异常: {e}", exc_info=True)

        # 标记为已释放
        self._released = True
        # 清空模型引用
        self.model = None

    def __del__(self):
        """
        析构函数，释放模型资源（作为备份，推荐使用release方法）
        """
        if not self._released:
            try:
                self.release()
            except Exception as e:
                logger.warning(f"析构函数中释放资源失败: {e}")

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

    def _draw_detections(self, image: np.ndarray, results: List[List[float]], image_path: Optional[str] = None):
        """
        在图像上绘制检测框并保存（Save模式）

        Args:
            image: 输入图像（cv2格式）
            results: 检测结果列表，每个元素为 [x1, y1, x2, y2, confidence, class_id, class_name]
            image_path: 原始图像路径，用于生成输出文件名
        """
        if not self.save:
            return

        try:
            # 复制图像以避免修改原图
            img_with_boxes = image.copy()

            # 定义颜色列表（BGR格式）
            colors = [
                (0, 255, 0),    # 绿色
                (255, 0, 0),    # 蓝色
                (0, 0, 255),    # 红色
                (255, 255, 0),  # 青色
                (255, 0, 255),  # 洋红色
                (0, 255, 255),  # 黄色
                (128, 0, 128),  # 紫色
                (255, 165, 0),  # 橙色
            ]

            # 获取图像尺寸
            img_h, img_w = image.shape[:2]

            # 绘制每个检测框
            for i, result in enumerate(results):
                if len(result) < 6:
                    continue

                x1, y1, x2, y2 = int(result[0]), int(result[1]), int(result[2]), int(result[3])
                confidence = result[4]
                class_id = int(result[5])
                class_name = result[6] if len(result) > 6 else f"class_{class_id}"

                # 裁剪坐标到图像范围内
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w - 1))
                y2 = max(0, min(y2, img_h - 1))

                # 检查坐标有效性
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"跳过无效的检测框: ({x1}, {y1}, {x2}, {y2})")
                    continue

                # 选择颜色（根据类别ID）
                color = colors[class_id % len(colors)]

                # 绘制边界框
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)

                # 准备标签文本
                label = f"{class_name}: {confidence:.2f}"

                # 计算文本大小
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # 绘制文本背景
                cv2.rectangle(
                    img_with_boxes,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )

                # 绘制文本
                cv2.putText(
                    img_with_boxes,
                    label,
                    (x1, y1 - baseline - 2),
                    font,
                    font_scale,
                    (255, 255, 255),  # 白色文本
                    thickness
                )

            # 生成输出文件名
            if image_path:
                # 从原始路径提取文件名
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_filename = f"{base_name}_detection_result.jpg"
            else:
                # 使用时间戳生成文件名
                import time
                timestamp = int(time.time())
                output_filename = f"detection_result_{timestamp}.jpg"

            output_path = os.path.join(self.output_dir, output_filename)

            # 保存图像
            cv2.imwrite(output_path, img_with_boxes)
            logger.info(f"检测结果已保存到 {output_path}，共检测到 {len(results)} 个目标")

        except Exception as e:
            logger.error(f"绘制检测框时出错: {str(e)}", exc_info=True)




def get_input_shape_from_onnx(onnx_model_path: str) -> Dict[str, Tuple[int, ...]]:
    """
    从 ONNX 模型文件中自动获取输入形状和名称
    
    Args:
        onnx_model_path (str): ONNX 模型文件路径
    
    Returns:
        Dict[str, Tuple[int, ...]]: 输入名称到形状的映射字典，例如 {"images": (1, 3, 640, 640)}
    
    Raises:
        ValueError: 当无法从模型中获取输入信息时
    """
    try:
        model = onnx.load(onnx_model_path)
        inputs = model.graph.input
        
        if not inputs:
            raise ValueError("无法从 ONNX 模型中获取输入信息")
        
        input_info = {}
        for input_node in inputs:
            input_name = input_node.name
            shape = []
            for dim in input_node.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(int(dim.dim_value))
                elif dim.dim_param:
                    shape.append(1)  # 参数化维度默认使用 1
                else:
                    shape.append(1)  # 未知维度默认使用 1
            input_info[input_name] = tuple(shape)
        
        if not input_info:
            raise ValueError("无法从 ONNX 模型中解析出有效的输入信息")
        
        return input_info
    except Exception as e:
        raise ValueError(f"从 ONNX 模型获取输入信息时出错: {str(e)}")

def detect_soc_version() -> str:
    """
    自动检测华为昇腾设备的 SOC 版本
    
    Returns:
        str: SOC 版本字符串，例如 "Ascend910B"、"Ascend310" 等。
             如果检测失败，返回默认值 "Ascend910B"。
    """
    try:
        import acl  # type: ignore
        # 设置设备
        acl.rt.set_device(0)
        # 创建上下文
        context, ret = acl.rt.create_context(0)
        # 获取 SOC 版本信息（参数 1 表示获取 SOC 版本）
        soc_version_bytes = acl.rt.get_device_info(1)
        # 解码为字符串
        soc_version = soc_version_bytes.decode('utf-8')
        # 清理资源
        acl.rt.destroy_context(context)
        acl.rt.reset_device(0)
        acl.finalize()
        return soc_version
    except Exception as e:
        logging.warning(f"SOC 版本自动检测失败: {e}，将使用默认值 Ascend910B")
        return "Ascend910B"

def onnx_to_om(
    onnx_model_path: str,
    output_om_path: str,
    soc_version: Optional[str] = None,
    precision_mode: str = "allow_fp32_to_fp16",
    log_level: str = "error",
    **kwargs
) -> bool:
    """
    将 ONNX 模型转换为华为昇腾设备支持的 OM 格式模型
    
    Args:
        onnx_model_path (str): ONNX 模型文件路径
        output_om_path (str): 输出 OM 模型文件路径
        soc_version (str, optional): 目标昇腾处理器版本。如果为 None，将自动检测。
        precision_mode (str, optional): 精度模式，默认为 "allow_fp32_to_fp16"
        log_level (str, optional): 日志级别，默认为 "error"
        **kwargs: 其他 ATC 工具参数
    
    Returns:
        bool: 转换是否成功
    
    Raises:
        FileNotFoundError: 当 ONNX 模型文件不存在时
        ValueError: 当无法获取输入信息或转换参数无效时
    """
    # 验证 ONNX 模型文件存在
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX 模型文件不存在: {onnx_model_path}")
    
    # 自动获取输入层名称和参数
    input_info = get_input_shape_from_onnx(onnx_model_path)
    if not input_info:
        raise ValueError("无法从 ONNX 模型中获取有效的输入信息")
    
    # 自动检测 SOC 版本
    if soc_version is None:
        soc_version = detect_soc_version()
    
    # 构建 ATC 命令
    atc_cmd = [
        "atc",
        f"--model={onnx_model_path}",
        f"--output={output_om_path}",
        f"--soc_version={soc_version}",
        f"--precision_mode={precision_mode}",
        f"--log={log_level}",
    ]
    
    # 添加输入形状参数
    for input_name, input_shape in input_info.items():
        shape_str = ",".join(map(str, input_shape))
        atc_cmd.append(f"--input_shape={input_name}:{shape_str}")
    
    # 添加其他参数
    for key, value in kwargs.items():
        atc_cmd.append(f"--{key}={value}")
    
    logging.info(f"执行 ATC 转换命令: {' '.join(atc_cmd)}")
    
    try:
        # 执行 ATC 命令
        result = subprocess.run(
            atc_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        logging.info(f"ATC 转换成功: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"ATC 转换失败: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"转换过程中发生未知错误: {str(e)}")
        return False

if __name__ == "__main__":
    # 修改loggging 打印代码行数
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
    # 当前代码文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    import sys
    # 接收参数debug和save
    debug = False
    save = False
    if len(sys.argv) > 1:
        if "debug" in sys.argv:
            debug = True

        if "save" in sys.argv:
            save = True

    # 示例：使用对象进行推理
    # 注意：这里使用示例模型路径，实际使用时请替换为真实的OM模型路径
    # model_name = "model_yolo11.ysy.om"
    # model_name = "model_yolo11.b2.om"
    model_name = "yolo11n.nonms.om"
    om_model_path = os.path.join(current_dir, "..", "model_demo",  model_name)
    # 图片路径 
    test_image_path = os.path.join(current_dir, "..", "000000000025.jpg")
    try:
        # 创建预测器实例
        # debug: 开启昇腾调试模式（设置环境变量）
        # save: 保存检测结果图片到output目录
        predictor = HUAWEI_910B_Predictor(om_model_path, debug=debug, save=save)
        
        # 创建测试图像（使用随机噪声或从文件读取）
        # 方法1：使用随机噪声作为测试图像
        # test_image = np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)
        
        # 方法2：从文件读取图像（如果有测试图像）
        # test_image_path = "/path/to/test/image.jpg"
        test_image = cv2.imread(test_image_path)
        
        # 执行目标检测推理
        print("\n执行目标检测推理...")
        detection_results = predictor.predict(test_image, image_path=test_image_path)
        print(f"检测结果: {detection_results}")
        if save:
            print(f"检测结果图片已保存到 output 目录")
        
        # 执行分类推理
        print("\n执行图像分类推理...")
        classification_results = predictor.predict_cls(test_image)
        print(f"分类结果: {classification_results}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        # 在非华为设备上，这里会进入模拟模式，仍然可以演示功能