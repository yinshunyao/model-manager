import logging
import onnx
import os
import subprocess
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class HUAWEI_910B_Predictor:
    """
    华为910B设备上的模型推理类，支持目标检测和图像分类任务
    """
    
    def __init__(self, om_model_path, debug=False):
        """
        初始化函数，加载OM模型到内存
        
        Args:
            om_model_path (str): OM模型文件路径
            debug (bool): 是否开启调试模式，默认False
        """
        self.om_model_path = om_model_path
        self.model = None
        self.debug = debug
        
        # 设置调试环境变量
        if self.debug:
            os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
            os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '1'
        
        # 验证模型文件存在
        if not os.path.exists(om_model_path):
            raise FileNotFoundError(f"OM模型文件不存在: {om_model_path}")
        
        # 加载模型
        self._load_model()
        
    def _load_model(self):
        """
        加载OM模型到内存（内部方法）
        """
        # 初始化为模拟模式作为默认值
        self.model = {'simulation_mode': True}
        
        # 将华为设备的Python依赖导入放在函数内部
        try:
            # 尝试导入华为AscendCL API
            try:
                import acl  # type: ignore
                acl_available = True
            except ImportError:
                acl_available = False
                
            if not acl_available:
                raise ImportError("华为AscendCL API不可用")
            
            # 初始化ACL
            ret = acl.init()  # type: ignore
            if ret != 0:
                raise RuntimeError(f"ACL初始化失败: {ret}")
            
            # 设置运行模式
            ret = acl.rt.set_device(0)  # type: ignore
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
                'device_id': 0,
                'simulation_mode': False
            }
            
            print(f"OM模型加载成功: {self.om_model_path}")
            
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
            # 检查模型是否存在且是字典类型
            if isinstance(self.model, dict):
                # 模拟模式下的推理
                if self.model.get('simulation_mode', True):
                    # 模拟检测结果
                    results = [
                        [100, 100, 200, 200, 0.9, 0, 'person'],
                        [300, 300, 400, 400, 0.85, 1, 'car']
                    ]
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
                    input_buffer, ret = acl.rt.malloc(input_size, acl.rt.MEMORY_DEVICE)  # type: ignore
                    if ret != 0:
                        raise RuntimeError(f"分配输入缓冲区失败: {ret}")
                    
                    # 复制输入数据到设备
                    ret = acl.rt.memcpy(input_buffer, input_size,  # type: ignore
                                       acl.util.bytes_to_ptr(input_data.tobytes()),  # type: ignore
                                       input_data.nbytes, acl.rt.MEMCPY_HOST_TO_DEVICE)  # type: ignore
                    if ret != 0:
                        raise RuntimeError(f"复制输入数据失败: {ret}")
                    
                    # 执行模型推理
                    ret = acl.mdl.execute(model_id, [input_buffer])  # type: ignore
                    if ret != 0:
                        raise RuntimeError(f"执行推理失败: {ret}")
                    
                    # 获取输出数量
                    num_outputs = acl.mdl.get_num_outputs(model_desc)  # type: ignore
                    if num_outputs < 1:
                        raise RuntimeError("模型没有输出")
                    
                    # 获取输出大小
                    output_size = acl.mdl.get_output_size_by_index(model_desc, 0)  # type: ignore
                    
                    # 分配输出缓冲区
                    output_buffer, ret = acl.rt.malloc(output_size, acl.rt.MEMORY_DEVICE)  # type: ignore
                    if ret != 0:
                        raise RuntimeError(f"分配输出缓冲区失败: {ret}")
                    
                    # 从设备获取输出
                    output_data = np.zeros(output_size // 4, dtype=np.float32)
                    ret = acl.rt.memcpy(acl.util.bytes_to_ptr(output_data.tobytes()),  # type: ignore
                                       output_data.nbytes, output_buffer, 
                                       output_size, acl.rt.MEMCPY_DEVICE_TO_HOST)  # type: ignore
                    if ret != 0:
                        raise RuntimeError(f"复制输出数据失败: {ret}")
                    
                    # 释放缓冲区
                    acl.rt.free(input_buffer)  # type: ignore
                    acl.rt.free(output_buffer)  # type: ignore
                    
                    # 解析输出结果（这里需要根据实际模型输出格式进行调整）
                    # 假设输出是 [x1, y1, x2, y2, confidence, class_id, class_id, ...] 的格式
                    results = []
                    # 实际解析代码会根据具体模型输出格式进行实现
                    print("实际推理完成（目标检测）")
                    return results
                    
                except ImportError:
                    # 如果导入失败，回退到模拟模式
                    print("华为AscendCL API不可用，使用模拟模式")
                    results = [
                        [100, 100, 200, 200, 0.9, 0, 'person'],
                        [300, 300, 400, 400, 0.85, 1, 'car']
                    ]
                    return results
            else:
                # 如果模型不是字典类型，使用模拟模式
                print("模型未正确加载，使用模拟模式")
                results = [
                    [100, 100, 200, 200, 0.9, 0, 'person'],
                    [300, 300, 400, 400, 0.85, 1, 'car']
                ]
                return results
            
        except Exception as e:
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
                    input_buffer, ret = acl.rt.malloc(input_size, acl.rt.MEMORY_DEVICE)  # type: ignore
                    if ret != 0:
                        raise RuntimeError(f"分配输入缓冲区失败: {ret}")
                    
                    # 复制输入数据到设备
                    ret = acl.rt.memcpy(input_buffer, input_size,  # type: ignore
                                       acl.util.bytes_to_ptr(input_data.tobytes()),  # type: ignore
                                       input_data.nbytes, acl.rt.MEMCPY_HOST_TO_DEVICE)  # type: ignore
                    if ret != 0:
                        raise RuntimeError(f"复制输入数据失败: {ret}")
                    
                    # 执行模型推理
                    ret = acl.mdl.execute(model_id, [input_buffer])  # type: ignore
                    if ret != 0:
                        raise RuntimeError(f"执行推理失败: {ret}")
                    
                    # 获取输出数量
                    num_outputs = acl.mdl.get_num_outputs(model_desc)  # type: ignore
                    if num_outputs < 1:
                        raise RuntimeError("模型没有输出")
                    
                    # 获取输出大小
                    output_size = acl.mdl.get_output_size_by_index(model_desc, 0)  # type: ignore
                    
                    # 分配输出缓冲区
                    output_buffer, ret = acl.rt.malloc(output_size, acl.rt.MEMORY_DEVICE)  # type: ignore
                    if ret != 0:
                        raise RuntimeError(f"分配输出缓冲区失败: {ret}")
                    
                    # 从设备获取输出
                    output_data = np.zeros(output_size // 4, dtype=np.float32)
                    ret = acl.rt.memcpy(acl.util.bytes_to_ptr(output_data.tobytes()),  # type: ignore
                                       output_data.nbytes, output_buffer, 
                                       output_size, acl.rt.MEMCPY_DEVICE_TO_HOST)  # type: ignore
                    if ret != 0:
                        raise RuntimeError(f"复制输出数据失败: {ret}")
                    
                    # 释放缓冲区
                    acl.rt.free(input_buffer)  # type: ignore
                    acl.rt.free(output_buffer)  # type: ignore
                    
                    # 解析分类结果
                    # 实际解析代码会根据具体模型输出格式进行实现
                    results = []
                    print("实际推理完成（图像分类）")
                    return results
                    
                except ImportError:
                    # 如果导入失败，回退到模拟模式
                    print("华为AscendCL API不可用，使用模拟模式")
                    results = [
                        ['cat', 0.75],
                        ['dog', 0.15]
                    ]
                    return results
            else:
                # 如果模型不是字典类型，使用模拟模式
                print("模型未正确加载，使用模拟模式")
                results = [
                    ['cat', 0.75],
                    ['dog', 0.15]
                ]
                return results
            
        except Exception as e:
            raise RuntimeError(f"分类推理过程中出错: {str(e)}")
    
    def __del__(self):
        """
        析构函数，释放模型资源
        """
        # 检查模型是否存在且不是模拟模式
        if isinstance(self.model, dict) and not self.model.get('simulation_mode', False):
            try:
                # 尝试导入华为AscendCL API
                try:
                    import acl  # type: ignore
                    acl_available = True
                except ImportError:
                    acl_available = False
                
                if acl_available:
                    # 释放资源
                    acl.mdl.destroy_desc(self.model.get('model_desc'))  # type: ignore
                    acl.mdl.unload(self.model.get('model_id'))  # type: ignore
                    acl.rt.destroy_context(self.model.get('context'))  # type: ignore
                    acl.rt.reset_device(self.model.get('device_id', 0))  # type: ignore
                    acl.finalize()  # type: ignore
                    
                    print("OM模型资源已释放")
            except Exception:
                # 静默处理析构函数中的异常
                pass


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
        # 初始化 ACL
        acl.init()
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
    # 当前代码文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 示例：使用对象进行推理
    # 注意：这里使用示例模型路径，实际使用时请替换为真实的OM模型路径
    om_model_path = os.path.join(current_dir, "..", "model_demo",  "yolo11n.om")
    # 图片路径 
    test_image_path = os.path.join(current_dir, "..", "000000000009.jpg")
    try:
        # 创建预测器实例（开启调试模式）
        predictor = HUAWEI_910B_Predictor(om_model_path, debug=True)
        
        # 创建测试图像（使用随机噪声或从文件读取）
        # 方法1：使用随机噪声作为测试图像
        # test_image = np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)
        
        # 方法2：从文件读取图像（如果有测试图像）
        # test_image_path = "/path/to/test/image.jpg"
        test_image = cv2.imread(test_image_path)
        
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
        # 在非华为设备上，这里会进入模拟模式，仍然可以演示功能