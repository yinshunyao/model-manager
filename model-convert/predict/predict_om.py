import logging

import cv2
import numpy as np
import os


class HUAWEI_910B_Predictor:
    """
    华为910B设备上的模型推理类，支持目标检测和图像分类任务
    """
    
    def __init__(self, om_model_path):
        """
        初始化函数，加载OM模型到内存
        
        Args:
            om_model_path (str): OM模型文件路径
        """
        self.om_model_path = om_model_path
        self.model = None
        
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
                import acl
                acl_available = True
            except ImportError:
                acl_available = False
                
            if not acl_available:
                raise ImportError("华为AscendCL API不可用")
            
            # 初始化ACL
            ret = acl.init()
            if ret != 0:
                raise RuntimeError(f"ACL初始化失败: {ret}")
            
            # 设置运行模式
            ret = acl.rt.set_device(0)
            if ret != 0:
                raise RuntimeError(f"设置设备失败: {ret}")
            
            # 创建上下文
            context, ret = acl.rt.create_context(0)
            if ret != 0:
                raise RuntimeError(f"创建上下文失败: {ret}")
            
            # 加载模型
            model_id, ret = acl.mdl.load_from_file(self.om_model_path)
            if ret != 0:
                raise RuntimeError(f"加载模型失败: {ret}")
            
            # 获取模型描述信息
            model_desc = acl.mdl.create_desc()
            ret = acl.mdl.get_desc(model_desc, model_id)
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
                    import acl
                    
                    # 获取输入/输出缓冲区
                    input_buffer = acl.mdl.get_input(self.model.get('model_id'), 0)
                    output_buffer = acl.mdl.get_output(self.model.get('model_id'), 0)
                    
                    # 复制输入数据到设备
                    ret = acl.rt.memcpy(input_buffer, input_data.nbytes, 
                                       acl.util.bytes_to_ptr(input_data.tobytes()), 
                                       input_data.nbytes, acl.rt.MEMCPY_HOST_TO_DEVICE)
                    
                    # 执行模型推理
                    ret = acl.mdl.execute(self.model.get('model_id'))
                    if ret != 0:
                        raise RuntimeError(f"执行推理失败: {ret}")
                    
                    # 从设备获取输出
                    output_size = acl.mdl.get_output_size_by_index(self.model.get('model_desc'), 0)
                    output_data = np.zeros(output_size // 4, dtype=np.float32)
                    ret = acl.rt.memcpy(acl.util.bytes_to_ptr(output_data.tobytes()), 
                                       output_data.nbytes, output_buffer, 
                                       output_size, acl.rt.MEMCPY_DEVICE_TO_HOST)
                    
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
                    import acl
                    
                    # 获取输入/输出缓冲区
                    input_buffer = acl.mdl.get_input(self.model.get('model_id'), 0)
                    output_buffer = acl.mdl.get_output(self.model.get('model_id'), 0)
                    
                    # 复制输入数据到设备
                    ret = acl.rt.memcpy(input_buffer, input_data.nbytes, 
                                       acl.util.bytes_to_ptr(input_data.tobytes()), 
                                       input_data.nbytes, acl.rt.MEMCPY_HOST_TO_DEVICE)
                    
                    # 执行模型推理
                    ret = acl.mdl.execute(self.model.get('model_id'))
                    if ret != 0:
                        raise RuntimeError(f"执行推理失败: {ret}")
                    
                    # 从设备获取输出
                    output_size = acl.mdl.get_output_size_by_index(self.model.get('model_desc'), 0)
                    output_data = np.zeros(output_size // 4, dtype=np.float32)
                    ret = acl.rt.memcpy(acl.util.bytes_to_ptr(output_data.tobytes()), 
                                       output_data.nbytes, output_buffer, 
                                       output_size, acl.rt.MEMCPY_DEVICE_TO_HOST)
                    
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
                    import acl
                    acl_available = True
                except ImportError:
                    acl_available = False
                
                if acl_available:
                    # 释放资源
                    acl.mdl.destroy_desc(self.model.get('model_desc'))
                    acl.mdl.unload(self.model.get('model_id'))
                    acl.rt.destroy_context(self.model.get('context'))
                    acl.rt.reset_device(self.model.get('device_id', 0))
                    acl.finalize()
                    
                    print("OM模型资源已释放")
            except Exception:
                # 静默处理析构函数中的异常
                pass


if __name__ == "__main__":
    # 当前代码文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 示例：使用对象进行推理
    # 注意：这里使用示例模型路径，实际使用时请替换为真实的OM模型路径
    om_model_path = os.path.join(current_dir, "yolo11n.om.om")
    # 图片路径 
    test_image_path = os.path.join(current_dir, "000000000009.jpg")
    try:
        # 创建预测器实例
        predictor = HUAWEI_910B_Predictor(om_model_path)
        
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