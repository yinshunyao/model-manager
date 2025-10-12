import logging
import os
import sys
import unittest
import shutil

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入要测试的模块
from yolo_to_onnx import convert_yolo11_to_onnx, verify_onnx_model

class TestYOLOToONNX(unittest.TestCase):
    
    def setUp(self):
        """测试前的准备工作"""
        # 设置测试目录和输出目录
        self.test_dir = os.path.join(os.path.dirname(__file__), "..", "model_demo")
        self.output_dir = os.path.join(self.test_dir, "out")

        logging.warning(f"测试目录: {self.test_dir}, 输出目录: {self.output_dir}")
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.test_model_path = os.path.join(self.test_dir, "yolo11n.pt")
        self.model_name = os.path.basename(self.test_model_path).replace('.pt', '')
        self.test_onnx_path = os.path.join(self.output_dir, "yolo11n.onnx")

    def tearDown(self):
        """测试后的清理工作"""
        # 不删除任何原始模型和转换后的ONNX模型
        pass
    
    def test_file_exists_check(self):
        """简单测试文件存在检查功能"""
        # 测试不存在的文件应该抛出异常
        with self.assertRaises(FileNotFoundError):
            convert_yolo11_to_onnx(
                model_path="nonexistent_model.pt",
                output_path="nonexistent_output.onnx"
            )
    
    def test_output_path_generation(self):
        """测试输出路径生成逻辑"""
        if os.path.exists(self.test_model_path):
            # 测试指定输出路径的情况
            result_path = convert_yolo11_to_onnx(
                model_path=self.test_model_path,
                output_path=self.test_onnx_path
            )
            # 验证文件确实存在于指定位置
            self.assertTrue(os.path.exists(self.test_onnx_path), 
                          f"ONNX文件未在预期位置找到: {self.test_onnx_path}")
            # 验证返回的路径正确
            self.assertEqual(result_path, self.test_onnx_path)
        else:
            self.skipTest(f"测试模型文件不存在: {self.test_model_path}")

if __name__ == '__main__':
    # 执行TestYOLOToONNX的单元测试

    unittest.main()