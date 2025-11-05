import os
import sys
import tempfile
import shutil

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_onnx_to_rknn_conversion():
    """
    测试ONNX到RKNN的转换功能
    """
    # 准备测试数据
    test_dir = os.path.join(os.path.dirname(__file__), "..", "model_demo")
    onnx_model_path = os.path.join(test_dir, "yolo11n.onnx")
    
    # 检查ONNX模型是否存在
    if not os.path.exists(onnx_model_path):
        print("ONNX模型文件不存在，请先运行YOLO到ONNX的转换")
        return False
    
    # 创建临时输出目录
    temp_output_dir = tempfile.mkdtemp(prefix="model_convert_test_onnx_to_rknn_")
    
    try:
        from convert.onnx_to_rknn import onnx_to_rknn
        
        # 执行转换
        output_rknn_path = os.path.join(temp_output_dir, "yolo11n.rknn")
        
        print(f"开始ONNX到RKNN转换: {onnx_model_path} -> {output_rknn_path}")
        
        success = onnx_to_rknn(
            onnx_model_path=onnx_model_path,
            output_rknn_path=output_rknn_path,
            input_shape="1,3,640,640",
            target_platform="rk3588",
            precision_mode="float32"
        )
        
        if success:
            print(f"ONNX到RKNN转换成功: {output_rknn_path}")
            # 检查输出文件是否存在
            if os.path.exists(output_rknn_path):
                print(f"输出RKNN模型文件已生成: {output_rknn_path}")
                return True
            else:
                print("输出RKNN模型文件未生成")
                return False
        else:
            print("ONNX到RKNN转换失败")
            return False
            
    except Exception as e:
        print(f"ONNX到RKNN转换异常: {str(e)}")
        return False
    finally:
        # 清理临时目录
        shutil.rmtree(temp_output_dir)
        print(f"清理临时目录: {temp_output_dir}")

if __name__ == "__main__":
    test_onnx_to_rknn_conversion()