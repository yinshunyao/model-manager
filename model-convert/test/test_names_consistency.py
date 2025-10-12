#!/usr/bin/env python3
"""
测试脚本：验证PT模型转换为ONNX后类别名称的一致性
"""

import os
import sys
import json

# 添加项目根目录到Python路径
parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(parent)

# 导入要测试的模块
from yolo_to_onnx import convert_yolo11_to_onnx, verify_onnx_model
from predict_onnx import ONNXPredictor

def test_names_consistency():
    """
    测试PT模型转换为ONNX后类别名称的一致性
    """
    print("=== 测试PT模型与ONNX模型类别名称一致性 ===")
    
    # 设置路径
    current_dir = parent
    pt_model_path = os.path.join(current_dir, "model_demo", "yolo11n.pt")
    onnx_model_path = os.path.join(current_dir, "model_demo", "out", "yolo11n_test.onnx")
    
    print(f"PT模型路径: {pt_model_path}")
    print(f"ONNX模型路径: {onnx_model_path}")
    
    # 检查PT模型是否存在
    if not os.path.exists(pt_model_path):
        print(f"错误: PT模型文件不存在: {pt_model_path}")
        return False
    
    try:
        # 1. 加载PT模型获取类别名称
        print("\n1. 加载PT模型获取类别名称...")
        from ultralytics import YOLO
        pt_model = YOLO(pt_model_path)
        pt_names = pt_model.names
        print(f"PT模型类别数量: {len(pt_names)}")
        print(f"PT模型类别名称示例: {list(pt_names.values())[:10]}")
        
        # 2. 转换为ONNX
        print("\n2. 转换PT模型为ONNX...")
        converted_onnx_path = convert_yolo11_to_onnx(
            model_path=pt_model_path,
            output_path=onnx_model_path,
            imgsz=640,
            simplify=True,
            opset_version=17
        )
        
        # 3. 验证ONNX模型
        print("\n3. 验证ONNX模型...")
        verify_onnx_model(converted_onnx_path)
        
        # 4. 使用ONNXPredictor加载ONNX模型并检查类别名称
        print("\n4. 使用ONNXPredictor加载ONNX模型...")
        predictor = ONNXPredictor(converted_onnx_path)
        onnx_names = predictor.class_names
        
        # 5. 比较类别名称
        print("\n5. 比较类别名称一致性...")
        if len(pt_names) == len(onnx_names):
            print("✓ 类别数量一致")
        else:
            print(f"✗ 类别数量不一致: PT={len(pt_names)}, ONNX={len(onnx_names)}")
            return False
        
        # 检查类别名称是否完全一致
        names_match = True
        for class_id, pt_name in pt_names.items():
            if class_id not in onnx_names:
                print(f"✗ 类别ID {class_id} 在ONNX模型中不存在")
                names_match = False
            elif onnx_names[class_id] != pt_name:
                print(f"✗ 类别ID {class_id} 名称不一致: PT='{pt_name}', ONNX='{onnx_names[class_id]}'")
                names_match = False
        
        if names_match:
            print("✓ 所有类别名称完全一致")
        else:
            print("✗ 类别名称存在不一致")
            return False
        
        # 6. 测试推理功能
        print("\n6. 测试推理功能...")
        # 创建一个测试图片（简单的随机图片）
        import numpy as np
        import cv2
        
        test_image_path = os.path.join(current_dir, "test_image.jpg")
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        cv2.imwrite(test_image_path, test_image)
        
        try:
            results = predictor.predict(test_image_path, conf_threshold=0.5)
            if isinstance(results, list):
                print(f"✓ 推理成功，检测到 {len(results)} 个目标")
                if len(results) > 0 and isinstance(results[0], dict):
                    first_result = results[0]
                    if 'class_name' in first_result:
                        print(f"✓ 类别名称功能正常，示例: {first_result['class_name']}")
                    else:
                        print("✗ 推理结果中缺少类别名称")
                        return False
            else:
                print("✓ 推理成功，返回原始输出")
        except Exception as e:
            print(f"✗ 推理测试失败: {e}")
            return False
        finally:
            # 清理测试图片
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
        
        print("\n=== 测试结果 ===")
        print("✓ 所有测试通过！PT模型和ONNX模型的类别名称完全一致")
        return True
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_names_consistency()
    sys.exit(0 if success else 1)
