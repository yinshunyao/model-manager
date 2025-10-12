#!/usr/bin/env python3
"""
ONNX推理可视化调试测试脚本
演示如何使用新的调试功能
"""

import os
import sys
import subprocess
from pathlib import Path

def run_debug_test():
    """运行调试测试"""
    print("=== ONNX推理可视化调试测试 ===")
    
    # 检查测试文件是否存在
    test_file = Path(__file__).parent / "test" / "test_predict_onnx.py"
    if not test_file.exists():
        print(f"错误: 测试文件不存在: {test_file}")
        return False
    
    # 检查ONNX模型是否存在
    onnx_model = Path(__file__).parent / "model_demo" / "out" / "yolo11n.onnx"
    if not onnx_model.exists():
        print(f"警告: ONNX模型不存在: {onnx_model}")
        print("请先运行模型转换生成ONNX文件")
        return False
    
    print(f"测试文件: {test_file}")
    print(f"ONNX模型: {onnx_model}")
    
    # 运行不同的测试配置
    test_configs = [
        {
            "name": "基本测试（无可视化）",
            "args": ["--test-onnx-only"]
        },
        {
            "name": "可视化调试测试",
            "args": ["--test-onnx-only", "--enable-visualization", "--conf-threshold", "0.3"]
        }
    ]
    
    for config in test_configs:
        print(f"\n--- {config['name']} ---")
        
        # 构建命令
        cmd = [sys.executable, str(test_file)] + config["args"]
        print(f"执行命令: {' '.join(cmd)}")
        
        try:
            # 运行测试
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ 测试成功完成")
                print("输出摘要:")
                # 显示关键输出
                lines = result.stdout.split('\n')
                for line in lines[-10:]:  # 显示最后10行
                    if line.strip():
                        print(f"  {line}")
            else:
                print("❌ 测试失败")
                print("错误输出:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print("⏰ 测试超时")
        except Exception as e:
            print(f"❌ 执行错误: {e}")
    
    print("\n=== 测试完成 ===")
    print("\n使用说明:")
    print("1. 基本测试: python test/test_predict_onnx.py --test-onnx-only")
    print("2. 可视化调试: python test/test_predict_onnx.py --test-onnx-only --enable-visualization")
    print("3. 查看生成的图片文件来调试推理结果")

def show_usage():
    """显示使用说明"""
    print("\n=== 使用说明 ===")
    print("\n1. 基本评估测试:")
    print("   python test/test_predict_onnx.py --test-onnx-only")
    
    print("\n2. 启用可视化调试:")
    print("   python test/test_predict_onnx.py --test-onnx-only --enable-visualization")
    
    print("\n3. 自定义配置:")
    print("   python test/test_predict_onnx.py \\")
    print("       --test-onnx-only \\")
    print("       --enable-visualization \\")
    print("       --output-dir my_debug_output \\")
    print("       --conf-threshold 0.4")
    
    print("\n4. 完整评估（PT + ONNX）:")
    print("   python test/test_predict_onnx.py --enable-visualization")
    
    print("\n5. 查看帮助:")
    print("   python test/test_predict_onnx.py --help")

def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_usage()
        return
    
    # 运行调试测试
    success = run_debug_test()
    
    if success:
        print("\n🎉 调试功能测试完成！")
    else:
        print("\n❌ 调试功能测试失败")
        print("请检查模型文件和依赖项")
    
    # 显示使用说明
    show_usage()

if __name__ == "__main__":
    main()
