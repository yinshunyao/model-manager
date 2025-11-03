import os
import argparse
import sys

# 确保核心依赖已正确导入
try:
    import torch
except ImportError:
    print("错误: 未安装torch。请运行 'pip install -r requirements.txt' 安装所需依赖。")
    sys.exit(1)
    
try:
    import numpy as np
except ImportError:
    print("错误: 未安装numpy。请运行 'pip install -r requirements.txt' 安装所需依赖。")
    sys.exit(1)
    
try:
    from ultralytics import YOLO
except ImportError:
    print("错误: 未安装ultralytics。请运行 'pip install -r requirements.txt' 安装所需依赖。")
    sys.exit(1)
    

def convert_yolo11_to_onnx(model_path, output_path=None, imgsz=640, simplify=False, opset_version=17):
    """
    将YOLOv11模型转换为ONNX格式
    
    Args:
        model_path (str): 输入的YOLOv11模型路径
        output_path (str): 输出的ONNX文件路径，如果不指定则在同一目录下生成
        imgsz (int): 输入图像的大小
        simplify (bool): 是否简化ONNX模型
        opset_version (int): ONNX的opset版本
    
    Returns:
        str: 生成的ONNX文件路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 如果未指定输出路径，则在同一目录下生成
    if output_path is None:
        base_name = os.path.basename(model_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(os.path.dirname(model_path), f"{name_without_ext}.onnx")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path) + os.sep
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取导出模型名称
    output_name = os.path.basename(output_path)
    
    print(f"开始加载YOLOv11模型: {model_path}")
    # 加载YOLOv11模型
    model = YOLO(model_path)
    
    # 创建示例输入
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_input = torch.zeros((1, 3, imgsz, imgsz), device=device)
    
    print(f"开始转换模型到ONNX格式...")
    print(f"输出目录: {output_dir}")
    print(f"图像大小: {imgsz}x{imgsz}")
    print(f"ONNX Opset版本: {opset_version}")
    print(f"是否简化模型: {simplify}")
    
    # 获取原模型的类别名称
    original_names = model.names if hasattr(model, 'names') else None
    print(f"原模型类别数量: {len(original_names) if original_names else '未知'}")
    if original_names:
        # 处理类别名称，无论是字典还是列表
        if isinstance(original_names, dict):
            print(f"类别名称示例: {list(original_names.values())[:10]}")
        else:
            # 假设是列表或其他可迭代对象
            print(f"类别名称示例: {list(original_names)[:10]}")
    
    # 导出为ONNX
    # 注意：YOLO的export方法可能不会完全按照save_dir和name参数工作
    # 我们需要手动处理输出路径
    try:
        exported_path = model.export(
            format='onnx',
            imgsz=imgsz,
            opset=opset_version,
            simplify=simplify,
            device=device,
            # 使用正确的参数确保输出完整的特征维度
            dynamic=False,
            batch=1
        )
        
        # 如果导出的文件不在期望的位置，移动它
        if exported_path != output_path:
            import shutil
            if os.path.exists(exported_path):
                shutil.move(exported_path, output_path)
                print(f"模型已移动到指定位置: {output_path}")
            else:
                print(f"警告：导出的模型文件未找到: {exported_path}")
        
        # 将类别名称写入同名的JSON文件，便于推理端读取
        if original_names:
            save_class_names_json(output_path, original_names)
            # 同时也保存在ONNX元数据中（可选，作为回退方案）
            preserve_model_names(output_path, original_names)
            
    except Exception as e:
        print(f"ONNX导出失败: {e}")
        print("这通常是由于NumPy版本兼容性问题导致的")
        print("建议解决方案:")
        print("1. 降级NumPy: pip install 'numpy<2'")
        print("2. 或者升级相关包: pip install --upgrade onnx onnxruntime")
        
        # 即使导出失败，我们仍然可以保存类别名称信息
        if original_names:
            save_class_names_json(output_path, original_names)
            print(f"类别名称已保存到JSON文件，即使ONNX导出失败")
        
        raise RuntimeError(f"ONNX导出失败: {e}")
    
    # 验证导出的模型
    if verify_onnx_model(output_path):
        print(f"\n转换成功！ONNX模型已保存至: {output_path}")
    else:
        print(f"\n警告：ONNX模型可能存在问题，请检查")
    
    return output_path

def preserve_model_names(onnx_path, class_names):
    """
    确保ONNX模型保留类别名称信息
    
    Args:
        onnx_path (str): ONNX模型路径
        class_names (dict): 类别名称字典 {class_id: class_name}
    """
    try:
        import onnx
        import json
        
        print("正在确保ONNX模型保留类别名称...")
        
        # 加载ONNX模型
        model = onnx.load(onnx_path)
        
        # 创建类别名称的元数据
        names_metadata = {
            "names": class_names,
            "nc": len(class_names)  # 类别数量
        }
        
        # 将类别名称信息添加到模型的元数据中
        if model.metadata_props:
            # 如果已有元数据，更新或添加
            for prop in model.metadata_props:
                if prop.key == "class_names":
                    prop.value = json.dumps(names_metadata)
                    break
            else:
                # 如果没有找到class_names，添加新的元数据
                # 直接创建StringStringEntryProto对象
                from onnx import StringStringEntryProto
                entry = StringStringEntryProto()
                entry.key = "class_names"
                entry.value = json.dumps(names_metadata)
                model.metadata_props.append(entry)
        else:
            # 如果没有元数据，创建新的
            # 直接创建StringStringEntryProto对象
            from onnx import StringStringEntryProto
            entry = StringStringEntryProto()
            entry.key = "class_names"
            entry.value = json.dumps(names_metadata)
            model.metadata_props.append(entry)
        
        # 保存更新后的模型
        onnx.save(model, onnx_path)
        
        print(f"类别名称已保存到ONNX模型元数据中")
        print(f"类别数量: {len(class_names)}")
        
    except ImportError:
        print("警告: 未安装onnx库，无法保存类别名称到模型元数据")
    except Exception as e:
        print(f"保存类别名称时出错: {e}")

def save_class_names_json(onnx_path, class_names):
    """
    将类别名称保存到与ONNX同名的JSON文件中（扩展名为 .names.json）

    Args:
        onnx_path (str): ONNX模型路径
        class_names: 类别名称（字典或列表格式）
    """
    try:
        import json
        base, _ = os.path.splitext(onnx_path)
        names_path = f"{base}.names.json"

        # 标准化类别名称格式，确保始终保存为字典格式
        if isinstance(class_names, dict):
            standardized_names = class_names
        else:
            # 如果是列表或其他可迭代对象，转换为字典 {索引: 名称}
            standardized_names = {i: name for i, name in enumerate(class_names)}

        # 注意：JSON的键为字符串，这里直接保存，由读取端恢复为整型键
        payload = {
            "names": standardized_names,
            "nc": len(standardized_names)
        }

        with open(names_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"类别名称已保存到文件: {names_path}")
    except Exception as e:
        print(f"保存类别名称JSON失败: {e}")

def extract_yolo_names_only(model_path, output_path=None):
    """
    仅提取YOLO模型的类别名称并保存到JSON文件（不进行ONNX转换）
    用于调试或当ONNX转换失败时仍能获取类别名称
    
    Args:
        model_path (str): YOLO模型路径
        output_path (str): 输出JSON文件路径，如果不指定则在同一目录下生成
        
    Returns:
        str: 生成的JSON文件路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 如果未指定输出路径，则在同一目录下生成
    if output_path is None:
        base_name = os.path.basename(model_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(os.path.dirname(model_path), f"{name_without_ext}.names.json")
    
    print(f"开始加载YOLOv11模型: {model_path}")
    # 加载YOLOv11模型
    model = YOLO(model_path)
    
    # 获取原模型的类别名称
    original_names = model.names if hasattr(model, 'names') else None
    print(f"原模型类别数量: {len(original_names) if original_names else '未知'}")
    if original_names:
        # 处理类别名称，无论是字典还是列表
        if isinstance(original_names, dict):
            print(f"类别名称示例: {list(original_names.values())[:10]}")
        else:
            # 假设是列表或其他可迭代对象
            print(f"类别名称示例: {list(original_names)[:10]}")
        
        # 保存类别名称
        save_class_names_json(output_path, original_names)
        print(f"类别名称已成功提取并保存到: {output_path}")
        return output_path
    else:
        print("警告: 模型中未找到类别名称信息")
        return None

def verify_onnx_model(onnx_path):
    """
    验证ONNX模型是否有效
    
    Args:
        onnx_path (str): ONNX模型路径
    
    Returns:
        bool: 模型是否有效
    """
    # 尝试导入onnx库
    try:
        import onnx
    except ImportError:
        print("未安装onnx库，跳过模型验证")
        return True  # 跳过验证，假设成功
    
    try:
        # 加载模型
        model = onnx.load(onnx_path)
        
        # 验证模型
        onnx.checker.check_model(model)
        print("ONNX模型验证通过")
        
        # 获取模型信息
        print(f"模型输入数量: {len(model.graph.input)}")
        print(f"模型输出数量: {len(model.graph.output)}")
        
        # 打印输入信息
        for input in model.graph.input:
            print(f"输入名称: {input.name}")
            print(f"输入形状: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")
        
        # 检查并显示类别名称信息
        if model.metadata_props:
            for prop in model.metadata_props:
                if prop.key == "class_names":
                    try:
                        import json
                        names_data = json.loads(prop.value)
                        if "names" in names_data:
                            names = names_data["names"]
                            print(f"模型包含类别名称: {len(names)} 个类别")
                            print(f"类别示例: {list(names.values())[:10]}")
                        break
                    except:
                        print("无法解析类别名称信息")
        
        return True
    except Exception as e:
        print(f"ONNX模型验证失败: {e}")
        return False