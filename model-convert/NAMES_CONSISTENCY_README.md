# ONNX模型类别名称一致性修改说明

## 概述

本次修改确保了从YOLO PT模型转换为ONNX模型后，类别名称（names）信息能够完整保留，并在推理时正确使用。

## 主要修改

### 1. 修改 `yolo_to_onnx.py`

#### 新增功能：
- **获取原模型类别名称**：在转换前从PT模型中提取类别名称信息
- **保存类别名称到ONNX元数据**：新增 `preserve_model_names()` 函数，将类别名称保存到ONNX模型的元数据中
- **验证时显示类别名称**：修改 `verify_onnx_model()` 函数，在验证时显示模型的类别名称信息

#### 关键代码变更：
```python
# 获取原模型的类别名称
original_names = model.names if hasattr(model, 'names') else None

# 确保ONNX模型保留类别名称信息
if original_names:
    preserve_model_names(output_path, original_names)
```

#### 新增函数：
- `preserve_model_names(onnx_path, class_names)`: 将类别名称信息保存到ONNX模型元数据中

### 2. 修改 `predict_onnx.py`

#### 新增功能：
- **从ONNX元数据加载类别名称**：新增 `_load_class_names()` 方法
- **推理结果包含类别名称**：修改 `postprocess_yolo()` 方法，在检测结果中包含类别名称
- **模型信息包含类别名称**：修改 `get_model_info()` 方法，返回类别名称信息

#### 关键代码变更：
```python
# 在初始化时加载类别名称
self.class_names = {}

# 从模型元数据中读取类别名称
self._load_class_names()

# 推理结果包含类别名称
results.append({
    'bbox': [x1, y1, x2, y2],
    'conf': conf,
    'class': int(class_id),
    'class_name': class_name
})
```

#### 新增方法：
- `_load_class_names()`: 从ONNX模型元数据中加载类别名称

### 3. 修改 `test/test_predict_onnx.py`

#### 新增功能：
- **检查ONNX模型类别名称**：在评估ONNX模型时检查是否成功加载类别名称
- **推理结果包含类别名称**：在评估过程中使用类别名称信息

### 4. 新增测试脚本

#### `test_names_consistency.py`
- 完整的端到端测试，验证PT模型转换为ONNX后类别名称的一致性
- 测试包括：类别数量一致性、类别名称一致性、推理功能测试

## 使用方法

### 1. 转换模型时保留类别名称
```python
from yolo_to_onnx import convert_yolo11_to_onnx

# 转换模型，自动保留类别名称
onnx_path = convert_yolo11_to_onnx(
    model_path="model.pt",
    output_path="model.onnx"
)
```

### 2. 使用ONNX模型推理
```python
from predict_onnx import ONNXPredictor

# 创建推理器，自动加载类别名称
predictor = ONNXPredictor("model.onnx")

# 进行推理，结果包含类别名称
results = predictor.predict("image.jpg")
for result in results:
    print(f"检测到: {result['class_name']}, 置信度: {result['conf']:.3f}")
```

### 3. 获取模型信息
```python
# 获取包含类别名称的模型信息
model_info = predictor.get_model_info()
print(f"类别数量: {model_info['num_classes']}")
print(f"类别名称: {model_info['class_names']}")
```

## 测试验证

运行测试脚本验证功能：
```bash
cd model-convert
python test_names_consistency.py
```

## 技术细节

### 类别名称存储格式
类别名称信息以JSON格式存储在ONNX模型的元数据中：
```json
{
    "names": {
        "0": "person",
        "1": "bicycle",
        "2": "car",
        ...
    },
    "nc": 80
}
```

### 兼容性
- 向后兼容：如果ONNX模型没有类别名称信息，推理器仍能正常工作
- 错误处理：如果读取类别名称失败，会显示警告但不影响推理功能
- 依赖检查：自动检查是否安装了必要的依赖库（onnx）

## 注意事项

1. **依赖要求**：需要安装 `onnx` 库才能保存和读取类别名称信息
2. **元数据大小**：类别名称会增加ONNX文件的大小，但增加量很小
3. **版本兼容**：确保使用的onnx库版本支持元数据操作

## 验证清单

- [x] PT模型类别名称正确提取
- [x] ONNX模型元数据正确保存类别名称
- [x] 推理器正确读取类别名称
- [x] 推理结果包含类别名称
- [x] 模型信息包含类别名称
- [x] 测试脚本验证功能正常
- [x] 错误处理和兼容性良好
