# ONNX推理可视化调试功能使用说明

## 功能概述

新增的可视化调试功能可以在ONNX模型推理过程中生成带标注的图片，帮助调试和验证推理结果。

## 主要特性

### 1. 可视化标注
- **绿色实线框**: ONNX模型预测结果
- **红色虚线框**: 真实标签（Ground Truth）
- **标签显示**: 类别名称和置信度
- **统计信息**: 预测数量和真实标签数量

### 2. 调试配置
- 支持命令行参数配置
- 可选择性启用/禁用可视化
- 自定义输出目录和置信度阈值

## 使用方法

### 基本用法
```bash
# 运行基本评估（不生成可视化）
python test/test_predict_onnx.py

# 启用可视化调试
python test/test_predict_onnx.py --enable-visualization

# 只测试ONNX模型（跳过PT模型）
python test/test_predict_onnx.py --test-onnx-only --enable-visualization
```

### 高级配置
```bash
# 自定义输出目录
python test/test_predict_onnx.py --enable-visualization --output-dir my_debug_output

# 调整置信度阈值
python test/test_predict_onnx.py --enable-visualization --conf-threshold 0.3

# 完整配置示例
python test/test_predict_onnx.py \
    --enable-visualization \
    --output-dir onnx_debug_results \
    --conf-threshold 0.4 \
    --test-onnx-only
```

## 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable-visualization` | flag | False | 启用可视化调试，生成带标注的图片 |
| `--output-dir` | string | `onnx_debug_output` | 可视化图片输出目录 |
| `--conf-threshold` | float | 0.5 | 置信度阈值 |
| `--test-onnx-only` | flag | False | 只测试ONNX模型，跳过PT模型评估 |

## 输出文件说明

### 可视化图片
- 文件名格式: `debug_XXX_原图片名.jpg`
- 包含预测框（绿色）和真实框（红色虚线）
- 显示类别名称、置信度和统计信息

### 评估报告
- `evaluation_results.json`: 详细的评估结果数据
- 包含精度、召回率、F1分数等指标

## 可视化图片解读

### 标注说明
```
绿色实线框 + 标签: ONNX模型预测结果
  - 显示: "类别名: 置信度"
  - 例如: "person: 0.85"

红色虚线框 + GT标签: 真实标签
  - 显示: "GT: 类别名"  
  - 例如: "GT: person"

图例信息:
  - Green: Predictions
  - Red: Ground Truth
  - Predictions: X, GT: Y
```

### 调试要点
1. **预测框位置**: 检查边界框是否准确
2. **类别识别**: 验证类别名称是否正确
3. **置信度**: 观察置信度是否合理
4. **漏检/误检**: 对比预测框和真实框

## 代码集成

### 在predict_onnx.py中使用

```python
from predict.predict_onnx import draw_detection_results

# 进行推理
results = predictor.predict("image.jpg")

# 生成可视化
output_path = draw_detection_results(
    "image.jpg",
    results,
    output_path="result_with_boxes.jpg",
    conf_threshold=0.5
)
```

### 在测试代码中使用
```python
# 启用可视化调试
evaluator.evaluate_onnx_model(
    model_path,
    save_visualizations=True,
    output_dir="debug_output"
)
```

## 调试技巧

### 1. 置信度调优
- 降低置信度阈值看到更多检测结果
- 提高置信度阈值过滤低质量检测

### 2. 类别验证
- 检查类别名称是否正确加载
- 验证类别ID是否在有效范围内（0-79）

### 3. 边界框检查
- 确认坐标转换是否正确
- 检查边界框是否超出图片范围

### 4. 性能分析
- 对比预测框和真实框的重叠情况
- 分析漏检和误检的原因

## 故障排除

### 常见问题
1. **OpenCV导入错误**: 确保安装了opencv-python
2. **图片无法保存**: 检查输出目录权限
3. **坐标显示异常**: 验证坐标转换逻辑

### 解决方案
```bash
# 安装OpenCV
pip install opencv-python

# 检查目录权限
ls -la onnx_debug_output/

# 验证坐标转换
python -c "
import numpy as np
# 测试坐标转换逻辑
bbox = [0.5, 0.5, 0.8, 0.8]  # 归一化坐标
img_width, img_height = 640, 480
x1 = int(bbox[0] * img_width)
print(f'转换后坐标: {x1}')
"
```

## 扩展功能

### 自定义可视化
可以修改 `_save_visualization_result` 方法来：
- 调整颜色方案
- 修改标签格式
- 添加更多统计信息
- 实现不同的绘制效果

### 批量处理
```python
# 批量生成可视化
for image_path in image_list:
    results = predictor.predict(image_path)
    draw_detection_results(image_path, results)
```

这个可视化调试功能大大提高了ONNX模型推理的调试效率，帮助快速定位和解决问题。
