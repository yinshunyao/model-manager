# 跨平台智能算法模型转化工具

## 项目简介
跨平台智能算法模型转化工具是一个功能强大的模型管理和转换系统，支持多种模型格式之间的转换（如 YOLO 到 ONNX，ONNX 到 OM），并提供模型推理、任务管理、可视化等功能。该工具旨在简化模型在不同平台之间的部署和使用流程，提高开发效率。

## 主要功能

### 模型转换
- **YOLO 到 ONNX**：支持将 YOLO 系列模型（如 YOLO11）转换为 ONNX 格式
- **ONNX 到 OM**：支持将 ONNX 模型转换为华为昇腾平台的 OM 格式
- 自动处理模型转换过程中的兼容性问题
- 支持批量转换多个模型

### 模型预测
- **ONNX 模型推理**：支持直接运行 ONNX 格式的模型进行推理
- **OM 模型推理**：支持在华为昇腾平台上运行 OM 格式的模型
- 支持图像分类、目标检测、语义分割等多种任务类型
- 提供可视化的推理结果输出

### 任务管理服务
- 提供 RESTful API 接口，支持远程调用模型转换和推理功能
- 任务调度和管理系统，支持任务的创建、查询、取消等操作
- 数据库存储任务历史和结果，方便查询和统计
- 支持任务优先级和并发控制

### 可视化界面
- 提供直观的 Web 界面，方便用户上传模型、配置转换参数、启动任务
- 实时显示任务进度和状态
- 可视化展示推理结果，包括图像标注、统计信息等
- 支持模型性能评估和对比

## 目录结构

```
├── model-convert/          # 模型转换核心模块
│   ├── convert/           # 转换算法实现
│   │   ├── onnx_to_om.py  # ONNX 到 OM 转换
│   │   └── yolo_to_onnx.py # YOLO 到 ONNX 转换
│   ├── predict/           # 模型推理模块
│   │   ├── predict_om.py  # OM 模型推理
│   │   └── predict_onnx.py # ONNX 模型推理
│   ├── service/           # 服务模块
│   │   ├── task_api.py    # API 接口实现
│   │   ├── task_manager.py # 任务管理
│   │   └── task_scheduler.py # 任务调度
│   ├── test/              # 测试模块
│   ├── config/            # 配置文件
│   ├── model_demo/        # 示例模型
│   └── output/            # 输出目录
├── example/               # 示例代码
├── 文档/                   # 项目文档
├── 前端程序.rar            # 前端程序包
├── 后端代码.rar            # 后端代码包
└── readme.md              # 项目说明文档
```

## 安装和配置

### 环境要求
- Python 3.8+ 
- 华为昇腾 NPU 环境（可选，用于 OM 模型转换和推理）
- 依赖库：numpy, onnx, onnxruntime, opencv-python, ultralytics 等

### 安装步骤

1. 克隆项目到本地
   ```bash
   git clone <仓库地址>
   cd model-manager
   ```

2. 安装依赖库
   ```bash
   pip install -r model-convert/requirements.txt
   # 如需使用 REST API 服务，安装额外依赖
   pip install -r model-convert/requirements-rest.txt
   ```

3. 配置环境
   - 编辑 `model-convert/config/config.yaml` 文件，设置模型转换和推理的参数
   - 如需使用华为昇腾平台，配置昇腾环境变量和相关参数

## 使用方法

### 模型转换示例

1. YOLO 到 ONNX 转换
   ```python
   from model_convert.convert.yolo_to_onnx import yolo_to_onnx
   
   # 转换 YOLO 模型到 ONNX
   yolo_to_onnx(
       model_path="model_demo/yolo11n.pt",
       output_path="output/yolo11n.onnx",
       input_shape=(640, 640),
       opset_version=13
   )
   ```

2. ONNX 到 OM 转换
   ```python
   from model_convert.convert.onnx_to_om import onnx_to_om
   
   # 转换 ONNX 模型到 OM
   onnx_to_om(
       onnx_path="output/yolo11n.onnx",
       om_path="output/yolo11n.om",
       config_path="config/config-910b.yaml",
       input_shape="640,640,3"
   )
   ```

### 模型推理示例

1. ONNX 模型推理
   ```python
   from model_convert.predict.predict_onnx import predict_onnx
   
   # 运行 ONNX 模型推理
   predict_onnx(
       model_path="output/yolo11n.onnx",
       image_path="dataset/coco8/images/train2017/000000000009.jpg",
       output_path="output/detection_result.jpg",
       task_type="detect"
   )
   ```

2. OM 模型推理
   ```python
   from model_convert.predict.predict_om import predict_om
   
   # 运行 OM 模型推理
   predict_om(
       model_path="output/yolo11n.om",
       image_path="dataset/coco8/images/train2017/000000000009.jpg",
       output_path="output/detection_result.jpg",
       task_type="detect"
   )
   ```

### REST API 服务使用

1. 启动服务
   ```bash
   cd model-convert
   python rest_convert.py
   ```

2. 调用 API 进行模型转换
   ```bash
   curl -X POST http://localhost:5000/convert/yolo-to-onnx \
        -H "Content-Type: application/json" \
        -d '{
            "model_path": "model_demo/yolo11n.pt",
            "output_path": "output/yolo11n.onnx",
            "input_shape": [640, 640],
            "opset_version": 13
        }'
   ```

3. 调用 API 进行模型推理
   ```bash
   curl -X POST http://localhost:5000/predict/onnx \
        -H "Content-Type: application/json" \
        -d '{
            "model_path": "output/yolo11n.onnx",
            "image_path": "dataset/coco8/images/train2017/000000000009.jpg",
            "output_path": "output/detection_result.jpg",
            "task_type": "detect"
        }'
   ```

## 服务模块说明

服务模块提供了完整的任务管理系统，包括：
- **任务 API**：提供 RESTful 接口，支持任务的创建、查询、取消等操作
- **任务管理器**：负责任务的生命周期管理，包括任务创建、执行、状态更新等
- **任务调度器**：支持任务的定时执行和优先级调度
- **数据库**：存储任务历史和结果，方便查询和统计

详细的 API 文档请参考 `model-convert/TASK_MANAGEMENT_API.md`。

## 测试说明

项目提供了完善的测试用例，覆盖模型转换、推理和服务模块：

1. 运行测试用例
   ```bash
   cd model-convert/test
   python -m pytest test_yolo_to_onnx.py -v
   python -m pytest test_rest_onnx_to_om.py -v
   ```

2. 性能测试
   - 支持模型转换速度测试
   - 支持推理性能测试（包括延迟、吞吐量等）
   - 测试结果保存在 `model-convert/test/evaluation_results.json`

## 文档

项目提供了详细的文档，包括：
- **技术方案文档**：`文档/研发文档/跨平台智能算法模型转化工具技术方案.docx`
- **使用说明书**：`文档/研发文档/跨平台智能算法模型转化工具使用说明书.docx`
- **API 文档**：`model-convert/TASK_MANAGEMENT_API.md`
- **架构图**：`文档/系统架构图.jpg`

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 贡献

欢迎提交 Issue 和 Pull Request。

## 联系方式

如有问题或建议，请联系项目团队：
- 邮箱：example@example.com
- 电话：123-4567-8901