# 跨平台模型转换任务管理系统

本系统提供了一个完整的任务管理框架，用于管理和执行跨平台模型转换任务，支持华为昇腾、瑞芯微和寒武纪平台。

## 系统架构

系统采用模块化设计，主要包含以下组件：

1. **数据库管理模块** (`database.py`): 负责任务信息的持久化存储
2. **任务管理模块** (`task_manager.py`): 负责任务的创建、查询和状态管理
3. **任务执行器模块** (`task_executor.py`): 负责实际执行模型转换任务
4. **任务调度器模块** (`task_scheduler.py`): 负责任务的调度和自动执行
5. **API接口模块** (`task_api.py`): 提供简单易用的函数接口
6. **服务主程序** (`task_service.py`): 作为微服务独立运行

## 功能特点

- ✅ 任务信息持久化到SQLite数据库
- ✅ 支持华为昇腾平台的ONNX转OM转换
- ✅ 预留瑞芯微和寒武纪平台的扩展接口
- ✅ 自动设置华为昇腾环境变量
- ✅ 任务状态管理（等待、运行中、已暂停、完成、失败）
- ✅ 支持并发任务执行控制
- ✅ 提供简单易用的API接口
- ✅ 支持守护进程方式运行
- ✅ 支持分页查询任务列表
- ✅ 支持任务暂停和恢复
- ✅ 支持批量删除任务

## 快速开始

### 1. 创建任务

使用API接口创建模型转换任务：

```python
from service.task_api import create_huawei_onnx_to_om_task

# 创建华为ONNX转OM任务
task_id = create_huawei_onnx_to_om_task(
    input_path='/path/to/your/model.onnx',
    output_path='/path/to/output/model.om',
    parameters={
        'input_shape': '1,3,640,640',
        'auto_input_shape': True,
        'soc_version': 'Ascend910B'
    }
)

print(f"任务创建成功，ID: {task_id}")
```

### 2. 查询任务状态

```python
from service.task_api import get_task_info

# 获取任务信息
task = get_task_info(task_id)
if task:
    print(f"任务状态: {task['status']}")
    print(f"创建时间: {task['created_at']}")
    if task.get('error_message'):
        print(f"错误信息: {task['error_message']}")
```

### 3. 启动任务调度服务

```bash
# 启动任务调度服务（前台运行）
python service/task_service.py

# 或者设置自定义参数
python service/task_service.py --interval 5 --max-concurrent 2

# 以守护进程方式运行
python service/task_service.py --daemon
```

### 4. 立即执行任务

如果需要立即执行某个任务，而不是等待调度器：

```python
from service.task_api import execute_task_immediately

# 立即执行任务
success = execute_task_immediately(task_id)
print(f"任务执行{'成功' if success else '失败'}")
```

### 5. 分页查询任务列表

```python
from service.task_api import get_tasks_paginated

# 获取第1页，每页10条任务
result = get_tasks_paginated(page=1, page_size=10)
print(f"总数: {result['total']}")
print(f"当前页任务数: {len(result['tasks'])}")
for task in result['tasks']:
    print(f"任务ID: {task['id']}, 状态: {task['status']}")
```

### 6. 暂停和恢复任务

```python
from service.task_api import pause_task, resume_task

# 暂停正在执行的任务
success = pause_task(task_id)
if success:
    print("任务已暂停")

# 恢复已暂停的任务
success = resume_task(task_id)
if success:
    print("任务已恢复")
```

### 7. 批量删除任务

```python
from service.task_api import delete_tasks_batch

# 批量删除任务（支持单个或多个）
task_ids = ["task1", "task2", "task3"]
results = delete_tasks_batch(task_ids)
for task_id, success in results.items():
    print(f"任务 {task_id}: {'删除成功' if success else '删除失败'}")
```

## 支持的参数

### 华为ONNX转OM转换参数

- `input_shape`: 模型输入形状，格式为字符串或元组，如 "1,3,640,640"
- `auto_input_shape`: 是否自动从模型获取输入形状
- `soc_version`: 目标昇腾处理器版本，默认为 "Ascend910B"
- `precision_mode`: 精度模式，默认为 "allow_fp32_to_fp16"
- `log_level`: 日志级别，默认为 "error"

## 任务状态说明

- `pending`: 任务等待执行
- `running`: 任务正在执行
- `paused`: 任务已暂停
- `completed`: 任务执行成功
- `failed`: 任务执行失败

## 配置说明

### 华为环境变量设置

系统会自动设置华为昇腾所需的环境变量：

- `ASCEND_HOME`: "/usr/local/Ascend/ascend-toolkit/8.0.0"
- `LD_LIBRARY_PATH`: 包含昇腾库路径
- `PATH`: 包含昇腾工具路径
- `PYTHONPATH`: 包含昇腾Python包路径

如果需要修改环境变量，请编辑 `task_executor.py` 文件中的 `setup_huawei_environment()` 函数。

## 日志

- 服务日志默认保存在 `service/task_service.log`
- 可以通过修改日志配置来自定义日志行为

## 测试

运行测试脚本来验证系统功能：

```bash
python service/test_task_manager.py
```

## 扩展平台支持

要添加对新平台的支持：

1. 在 `task_manager.py` 中的 `SUPPORTED_PLATFORMS` 和 `TASK_TYPES` 中添加新平台
2. 在 `task_executor.py` 中实现对应的转换函数
3. 在 `task_executor.py` 的 `execute_task` 方法中添加平台判断逻辑

## 注意事项

1. 确保华为昇腾工具链已正确安装在 `/usr/local/Ascend/ascend-toolkit/8.0.0`
2. 执行任务时需要确保输入文件存在且可访问
3. 输出目录需要有写入权限
4. 对于MinIO存储的文件，系统会先转换到本地临时目录，然后上传

## 故障排除

1. 转换失败时，检查任务的 `error_message` 字段获取详细错误信息
2. 查看日志文件了解更多执行细节
3. 确保华为环境变量设置正确
4. 验证输入模型文件是否有效