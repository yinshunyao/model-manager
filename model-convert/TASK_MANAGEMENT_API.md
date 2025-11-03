# 任务管理REST API文档

本文档描述了模型转换服务的任务管理REST API接口。

## 新增功能概览

本次更新新增了以下功能：

1. ✅ **任务状态扩展**：添加 `paused`（已暂停）状态
2. ✅ **分页查询任务列表**：支持按页码查询任务
3. ✅ **任务暂停/恢复**：支持暂停正在执行的任务并恢复
4. ✅ **批量删除任务**：支持同时删除多个任务

## API接口列表

### 1. 单任务查询

**接口**: `POST /tasks/query`

**描述**: 根据任务ID查询任务的详细信息

**请求体**:
```json
{
  "task_id": "task_1234567890_1234"
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "获取任务信息成功",
  "data": {
    "task_id": "task_1234567890_1234",
    "task_info": {
      "id": "task_1234567890_1234",
      "task_type": "onnx_to_om",
      "platform": "huawei",
      "status": "pending",
      "input_path": "/path/to/input.onnx",
      "output_path": "/path/to/output.om",
      "created_at": "2024-01-01T12:00:00"
    }
  }
}
```

### 2. 获取任务状态

**接口**: `GET /tasks/{task_id}/status`

**描述**: 获取指定任务的当前状态

**响应示例**:
```json
{
  "success": true,
  "message": "获取任务状态成功",
  "data": {
    "task_id": "task_1234567890_1234",
    "status": "running",
    "is_completed": false,
    "is_failed": false
  }
}
```

### 3. 分页查询任务列表 ✨ 新增

**接口**: `POST /tasks/list`

**描述**: 分页查询任务列表

**请求体**:
```json
{
  "page": 1,
  "page_size": 10
}
```

**请求参数**:
- `page` (int, 可选): 页码，从1开始，默认值为1
- `page_size` (int, 可选): 每页任务数量，默认值为10，最大值为100

**响应示例**:
```json
{
  "success": true,
  "message": "获取任务列表成功",
  "data": {
    "tasks": [
      {
        "id": "task_1234567890_1234",
        "task_type": "onnx_to_om",
        "platform": "huawei",
        "status": "pending",
        "input_path": "/path/to/input.onnx",
        "output_path": "/path/to/output.om",
        "created_at": "2024-01-01T12:00:00"
      }
    ],
    "total": 100,
    "page": 1,
    "page_size": 10,
    "total_pages": 10
  }
}
```

### 4. 批量删除任务 ✨ 新增

**接口**: `POST /tasks/batch-delete`

**描述**: 批量删除任务（兼容单任务删除）

**请求体**:
```json
{
  "task_ids": ["task_1234567890_1234", "task_1234567890_5678"]
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "批量删除完成: 成功删除 2/2 个任务",
  "data": {
    "total_count": 2,
    "success_count": 2,
    "failed_count": 0,
    "results": {
      "task_1234567890_1234": true,
      "task_1234567890_5678": true
    }
  }
}
```

**注意事项**:
- 支持单个或多个任务ID
- 返回每个任务ID的删除结果
- 部分任务删除失败不影响其他任务

### 5. 暂停任务 ✨ 新增

**接口**: `POST /tasks/pause`

**描述**: 暂停正在执行的任务

**请求体**:
```json
{
  "task_id": "task_1234567890_1234"
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "任务暂停成功",
  "data": {
    "task_id": "task_1234567890_1234",
    "status": "paused"
  }
}
```

**状态限制**:
- 只能暂停状态为 `running` 的任务
- 其他状态的任务暂停会失败

### 6. 恢复任务 ✨ 新增

**接口**: `POST /tasks/resume`

**描述**: 恢复已暂停的任务

**请求体**:
```json
{
  "task_id": "task_1234567890_1234"
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "任务恢复成功",
  "data": {
    "task_id": "task_1234567890_1234",
    "status": "running"
  }
}
```

**状态限制**:
- 只能恢复状态为 `paused` 的任务
- 其他状态的任务恢复会失败

## 任务状态说明

完整的任务状态包括：

- `pending`: 任务等待执行
- `running`: 任务正在执行
- `paused`: 任务已暂停 ✨ 新增
- `completed`: 任务执行成功
- `failed`: 任务执行失败

## 错误处理

所有接口遵循统一的错误响应格式：

```json
{
  "success": false,
  "message": "错误描述",
  "data": null
}
```

常见HTTP状态码：
- `200`: 请求成功
- `400`: 请求参数错误
- `404`: 任务不存在
- `500`: 服务器内部错误

## 使用示例

### Python示例

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. 分页查询任务列表
response = requests.post(
    f"{BASE_URL}/tasks/list",
    json={"page": 1, "page_size": 10}
)
print(response.json())

# 2. 暂停任务
response = requests.post(
    f"{BASE_URL}/tasks/pause",
    json={"task_id": "task_1234567890_1234"}
)
print(response.json())

# 3. 恢复任务
response = requests.post(
    f"{BASE_URL}/tasks/resume",
    json={"task_id": "task_1234567890_1234"}
)
print(response.json())

# 4. 批量删除任务
response = requests.post(
    f"{BASE_URL}/tasks/batch-delete",
    json={
        "task_ids": [
            "task_1234567890_1234",
            "task_1234567890_5678"
        ]
    }
)
print(response.json())
```

### cURL示例

```bash
# 1. 分页查询任务列表
curl -X POST "http://localhost:8000/tasks/list" \
  -H "Content-Type: application/json" \
  -d '{"page": 1, "page_size": 10}'

# 2. 暂停任务
curl -X POST "http://localhost:8000/tasks/pause" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1234567890_1234"}'

# 3. 恢复任务
curl -X POST "http://localhost:8000/tasks/resume" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1234567890_1234"}'

# 4. 批量删除任务
curl -X POST "http://localhost:8000/tasks/batch-delete" \
  -H "Content-Type: application/json" \
  -d '{"task_ids": ["task_1234567890_1234", "task_1234567890_5678"]}'
```

## API交互文档

启动服务后，访问以下URL查看完整的交互式API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 实现细节

### 数据库层 (`database.py`)

新增方法：
- `get_tasks_paginated()`: 分页获取任务列表
- `delete_tasks_batch()`: 批量删除任务

### 任务管理层 (`task_manager.py`)

新增方法：
- `pause_task()`: 暂停任务
- `resume_task()`: 恢复任务

新增状态：
- `TASK_STATUS['PAUSED'] = 'paused'`

### API层 (`task_api.py`)

新增函数：
- `get_tasks_paginated()`: 分页查询封装
- `pause_task()`: 暂停任务封装
- `resume_task()`: 恢复任务封装
- `delete_tasks_batch()`: 批量删除封装

### REST接口层 (`rest_convert.py`)

新增端点：
- `POST /tasks/list`: 分页查询任务列表
- `POST /tasks/pause`: 暂停任务
- `POST /tasks/resume`: 恢复任务
- `POST /tasks/batch-delete`: 批量删除任务

## 测试

运行系统测试以验证所有功能：

```bash
# 进入项目目录
cd model-convert

# 运行测试（需要先确保数据库和服务环境已配置）
python -m pytest test/ -v
```

## 更新日期

- **版本**: 1.0.0
- **更新日期**: 2024年
- **更新内容**: 
  - 新增任务分页查询功能
  - 新增任务暂停/恢复功能
  - 新增批量删除任务功能
  - 扩展任务状态为包含paused状态

## 注意事项

1. 任务暂停功能仅适用于 `running` 状态的任务
2. 任务恢复功能仅适用于 `paused` 状态的任务
3. 批量删除时，个别任务删除失败不会影响其他任务
4. 分页查询的 `page_size` 建议不超过100
5. 所有接口都支持MinIO存储的路径格式 `bucket/object`
