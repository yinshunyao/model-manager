
# 服务列表

| Name     | Type   | 地址                              |
|----------|--------|---------------------------------|
| 华为测试服务器  | 910B2  | http://8.137.18.24:49000/docs#/ |
| 瑞芯微测试服务器 | rk3588 | http://8.137.18.24:49001/docs#/ |
| 寒武纪测试服务器 | 思元370  | http://8.137.18.24:49002/docs#/ |

# 测试数据范例
## onnx转om
onnx模型可以使用 yolo11n.onnx 或者 9-yolo11n.onnx
可以使用本地路径测试 /root/miniconda3/convert/predict/../model_demo/yolo11n.onnx
```json
{
  "task_id": 2,
  "model_id": 2,
  "model_type": "yolo",
  "platform": "Huawei",
  "source_file": "model1",
  "model_file": "model2",
  "onnx_file": "/root/miniconda3/convert/predict/../model_demo/yolo11n.onnx",
  "engine_file": ""
}
```

## 回调
```json
{
  "task_id": 12, 
  "model_id": 9, 
  "platform": null, 
  "result": "任务执行异常: 从MinIO下载文件失败: onnx-file/9-yolo11n.onnx", 
  "target_file": "", 
  "engine_file": ""
}
```