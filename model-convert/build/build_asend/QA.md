# 报错 50002
# 日志
```shell
ERROR:root:无法加载华为ACL，使用模拟模式: 加载模型失败: 500002
Traceback (most recent call last):
  File "/root/miniconda3/convert/predict/predict_om.py", line 67, in _load_model
    raise RuntimeError(f"加载模型失败: {ret}")
RuntimeError: 加载模型失败: 500002
```

## 日志放开
```shell
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1 
```

## 详细日志
```shell
[ERROR] RUNTIME(34201,python3):2025-11-06-00:03:33.914.534 [api_c.cc:4856]34201 rtModelCheckCompatibility:[LOAD][DEFAULT]report error module_name=EE1001
[ERROR] RUNTIME(34201,python3):2025-11-06-00:03:33.914.465 [runtime.cc:5907]34201 ModelCheckArchVersion:[LOAD][DEFAULT]ModelCheckArchVersion failed, omArchVersion=3, archType=0
[ERROR] GE(34201,python3):2025-11-06-00:03:33.915.355 [om_file_helper.cc:265]34201 CheckModelCompatibility: ErrorNo: 1343225859(Failed to call runtime API!) [LOAD][DEFAULT]Call rt api failed, ret: 0x7BC80
[INFO] GE(34201,python3):2025-11-06-00:03:33.915.376 [error_manager.cc:358]34201 ReportInterErrMessage:report error_message, error_code:E19999, work_stream_id:3420134201, error_mode:0
[ERROR] GE(34201,python3):2025-11-06-00:03:33.915.394 [model_helper.cc:1198]34201 LoadModelData: ErrorNo: 4294967295(failed) [LOAD][DEFAULT]Check model compatibility failed.
[ERROR] GE(34201,python3):2025-11-06-00:03:33.919.603 [model_helper.cc:1119]34201 GenerateGeRootModel: ErrorNo: 4294967295(failed) [LOAD][DEFAULT][Generate][GERootModel]Failed
[INFO] GE(34201,python3):2025-11-06-00:03:33.919.634 [error_manager.cc:358]34201 ReportInterErrMessage:report error_message, error_code:E19999, work_stream_id:3420134201, error_mode:0
[ERROR] GE(34201,python3):2025-11-06-00:03:33.919.658 [model_helper.cc:1048]34201 LoadRootModel: ErrorNo: 4294967295(failed) [LOAD][DEFAULT]Assert ((GenerateGeRootModel(om_load_helper, model_data)) == ge::SUCCESS) failed
[ERROR] GE(34201,python3):2025-11-06-00:03:33.919.672 [model_manager.cc:1538]34201 LoadModelOffline: ErrorNo: 1343225857(Parameter invalid!) [LOAD][DEFAULT][Load][RootModel] failed, ret:1343225857, model_id:1.
[ERROR] GE(34201,python3):2025-11-06-00:03:33.919.692 [graph_loader.cc:136]34201 LoadModelFromData: ErrorNo: 1343225857(Parameter invalid!) [LOAD][DEFAULT][Load][Model] failed, model_id:1.
[ERROR] ASCENDCL(34201,python3):2025-11-06-00:03:33.919.704 [model.cpp:282]34201 ModelLoadFromFileWithMem: [LOAD][DEFAULT][Model][FromData]load model from data failed, ge result[1343225857]
[INFO] GE(34201,python3):2025-11-06-00:03:33.919.720 [error_manager.cc:358]34201 ReportInterErrMessage:report error_message, error_code:EH9999, work_stream_id:3420134201, error_mode:0
[ERROR] ASCENDCL(34201,python3):2025-11-06-00:03:33.923.895 [model.cpp:1936]34201 aclmdlLoadFromFile: [LOAD][DEFAULT]Load model from file failed!
ERROR:root:无法加载华为ACL，使用模拟模式: 加载模型失败: 500002
Traceback (most recent call last):
  File "/root/miniconda3/convert/predict/predict_om.py", line 74, in _load_model
    raise RuntimeError(f"加载模型失败: {ret}")
RuntimeError: 加载模型失败: 500002
```

# 推理错误
## 日志
```shell
执行目标检测推理...
错误: 推理过程中出错: module 'acl.mdl' has no attribute 'get_input'
```