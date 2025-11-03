#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/13
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 模型转换RESTful API服务
# @Software: PyCharm
import os

# 手动设置昇腾环境变量（路径替换为实际值，参考终端中env的输出）
os.environ["ASCEND_HOME"] = "/usr/local/Ascend/ascend-toolkit/8.0.0"
os.environ["LD_LIBRARY_PATH"] = (
    f"{os.environ['ASCEND_HOME']}/lib64:"
    "/usr/local/Ascend/driver/lib64:"
    f"{os.environ.get('LD_LIBRARY_PATH', '')}"
)
os.environ["PATH"] = f"{os.environ['ASCEND_HOME']}/bin:{os.environ.get('PATH', '')}"
os.environ["PYTHONPATH"] = (
    f"{os.environ['ASCEND_HOME']}/python/site-packages:"
    f"{os.environ.get('PYTHONPATH', '')}"
)

import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Union, Dict, Any, List, Tuple
import os
import sys
import uvicorn
import tempfile

# 添加项目路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入任务管理API
from service.task_api import (
    create_huawei_onnx_to_om_task,
    get_task_info,
    get_task_status,
    is_task_completed,
    is_task_failed,
    get_tasks_paginated,
    pause_task,
    resume_task,
    delete_tasks_batch,
    SUPPORTED_PLATFORMS,
    TASK_TYPES,
    TASK_STATUS
)
# 导入配置加载器
from config.config_loader import config_loader
# 导入MinIO工具
from tools.handle_file_minio import minio_handler

# 创建FastAPI应用实例
app = FastAPI(
    title="跨平台模型转换服务",
    description="支持华为昇腾、瑞芯微和寒武纪等平台的模型转换接口",
    version="1.0.0"
)


# 基础响应模型
class BaseResponse(BaseModel):
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")


# 华为平台ONNX到OM转换请求模型
class ONNXToOMRequest(BaseModel):
    input_model_path: str = Field(..., description="输入ONNX模型路径（支持本地路径或MinIO路径格式bucket/object）")
    output_model_path: str = Field(..., description="输出OM模型路径（支持本地路径或MinIO路径格式bucket/object）")
    input_shape: Optional[Union[str, List[int]]] = Field(
        None, 
        description="模型输入形状，支持字符串格式(如'1,3,640,640'或'input_name:1,3,640,640')或整数列表"
    )
    soc_version: str = Field("Ascend910B", description="目标昇腾处理器版本")
    precision_mode: str = Field("allow_fp32_to_fp16", description="精度模式")
    log_level: str = Field("error", description="日志级别")
    auto_input_shape: bool = Field(True, description="是否自动从模型中获取输入形状")
    extra_params: Optional[Dict[str, str]] = Field(None, description="额外的ATC工具参数")


# 通用模型转换请求模型
class ModelConvertRequest(BaseModel):
    input_model_path: str = Field(..., description="输入模型路径（支持本地路径或MinIO路径格式bucket/object）")
    output_model_path: str = Field(..., description="输出模型路径（支持本地路径或MinIO路径格式bucket/object）")
    platform: str = Field(..., description="目标平台，可选值: ascend, rockchip, cambricon")
    model_type: str = Field(..., description="模型类型，如: onnx, pytorch等")
    params: Optional[Dict[str, Any]] = Field(None, description="平台特定参数")


@app.post("/convert/onnx-to-om", response_model=BaseResponse, summary="华为平台ONNX转OM")
async def convert_onnx_to_om(request: ONNXToOMRequest):
    """
    将ONNX模型转换为华为昇腾平台支持的OM模型（创建任务）
    
    - **input_model_path**: 输入ONNX模型路径（支持本地路径或MinIO路径格式bucket/object）
    - **output_model_path**: 输出OM模型路径
    - **input_shape**: 模型输入形状，可选参数
    - **soc_version**: 目标昇腾处理器版本，默认为Ascend910B
    - **precision_mode**: 精度模式，默认为allow_fp32_to_fp16
    - **log_level**: 日志级别，默认为error
    - **auto_input_shape**: 是否自动从模型中获取输入形状，默认为True
    - **extra_params**: 额外的ATC工具参数
    """
    try:
        # 构建转换参数
        parameters = {
            "soc_version": request.soc_version,
            "precision_mode": request.precision_mode,
            "log_level": request.log_level,
            "auto_input_shape": request.auto_input_shape
        }
        
        # 处理输入形状
        if request.input_shape is not None:
            parameters["input_shape"] = request.input_shape
        
        # 添加额外参数
        if request.extra_params:
            parameters.update(request.extra_params)
        
        # 调用任务管理API创建转换任务
        task_id = create_huawei_onnx_to_om_task(
            input_path=request.input_model_path,
            output_path=request.output_model_path,
            parameters=parameters
        )
        
        logger.info(f"成功创建华为平台ONNX转OM任务: {task_id}")
        
        return BaseResponse(
            success=True,
            message="任务创建成功，请通过任务ID查询状态",
            data={
                "task_id": task_id,
                "input_model": request.input_model_path,
                "output_model": request.output_model_path,
                "platform": "huawei",
                "task_type": "onnx_to_om"
            }
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"文件不存在: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"参数错误: {str(e)}")
    except Exception as e:
        logger.error(f"创建任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")


@app.post("/convert/rockchip", response_model=BaseResponse, summary="瑞芯微平台模型转换")
async def convert_to_rockchip(request: ModelConvertRequest):
    """
    瑞芯微平台模型转换接口（预留）
    注意：此功能暂未实现
    """
    return BaseResponse(
        success=False,
        message="瑞芯微平台模型转换功能暂未实现",
        data={"platform": "rockchip", "status": "not_implemented"}
    )


@app.post("/convert/cambricon", response_model=BaseResponse, summary="寒武纪平台模型转换")
async def convert_to_cambricon(request: ModelConvertRequest):
    """
    寒武纪平台模型转换接口（预留）
    注意：此功能暂未实现
    """
    return BaseResponse(
        success=False,
        message="寒武纪平台模型转换功能暂未实现",
        data={"platform": "cambricon", "status": "not_implemented"}
    )


@app.post("/convert", response_model=BaseResponse, summary="通用模型转换接口")
async def convert_model(request: ModelConvertRequest):
    """
    通用模型转换接口，根据平台选择对应的转换功能（创建任务）
    
    - **input_model_path**: 输入模型路径（支持本地路径或MinIO路径格式bucket/object）
    - **output_model_path**: 输出模型路径
    - **platform**: 目标平台，可选值: ascend, rockchip, cambricon
    - **model_type**: 模型类型，如: onnx, pytorch等
    - **params**: 平台特定参数
    """
    try:
        # 根据平台和模型类型创建相应的转换任务
        if request.platform == "ascend" and request.model_type == "onnx":
            # 调用华为平台ONNX转OM任务创建接口
            task_id = create_huawei_onnx_to_om_task(
                input_path=request.input_model_path,
                output_path=request.output_model_path,
                parameters=request.params
            )
            
            return BaseResponse(
                success=True,
                message="任务创建成功，请通过任务ID查询状态",
                data={
                    "task_id": task_id,
                    "input_model": request.input_model_path,
                    "output_model": request.output_model_path,
                    "platform": request.platform,
                    "model_type": request.model_type
                }
            )
            
        elif request.platform == "rockchip":
            return await convert_to_rockchip(request)
        elif request.platform == "cambricon":
            return await convert_to_cambricon(request)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的平台或模型类型组合: platform={request.platform}, model_type={request.model_type}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")


@app.get("/health", response_model=BaseResponse, summary="健康检查接口")
async def health_check():
    """
    服务健康检查接口
    """
    return BaseResponse(
        success=True,
        message="服务运行正常",
        data={
            "status": "healthy",
            "platforms": {
                "ascend": {"onnx_to_om": "available"},
                "rockchip": "not_implemented",
                "cambricon": "not_implemented"
            },
            "supported_platforms": list(SUPPORTED_PLATFORMS.values()),
            "supported_task_types": list(TASK_TYPES.values()),
            "supported_task_statuses": list(TASK_STATUS.values())
        }
    )


# 新增任务查询接口
class TaskQueryRequest(BaseModel):
    task_id: str = Field(..., description="任务ID")


@app.post("/tasks/query", response_model=BaseResponse, summary="查询任务信息")
async def query_task(request: TaskQueryRequest):
    """
    查询指定任务的详细信息
    
    - **task_id**: 任务ID（必选）
    """
    try:
        logger.info(f"查询任务信息: {request.task_id}")
        
        # 获取任务信息
        task_info = get_task_info(request.task_id)
        
        if task_info:
            return BaseResponse(
                success=True,
                message="获取任务信息成功",
                data={
                    "task_id": request.task_id,
                    "task_info": task_info
                }
            )
        else:
            raise HTTPException(status_code=404, detail=f"任务不存在: {request.task_id}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询任务失败: {str(e)}")


@app.get("/tasks/{task_id}/status", response_model=BaseResponse, summary="获取任务状态")
async def get_task_status_api(task_id: str):
    """
    获取指定任务的当前状态
    
    - **task_id**: 任务ID（路径参数）
    """
    try:
        logger.info(f"获取任务状态: {task_id}")
        
        status = get_task_status(task_id)
        
        if status is not None:
            return BaseResponse(
                success=True,
                message="获取任务状态成功",
                data={
                    "task_id": task_id,
                    "status": status,
                    "is_completed": is_task_completed(task_id),
                    "is_failed": is_task_failed(task_id)
                }
            )
        else:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")


# 任务列表分页查询请求模型
class TaskListRequest(BaseModel):
    page: int = Field(1, description="页码，从1开始")
    page_size: int = Field(10, description="每页任务数量")


@app.post("/tasks/list", response_model=BaseResponse, summary="分页查询任务列表")
async def list_tasks(request: TaskListRequest):
    """
    分页查询任务列表
    
    - **page**: 页码，从1开始（默认为1）
    - **page_size**: 每页任务数量（默认为10）
    """
    try:
        logger.info(f"分页查询任务列表: page={request.page}, page_size={request.page_size}")
        
        # 验证参数
        if request.page < 1:
            raise ValueError("页码必须大于等于1")
        if request.page_size < 1 or request.page_size > 100:
            raise ValueError("每页任务数量必须在1-100之间")
        
        # 获取分页结果
        result = get_tasks_paginated(page=request.page, page_size=request.page_size)
        
        return BaseResponse(
            success=True,
            message="获取任务列表成功",
            data=result
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"获取任务列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


# 批量删除任务请求模型
class BatchDeleteRequest(BaseModel):
    task_ids: List[str] = Field(..., description="任务ID列表")


@app.post("/tasks/batch-delete", response_model=BaseResponse, summary="批量删除任务")
async def batch_delete_tasks(request: BatchDeleteRequest):
    """
    批量删除任务（兼容单任务删除）
    
    - **task_ids**: 任务ID列表，支持单个或多个任务
    """
    try:
        logger.info(f"批量删除任务: {request.task_ids}")
        
        # 验证参数
        if not request.task_ids:
            raise ValueError("任务ID列表不能为空")
        
        # 执行批量删除
        results = delete_tasks_batch(request.task_ids)
        
        # 统计结果
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        return BaseResponse(
            success=True,
            message=f"批量删除完成: 成功删除 {success_count}/{total_count} 个任务",
            data={
                "total_count": total_count,
                "success_count": success_count,
                "failed_count": total_count - success_count,
                "results": results
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"批量删除任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量删除任务失败: {str(e)}")


# 暂停/恢复任务请求模型
class TaskControlRequest(BaseModel):
    task_id: str = Field(..., description="任务ID")


@app.post("/tasks/pause", response_model=BaseResponse, summary="暂停任务")
async def pause_task_api(request: TaskControlRequest):
    """
    暂停正在执行的任务
    
    - **task_id**: 任务ID（必选）
    """
    try:
        logger.info(f"暂停任务: {request.task_id}")
        
        # 暂停任务
        success = pause_task(request.task_id)
        
        if success:
            return BaseResponse(
                success=True,
                message="任务暂停成功",
                data={"task_id": request.task_id, "status": "paused"}
            )
        else:
            raise HTTPException(
                status_code=400, 
                detail="任务暂停失败：任务不存在或当前状态不允许暂停"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"暂停任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"暂停任务失败: {str(e)}")


@app.post("/tasks/resume", response_model=BaseResponse, summary="恢复任务")
async def resume_task_api(request: TaskControlRequest):
    """
    恢复已暂停的任务
    
    - **task_id**: 任务ID（必选）
    """
    try:
        logger.info(f"恢复任务: {request.task_id}")
        
        # 恢复任务
        success = resume_task(request.task_id)
        
        if success:
            return BaseResponse(
                success=True,
                message="任务恢复成功",
                data={"task_id": request.task_id, "status": "running"}
            )
        else:
            raise HTTPException(
                status_code=400, 
                detail="任务恢复失败：任务不存在或当前状态不允许恢复"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"恢复任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"恢复任务失败: {str(e)}")


def process_model_file(file_path: str) -> str:
    """
    处理模型文件路径，如果是MinIO路径则下载到本地临时文件
    
    Args:
        file_path: 输入文件路径，可以是本地路径或MinIO路径格式(bucket/object)
        
    Returns:
        str: 本地文件路径
    """
    # 判断是否是MinIO路径格式 (bucket/object)
    if minio_handler and '/' in file_path and len(file_path.split('/')) >= 2 and not os.path.isabs(file_path):
        try:
            # 解析bucket和object
            bucket_name, object_name = file_path.split('/', 1)
            
            # 创建临时文件
            fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(object_name)[1])
            os.close(fd)
            
            # 从MinIO下载文件
            if minio_handler.download_file(bucket_name, object_name, temp_path):
                logger.info(f"成功从MinIO下载文件: {bucket_name}/{object_name} -> {temp_path}")
                return temp_path
            else:
                raise ValueError(f"从MinIO下载文件失败: {file_path}")
        except Exception as e:
            logger.error(f"处理MinIO文件失败: {str(e)}")
            raise
    
    # 本地文件直接返回
    return file_path

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # 从配置文件获取端口
    server_config = config_loader.get_server_config()
    port = server_config.get('port', 8000)
    
    # 启动服务
    uvicorn.run(
        "rest_convert:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
