#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/29 11:33
# @Author  : shunyaoyin
# @Email   : shunyaoyin@xxx.com
# @Detail  : 华为910b平台推理和模型转换服务
# @Software: PyCharm
import os
import sys
import tempfile

# 添加项目根目录和model-convert目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_convert_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, model_convert_path)

# 确保config模块可以被导入
config_path = os.path.join(model_convert_path, 'config')
if config_path not in sys.path:
    sys.path.insert(0, config_path)

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
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Union
import asyncio
import httpx
from config.config_loader import config_loader

from predict.predict_om import HUAWEI_910B_Predictor
from convert.onnx_to_om import onnx_to_om
import cv2

# 导入MinIO处理模块
from tools.handle_file_minio import minio_handler, init_minio_handler

from rest_simple.utils import draw_detection_results, BUCKET_ENGINE, BUCKET_ONNX, BUCKET_SOURCE, BUCKET_MODEL, BUCKET_TARGET

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="华为910b平台推理和模型转换服务",
    description="提供华为910b平台上的图片推理和模型转换任务执行",
    version="1.0.0"
)

# 从配置文件获取回调URL
CALLBACK_URL = config_loader.get_config_value("callback.url", "http://localhost:8000/api/predict/callback")


class PredictRequest(BaseModel):
    task_id: int
    model_id: int
    model_type: str
    platform: str
    source_file: str
    model_file: str
    onnx_file: str
    engine_file: Optional[str] = None

    def _convert_path(self, bucket_name: str, object_name: str):
        return f"{bucket_name}/{object_name}"

    @property
    def source_file_bucket(self):
        return self._convert_path(BUCKET_SOURCE, self.source_file)

    @property
    def model_file_bucket(self):
        return self._convert_path(BUCKET_MODEL, self.model_file)

    @property
    def onnx_file_bucket(self):
        return self._convert_path(BUCKET_ONNX, self.onnx_file)

    @property
    def engine_file_bucket(self):
        return self._convert_path(BUCKET_ENGINE, self.engine_file)

    def convert_path(self):
        self.source_file = self._convert_path(BUCKET_SOURCE, self.source_file)
        self.model_file = self._convert_path(BUCKET_MODEL, self.model_file)
        self.engine_file = self._convert_path(BUCKET_ENGINE, self.engine_file)
        if self.onnx_file:
            self.onnx_file = self._convert_path(BUCKET_ONNX, self.onnx_file)


class CallbackRequest(BaseModel):
    task_id: int
    model_id: int
    result: str
    target_file: str
    engine_file: Optional[str] = None
    platform: str = "Huawei"

    # def clear_bucket(self):
    #     self.target_file = self.target_file.replace(BUCKET_TARGET+"/", "")
    #     self.engine_file = self.engine_file.replace(BUCKET_ENGINE+"/", "")


async def send_callback(callback_data: dict):
    """发送回调请求到远程接口"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(CALLBACK_URL, json=callback_data, timeout=30.0)
            logger.info(f"回调请求发送成功，状态码: {response.status_code}")
            return True
    except Exception as e:
        logger.error(f"发送回调请求失败: {str(e)}")
        return False


def process_minio_file(file_path: str) -> tuple[str, bool]:
    """
    处理MinIO文件路径，如果是MinIO路径则下载到本地临时文件
    
    Args:
        file_path: 输入文件路径，可以是本地路径或MinIO路径格式(bucket/object)
        
    Returns:
        tuple: (本地文件路径, 是否为MinIO文件)
    """
    # 确保MinIO处理器已初始化
    if minio_handler is None:
        init_minio_handler()
    
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
                return temp_path, True
            else:
                raise ValueError(f"从MinIO下载文件失败: {file_path}")
        except Exception as e:
            logger.error(f"处理MinIO文件失败: {str(e)}")
            raise
    
    # 本地文件直接返回
    return file_path, False


def cleanup_temp_file(file_path: str, is_minio_file: bool):
    """清理临时文件"""
    if is_minio_file and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"临时文件已清理: {file_path}")
        except Exception as e:
            logger.error(f"清理临时文件失败: {str(e)}")


def run_inference(engine_file: str, source_file: str, target_file: str) -> Union[bool, dict]:
    """执行推理任务"""
    # 处理MinIO文件
    local_engine_file, is_engine_minio = process_minio_file(engine_file)
    local_source_file, is_source_minio = process_minio_file(source_file)
    
    try:
        # 初始化预测器
        predictor = HUAWEI_910B_Predictor(om_model_path=local_engine_file)
        
        # 执行推理
        result = predictor.predict(image=local_source_file)
        
        # 绘制检测结果并保存
        drawn_target_file = draw_detection_results(
            image_path=local_source_file,
            results=result,
            output_path=target_file,
            conf_threshold=0.5
        )
        
        if drawn_target_file is None:
            logger.error(f"绘制检测结果失败: {target_file}")
            return False
            
        # 返回推理结果，供回调使用
        return result
    except Exception as e:
        logger.error(f"推理任务执行失败: {str(e)}")
        return False
    finally:
        # 清理临时文件
        cleanup_temp_file(local_engine_file, is_engine_minio)
        cleanup_temp_file(local_source_file, is_source_minio)


def run_conversion(onnx_file: str, om_file: str) -> bool:
    """执行模型转换任务"""
    # 处理MinIO文件
    local_onnx_file, is_onnx_minio = process_minio_file(onnx_file)
    
    try:
        # 调用onnx_to_om函数进行转换
        # 传递额外的参数以确保转换成功
        # 传入的output，实际输出是会添加.om
        success = onnx_to_om(
            onnx_model_path=local_onnx_file,
            output_om_path=om_file,
            auto_input_shape=True,
            soc_version=config_loader.get_config_value("ascend.soc_version", "Ascend910B"),
            precision_mode=config_loader.get_config_value("ascend.precision_mode", "allow_fp32_to_fp16"),
            log_level=config_loader.get_config_value("ascend.log_level", "info")
        )

        om_file = f"{om_file}.om"
        # 检查生成的OM文件是否存在
        if success and not os.path.exists(om_file):
            logger.error(f"模型转换报告成功，但输出文件不存在: {om_file}")
            return False
            
        return success
    except Exception as e:
        logger.error(f"模型转换任务执行失败: {str(e)}")
        return False
    finally:
        # 清理临时文件
        cleanup_temp_file(local_onnx_file, is_onnx_minio)


def upload_to_minio_if_needed(file_path: str, minio_path: str) -> str:
    """
    如果需要，将文件上传到MinIO
    
    Args:
        file_path: 本地文件路径
        minio_path: MinIO路径 (bucket/object)
        
    Returns:
        str: 最终文件路径（可能是MinIO路径或本地路径）
    """
    # 确保MinIO处理器已初始化
    if minio_handler is None:
        init_minio_handler()
    
    # 检查本地文件是否存在
    if not os.path.exists(file_path):
        logger.error(f"本地文件不存在，无法上传到MinIO: {file_path}")
        return ""
    
    # 如果minio_path是MinIO路径格式，则上传文件到MinIO
    if minio_handler and '/' in minio_path and len(minio_path.split('/')) >= 2:
        try:
            # 解析bucket和object
            bucket_name, object_name = minio_path.split('/', 1)
            
            # 上传文件到MinIO
            if minio_handler.upload_file(bucket_name, object_name, file_path):
                logger.info(f"成功上传文件到MinIO: {bucket_name}/{object_name}")
                return minio_path  # 返回MinIO路径
            else:
                logger.error(f"上传文件到MinIO失败: {bucket_name}/{object_name}")
                return ""  # 上传失败，返回本地路径
        except Exception as e:
            logger.error(f"上传文件到MinIO时出错: {str(e)}")
            return ""  # 出错，返回本地路径
    
    # 如果不需要上传，返回本地文件路径
    return file_path


async def process_task(request: PredictRequest):
    """后台处理任务"""
    # 初始化回调数据
    callback_data = {
        "task_id": request.task_id,
        "model_id": request.model_id,
        "result": "",
        "target_file": "",
        "engine_file": request.engine_file,
        "platform": request.platform,
    }
    
    try:
        if request.engine_file:
            # 推理任务
            logger.info(f"开始执行推理任务，任务ID: {request.task_id}")
            
            # 定义输出文件路径
            target_file = f"/tmp/result_{request.task_id}.jpg"
            
            # 执行推理
            result = run_inference(request.engine_file_bucket, request.source_file_bucket, target_file)
            
            if result:
                # 如果source_file是MinIO路径，将结果上传到相同的bucket
                result_object_name = f"result_{request.task_id}.jpg"
                try:
                    # 上传失败
                    if not upload_to_minio_if_needed(target_file, f"{BUCKET_TARGET}/{result_object_name}"):
                        callback_data.update({
                            "result": "上传结果文件到MinIO失败",
                            "target_file": ""
                        })
                    else:
                        # 根据接口文档，result字段应该包含实际的推理结果
                        # 将推理结果转换为JSON字符串
                        import json
                        result_str = json.dumps(result, ensure_ascii=False)

                        callback_data.update({
                            "result": result_str,
                            "target_file": result_object_name
                        })
                        logger.info(f"推理任务执行成功，任务ID: {request.task_id}")

                except Exception as e:
                    logger.error(f"上传结果文件到MinIO时出错: {str(e)}")
                    callback_data.update({
                        "result": "上传结果文件到MinIO失败",
                        "target_file": ""
                    })
        else:
            # 模型转换任务
            logger.info(f"开始执行模型转换任务，任务ID: {request.task_id}")
            
            # 定义输出OM文件路径
            om_file = f"/tmp/model_{request.task_id}"
            
            # 执行转换
            success = run_conversion(request.onnx_file_bucket, om_file)

            # 工具会自动添加后缀
            om_file = f"{om_file}.om"

            if success:
                # 检查生成的OM文件是否存在
                if not os.path.exists(om_file):
                    logger.error(f"模型转换报告成功，但输出文件不存在: {om_file}")
                    callback_data.update({
                        "result": "转换失败：输出文件不存在",
                        "engine_file": None
                    })
                    logger.error(f"模型转换任务执行失败，任务ID: {request.task_id}")
                else:
                    # 如果onnx_file是MinIO路径，将结果上传到相同的bucket
                    # final_om_file = om_file
                    # if '/' in request. and len(request.onnx_file.split('/')) >= 2 and not os.path.isabs(request.onnx_file):
                    try:
                        bucket_name = BUCKET_ENGINE
                        # 生成OM文件名
                        om_object_name = request.model_file.replace('.onnx', '.om') if '.onnx' in request.model_file else f"{request.model_file}.om"
                        final_om_file = upload_to_minio_if_needed(om_file, f"{bucket_name}/{om_object_name}")
                        callback_data.update({
                            "result": "转换成功",
                            "engine_file": om_object_name
                        })
                        logger.info(f"模型转换任务执行成功，任务ID: {request.task_id}")
                    except Exception as e:
                        logger.error(f"上传OM文件到MinIO时出错: {str(e)}")
                        final_om_file = om_file  # 出错时使用本地路径

                        callback_data.update({
                            "result": "转换失败，结果上传Minio失败",
                            "engine_file": ""
                        })
                        logger.error(f"模型转换任务执行成功，任务ID: {request.task_id}")
            else:
                callback_data.update({
                    "result": "转换失败",
                    "engine_file": ""
                })
                logger.error(f"模型转换任务执行失败，任务ID: {request.task_id}")
        
        # 发送回调
        await send_callback(callback_data)
        
    except Exception as e:
        logger.error(f"任务处理过程中出现异常: {str(e)}")
        callback_data.update({
            "result": f"任务执行异常: {str(e)}",
            "target_file": "",
            "engine_file": ""
        })
        await send_callback(callback_data)


@app.post("/api/v1/predict")
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    """处理推理或模型转换任务"""
    logger.info(f"接收到任务请求，任务ID: {request.task_id}")
    
    # 检查必要参数
    if not request.task_id or not request.model_id:
        raise HTTPException(status_code=400, detail="缺少必要参数")
    
    # 将任务添加到后台执行
    background_tasks.add_task(process_task, request)
    
    # 立即返回响应
    return {
        "code": 200,
        "data": {
            "task_id": str(request.task_id),
            "status": "pending",
            "message": "预测任务已接受，正在处理中"
        },
        "message": "success"
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=39000)