#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/29 11:33
# @Author  : shunyaoyin
# @Email   : shunyaoyin@xxx.com
# @Detail  : 瑞芯微RK3588平台推理和模型转换服务
# @Software: PyCharm
import logging
# 配置日志，包含文件名和行号信息
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)
import os
import shutil
import threading

import sys
import tempfile

rknn_lock = threading.RLock()
# 添加项目根目录和model-convert目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_convert_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, model_convert_path)

# 确保config模块可以被导入
config_path = os.path.join(model_convert_path, 'config')
if config_path not in sys.path:
    sys.path.insert(0, config_path)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Union
import httpx
from config.config_loader import config_loader

# 设置RKNN环境变量（如果需要）
# RKNN Toolkit 2通常不需要特殊的环境变量设置，但可以根据实际情况添加
rknn_toolkit_path = config_loader.get_config_value("rockchip.tool_path", "/usr/local/rockchip/rknn-toolkit2")
if os.path.exists(rknn_toolkit_path):
    rknn_python_path = os.path.join(rknn_toolkit_path, "python")
    if os.path.exists(rknn_python_path):
        if rknn_python_path not in sys.path:
            sys.path.insert(0, rknn_python_path)

from predict.predict_rknn import RK3588_Predictor
from convert.onnx_to_rknn import onnx_to_rknn

# 导入MinIO处理模块
from tools.handle_file_minio import minio_handler, init_minio_handler

from rest_simple.utils import draw_detection_results, BUCKET_ENGINE, BUCKET_ONNX, BUCKET_SOURCE, BUCKET_MODEL, BUCKET_TARGET


app = FastAPI(
    title="瑞芯微RK3588平台推理和模型转换服务",
    description="提供瑞芯微RK3588平台上的图片推理和模型转换任务执行",
    version="1.0.0"
)

# 从配置文件获取回调URL
CALLBACK_URL = config_loader.get_config_value("callback.url", "http://localhost:8000/api/predict/callback")


class PredictRequest(BaseModel):
    task_id: int
    model_id: int
    model_type: Optional[str] = None
    platform: Optional[str] = None
    source_file: Optional[str] = None
    model_file: Optional[str] = None
    onnx_file: Optional[str] = None
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
    platform: str
    result: Optional[str] = None
    target_file: Optional[str] = None
    engine_file: Optional[str] = None


async def send_callback(callback_data: dict):
    """发送回调请求到远程接口"""
    try:
        logger.warning(f"callback_data:{callback_data}")
        async with httpx.AsyncClient() as client:
            response = await client.post(CALLBACK_URL, json=callback_data, timeout=5.0)
            logger.info(f"回调请求发送成功，状态码: {response.status_code}")
            return True
    except Exception as e:
        logger.error(f"发送回调请求失败: {str(e)}")
        return False


def process_minio_file(file_path: str, bucket_name: str = None) -> tuple[str, bool]:
    """
    处理MinIO文件路径，如果是MinIO路径则下载到本地临时文件
    
    Args:
        file_path: 输入文件路径，可以是：
                  - 本地路径（以/开头）
                  - MinIO路径格式(bucket/object)，如 "source-file/1-car.jpeg"
                  - 对象名称（需要提供bucket_name）
        bucket_name: 如果file_path只是对象名称，需要提供bucket名称
        
    Returns:
        tuple: (本地文件路径, 是否为MinIO文件)
    """
    # 确保MinIO处理器已初始化
    if minio_handler is None:
        init_minio_handler()

    filename = os.path.basename(file_path)

    # 如果/开头，是本地路径
    if file_path.startswith('/'):
        logger.warning(f"输入文件路径是本地路径 {file_path}")
        fd, temp_path = tempfile.mkstemp(suffix=filename)
        os.close(fd)
        # 复制到temp_path
        shutil.copy(file_path, temp_path)
        return temp_path, True

    # 判断是否是MinIO路径格式 (bucket/object)
    # 如果包含/且不是绝对路径，可能是bucket/object格式
    if '/' in file_path and len(file_path.split('/')) >= 2 and not os.path.isabs(file_path):
        try:
            # 解析bucket和object
            parts = file_path.split('/', 1)
            actual_bucket_name = parts[0]
            object_name = parts[1]
            
            # 创建临时文件
            fd, temp_path = tempfile.mkstemp(suffix=os.path.basename(object_name))
            os.close(fd)

            # 从MinIO下载文件
            if minio_handler.download_file(actual_bucket_name, object_name, temp_path):
                logger.info(f"成功从MinIO下载文件: {actual_bucket_name}/{object_name} -> {temp_path}")
                return temp_path, True
            else:
                raise ValueError(f"从MinIO下载文件失败: {actual_bucket_name}/{object_name}")
        except Exception as e:
            logger.error(f"处理MinIO文件失败: {str(e)}")
            raise
    
    # 如果只是对象名称，使用提供的bucket_name
    if bucket_name:
        try:
            # 创建临时文件
            fd, temp_path = tempfile.mkstemp(suffix=filename)
            os.close(fd)

            # 从MinIO下载文件
            if minio_handler.download_file(bucket_name, file_path, temp_path):
                logger.info(f"成功从MinIO下载文件: {bucket_name}/{file_path} -> {temp_path}")
                return temp_path, True
            else:
                raise ValueError(f"从MinIO下载文件失败: {bucket_name}/{file_path}")
        except Exception as e:
            logger.error(f"处理MinIO文件失败: {str(e)}")
            raise
    
    # 如果既不是bucket/object格式，也没有提供bucket_name，报错
    raise ValueError(f"无法处理文件路径: {file_path}，需要提供bucket_name或使用bucket/object格式")


def cleanup_temp_file(file_path: str, is_minio_file: bool):
    """清理临时文件"""
    if is_minio_file and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"临时文件已清理: {file_path}")
        except Exception as e:
            logger.error(f"清理临时文件失败: {str(e)}")


def run_inference(engine_file: str, source_file: str, target_file: str, bucket_name: str) -> Union[bool, dict]:
    """执行推理任务"""
    # 处理MinIO文件
    # engine_file 从 BUCKET_ENGINE 下载
    local_engine_file, is_engine_minio = process_minio_file(engine_file, bucket_name)
    # source_file 可能是 bucket/object 格式，process_minio_file 会自动解析
    local_source_file, is_source_minio = process_minio_file(source_file)
    
    predictor = None
    try:
        # 初始化预测器
        predictor = RK3588_Predictor(rknn_model_path=local_engine_file)
        
        # 执行推理
        with rknn_lock:
            result = predictor.predict(image=local_source_file)
            # 先释放资源
            predictor.release()
            predictor = None
        
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
        logger.error(f"推理任务执行失败: {str(e)}", exc_info=True)
        return False
    finally:
        # 确保释放predictor资源
        if predictor is not None:
            try:
                predictor.release()
                logger.info("Predictor资源已释放")
            except Exception as e:
                logger.error(f"释放Predictor资源失败: {str(e)}", exc_info=True)
        
        # 清理临时文件
        cleanup_temp_file(local_engine_file, is_engine_minio)
        cleanup_temp_file(local_source_file, is_source_minio)


def run_conversion(onnx_file: str, rknn_file: str, bucket_name: str) -> bool:
    """执行模型转换任务"""
    # 处理MinIO文件
    local_onnx_file, is_onnx_minio = process_minio_file(onnx_file, bucket_name)
    
    try:
        # 调用onnx_to_rknn函数进行转换
        # 传递额外的参数以确保转换成功
        with rknn_lock:
            success = onnx_to_rknn(
                onnx_model_path=local_onnx_file,
                output_rknn_path=rknn_file,
                auto_input_shape=True,
                target_platform=config_loader.get_config_value("rockchip.params.default_target_platform", "rk3588"),
                precision_mode=config_loader.get_config_value("rockchip.params.default_precision", "float32")
            )

        # 检查生成的RKNN文件是否存在
        if success and not os.path.exists(rknn_file):
            logger.error(f"模型转换报告成功，但输出文件不存在: {rknn_file}")
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
    # 根据接口文档，platform字段应该是 "Huawei"、"Rockchip" 或 "Cambricon"
    # 对于瑞芯微平台，默认为 "Rockchip"
    callback_data = {
        "task_id": request.task_id,
        "model_id": request.model_id,
        "platform": request.platform or "Rockchip",
        "result": "",
        "target_file": "",
        "engine_file": request.engine_file or "",
    }
    # CallbackRequest(**callback_data)
    
    try:
        # 根据 engine_file 字段识别子任务
        # 推理任务是必须执行的，模型转换任务是可选任务
        # 如果 engine_file 为空，需要先进行模型转换任务，然后执行推理
        # 如果 engine_file 不为空，直接执行推理任务
        
        engine_file_path = None
        
        if not request.engine_file:
            # engine_file 为空，需要先进行模型转换任务（可选任务）
            logger.info(f"开始执行模型转换任务，任务ID: {request.task_id}")
            
            # 定义输出RKNN文件路径
            rknn_file = f"/tmp/model_{request.task_id}.rknn"
            
            # 执行转换
            success = run_conversion(request.onnx_file, rknn_file, bucket_name=BUCKET_ONNX)

            if success:
                # 检查生成的RKNN文件是否存在
                if not os.path.exists(rknn_file):
                    logger.error(f"模型转换报告成功，但输出文件不存在: {rknn_file}")
                    callback_data.update({
                        "result": "转换失败：输出文件不存在",
                        "engine_file": ""
                    })
                    logger.error(f"模型转换任务执行失败，任务ID: {request.task_id}")
                    await send_callback(callback_data)
                    return
                else:
                    # 上传RKNN文件到MinIO
                    try:
                        bucket_name = BUCKET_ENGINE
                        # 生成RKNN文件名
                        rknn_object_name = request.model_file.replace('.onnx', '.rknn') if '.onnx' in request.model_file else f"{request.model_file}.rknn"
                        # 上传RKNN文件到MinIO
                        final_rknn_file = upload_to_minio_if_needed(rknn_file, f"{bucket_name}/{rknn_object_name}")
                        if final_rknn_file:
                            logger.info(f"模型转换任务执行成功，任务ID: {request.task_id}")
                            # 设置生成的 engine_file 路径，用于后续推理
                            engine_file_path = f"{bucket_name}/{rknn_object_name}"
                            # 保存到callback_data，供推理回调使用
                            callback_data["engine_file"] = rknn_object_name
                            
                            # 模型转换任务完成后，发送回调
                            conversion_callback = {
                                "task_id": request.task_id,
                                "model_id": request.model_id,
                                "platform": request.platform or "Rockchip",
                                "result": "转换成功",
                                "target_file": "",
                                "engine_file": rknn_object_name
                            }
                            await send_callback(conversion_callback)
                            
                            # 上传成功后清理本地临时文件
                            cleanup_temp_file(rknn_file, True)
                        else:
                            callback_data.update({
                                "result": "转换成功，但上传到MinIO失败",
                                "engine_file": ""
                            })
                            logger.error(f"模型转换成功但上传到MinIO失败，任务ID: {request.task_id}")
                            await send_callback(callback_data)
                            return
                    except Exception as e:
                        logger.error(f"上传RKNN文件到MinIO时出错: {str(e)}", exc_info=True)
                        callback_data.update({
                            "result": "转换失败，结果上传Minio失败",
                            "engine_file": ""
                        })
                        logger.error(f"模型转换任务执行失败，任务ID: {request.task_id}")
                        await send_callback(callback_data)
                        return
            else:
                callback_data.update({
                    "result": "转换失败",
                    "engine_file": ""
                })
                logger.error(f"模型转换任务执行失败，任务ID: {request.task_id}")
                await send_callback(callback_data)
                return
        else:
            # engine_file 不为空，直接使用提供的 engine_file
            engine_file_path = request.engine_file
        
        # 推理任务是必须执行的
        logger.info(f"开始执行推理任务，任务ID: {request.task_id}")
        
        # 初始化推理回调数据（保持engine_file，如果模型转换已设置）
        inference_callback_data = {
            "task_id": request.task_id,
            "model_id": request.model_id,
            "platform": request.platform or "Rockchip",
            "result": "",
            "target_file": "",
            "engine_file": callback_data["engine_file"]
        }
        
        # 定义输出文件路径
        target_file = f"/tmp/result_{request.task_id}.jpg"
        
        # 执行推理
        result = run_inference(engine_file_path, request.source_file_bucket, target_file, bucket_name=BUCKET_ENGINE)
        
        if result:
            # 将结果上传到MinIO
            result_object_name = f"result_{request.task_id}.jpg"
            try:
                # 上传结果图片到MinIO
                final_result_file = upload_to_minio_if_needed(target_file, f"{BUCKET_TARGET}/{result_object_name}")
                if not final_result_file:
                    inference_callback_data.update({
                        "result": "上传结果文件到MinIO失败",
                        "target_file": ""
                    })
                else:
                    # 根据接口文档，result字段应该包含实际的推理结果
                    # 将推理结果转换为JSON字符串
                    import json
                    result_str = json.dumps(result, ensure_ascii=False)

                    inference_callback_data.update({
                        "result": result_str,
                        "target_file": result_object_name
                    })
                    logger.info(f"推理任务执行成功，任务ID: {request.task_id}")
                    
                    # 上传成功后清理本地临时文件
                    cleanup_temp_file(target_file, True)
            except Exception as e:
                logger.error(f"上传结果文件到MinIO时出错: {str(e)}", exc_info=True)
                inference_callback_data.update({
                    "result": "上传结果文件到MinIO失败",
                    "target_file": ""
                })
        else:
            # 推理失败
            inference_callback_data.update({
                "result": "推理失败",
                "target_file": ""
            })
            logger.error(f"推理任务执行失败，任务ID: {request.task_id}")
        
        # 推理任务完成后，发送回调
        await send_callback(inference_callback_data)
        
    except Exception as e:
        logger.error(f"任务处理过程中出现异常: {str(e)}", exc_info=True)
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
    
    # 立即返回响应，根据接口文档返回202表示任务已接收
    return {
        "code": 202,
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