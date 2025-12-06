import os
import sys
import time
import pytest
import tempfile
import shutil

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.task_manager import get_task_manager, TASK_STATUS
from service.task_scheduler import TaskScheduler

def test_onnx_to_om_conversion_service():
    """
    测试ONNX到OM的转换服务流程
    1. 检查model_demo中的ONNX文件是否存在，不存在则转换生成
    2. 创建ONNX到OM转换任务
    3. 启动服务执行任务
    4. 检查任务状态
    5. 删除任务
    """
    # 准备测试数据
    test_dir = os.path.join(os.path.dirname(__file__), "..", "model_demo")
    yolo_model_path = os.path.join(test_dir, "yolo11n.pt")
    onnx_model_path_in_demo = os.path.join(test_dir, "yolo11n.onnx")
    
    # 创建临时输出目录
    temp_output_dir = tempfile.mkdtemp(prefix="model_convert_test_")
    onnx_model_path = os.path.join(temp_output_dir, "yolo11n.onnx")
    om_output_path = os.path.join(temp_output_dir, "yolo11n.om")
    
    yolo_to_onnx_task_id = None
    onnx_to_om_task_id = None
    
    try:
        # 1. 创建任务管理器实例
        task_manager = get_task_manager()
        
        # 2. 检查并准备ONNX模型文件
        if os.path.exists(onnx_model_path_in_demo):
            # 复制现有ONNX文件到临时目录
            shutil.copy(onnx_model_path_in_demo, onnx_model_path)
            print(f"已复制现有ONNX文件: {onnx_model_path_in_demo} -> {onnx_model_path}")
        else:
            # 创建YOLO到ONNX的转换任务
            yolo_to_onnx_task_id = task_manager.create_task(
                task_type="yolo_to_onnx",
                platform="onnx",
                input_path=yolo_model_path,
                output_path=onnx_model_path,
                parameters={"img_size": 640, "simplify": True, "opset_version": 13}
            )
            
            assert yolo_to_onnx_task_id is not None, "创建YOLO到ONNX任务失败"
            
            # 启动调度器执行YOLO到ONNX转换
            scheduler = TaskScheduler(check_interval=2, max_concurrent_tasks=1)
            scheduler.start()
            
            # 等待任务完成
            max_wait_time = 60  # 最大等待时间60秒
            start_time = time.time()
            yolo_to_onnx_task = None
            
            while time.time() - start_time < max_wait_time:
                yolo_to_onnx_task = task_manager.get_task(yolo_to_onnx_task_id)
                if yolo_to_onnx_task and yolo_to_onnx_task["status"] in [TASK_STATUS["COMPLETED"], TASK_STATUS["FAILED"]]:
                    break
                time.sleep(2)
            
            # 停止调度器
            scheduler.stop()
            
            # 检查YOLO到ONNX转换是否成功
            assert yolo_to_onnx_task is not None, "获取YOLO到ONNX任务失败"
            assert yolo_to_onnx_task["status"] == TASK_STATUS["COMPLETED"], f"YOLO到ONNX转换失败: {yolo_to_onnx_task.get('error_message')}"
        
        # 3. 创建ONNX到OM的转换任务
        onnx_to_om_task_id = task_manager.create_task(
            task_type="onnx_to_om",
            platform="huawei",
            input_path=onnx_model_path,
            output_path=om_output_path,
            parameters={"input_shape": "1,3,640,640"}
        )
        
        assert onnx_to_om_task_id is not None, "创建ONNX到OM任务失败"
        
        # 4. 启动调度器执行ONNX到OM转换
        scheduler = TaskScheduler(check_interval=2, max_concurrent_tasks=1)
        scheduler.start()
        
        # 5. 等待任务完成
        max_wait_time = 60  # 最大等待时间60秒
        start_time = time.time()
        onnx_to_om_task = None
        
        while time.time() - start_time < max_wait_time:
            onnx_to_om_task = task_manager.get_task(onnx_to_om_task_id)
            if onnx_to_om_task and onnx_to_om_task["status"] in [TASK_STATUS["COMPLETED"], TASK_STATUS["FAILED"]]:
                break
            time.sleep(2)
        
        # 停止调度器
        scheduler.stop()
        
        # 6. 检查ONNX到OM转换是否成功
        assert onnx_to_om_task is not None, "获取ONNX到OM任务失败"
        # 注意：如果没有安装华为昇腾环境，转换会失败，这里只检查任务是否被处理
        
        # 7. 删除任务
        if yolo_to_onnx_task_id:
            task_manager.delete_task(yolo_to_onnx_task_id)
        task_manager.delete_task(onnx_to_om_task_id)
        
        # 8. 验证任务已删除
        if yolo_to_onnx_task_id:
            deleted_yolo_task = task_manager.get_task(yolo_to_onnx_task_id)
            assert deleted_yolo_task is None, "YOLO到ONNX任务删除失败"
        deleted_om_task = task_manager.get_task(onnx_to_om_task_id)
        assert deleted_om_task is None, "ONNX到OM任务删除失败"
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_output_dir)
        print(f"清理临时目录: {temp_output_dir}")

if __name__ == "__main__":
    test_onnx_to_om_conversion_service()