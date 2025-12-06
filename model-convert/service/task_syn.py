import logging
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入task_rmq模块中的RabbitMQTaskManager类
from common.task_rmq import RabbitMQTaskManager
# 导入配置加载器
from config.config_loader import ConfigLoader

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sync_task_status() -> int:
    """
    同步任务状态：从RabbitMQ的result队列获取任务状态并更新到SQLite
    
    Returns:
        int: 更新的任务数量
    """
    try:
        # 从配置加载器获取RabbitMQ配置
        config_loader = ConfigLoader()
        rmq_conn_config = config_loader.get_rabbitmq_connection_config()
        rmq_queues_config = config_loader.get_rabbitmq_queues_config()
        
        # 创建RabbitMQTaskManager实例
        rmq_manager = RabbitMQTaskManager(
            host=rmq_conn_config.get('host', 'localhost'),
            port=rmq_conn_config.get('port', 5672),
            username=rmq_conn_config.get('username', 'guest'),
            password=rmq_conn_config.get('password', 'guest'),
            virtual_host=rmq_conn_config.get('virtual_host', '/'),
            task_queue=rmq_queues_config.get('task_queue', 'model_convert_tasks'),
            result_queue=rmq_queues_config.get('result_queue', 'model_convert_results')
        )
        
        # 调用任务状态拉取函数
        updated_count = rmq_manager.update_task_from_result_queue()
        
        logger.info(f"任务状态同步完成，共更新了{updated_count}个任务")
        return updated_count
        
    except Exception as e:
        logger.error(f"任务状态同步失败: {str(e)}")
        return 0


def main():
    """
    主函数，用于直接调用任务状态同步功能
    """
    logger.info("开始执行任务状态同步...")
    updated_count = sync_task_status()
    logger.info(f"任务状态同步结束，更新数量: {updated_count}")


if __name__ == "__main__":
    main()