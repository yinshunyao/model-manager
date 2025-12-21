import pika
import json
import logging
import time
from typing import Dict, Optional, Any

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RabbitMQTaskManager:
    """
    RabbitMQ任务管理器类，负责处理RabbitMQ连接和任务管理
    """
    
    def __init__(self, host: str, port: int, username: str, password: str, virtual_host: str = '/', task_queue: str = 'model_convert_tasks', result_queue: str = 'model_convert_results'):
        """
        初始化RabbitMQ任务管理器
        
        Args:
            host: RabbitMQ服务器地址
            port: RabbitMQ服务器端口
            username: RabbitMQ用户名
            password: RabbitMQ密码
            virtual_host: RabbitMQ虚拟主机
            task_queue: 任务队列名称
            result_queue: 结果队列名称
        """
        # 保存RabbitMQ连接配置
        self.rabbitmq_conn_config = {
            'host': host,
            'port': port,
            'username': username,
            'password': password,
            'virtual_host': virtual_host
        }
        # 保存队列配置
        self.rabbitmq_queues_config = {
            'task_queue': task_queue,
            'result_queue': result_queue
        }
        # 初始化连接和通道
        self.connection: Optional[Any] = None
        self.channel: Optional[Any] = None
    
    def _connect(self):
        """
        连接到RabbitMQ服务器
        
        Returns:
            bool: 连接是否成功
        """
        try:
            # 构建连接参数
            credentials = pika.PlainCredentials(
                self.rabbitmq_conn_config.get('username', 'guest'),
                self.rabbitmq_conn_config.get('password', 'guest')
            )
            parameters = pika.ConnectionParameters(
                host=self.rabbitmq_conn_config.get('host', 'localhost'),
                port=self.rabbitmq_conn_config.get('port', 5672),
                virtual_host=self.rabbitmq_conn_config.get('virtual_host', '/'),
                credentials=credentials,
                heartbeat=self.rabbitmq_conn_config.get('heartbeat', 30)
            )
            
            # 建立连接和通道
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # 声明队列
            task_queue = self.rabbitmq_queues_config.get('task_queue', 'task')
            result_queue = self.rabbitmq_queues_config.get('result_queue', 'result')
            if self.channel:
                self.channel.queue_declare(queue=task_queue, durable=True)
                self.channel.queue_declare(queue=result_queue, durable=True)
                
                logger.info("成功连接到RabbitMQ服务器")
                return True
            else:
                logger.error("RabbitMQ通道创建失败")
                return False
        except Exception as e:
            logger.error(f"连接RabbitMQ失败: {e}")
            self._close()
            return False
    
    def _close(self):
        """
        关闭RabbitMQ连接
        """
        try:
            if self.channel:
                self.channel.close()
            if self.connection:
                self.connection.close()
            logger.info("RabbitMQ连接已关闭")
        except Exception as e:
            logger.error(f"关闭RabbitMQ连接时出错: {e}")
        finally:
            self.channel = None
            self.connection = None
    
    def get_task_from_rmq(self, update_sqlite: bool = True) -> Optional[Dict[str, Any]]:
        """
        从RabbitMQ获取任务并可选择性地缓存到本地SQLite
        - 查询SQLite，如果本地无正在执行任务，才从RabbitMQ中拉取
        - 拉取到任务，设置为运行中，存储到SQLite中
        
        Args:
            update_sqlite: 是否更新SQLite，默认为True
            
        Returns:
            Optional[Dict]: 任务数据字典，如果没有可用任务则返回None
        """
        method_frame = None
        try:
            # 导入task_manager获取任务管理器实例
            from service.task_manager import get_task_manager, TASK_STATUS
            task_manager = get_task_manager()
            
            # 检查本地是否有正在执行的任务
            has_running_task = False
            try:
                # 获取所有任务，然后筛选运行中的任务
                all_tasks = task_manager.get_all_tasks()
                running_tasks = [task for task in all_tasks if task.get('status') == TASK_STATUS['RUNNING']]
                has_running_task = len(running_tasks) > 0
            except Exception as e:
                logger.warning(f"检查运行中任务时出错: {e}，默认假设无运行中任务")
            
            if has_running_task:
                # 计算运行中任务数量
                try:
                    all_tasks = task_manager.get_all_tasks()
                    running_count = sum(1 for task in all_tasks if task.get('status') == TASK_STATUS['RUNNING'])
                except Exception:
                    running_count = 1
                logger.info(f"本地已有{running_count}个正在执行的任务，暂不拉取新任务")
                return None
            
            # 连接RabbitMQ
            if not self._connect():
                logger.error("无法连接到RabbitMQ，无法获取任务")
                return None
            
            # 确保channel不为None
            if not self.channel:
                logger.error("RabbitMQ通道未初始化")
                return None
            
            # 尝试获取一个任务（非阻塞）
            task_queue = self.rabbitmq_queues_config.get('task_queue', 'task')
            method_frame, header_frame, body = self.channel.basic_get(queue=task_queue, auto_ack=False)
            
            if method_frame and body:
                # 解析任务数据
                task_data = json.loads(body.decode('utf-8'))
                task_id = task_data.get('task_id', str(int(time.time() * 1000)))
                
                # 如果需要更新SQLite，则使用task_manager创建任务
                if update_sqlite:
                    try:
                        # 从任务数据中提取必要参数
                        task_type = task_data.get('task_type', 'onnx_to_om')
                        platform = task_data.get('platform', 'huawei')
                        input_path = task_data.get('input_path', '')
                        output_path = task_data.get('output_path', '')
                        parameters = task_data.get('parameters', {})
                        
                        # 使用task_manager创建任务并开始执行
                        created_task_id = task_manager.create_task(
                            task_type=task_type,
                            platform=platform,
                            input_path=input_path,
                            output_path=output_path,
                            parameters=parameters,
                            task_id=task_id
                        )
                        
                        # 开始执行任务
                        task_manager.start_task(created_task_id)
                        
                        logger.info(f"成功创建并启动任务: {created_task_id}")
                    except Exception as e:
                        logger.error(f"使用task_manager创建任务失败: {e}")
                        # 即使创建任务失败，也继续处理消息
                
                # 确认任务已接收
                if self.channel:
                    self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                logger.info(f"成功获取任务: {task_id}")
                return task_data
            else:
                logger.info("RabbitMQ中没有可用任务")
                return None
        except Exception as e:
            logger.error(f"获取任务时出错: {e}")
            # 如果有消息但处理失败，拒绝消息
            if method_frame and self.channel:
                self.channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=True)
            return None
        finally:
            self._close()
    
    def push_task_status(self, task_id: str, status: str, result: Optional[Dict[str, Any]] = None):
        """
        推送任务状态到RabbitMQ的result队列中
        
        Args:
            task_id: 任务ID
            status: 任务状态 (pending, running, completed, failed)
            result: 任务执行结果，可选
        
        Returns:
            bool: 推送是否成功
        """
        try:
            # 验证task_id不为None
            if task_id is None:
                logger.error("任务ID不能为空")
                return False
            
            # 构建状态消息
            status_message = {
                'task_id': task_id,
                'status': status,
                'timestamp': time.time(),
                'result': result if result else {}
            }
            
            # 连接RabbitMQ
            if not self._connect():
                logger.error("无法连接到RabbitMQ，无法推送任务状态")
                return False
            
            # 确保channel不为None
            if not self.channel:
                logger.error("RabbitMQ通道未初始化")
                return False
            
            # 推送消息到result队列
            result_queue = self.rabbitmq_queues_config.get('result_queue', 'result')
            if self.channel:
                self.channel.basic_publish(
                    exchange='',
                    routing_key=result_queue,
                    body=json.dumps(status_message),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # 持久化消息
                        content_type='application/json'
                    )
                )
            
            logger.info(f"成功推送任务状态到RabbitMQ: {task_id}, status: {status}")
            return True
            
        except Exception as e:
            logger.error(f"推送任务状态时出错: {e}")
            return False
        finally:
            self._close()
    
    def update_task_from_result_queue(self, update_sqlite: bool = True):
        """
        拉取任务状态并可选择性地更新SQLite中任务对应状态
        
        Args:
            update_sqlite: 是否更新SQLite，默认为True
            
        Returns:
            int: 更新的任务数量
        """
        updated_count = 0
        try:
            # 连接RabbitMQ
            if not self._connect():
                logger.error("无法连接到RabbitMQ，无法更新任务状态")
                return 0
            
            # 确保channel和connection不为None
            if not self.channel or not self.connection:
                logger.error("RabbitMQ通道或连接未初始化")
                return 0
            
            # 定义回调函数处理消息
            def callback(ch, method, properties, body):
                nonlocal updated_count
                try:
                    # 解析状态消息
                    status_message = json.loads(body.decode('utf-8'))
                    task_id = status_message.get('task_id')
                    new_status = status_message.get('status')
                    error_message = status_message.get('error_message', '')
                    
                    # 如果需要更新SQLite，则使用task_manager更新任务状态
                    if update_sqlite and task_id and new_status:
                        # 使用任务管理器更新任务状态
                        from service.task_manager import get_task_manager, TASK_STATUS
                        task_manager = get_task_manager()
                        
                        try:
                            # 根据新状态选择合适的更新方法
                            if new_status == TASK_STATUS['COMPLETED']:
                                task_manager.complete_task(task_id, error_message=None)
                                updated_count += 1
                                logger.info(f"更新任务状态: {task_id} -> {new_status}")
                            elif new_status == TASK_STATUS['FAILED']:
                                task_manager.complete_task(task_id, error_message=error_message)
                                updated_count += 1
                                logger.info(f"更新任务状态: {task_id} -> {new_status}")
                            elif new_status == TASK_STATUS['RUNNING']:
                                task_manager.start_task(task_id)
                                updated_count += 1
                                logger.info(f"更新任务状态: {task_id} -> {new_status}")
                            else:
                                # 对于其他状态，尝试直接更新
                                task_manager.update_task_status(task_id, new_status, error_message=error_message)
                                updated_count += 1
                                logger.info(f"更新任务状态: {task_id} -> {new_status}")
                        except Exception as update_error:
                            logger.error(f"更新任务状态失败: {update_error}")
                    
                    # 确认消息已处理
                    if ch and method:
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                    
                except Exception as e:
                    logger.error(f"处理状态消息时出错: {e}")
                    # 拒绝消息，不放回队列以避免死循环
                    if ch:
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            
            # 设置消费，最多处理10条消息
            result_queue = self.rabbitmq_queues_config.get('result_queue', 'result')
            if self.channel:
                self.channel.basic_consume(queue=result_queue, on_message_callback=callback, auto_ack=False)
                
                # 尝试消费消息，设置超时为5秒
                self.connection.process_data_events(time_limit=5)
            else:
                logger.error("RabbitMQ通道未初始化，无法设置消费者")
                return 0
            
            logger.info(f"总共更新了{updated_count}个任务状态")
            return updated_count
            
        except Exception as e:
            logger.error(f"更新任务状态时出错: {e}")
            return 0
        finally:
            self._close()
    
    def __del__(self):
        """
        析构函数，确保连接被关闭
        """
        self._close()