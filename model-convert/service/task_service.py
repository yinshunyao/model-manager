#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务管理服务主程序

此脚本作为独立的微服务运行，负责从数据库中搜索需要执行的任务并执行。
"""
import os
import sys
import time
import logging
import argparse

# 添加项目路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'task_service.log'))
    ]
)
logger = logging.getLogger('task_service')

def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='跨平台模型转换任务管理服务')
    
    parser.add_argument(
        '--interval', 
        type=int, 
        default=10, 
        help='检查待执行任务的间隔时间（秒）'
    )
    
    parser.add_argument(
        '--max-concurrent', 
        type=int, 
        default=1, 
        help='最大并发执行的任务数'
    )
    
    parser.add_argument(
        '--daemon', 
        action='store_true', 
        help='以守护进程方式运行'
    )
    
    return parser.parse_args()

def start_service(interval: int, max_concurrent: int):
    """
    启动任务管理服务
    
    Args:
        interval: 检查间隔时间（秒）
        max_concurrent: 最大并发任务数
    """
    try:
        logger.info(f"正在启动任务管理服务...")
        logger.info(f"检查间隔: {interval}秒, 最大并发任务数: {max_concurrent}")
        
        # 导入任务调度器
        from service.task_scheduler import TaskScheduler
        
        # 创建并启动调度器
        scheduler = TaskScheduler(
            check_interval=interval,
            max_concurrent_tasks=max_concurrent
        )
        
        scheduler.start()
        
        logger.info("任务管理服务已成功启动")
        logger.info("服务将定期检查数据库中的待执行任务并执行")
        
        # 保持主程序运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("接收到中断信号，正在停止服务...")
        finally:
            scheduler.stop()
            logger.info("任务管理服务已停止")
            
    except ImportError as e:
        logger.error(f"导入模块失败: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        sys.exit(1)

def daemonize():
    """
    将进程守护化
    """
    try:
        # 第一次fork，创建子进程，脱离父进程
        pid = os.fork()
        if pid > 0:
            # 父进程退出
            sys.exit(0)
    except OSError as err:
        logger.error(f"fork #1 failed: {err}")
        sys.exit(1)
    
    # 子进程继续运行
    # 修改工作目录
    os.chdir("/")
    # 创建新的会话，脱离控制终端
    os.setsid()
    # 设置文件权限掩码
    os.umask(0)
    
    try:
        # 第二次fork，防止进程重新获取控制终端
        pid = os.fork()
        if pid > 0:
            # 第一个子进程退出
            sys.exit(0)
    except OSError as err:
        logger.error(f"fork #2 failed: {err}")
        sys.exit(1)
    
    # 第二个子进程继续运行
    # 重定向标准文件描述符
    sys.stdout.flush()
    sys.stderr.flush()
    
    si = open(os.devnull, 'r')
    so = open(os.devnull, 'a+')
    se = open(os.devnull, 'a+')
    
    os.dup2(si.fileno(), sys.stdin.fileno())
    os.dup2(so.fileno(), sys.stdout.fileno())
    os.dup2(se.fileno(), sys.stderr.fileno())

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 如果需要以守护进程方式运行
    if args.daemon:
        logger.info("以守护进程方式启动服务")
        daemonize()
    
    # 启动服务
    start_service(args.interval, args.max_concurrent)

if __name__ == "__main__":
    main()