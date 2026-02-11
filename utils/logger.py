import logging
import os
from datetime import datetime


# -> it seems to adjust the setting of the logging package
def set_logging_level(logging_level, log_file_path=None):
    logging_level = logging_level.lower()

    if logging_level == "critical":
        level = logging.CRITICAL
    elif logging_level == "warning":
        level = logging.WARNING
    elif logging_level == "info":
        level = logging.INFO
    else:
        level = logging.DEBUG

    # 清除现有的handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 创建formatter - 添加行号信息
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(filename)s:%(lineno)d]: %(message)s', 
        datefmt='%Y-%m-%d:%H:%M:%S'
    )
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # 添加handlers到root logger
    logging.root.addHandler(console_handler)
    
    # 如果指定了日志文件路径，创建文件handler
    if log_file_path:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
        
        print(f"日志将保存到: {log_file_path}")
    
    # 设置root logger的级别
    logging.root.setLevel(level)


def get_default_log_path(label=None, exp_dir=None):
    """生成默认的日志文件路径"""
    if exp_dir:
        log_dir = os.path.join(exp_dir, "logs")
    else:
        log_dir = "logs"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if label:
        log_filename = f"{label}_{timestamp}.log"
    else:
        log_filename = f"training_{timestamp}.log"
    
    return os.path.join(log_dir, log_filename)


