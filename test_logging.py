#!/usr/bin/env python3
"""
测试日志功能的简单脚本
"""
import logging
import sys
import os

# 添加项目路径到sys.path
sys.path.append('.')

from utils import logger as logger_lib

def test_logging():
    """测试日志功能"""
    print("=== 测试日志功能（包含行号显示）===")
    
    # 测试1: 使用默认路径
    print("\n1. 测试默认日志路径:")
    default_path = logger_lib.get_default_log_path(label="test_experiment")
    print(f"默认日志路径: {default_path}")
    
    # 测试2: 设置日志到文件
    print("\n2. 测试日志输出到文件:")
    logger_lib.set_logging_level("info", default_path)
    
    # 获取logger并测试输出
    logger = logging.getLogger(__name__)
    logger.info("这是一条测试信息 - 应该同时出现在控制台和文件中")  # 第29行
    logger.warning("这是一条警告信息")  # 第30行
    logger.debug("这是一条调试信息 (可能不会显示，取决于日志级别)")  # 第31行
    
    # 测试不同函数中的日志
    test_function_logging()
    
    # 检查文件是否创建
    if os.path.exists(default_path):
        print(f"✓ 日志文件已创建: {default_path}")
        with open(default_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"文件内容预览:\n{content}")
    else:
        print(f"✗ 日志文件未创建: {default_path}")
    
    # 测试3: 使用自定义路径
    print("\n3. 测试自定义日志路径:")
    custom_path = "logs/custom_test.log"
    logger_lib.set_logging_level("debug", custom_path)
    
    logger.info("这是使用自定义路径的测试信息")  # 第47行
    logger.debug("这是调试级别的信息 (现在应该显示)")  # 第48行
    
    if os.path.exists(custom_path):
        print(f"✓ 自定义日志文件已创建: {custom_path}")
        with open(custom_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"自定义日志文件内容:\n{content}")
    else:
        print(f"✗ 自定义日志文件未创建: {custom_path}")

def test_function_logging():
    """测试在不同函数中的日志输出"""
    logger = logging.getLogger(__name__)
    logger.info("这是来自test_function_logging函数的日志信息")  # 第60行
    logger.warning("这是来自test_function_logging函数的警告信息")  # 第61行

if __name__ == "__main__":
    test_logging() 