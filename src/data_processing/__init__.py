#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理模块

这个模块包含了将LCCC数据集转换为二进制格式的所有工具和脚本。

主要功能:
1. 数据预处理 - 将原始JSON数据转换为模型可用的格式
2. 二进制转换 - 将处理后的数据转换为PyTorch张量并保存
3. 数据合并 - 将分块的数据合并为最终的训练/验证/测试文件
4. 工具脚本 - 各种辅助工具和调试脚本

文件说明:
- prepare_binary_data.py: 主要的数据转换脚本
- merge_tools.py: 数据合并工具集合
- data_utils.py: 数据处理工具函数
- debug_tools.py: 调试和检查工具
- README.md: 详细使用说明
"""

__version__ = "1.0.0"
__author__ = "HGD-MemNet Team"

# 导入主要功能
from .prepare_binary_data import convert_to_binary, process_dialogue_to_tensors
from .merge_tools import merge_chunks_optimized, verify_merged_files
from .data_utils import check_data_integrity, estimate_processing_time
from .debug_tools import test_chunk_loading, analyze_processing_status
from .vocabulary import VocabularyBuilder

__all__ = [
    'convert_to_binary',
    'process_dialogue_to_tensors',
    'merge_chunks_optimized',
    'verify_merged_files',
    'check_data_integrity',
    'estimate_processing_time',
    'test_chunk_loading',
    'analyze_processing_status',
    'VocabularyBuilder'
]
