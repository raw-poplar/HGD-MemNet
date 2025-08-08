#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
词汇表构建模块

这个模块包含了从JSONL数据文件构建词汇表的工具。

主要功能:
1. 从多个JSONL文件中提取词汇
2. 统计词频并过滤低频词
3. 生成指定大小的词汇表
4. 支持中文分词和文本预处理
5. 输出JSON格式的词汇表文件

使用方法:
    from src.data_processing.vocabulary import VocabularyBuilder
    
    builder = VocabularyBuilder(data_path="path/to/data", vocab_size=30000)
    vocab = builder.build_vocabulary()
    builder.save_vocabulary(vocab)
"""

__version__ = "1.0.0"
__author__ = "HGD-MemNet Team"

# 导入主要功能
from .build_vocabulary import VocabularyBuilder

__all__ = [
    'VocabularyBuilder'
]
