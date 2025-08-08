#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
词汇表构建快速启动脚本

这是一个简化的启动脚本，用于快速构建词汇表。
"""

import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import config
from src.data_processing.vocabulary.build_vocabulary import VocabularyBuilder


def main():
    """主函数"""
    print("词汇表构建工具 - 快速启动")
    print("=" * 50)
    
    # 配置参数
    data_path = config.LCCC_PROCESSED_PATH  # 使用config中的数据路径
    vocab_size = 30000  # 词汇表大小
    min_freq = 2  # 最小词频
    
    print(f"配置信息:")
    print(f"   数据路径: {data_path}")
    print(f"   词汇表大小: {vocab_size}")
    print(f"   最小词频: {min_freq}")
    
    # 检查数据路径是否存在
    if not os.path.exists(data_path):
        print(f"数据路径不存在: {data_path}")
        print(f"请确认以下文件存在:")
        print(f"   - {os.path.join(data_path, 'train.jsonl')}")
        print(f"   - {os.path.join(data_path, 'valid.jsonl')}")
        print(f"   - {os.path.join(data_path, 'test.jsonl')}")
        return 1
    
    try:
        # 创建词汇表构建器
        builder = VocabularyBuilder(
            data_path=data_path,
            vocab_size=vocab_size,
            min_freq=min_freq
        )
        
        # 构建词汇表
        vocab = builder.build_vocabulary()
        
        # 打印统计信息
        builder.print_vocabulary_stats(vocab)
        
        # 保存词汇表
        output_path = builder.save_vocabulary(vocab)
        
        print(f"\n词汇表构建完成!")
        print(f"   输出文件: {output_path}")
        print(f"\n使用提示:")
        print(f"   1. 词汇表文件: vocabulary.json")
        print(f"   2. 统计信息文件: vocabulary_stats.json")
        print(f"   3. 可以在训练脚本中加载使用")
        
    except Exception as e:
        print(f"构建过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
