#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试当前设置
"""

import os
import json
import sys
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.dataset import Vocabulary

def test_current_setup():
    """测试当前设置是否正确"""
    print("测试当前设置...")
    
    # 1. 检查路径配置
    print(f"LCCC_PROCESSED_PATH: {config.LCCC_PROCESSED_PATH}")
    
    # 2. 检查词汇表文件
    vocab_path = os.path.join(config.LCCC_PROCESSED_PATH, "vocabulary.json")
    print(f"词汇表路径: {vocab_path}")
    print(f"词汇表存在: {os.path.exists(vocab_path)}")
    
    if os.path.exists(vocab_path):
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_dict = json.load(f)
            vocab = Vocabulary("lccc")
            vocab.__dict__.update(vocab_dict)
            print(f"词汇表加载成功，大小: {vocab.num_words}")
        except Exception as e:
            print(f"词汇表加载失败: {e}")
            return False
    else:
        print(f"词汇表文件不存在")
        return False
    
    # 3. 检查数据文件
    data_types = ["train", "valid", "test"]
    for data_type in data_types:
        data_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{data_type}.jsonl")
        print(f"{data_type}.jsonl 存在: {os.path.exists(data_file)}")
        
        if os.path.exists(data_file):
            # 检查文件大小
            size_mb = os.path.getsize(data_file) / (1024 * 1024)
            print(f"   文件大小: {size_mb:.1f} MB")
            
            # 检查是否可以读取
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if first_line:
                        dialogue = json.loads(first_line)
                        print(f"   可以读取，样本对话长度: {len(dialogue)}")
                    else:
                        print(f"   文件为空")
            except Exception as e:
                print(f"   读取失败: {e}")
    
    # 4. 检查现有chunk文件
    for data_type in data_types:
        chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
        if os.path.exists(chunk_dir):
            chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".pt")]
            print(f"{data_type} 现有chunk数: {len(chunk_files)}")
            
            if chunk_files:
                # 找到最大编号
                chunk_numbers = []
                for f in chunk_files:
                    try:
                        num = int(f.split('_')[1].split('.')[0])
                        chunk_numbers.append(num)
                    except:
                        pass
                
                if chunk_numbers:
                    max_chunk = max(chunk_numbers)
                    print(f"   最大chunk编号: {max_chunk}")
                    print(f"   下一个chunk将是: chunk_{max_chunk + 1}.pt")
        else:
            print(f"{data_type} chunk目录不存在")
    
    # 5. 检查resume文件
    for data_type in data_types:
        chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
        resume_file = os.path.join(chunk_dir, f"resume_{data_type}.json")
        if os.path.exists(resume_file):
            try:
                with open(resume_file, 'r') as f:
                    resume_data = json.load(f)
                print(f"{data_type} resume信息: {resume_data}")
            except Exception as e:
                print(f"{data_type} resume文件损坏: {e}")
        else:
            print(f"{data_type} 无resume文件（将从头开始）")
    
    return True

def estimate_processing_time():
    """估算处理时间"""
    print(f"\n处理时间估算:")
    
    # 检查文件大小
    total_size = 0
    for data_type in ["train", "valid", "test"]:
        data_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{data_type}.jsonl")
        if os.path.exists(data_file):
            size_mb = os.path.getsize(data_file) / (1024 * 1024)
            total_size += size_mb
            print(f"   {data_type}: {size_mb:.1f} MB")
    
    print(f"   总大小: {total_size:.1f} MB")
    
    # 基于经验估算
    # 假设处理速度为 50-200 it/s，每个对话约 0.5KB
    estimated_dialogues = total_size * 1024 / 0.5  # 估算对话数
    
    speeds = {
        "单线程": 100,
        "4线程": 300,
        "8线程": 400
    }
    
    print(f"   估算对话数: {estimated_dialogues:,.0f}")
    print(f"   预计处理时间:")
    
    for mode, speed in speeds.items():
        time_seconds = estimated_dialogues / speed
        time_minutes = time_seconds / 60
        time_hours = time_minutes / 60
        
        if time_hours > 1:
            print(f"     {mode}: {time_hours:.1f} 小时")
        elif time_minutes > 1:
            print(f"     {mode}: {time_minutes:.1f} 分钟")
        else:
            print(f"     {mode}: {time_seconds:.1f} 秒")

if __name__ == "__main__":
    print("测试当前设置...")
    
    if test_current_setup():
        print(f"\n设置检查通过！")
        estimate_processing_time()
        
        print(f"\n建议:")
        print(f"1. 使用 4-8 个worker进程以获得最佳性能")
        print(f"2. 监控内存使用，确保不超过系统限制")
        print(f"3. 处理过程中可以随时中断，支持断点续传")
        print(f"4. 完成后会自动合并为最终的train/valid/test文件")
    else:
        print(f"\n设置检查失败！")
        print(f"请检查路径配置和文件完整性")
