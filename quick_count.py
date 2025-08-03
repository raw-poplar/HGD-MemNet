#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速统计处理进度
"""

import os
import glob

def quick_count():
    """快速统计"""
    print("📊 快速统计处理进度...")
    
    # 直接搜索chunk文件
    chunk_pattern = "F:/modelTrain/data/lccc_processed/train/chunk_*.pt"
    chunk_files = glob.glob(chunk_pattern)
    
    if not chunk_files:
        print("❌ 未找到chunk文件")
        return
    
    print(f"📦 找到 {len(chunk_files)} 个chunk文件")
    
    # 获取chunk编号
    chunk_numbers = []
    for file_path in chunk_files:
        filename = os.path.basename(file_path)
        try:
            num = int(filename.split('_')[1].split('.')[0])
            chunk_numbers.append(num)
        except:
            pass
    
    if chunk_numbers:
        chunk_numbers.sort()
        print(f"🔢 编号范围: {min(chunk_numbers)} - {max(chunk_numbers)}")
        print(f"📈 最大chunk编号: {max(chunk_numbers)}")
        print(f"📊 总chunk数: {len(chunk_numbers)}")
        
        # 检查连续性
        expected = list(range(min(chunk_numbers), max(chunk_numbers) + 1))
        missing = set(expected) - set(chunk_numbers)
        if missing:
            print(f"⚠️  缺失编号: {sorted(missing)}")
        else:
            print(f"✅ 编号连续")
        
        # 估算处理进度
        # 假设每个chunk包含约10000个对话
        estimated_dialogues = len(chunk_numbers) * 10000
        print(f"\n📊 估算统计:")
        print(f"   🗣️  估算已处理对话数: {estimated_dialogues:,}")
        
        # 基于chunk数量估算处理的行数
        # train.jsonl 大约有 15,220,604 行（从之前的估算）
        total_estimated_lines = 15220604
        
        # 假设每个chunk对应约 20000 行原始数据
        estimated_processed_lines = len(chunk_numbers) * 20000
        
        print(f"   📄 估算已处理行数: {estimated_processed_lines:,}")
        
        if total_estimated_lines > 0:
            progress = min(estimated_processed_lines / total_estimated_lines * 100, 100)
            print(f"   📈 估算处理进度: {progress:.1f}%")
            
            remaining_lines = max(0, total_estimated_lines - estimated_processed_lines)
            print(f"   ⏳ 估算剩余行数: {remaining_lines:,}")
    
    # 检查文件大小
    print(f"\n💾 文件大小统计:")
    total_size = 0
    for file_path in chunk_files[:5]:  # 检查前5个文件
        try:
            size = os.path.getsize(file_path)
            total_size += size
            filename = os.path.basename(file_path)
            print(f"   {filename}: {size / (1024*1024):.2f} MB")
        except:
            pass
    
    if len(chunk_files) > 0:
        avg_size = total_size / min(len(chunk_files), 5)
        estimated_total_size = avg_size * len(chunk_files)
        print(f"   📊 估算总大小: {estimated_total_size / (1024*1024*1024):.2f} GB")

if __name__ == "__main__":
    quick_count()
