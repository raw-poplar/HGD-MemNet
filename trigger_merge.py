#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
触发原始的合并逻辑
"""

import os
import sys
import glob
sys.path.append('.')
import config
from src.prepare_binary_data import merge_chunks

def cleanup_and_merge():
    """清理partial文件并触发合并"""
    print("🔧 清理partial文件并触发合并...")
    
    data_types = ["train", "valid", "test"]
    
    for data_type in data_types:
        chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
        if not os.path.exists(chunk_dir):
            print(f"❌ {data_type} 目录不存在")
            continue
            
        print(f"\n📁 处理 {data_type}:")
        
        # 1. 清理partial文件
        partial_pattern = os.path.join(chunk_dir, "partial_*.pt")
        partial_files = glob.glob(partial_pattern)
        
        if partial_files:
            print(f"   🔄 找到 {len(partial_files)} 个partial文件")
            for partial_file in partial_files:
                try:
                    partial_index = int(os.path.basename(partial_file).split('_')[1].split('.')[0])
                    target_chunk = os.path.join(chunk_dir, f"chunk_{partial_index}.pt")
                    
                    if os.path.exists(target_chunk):
                        print(f"   ⚠️  chunk_{partial_index}.pt 已存在，删除partial文件")
                        os.remove(partial_file)
                    else:
                        print(f"   🔄 重命名: partial_{partial_index}.pt -> chunk_{partial_index}.pt")
                        os.rename(partial_file, target_chunk)
                        
                except Exception as e:
                    print(f"   ❌ 处理 {partial_file} 失败: {e}")
        else:
            print(f"   ✅ 无partial文件")
        
        # 2. 检查chunk文件
        chunk_files = glob.glob(os.path.join(chunk_dir, "chunk_*.pt"))
        print(f"   📦 找到 {len(chunk_files)} 个chunk文件")
        
        if len(chunk_files) == 0:
            print(f"   ❌ 无chunk文件可合并")
            continue
        
        # 3. 检查是否已有合并文件
        merged_file = os.path.join(chunk_dir, f"{data_type}.pt")
        if os.path.exists(merged_file):
            response = input(f"   ⚠️  {data_type}.pt 已存在，是否重新合并? (y/n): ")
            if response.lower() != 'y':
                print(f"   ⏭️  跳过 {data_type}")
                continue
        
        # 4. 执行合并
        try:
            print(f"   🔄 开始合并...")
            merge_chunks(chunk_dir, data_type, delete_chunks_after_merge=False)  # 不删除chunk文件
            print(f"   ✅ {data_type} 合并完成")
        except Exception as e:
            print(f"   ❌ {data_type} 合并失败: {e}")

def verify_merged_files():
    """验证合并后的文件"""
    print(f"\n📊 验证合并后的文件...")
    
    data_types = ["train", "valid", "test"]
    
    for data_type in data_types:
        merged_file = os.path.join(config.LCCC_PROCESSED_PATH, data_type, f"{data_type}.pt")
        
        if os.path.exists(merged_file):
            try:
                import torch
                data = torch.load(merged_file, weights_only=True)
                size_gb = os.path.getsize(merged_file) / (1024**3)
                print(f"   ✅ {data_type}.pt: {len(data):,} 个对话, {size_gb:.2f} GB")
            except Exception as e:
                print(f"   ❌ {data_type}.pt 验证失败: {e}")
        else:
            print(f"   ❌ {data_type}.pt 不存在")

if __name__ == "__main__":
    print("🔧 触发原始合并逻辑")
    print("=" * 40)
    
    cleanup_and_merge()
    verify_merged_files()
    
    print(f"\n🎯 完成!")
    print(f"💡 合并后的文件位置:")
    print(f"   📁 F:/modelTrain/data/lccc_processed/train/train.pt")
    print(f"   📁 F:/modelTrain/data/lccc_processed/valid/valid.pt")
    print(f"   📁 F:/modelTrain/data/lccc_processed/test/test.pt")
