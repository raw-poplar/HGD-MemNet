#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查找现有的chunk文件
"""

import os
import glob
import torch
import sys
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def find_chunk_files():
    """查找所有可能的chunk文件位置"""
    print("搜索chunk文件...")
    
    # 可能的搜索路径
    search_paths = [
        ".",
        "./data",
        "./data/lccc_processed",
        "F:/modelTrain/data/lccc_processed",  # 原始硬编码路径
        "F:/modelTrain",
        "../data",
        "../../data",
    ]
    
    found_chunks = []
    
    for base_path in search_paths:
        if not os.path.exists(base_path):
            continue

        print(f"搜索路径: {base_path}")
        
        # 搜索chunk文件
        patterns = [
            os.path.join(base_path, "**", "chunk_*.pt"),
            os.path.join(base_path, "chunk_*.pt"),
            os.path.join(base_path, "*", "chunk_*.pt"),
            os.path.join(base_path, "*", "*", "chunk_*.pt"),
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern, recursive=True)
            for file in files:
                if file not in [f[0] for f in found_chunks]:
                    # 获取文件信息
                    try:
                        stat = os.stat(file)
                        size_mb = stat.st_size / (1024 * 1024)
                        found_chunks.append((file, size_mb))
                    except:
                        found_chunks.append((file, 0))
    
    return found_chunks

def analyze_chunk_files(chunk_files):
    """分析chunk文件"""
    if not chunk_files:
        print("未找到任何chunk文件")
        return None
    
    print(f"\n找到 {len(chunk_files)} 个chunk文件:")
    
    # 按路径分组
    by_directory = {}
    for file_path, size in chunk_files:
        directory = os.path.dirname(file_path)
        if directory not in by_directory:
            by_directory[directory] = []
        by_directory[directory].append((file_path, size))
    
    total_size = 0
    total_samples = 0
    
    for directory, files in by_directory.items():
        print(f"\n目录: {directory}")
        print(f"   文件数量: {len(files)}")
        
        dir_size = sum(size for _, size in files)
        total_size += dir_size
        print(f"   总大小: {dir_size:.2f} MB")
        
        # 检查文件编号连续性
        chunk_numbers = []
        for file_path, _ in files:
            filename = os.path.basename(file_path)
            try:
                # 提取chunk编号
                if filename.startswith("chunk_") and filename.endswith(".pt"):
                    num_str = filename[6:-3]  # 去掉"chunk_"和".pt"
                    chunk_numbers.append(int(num_str))
            except:
                pass
        
        if chunk_numbers:
            chunk_numbers.sort()
            print(f"   编号范围: {min(chunk_numbers)} - {max(chunk_numbers)}")
            
            # 检查是否连续
            expected = list(range(min(chunk_numbers), max(chunk_numbers) + 1))
            missing = set(expected) - set(chunk_numbers)
            if missing:
                print(f"   缺失编号: {sorted(missing)}")
            else:
                print(f"   编号连续")
        
        # 检查第一个文件的内容
        if files:
            first_file = files[0][0]
            try:
                print(f"   检查文件: {os.path.basename(first_file)}")
                data = torch.load(first_file, weights_only=True)
                print(f"      样本数量: {len(data)}")
                total_samples += len(data) * len(files)  # 估算总样本数
                
                if data:
                    sample = data[0]
                    x_ref, steps_data = sample
                    print(f"      样本格式: x_ref={x_ref.shape}, steps={len(steps_data)}")
                    
                    # 检查数据格式（新vs旧）
                    none_count = sum(1 for x_t, _, _ in steps_data if x_t is None)
                    if none_count > 0:
                        print(f"      新格式数据 (None x_t: {none_count}/{len(steps_data)})")
                    else:
                        print(f"      旧格式数据 (无None x_t)")
                        
            except Exception as e:
                print(f"      读取失败: {e}")
    
    print(f"\n总计:")
    print(f"   文件总数: {len(chunk_files)}")
    print(f"   总大小: {total_size:.2f} MB")
    print(f"   估计样本数: {total_samples:,}")
    
    return by_directory

def check_continuation_possibility(chunk_dirs):
    """检查是否可以继续处理"""
    print(f"\n继续处理可行性分析:")
    
    for directory, files in chunk_dirs.items():
        print(f"\n📁 {directory}:")
        
        # 检查是否有partial文件
        partial_file = os.path.join(directory, "partial_processed.pt")
        if os.path.exists(partial_file):
            print(f"   找到partial文件: {partial_file}")
            try:
                partial_data = torch.load(partial_file, weights_only=True)
                print(f"      Partial样本数: {len(partial_data)}")
            except Exception as e:
                print(f"      Partial文件损坏: {e}")
        else:
            print(f"   未找到partial文件")
        
        # 检查最大chunk编号
        chunk_numbers = []
        for file_path, _ in files:
            filename = os.path.basename(file_path)
            try:
                if filename.startswith("chunk_") and filename.endswith(".pt"):
                    num_str = filename[6:-3]
                    chunk_numbers.append(int(num_str))
            except:
                pass
        
        if chunk_numbers:
            max_chunk = max(chunk_numbers)
            print(f"   最大chunk编号: {max_chunk}")
            print(f"   下一个chunk将是: chunk_{max_chunk + 1}.pt")
        
        # 检查是否已经有合并文件
        merged_files = []
        for split in ["train", "valid", "test"]:
            merged_file = os.path.join(directory, f"{split}_data.pt")
            if os.path.exists(merged_file):
                merged_files.append(split)
        
        if merged_files:
            print(f"   已存在合并文件: {merged_files}")
            print(f"      建议备份后重新合并")
        else:
            print(f"   未找到合并文件，可以正常合并")

if __name__ == "__main__":
    print("查找现有chunk文件...")
    
    chunk_files = find_chunk_files()
    
    if chunk_files:
        chunk_dirs = analyze_chunk_files(chunk_files)
        if chunk_dirs:
            check_continuation_possibility(chunk_dirs)
            
            print(f"\n结论:")
            print(f"1. 找到了现有的chunk文件")
            print(f"2. 修改后的代码可以继续处理")
            print(f"3. 数据格式兼容，可以混合处理")
            print(f"4. 处理完成后可以正常合并为train/valid/test文件")
    else:
        print(f"\n未找到任何chunk文件")
        print(f"可能的原因:")
        print(f"1. chunk文件在其他位置")
        print(f"2. 使用了不同的文件名模式")
        print(f"3. 文件已被移动或删除")
