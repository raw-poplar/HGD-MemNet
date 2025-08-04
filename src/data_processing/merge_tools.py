#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据合并工具集合

包含各种合并chunk文件的工具和方法：
1. 优化版合并 - 内存友好的流式合并
2. 简单合并 - 基础的一次性合并
3. 大文件合并 - 专门处理大chunk文件
4. 验证工具 - 检查合并结果
"""

import os
import torch
import sys
import json
import gc
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

def get_chunk_files(data_type):
    """获取指定数据类型的所有chunk文件"""
    chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
    if not os.path.exists(chunk_dir):
        return []
    
    chunk_files = []
    for file in os.listdir(chunk_dir):
        if file.startswith("chunk_") and file.endswith(".pt"):
            try:
                chunk_num = int(file.split('_')[1].split('.')[0])
                chunk_files.append((chunk_num, file))
            except:
                pass
    
    return sorted(chunk_files, key=lambda x: x[0])

def merge_chunks_simple(data_type):
    """简单合并 - 一次性加载所有chunk"""
    print(f"🔄 简单合并 {data_type}...")
    
    chunk_files = get_chunk_files(data_type)
    if not chunk_files:
        print(f"❌ 未找到chunk文件")
        return False
    
    chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
    output_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{data_type}.pt")
    
    print(f"📦 找到 {len(chunk_files)} 个chunk文件")
    
    # 检查输出文件
    if os.path.exists(output_file):
        response = input(f"⚠️  {output_file} 已存在，是否覆盖? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # 合并数据
    all_data = []
    total_dialogues = 0
    
    for chunk_num, chunk_file in tqdm(chunk_files, desc=f"合并{data_type}"):
        chunk_path = os.path.join(chunk_dir, chunk_file)
        try:
            data = torch.load(chunk_path, weights_only=True)
            all_data.extend(data)
            total_dialogues += len(data)
            del data
            gc.collect()
        except Exception as e:
            print(f"❌ 加载 {chunk_file} 失败: {e}")
            return False
    
    # 保存结果
    print(f"💾 保存到 {output_file}...")
    torch.save(all_data, output_file)
    
    # 验证
    file_size = os.path.getsize(output_file) / (1024**3)
    print(f"✅ {data_type}.pt 保存成功")
    print(f"📊 总对话数: {total_dialogues:,}")
    print(f"📊 文件大小: {file_size:.2f} GB")
    
    return True

def merge_chunks_optimized(data_type, max_workers=2, batch_size=10):
    """优化版合并 - 流式处理，内存友好"""
    print(f"🔄 优化合并 {data_type} (流式处理)...")
    
    chunk_files = get_chunk_files(data_type)
    if not chunk_files:
        print(f"❌ 未找到chunk文件")
        return False
    
    chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
    output_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{data_type}.pt")
    temp_file = output_file + ".tmp"
    
    print(f"📦 找到 {len(chunk_files)} 个chunk文件")
    
    # 检查输出文件
    if os.path.exists(output_file):
        response = input(f"⚠️  {output_file} 已存在，是否覆盖? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # 初始化临时文件
    torch.save([], temp_file)
    total_dialogues = 0
    
    def load_chunk_worker(chunk_info):
        """工作线程：加载chunk"""
        chunk_num, chunk_file = chunk_info
        chunk_path = os.path.join(chunk_dir, chunk_file)
        try:
            data = torch.load(chunk_path, weights_only=True)
            return chunk_num, data, len(data)
        except Exception as e:
            print(f"❌ 加载 {chunk_file} 失败: {e}")
            return chunk_num, None, 0
    
    def append_to_temp_file(new_data):
        """追加数据到临时文件"""
        existing_data = torch.load(temp_file, weights_only=True)
        existing_data.extend(new_data)
        torch.save(existing_data, temp_file)
        del existing_data
        gc.collect()
    
    # 分批处理
    with tqdm(total=len(chunk_files), desc=f"合并{data_type}") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(0, len(chunk_files), batch_size):
                batch = chunk_files[i:i+batch_size]
                
                # 提交加载任务
                futures = [executor.submit(load_chunk_worker, chunk_info) for chunk_info in batch]
                
                # 收集结果
                batch_data = []
                for future in futures:
                    chunk_num, data, count = future.result()
                    if data is not None:
                        batch_data.extend(data)
                        total_dialogues += count
                        del data
                
                # 追加到文件
                if batch_data:
                    append_to_temp_file(batch_data)
                    del batch_data
                    gc.collect()
                
                pbar.update(len(batch))
    
    # 完成合并
    if os.path.exists(output_file):
        os.remove(output_file)
    os.rename(temp_file, output_file)
    
    # 验证
    file_size = os.path.getsize(output_file) / (1024**3)
    print(f"✅ {data_type}.pt 保存成功")
    print(f"📊 总对话数: {total_dialogues:,}")
    print(f"📊 文件大小: {file_size:.2f} GB")
    
    return True

def merge_chunks_large_files(data_type, timeout_seconds=300):
    """大文件合并 - 专门处理大chunk文件"""
    print(f"🔄 大文件合并 {data_type}...")
    
    chunk_files = get_chunk_files(data_type)
    if not chunk_files:
        print(f"❌ 未找到chunk文件")
        return False
    
    chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
    output_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{data_type}.pt")
    
    # 显示文件信息
    total_size = 0
    for chunk_num, chunk_file in chunk_files:
        chunk_path = os.path.join(chunk_dir, chunk_file)
        size = os.path.getsize(chunk_path)
        size_mb = size / (1024*1024)
        total_size += size
        print(f"   📄 {chunk_file}: {size_mb:.1f} MB")
    
    print(f"📊 总大小: {total_size / (1024*1024*1024):.2f} GB")
    
    # 检查输出文件
    if os.path.exists(output_file):
        response = input(f"⚠️  {output_file} 已存在，是否覆盖? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # 逐个处理大文件
    all_data = []
    total_dialogues = 0
    
    for i, (chunk_num, chunk_file) in enumerate(chunk_files):
        chunk_path = os.path.join(chunk_dir, chunk_file)
        size_mb = os.path.getsize(chunk_path) / (1024*1024)
        
        print(f"\n📦 处理 {chunk_file} ({i+1}/{len(chunk_files)})...")
        print(f"   📊 文件大小: {size_mb:.1f} MB")
        
        start_time = time.time()
        
        try:
            print(f"   🔄 开始加载... (超时: {timeout_seconds}秒)")
            data = torch.load(chunk_path, weights_only=True)
            
            load_time = time.time() - start_time
            dialogue_count = len(data)
            
            print(f"   ✅ 加载成功!")
            print(f"   ⏱️  加载时间: {load_time:.1f}秒")
            print(f"   📊 对话数: {dialogue_count:,}")
            
            # 合并数据
            all_data.extend(data)
            total_dialogues += dialogue_count
            
            print(f"   📊 累计对话数: {total_dialogues:,}")
            
            # 清理内存
            del data
            gc.collect()
            
        except Exception as e:
            load_time = time.time() - start_time
            print(f"   ❌ 加载失败 (用时 {load_time:.1f}秒): {e}")
            
            response = input(f"   是否跳过此文件继续? (y/n): ")
            if response.lower() != 'y':
                return False
            continue
    
    # 保存结果
    print(f"\n💾 保存合并结果...")
    save_start = time.time()
    torch.save(all_data, output_file)
    save_time = time.time() - save_start
    
    print(f"✅ 保存成功!")
    print(f"⏱️  保存时间: {save_time:.1f}秒")
    
    # 验证
    file_size = os.path.getsize(output_file) / (1024**3)
    print(f"📊 输出文件大小: {file_size:.2f} GB")
    print(f"📊 总对话数: {total_dialogues:,}")
    
    return True

def verify_merged_files():
    """验证合并后的文件"""
    print("🔍 验证合并后的文件...")
    
    datasets = ["train", "valid", "test"]
    
    for dataset in datasets:
        output_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{dataset}.pt")
        
        if os.path.exists(output_file):
            try:
                print(f"\n📄 验证 {dataset}.pt...")
                
                # 检查文件大小
                file_size = os.path.getsize(output_file) / (1024**3)
                print(f"   📊 文件大小: {file_size:.2f} GB")
                
                # 加载并检查数据
                data = torch.load(output_file, weights_only=True)
                print(f"   📊 对话数量: {len(data):,}")
                
                # 检查数据结构
                if data:
                    sample = data[0]
                    if isinstance(sample, tuple) and len(sample) == 2:
                        x_ref, steps = sample
                        print(f"   ✅ 数据结构正确")
                        print(f"   📊 样本: x_ref={x_ref.shape}, steps={len(steps)}")
                    else:
                        print(f"   ⚠️  数据结构异常: {type(sample)}")
                
                del data
                gc.collect()
                
                print(f"   ✅ {dataset}.pt 验证通过")
                
            except Exception as e:
                print(f"   ❌ {dataset}.pt 验证失败: {e}")
        else:
            print(f"   ❌ {dataset}.pt 不存在")

def cleanup_chunk_files(data_type, confirm=True):
    """清理chunk文件（在合并完成后）"""
    chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
    if not os.path.exists(chunk_dir):
        return
    
    chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".pt")]
    
    if not chunk_files:
        print(f"📁 {data_type}: 无chunk文件需要清理")
        return
    
    print(f"📁 {data_type}: 找到 {len(chunk_files)} 个chunk文件")
    
    if confirm:
        response = input(f"是否删除 {data_type} 的chunk文件释放空间? (y/n): ")
        if response.lower() != 'y':
            return
    
    # 删除chunk文件
    deleted_size = 0
    for chunk_file in chunk_files:
        chunk_path = os.path.join(chunk_dir, chunk_file)
        deleted_size += os.path.getsize(chunk_path)
        os.remove(chunk_path)
    
    print(f"✅ 已删除 {len(chunk_files)} 个chunk文件")
    print(f"💾 释放空间: {deleted_size / (1024**3):.2f} GB")

def main():
    """主函数 - 合并工具的命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据合并工具")
    parser.add_argument("--method", choices=["simple", "optimized", "large"], 
                       default="optimized", help="合并方法")
    parser.add_argument("--dataset", choices=["train", "valid", "test", "all"],
                       default="all", help="要合并的数据集")
    parser.add_argument("--workers", type=int, default=2, help="工作线程数")
    parser.add_argument("--verify", action="store_true", help="验证合并结果")
    parser.add_argument("--cleanup", action="store_true", help="清理chunk文件")
    
    args = parser.parse_args()
    
    print("🔧 数据合并工具")
    print("=" * 40)
    
    # 选择数据集
    if args.dataset == "all":
        datasets = ["valid", "test", "train"]
    else:
        datasets = [args.dataset]
    
    # 选择合并方法
    merge_func = {
        "simple": merge_chunks_simple,
        "optimized": lambda dt: merge_chunks_optimized(dt, args.workers),
        "large": merge_chunks_large_files
    }[args.method]
    
    print(f"🔧 使用方法: {args.method}")
    print(f"📊 数据集: {datasets}")
    
    # 执行合并
    success_count = 0
    for dataset in datasets:
        print(f"\n{'='*20} {dataset.upper()} {'='*20}")
        
        if merge_func(dataset):
            success_count += 1
            print(f"✅ {dataset} 合并成功")
            
            # 清理chunk文件
            if args.cleanup:
                cleanup_chunk_files(dataset, confirm=False)
        else:
            print(f"❌ {dataset} 合并失败")
    
    # 验证结果
    if args.verify:
        verify_merged_files()
    
    print(f"\n🎯 合并完成: {success_count}/{len(datasets)} 个数据集")

if __name__ == "__main__":
    main()
