#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计已处理的对话数量
"""

import os
import torch
import sys
import time
sys.path.append('.')
import config

def count_processed_dialogues():
    """统计已处理的对话数量"""
    print("📊 统计已处理的对话数量...")
    
    data_types = ["train", "valid", "test"]
    total_processed = 0
    
    for data_type in data_types:
        chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
        if not os.path.exists(chunk_dir):
            print(f"📁 {data_type}: 目录不存在")
            continue
            
        # 查找所有chunk文件
        chunk_files = []
        for file in os.listdir(chunk_dir):
            if file.startswith("chunk_") and file.endswith(".pt"):
                try:
                    chunk_num = int(file.split('_')[1].split('.')[0])
                    chunk_files.append((chunk_num, file))
                except:
                    pass
        
        if not chunk_files:
            print(f"📁 {data_type}: 无chunk文件")
            continue
            
        # 按编号排序
        chunk_files.sort(key=lambda x: x[0])
        
        print(f"\n📁 {data_type} 数据集:")
        print(f"   📦 找到 {len(chunk_files)} 个chunk文件")
        print(f"   🔢 编号范围: {chunk_files[0][0]} - {chunk_files[-1][0]}")
        
        # 统计对话数量
        type_total = 0
        sample_chunks = []
        
        # 采样统计（前5个、中间5个、后5个）
        if len(chunk_files) <= 15:
            sample_chunks = chunk_files
        else:
            sample_chunks.extend(chunk_files[:5])  # 前5个
            mid_start = len(chunk_files) // 2 - 2
            sample_chunks.extend(chunk_files[mid_start:mid_start+5])  # 中间5个
            sample_chunks.extend(chunk_files[-5:])  # 后5个
        
        print(f"   🔍 采样统计 {len(sample_chunks)} 个chunk文件...")
        
        sample_total = 0
        sample_count = 0
        
        for chunk_num, chunk_file in sample_chunks:
            chunk_path = os.path.join(chunk_dir, chunk_file)
            try:
                data = torch.load(chunk_path, weights_only=True)
                dialogues_count = len(data)
                sample_total += dialogues_count
                sample_count += 1
                
                if sample_count <= 3:  # 显示前3个的详细信息
                    print(f"      chunk_{chunk_num}: {dialogues_count} 个对话")
                    
            except Exception as e:
                print(f"      ❌ chunk_{chunk_num} 加载失败: {e}")
        
        if sample_count > 0:
            # 计算平均值并估算总数
            avg_per_chunk = sample_total / sample_count
            estimated_total = int(avg_per_chunk * len(chunk_files))
            
            print(f"   📈 采样统计:")
            print(f"      平均每chunk: {avg_per_chunk:.1f} 个对话")
            print(f"      估算总数: {estimated_total:,} 个对话")
            
            type_total = estimated_total
        
        total_processed += type_total
        print(f"   ✅ {data_type} 估算总计: {type_total:,} 个对话")
    
    print(f"\n🎯 总计已处理: {total_processed:,} 个对话")
    return total_processed

def estimate_progress():
    """估算处理进度"""
    print(f"\n📊 处理进度估算...")
    
    # 读取原始数据文件大小来估算总量
    total_lines = 0
    processed_lines = 0
    
    data_info = {}
    
    for data_type in ["train", "valid", "test"]:
        data_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{data_type}.jsonl")
        if os.path.exists(data_file):
            # 估算总行数（通过文件大小）
            file_size = os.path.getsize(data_file)
            
            # 采样估算每行大小
            with open(data_file, 'r', encoding='utf-8') as f:
                sample_lines = []
                for i, line in enumerate(f):
                    if i >= 100:  # 采样前100行
                        break
                    sample_lines.append(len(line.encode('utf-8')))
            
            if sample_lines:
                avg_line_size = sum(sample_lines) / len(sample_lines)
                estimated_lines = int(file_size / avg_line_size)
                total_lines += estimated_lines
                
                data_info[data_type] = {
                    'file_size_mb': file_size / (1024 * 1024),
                    'estimated_lines': estimated_lines,
                    'avg_line_size': avg_line_size
                }
                
                print(f"   📄 {data_type}.jsonl:")
                print(f"      文件大小: {file_size / (1024 * 1024):.1f} MB")
                print(f"      估算行数: {estimated_lines:,}")
    
    # 检查已处理的chunk数量
    train_chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, "train")
    if os.path.exists(train_chunk_dir):
        chunk_files = [f for f in os.listdir(train_chunk_dir) if f.startswith("chunk_") and f.endswith(".pt")]
        if chunk_files:
            max_chunk = max([int(f.split('_')[1].split('.')[0]) for f in chunk_files])
            print(f"\n   📦 train已完成chunk: 0 - {max_chunk} ({max_chunk + 1} 个)")
            
            # 基于chunk数量估算已处理行数
            if 'train' in data_info:
                total_train_lines = data_info['train']['estimated_lines']
                processed_ratio = (max_chunk + 1) / (total_train_lines / 10000)  # 假设每chunk约10000行
                estimated_processed = int(total_train_lines * min(processed_ratio, 1.0))
                
                print(f"   📈 估算已处理: {estimated_processed:,} 行")
                print(f"   📊 处理进度: {estimated_processed / total_train_lines * 100:.1f}%")
                
                processed_lines = estimated_processed
    
    return total_lines, processed_lines

def check_current_processing_status():
    """检查当前处理状态"""
    print(f"\n🔄 当前处理状态检查...")
    
    # 检查是否有进程在运行
    try:
        import psutil
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = proc.info['cmdline']
                    if cmdline and any('prepare_binary_data' in str(cmd) for cmd in cmdline):
                        python_processes.append(proc.info)
            except:
                pass
        
        if python_processes:
            print(f"   🔄 发现 {len(python_processes)} 个数据处理进程正在运行")
            for proc in python_processes:
                print(f"      PID: {proc['pid']}")
        else:
            print(f"   ⏸️  当前无数据处理进程运行")
            
    except ImportError:
        print(f"   ⚠️  psutil未安装，无法检查进程状态")
    
    # 检查最近修改的文件
    train_chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, "train")
    if os.path.exists(train_chunk_dir):
        chunk_files = [f for f in os.listdir(train_chunk_dir) if f.startswith("chunk_") and f.endswith(".pt")]
        if chunk_files:
            # 找到最新的文件
            latest_file = None
            latest_time = 0
            
            for chunk_file in chunk_files:
                file_path = os.path.join(train_chunk_dir, chunk_file)
                mtime = os.path.getmtime(file_path)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_file = chunk_file
            
            if latest_file:
                import datetime
                latest_time_str = datetime.datetime.fromtimestamp(latest_time).strftime('%Y-%m-%d %H:%M:%S')
                print(f"   📅 最新chunk文件: {latest_file}")
                print(f"   ⏰ 最后修改时间: {latest_time_str}")

def main():
    print("📊 处理进度统计工具")
    print("=" * 50)
    
    # 1. 统计已处理的对话数量
    processed_dialogues = count_processed_dialogues()
    
    # 2. 估算处理进度
    total_lines, processed_lines = estimate_progress()
    
    # 3. 检查当前状态
    check_current_processing_status()
    
    # 4. 总结
    print(f"\n🎯 处理进度总结:")
    print(f"   📊 已处理对话数: {processed_dialogues:,}")
    if total_lines > 0 and processed_lines > 0:
        progress_percent = processed_lines / total_lines * 100
        print(f"   📈 估算处理进度: {progress_percent:.1f}%")
        print(f"   📄 已处理行数: {processed_lines:,} / {total_lines:,}")
        
        remaining_lines = total_lines - processed_lines
        print(f"   ⏳ 剩余行数: {remaining_lines:,}")
        
        # 估算剩余时间（基于之前的处理速度）
        if processed_lines > 0:
            # 假设平均处理速度为200 it/s
            estimated_speed = 200
            remaining_seconds = remaining_lines / estimated_speed
            remaining_hours = remaining_seconds / 3600
            
            print(f"   ⏱️  估算剩余时间: {remaining_hours:.1f} 小时 (假设200 it/s)")

if __name__ == "__main__":
    main()
