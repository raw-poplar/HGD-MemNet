#!/usr/bin/env python3
"""
调试合并工具 - 用于诊断合并过程中的问题
"""

import os
import sys
import time
import torch
import gc
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config

def get_chunk_files(data_type):
    """获取chunk文件列表"""
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

def test_single_chunk_load(data_type, chunk_index=0):
    """测试加载单个chunk文件"""
    print(f"🧪 测试加载单个chunk文件...")
    
    chunk_files = get_chunk_files(data_type)
    if not chunk_files:
        print(f"❌ 未找到chunk文件")
        return False
    
    if chunk_index >= len(chunk_files):
        chunk_index = 0
    
    chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
    chunk_num, chunk_file = chunk_files[chunk_index]
    chunk_path = os.path.join(chunk_dir, chunk_file)
    
    print(f"📄 测试文件: {chunk_file}")
    print(f"📊 文件大小: {os.path.getsize(chunk_path) / (1024*1024):.1f} MB")
    
    try:
        print(f"🔄 开始加载...")
        start_time = time.time()
        data = torch.load(chunk_path, weights_only=True)
        load_time = time.time() - start_time
        
        print(f"✅ 加载成功!")
        print(f"⏱️  加载时间: {load_time:.1f}秒")
        print(f"📊 数据量: {len(data):,} 个对话")
        
        # 检查数据结构
        if data:
            sample = data[0]
            print(f"📊 数据结构: {type(sample)}")
            if isinstance(sample, tuple):
                print(f"📊 元组长度: {len(sample)}")
        
        del data
        gc.collect()
        return True
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False

def test_multiple_chunks_load(data_type, num_chunks=5):
    """测试加载多个chunk文件"""
    print(f"🧪 测试加载多个chunk文件 (前{num_chunks}个)...")
    
    chunk_files = get_chunk_files(data_type)
    if not chunk_files:
        print(f"❌ 未找到chunk文件")
        return False
    
    chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
    test_chunks = chunk_files[:min(num_chunks, len(chunk_files))]
    
    total_dialogues = 0
    total_time = 0
    
    for i, (chunk_num, chunk_file) in enumerate(test_chunks):
        chunk_path = os.path.join(chunk_dir, chunk_file)
        
        print(f"\n📦 测试 {i+1}/{len(test_chunks)}: {chunk_file}")
        print(f"📊 文件大小: {os.path.getsize(chunk_path) / (1024*1024):.1f} MB")
        
        try:
            start_time = time.time()
            data = torch.load(chunk_path, weights_only=True)
            load_time = time.time() - start_time
            
            total_dialogues += len(data)
            total_time += load_time
            
            print(f"✅ 加载成功! 时间: {load_time:.1f}秒, 数据: {len(data):,}")
            
            del data
            gc.collect()
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False
    
    print(f"\n📊 总结:")
    print(f"✅ 成功加载 {len(test_chunks)} 个文件")
    print(f"📊 总对话数: {total_dialogues:,}")
    print(f"⏱️  总时间: {total_time:.1f}秒")
    print(f"📊 平均速度: {total_dialogues/total_time:.0f} 对话/秒")
    
    return True

def diagnose_system_resources():
    """诊断系统资源"""
    print(f"🔍 系统资源诊断...")
    
    try:
        import psutil
        
        # 内存信息
        memory = psutil.virtual_memory()
        print(f"💾 内存: {memory.total / (1024**3):.1f} GB 总量")
        print(f"💾 可用: {memory.available / (1024**3):.1f} GB ({memory.percent:.1f}% 已使用)")
        
        # CPU信息
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"🖥️  CPU使用率: {cpu_percent:.1f}%")
        print(f"🖥️  CPU核心数: {psutil.cpu_count()}")
        
        # 磁盘信息
        disk = psutil.disk_usage('F:')
        print(f"💿 F盘: {disk.total / (1024**3):.1f} GB 总量")
        print(f"💿 可用: {disk.free / (1024**3):.1f} GB ({(disk.used/disk.total)*100:.1f}% 已使用)")
        
    except ImportError:
        print("⚠️  psutil未安装，无法获取详细系统信息")
        print("💡 可以运行: pip install psutil")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="调试合并工具")
    parser.add_argument("--dataset", choices=["train", "valid", "test"], 
                       default="train", help="要测试的数据集")
    parser.add_argument("--test-single", action="store_true", help="测试单个文件加载")
    parser.add_argument("--test-multiple", type=int, default=5, help="测试多个文件加载")
    parser.add_argument("--chunk-index", type=int, default=0, help="测试的chunk索引")
    parser.add_argument("--diagnose", action="store_true", help="诊断系统资源")
    
    args = parser.parse_args()
    
    print("🔧 调试合并工具")
    print("=" * 40)
    
    if args.diagnose:
        diagnose_system_resources()
        print()
    
    if args.test_single:
        test_single_chunk_load(args.dataset, args.chunk_index)
        print()
    
    if args.test_multiple > 0:
        test_multiple_chunks_load(args.dataset, args.test_multiple)

if __name__ == "__main__":
    main()
