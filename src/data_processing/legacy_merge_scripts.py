#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遗留合并脚本集合

包含之前创建的各种合并脚本，保留作为备用方案：
1. final_merge - 大文件优化版
2. optimized_merge - 流式处理版
3. simple_merge - 简单版本
"""

import os
import torch
import sys
import gc
import time
import threading
import queue
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

def merge_large_chunks_legacy(data_type, timeout_seconds=300):
    """遗留版本：大文件合并"""
    print(f"🔄 合并 {data_type} (大文件优化版)...")
    
    chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
    output_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{data_type}.pt")
    
    if not os.path.exists(chunk_dir):
        print(f"❌ 目录不存在: {chunk_dir}")
        return False
    
    # 获取chunk文件
    chunk_files = []
    for file in os.listdir(chunk_dir):
        if file.startswith("chunk_") and file.endswith(".pt"):
            try:
                num = int(file.split('_')[1].split('.')[0])
                size = os.path.getsize(os.path.join(chunk_dir, file))
                chunk_files.append((num, file, size))
            except:
                pass
    
    chunk_files.sort()
    print(f"📦 找到 {len(chunk_files)} 个chunk文件")
    
    # 显示文件信息
    total_size = 0
    for num, filename, size in chunk_files:
        size_mb = size / (1024*1024)
        total_size += size
        print(f"   📄 {filename}: {size_mb:.1f} MB")
    
    print(f"📊 总大小: {total_size / (1024*1024*1024):.2f} GB")
    
    # 检查输出文件
    if os.path.exists(output_file):
        response = input(f"⚠️  {output_file} 已存在，是否覆盖? (y/n): ")
        if response.lower() != 'y':
            return True
    
    # 开始合并
    all_data = []
    total_dialogues = 0
    
    for i, (num, filename, size) in enumerate(chunk_files):
        chunk_path = os.path.join(chunk_dir, filename)
        size_mb = size / (1024*1024)
        
        print(f"\n📦 处理 {filename} ({i+1}/{len(chunk_files)})...")
        print(f"   📊 文件大小: {size_mb:.1f} MB")
        
        # 设置超时加载
        start_time = time.time()
        
        try:
            print(f"   🔄 开始加载... (超时: {timeout_seconds}秒)")
            
            # 尝试加载数据
            data = torch.load(chunk_path, weights_only=True)
            
            load_time = time.time() - start_time
            dialogue_count = len(data)
            
            print(f"   ✅ 加载成功!")
            print(f"   ⏱️  加载时间: {load_time:.1f}秒")
            print(f"   📊 对话数: {dialogue_count:,}")
            
            # 添加到总数据
            print(f"   🔄 合并数据...")
            all_data.extend(data)
            total_dialogues += dialogue_count
            
            print(f"   📊 累计对话数: {total_dialogues:,}")
            
            # 清理内存
            del data
            gc.collect()
            
            print(f"   🧹 内存清理完成")
            
        except Exception as e:
            load_time = time.time() - start_time
            print(f"   ❌ 加载失败 (用时 {load_time:.1f}秒): {e}")
            
            # 询问是否跳过
            response = input(f"   是否跳过此文件继续? (y/n): ")
            if response.lower() != 'y':
                return False
            continue
    
    # 保存结果
    print(f"\n💾 保存合并结果...")
    print(f"📊 总对话数: {total_dialogues:,}")
    
    try:
        save_start = time.time()
        torch.save(all_data, output_file)
        save_time = time.time() - save_start
        
        print(f"✅ 保存成功!")
        print(f"⏱️  保存时间: {save_time:.1f}秒")
        
        # 验证文件
        file_size = os.path.getsize(output_file) / (1024**3)
        print(f"📊 输出文件大小: {file_size:.2f} GB")
        
        # 快速验证
        print(f"🔍 验证文件...")
        verify_data = torch.load(output_file, weights_only=True)
        print(f"✅ 验证成功: {len(verify_data):,} 个对话")
        
        del all_data, verify_data
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False

class StreamMergerLegacy:
    """遗留版本：流式合并器"""
    def __init__(self, data_type, max_workers=4):
        self.data_type = data_type
        self.max_workers = max_workers
        self.chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
        self.output_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{data_type}.pt")
        self.resume_file = os.path.join(config.LCCC_PROCESSED_PATH, f"merge_resume_{data_type}.json")
        self.temp_file = self.output_file + ".tmp"
        
    def get_chunk_files(self):
        """获取所有chunk文件"""
        if not os.path.exists(self.chunk_dir):
            return []
        
        chunk_files = []
        for file in os.listdir(self.chunk_dir):
            if file.startswith("chunk_") and file.endswith(".pt"):
                try:
                    chunk_num = int(file.split('_')[1].split('.')[0])
                    chunk_files.append((chunk_num, file))
                except:
                    pass
        
        return sorted(chunk_files, key=lambda x: x[0])
    
    def load_resume_state(self):
        """加载断点续传状态"""
        if os.path.exists(self.resume_file):
            try:
                with open(self.resume_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"processed_chunks": 0, "total_dialogues": 0}
    
    def save_resume_state(self, processed_chunks, total_dialogues):
        """保存断点续传状态"""
        state = {
            "processed_chunks": processed_chunks,
            "total_dialogues": total_dialogues,
            "timestamp": time.time()
        }
        with open(self.resume_file, 'w') as f:
            json.dump(state, f)
    
    def load_chunk_worker(self, chunk_info):
        """工作线程：加载单个chunk"""
        chunk_num, chunk_file = chunk_info
        chunk_path = os.path.join(self.chunk_dir, chunk_file)
        
        try:
            data = torch.load(chunk_path, weights_only=True)
            return chunk_num, data, len(data)
        except Exception as e:
            print(f"❌ 加载 {chunk_file} 失败: {e}")
            return chunk_num, None, 0
    
    def append_to_file(self, new_data):
        """追加数据到文件"""
        try:
            # 加载现有数据
            if os.path.exists(self.temp_file):
                existing_data = torch.load(self.temp_file, weights_only=True)
            else:
                existing_data = []
            
            # 追加新数据
            existing_data.extend(new_data)
            
            # 保存回文件
            torch.save(existing_data, self.temp_file)
            
            # 清理内存
            del existing_data
            
        except Exception as e:
            print(f"❌ 追加数据失败: {e}")
            raise
    
    def stream_merge(self):
        """流式合并"""
        print(f"\n🔄 开始流式合并 {self.data_type}...")
        
        # 获取chunk文件列表
        chunk_files = self.get_chunk_files()
        if not chunk_files:
            print(f"❌ 未找到chunk文件")
            return False
        
        print(f"📊 找到 {len(chunk_files)} 个chunk文件")
        
        # 加载断点续传状态
        resume_state = self.load_resume_state()
        start_chunk = resume_state.get("processed_chunks", 0)
        total_dialogues = resume_state.get("total_dialogues", 0)
        
        if start_chunk > 0:
            print(f"🔄 从第 {start_chunk} 个chunk继续 (已处理 {total_dialogues:,} 个对话)")
        
        # 检查是否已存在输出文件
        if os.path.exists(self.output_file) and start_chunk == 0:
            response = input(f"⚠️  {self.data_type}.pt 已存在，是否覆盖? (y/n): ")
            if response.lower() != 'y':
                return False
        
        # 初始化输出文件
        if start_chunk == 0:
            # 创建新文件
            torch.save([], self.temp_file)
            print(f"📄 创建临时文件: {self.temp_file}")
        
        # 流式处理
        processed_chunks = start_chunk
        chunk_queue = queue.Queue(maxsize=self.max_workers * 2)
        
        # 启动加载线程
        def chunk_loader():
            for i in range(start_chunk, len(chunk_files)):
                chunk_queue.put(chunk_files[i])
            chunk_queue.put(None)  # 结束标记
        
        loader_thread = threading.Thread(target=chunk_loader)
        loader_thread.start()
        
        # 处理chunk
        with tqdm(total=len(chunk_files) - start_chunk, desc=f"合并{self.data_type}") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                
                while True:
                    chunk_info = chunk_queue.get()
                    if chunk_info is None:
                        break
                    
                    # 提交加载任务
                    future = executor.submit(self.load_chunk_worker, chunk_info)
                    
                    try:
                        chunk_num, chunk_data, dialogue_count = future.result(timeout=60)
                        
                        if chunk_data is not None:
                            # 追加到文件
                            self.append_to_file(chunk_data)
                            total_dialogues += dialogue_count
                            processed_chunks += 1
                            
                            # 更新进度
                            pbar.update(1)
                            pbar.set_postfix({
                                'chunk': chunk_num,
                                'dialogues': f"{total_dialogues:,}"
                            })
                            
                            # 定期保存断点
                            if processed_chunks % 10 == 0:
                                self.save_resume_state(processed_chunks, total_dialogues)
                        
                        # 清理内存
                        del chunk_data
                        
                    except Exception as e:
                        print(f"❌ 处理chunk失败: {e}")
                        continue
        
        loader_thread.join()
        
        # 完成合并
        if os.path.exists(self.temp_file):
            # 重命名临时文件
            if os.path.exists(self.output_file):
                os.remove(self.output_file)
            os.rename(self.temp_file, self.output_file)
            
            # 清理断点文件
            if os.path.exists(self.resume_file):
                os.remove(self.resume_file)
            
            print(f"✅ {self.data_type}.pt 合并完成")
            print(f"📊 总对话数: {total_dialogues:,}")
            
            # 验证文件
            self.verify_merged_file()
            return True
        
        return False
    
    def verify_merged_file(self):
        """验证合并后的文件"""
        try:
            data = torch.load(self.output_file, weights_only=True)
            file_size = os.path.getsize(self.output_file) / (1024**3)
            print(f"✅ 验证成功: {len(data):,} 个对话, {file_size:.2f} GB")
            del data
        except Exception as e:
            print(f"❌ 验证失败: {e}")

def simple_merge_legacy(data_type):
    """遗留版本：简单合并"""
    print(f"🔄 简单合并 {data_type}...")
    
    chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
    output_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{data_type}.pt")
    
    print(f"📁 chunk目录: {chunk_dir}")
    print(f"📄 输出文件: {output_file}")
    
    if not os.path.exists(chunk_dir):
        print(f"❌ 目录不存在")
        return False
    
    # 获取chunk文件
    chunk_files = []
    for file in os.listdir(chunk_dir):
        if file.startswith("chunk_") and file.endswith(".pt"):
            try:
                num = int(file.split('_')[1].split('.')[0])
                chunk_files.append((num, file))
            except:
                pass
    
    chunk_files.sort()
    print(f"📦 找到 {len(chunk_files)} 个chunk文件")
    
    if not chunk_files:
        print(f"❌ 无chunk文件")
        return False
    
    # 检查输出文件
    if os.path.exists(output_file):
        print(f"⚠️  输出文件已存在")
        return True
    
    # 开始合并
    all_data = []
    
    for i, (num, filename) in enumerate(chunk_files):
        chunk_path = os.path.join(chunk_dir, filename)
        print(f"📦 处理 {filename} ({i+1}/{len(chunk_files)})...")
        
        try:
            # 检查文件大小
            size_mb = os.path.getsize(chunk_path) / (1024*1024)
            print(f"   大小: {size_mb:.1f} MB")
            
            # 加载数据
            print(f"   加载中...")
            data = torch.load(chunk_path, weights_only=True)
            print(f"   ✅ 加载成功: {len(data)} 个对话")
            
            # 添加到总数据
            all_data.extend(data)
            print(f"   📊 累计: {len(all_data)} 个对话")
            
            # 清理
            del data
            gc.collect()
            
            # 每处理几个文件就强制垃圾回收
            if (i + 1) % 5 == 0:
                print(f"   🧹 强制垃圾回收...")
                gc.collect()
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            return False
    
    # 保存结果
    print(f"💾 保存到 {output_file}...")
    print(f"📊 总对话数: {len(all_data)}")
    
    try:
        torch.save(all_data, output_file)
        print(f"✅ 保存成功")
        
        # 验证
        file_size = os.path.getsize(output_file) / (1024**3)
        print(f"📊 文件大小: {file_size:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False

def main():
    """主函数 - 遗留合并脚本的命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="遗留合并脚本")
    parser.add_argument("--method", choices=["simple", "stream", "large"], 
                       default="large", help="合并方法")
    parser.add_argument("--dataset", choices=["train", "valid", "test"],
                       required=True, help="要合并的数据集")
    parser.add_argument("--workers", type=int, default=2, help="工作线程数")
    
    args = parser.parse_args()
    
    print("🔧 遗留合并脚本")
    print("=" * 40)
    
    if args.method == "simple":
        success = simple_merge_legacy(args.dataset)
    elif args.method == "stream":
        merger = StreamMergerLegacy(args.dataset, args.workers)
        success = merger.stream_merge()
    elif args.method == "large":
        success = merge_large_chunks_legacy(args.dataset)
    
    if success:
        print(f"✅ {args.dataset} 合并成功")
    else:
        print(f"❌ {args.dataset} 合并失败")

if __name__ == "__main__":
    main()
