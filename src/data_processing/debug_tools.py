#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试工具集合

包含各种调试和检查工具：
1. chunk文件加载测试
2. 处理状态分析
3. 数据兼容性检查
4. 性能测试工具
"""

import os
import torch
import json
import sys
import traceback
import time
import gc
from typing import Dict, List, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.dataset import Vocabulary

def test_chunk_loading(data_type: str = None, max_test: int = 3) -> Dict:
    """测试chunk文件加载"""
    print("🔍 测试chunk文件加载...")
    
    if data_type:
        data_types = [data_type]
    else:
        data_types = ["train", "valid", "test"]
    
    results = {}
    
    for dtype in data_types:
        print(f"\n📁 测试 {dtype}:")
        
        chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, dtype)
        if not os.path.exists(chunk_dir):
            print(f"   ❌ 目录不存在: {chunk_dir}")
            results[dtype] = {"error": "directory_not_found"}
            continue
        
        # 查找chunk文件
        chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".pt")]
        chunk_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        print(f"   📦 找到 {len(chunk_files)} 个chunk文件")
        
        if not chunk_files:
            results[dtype] = {"error": "no_chunks"}
            continue
        
        # 测试前几个chunk
        test_files = chunk_files[:max_test]
        test_results = []
        
        for chunk_file in test_files:
            chunk_path = os.path.join(chunk_dir, chunk_file)
            chunk_result = {
                "file": chunk_file,
                "path": chunk_path,
                "size_mb": 0,
                "load_time": 0,
                "dialogue_count": 0,
                "success": False,
                "error": None
            }
            
            try:
                # 文件大小
                chunk_result["size_mb"] = os.path.getsize(chunk_path) / (1024*1024)
                print(f"   🔍 测试 {chunk_file} ({chunk_result['size_mb']:.1f} MB)")
                
                # 加载测试
                start_time = time.time()
                data = torch.load(chunk_path, weights_only=True)
                chunk_result["load_time"] = time.time() - start_time
                chunk_result["dialogue_count"] = len(data)
                chunk_result["success"] = True
                
                print(f"      ✅ 加载成功: {chunk_result['dialogue_count']} 个对话, {chunk_result['load_time']:.2f}秒")
                
                # 检查数据结构
                if data:
                    sample = data[0]
                    if isinstance(sample, tuple) and len(sample) == 2:
                        x_ref, steps = sample
                        print(f"      📊 数据结构: x_ref={x_ref.shape}, steps={len(steps)}")
                        
                        # 检查steps结构
                        if steps:
                            x_t, target, gate = steps[0]
                            x_t_info = f"{x_t.shape}" if x_t is not None else "None"
                            print(f"      📊 步骤结构: x_t={x_t_info}, target={target.shape}, gate={gate.shape}")
                    else:
                        print(f"      ⚠️  数据结构异常: {type(sample)}")
                
                del data
                gc.collect()
                
            except Exception as e:
                chunk_result["error"] = str(e)
                print(f"      ❌ 加载失败: {e}")
                print(f"      📋 详细错误:")
                traceback.print_exc()
            
            test_results.append(chunk_result)
        
        results[dtype] = {
            "total_chunks": len(chunk_files),
            "tested_chunks": len(test_results),
            "test_results": test_results
        }
    
    return results

def analyze_processing_status() -> Dict:
    """分析处理状态"""
    print("📊 分析处理状态...")
    
    status = {}
    
    for data_type in ["train", "valid", "test"]:
        print(f"\n📁 分析 {data_type}:")
        
        type_status = {
            "original_file": None,
            "chunks": {"count": 0, "total_size_gb": 0},
            "partials": {"count": 0, "total_size_gb": 0},
            "merged_file": None,
            "resume_info": None,
            "processing_stage": "unknown"
        }
        
        # 检查原始文件
        original_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{data_type}.jsonl")
        if os.path.exists(original_file):
            size_gb = os.path.getsize(original_file) / (1024**3)
            type_status["original_file"] = {"exists": True, "size_gb": size_gb}
            print(f"   📄 原始文件: {size_gb:.2f} GB")
        else:
            type_status["original_file"] = {"exists": False}
            print(f"   📄 原始文件: 不存在")
        
        # 检查chunk目录
        chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
        if os.path.exists(chunk_dir):
            # chunk文件
            chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".pt")]
            if chunk_files:
                total_size = sum(os.path.getsize(os.path.join(chunk_dir, f)) for f in chunk_files)
                type_status["chunks"] = {
                    "count": len(chunk_files),
                    "total_size_gb": total_size / (1024**3)
                }
                print(f"   📦 chunk文件: {len(chunk_files)} 个, {total_size / (1024**3):.2f} GB")
            
            # partial文件
            partial_files = [f for f in os.listdir(chunk_dir) if f.startswith("partial_") and f.endswith(".pt")]
            if partial_files:
                total_size = sum(os.path.getsize(os.path.join(chunk_dir, f)) for f in partial_files)
                type_status["partials"] = {
                    "count": len(partial_files),
                    "total_size_gb": total_size / (1024**3)
                }
                print(f"   🔄 partial文件: {len(partial_files)} 个, {total_size / (1024**3):.2f} GB")
            
            # resume文件
            resume_file = os.path.join(chunk_dir, f"resume_{data_type}.json")
            if os.path.exists(resume_file):
                try:
                    with open(resume_file, 'r') as f:
                        resume_data = json.load(f)
                    type_status["resume_info"] = resume_data
                    print(f"   📋 resume信息: {resume_data}")
                except:
                    type_status["resume_info"] = {"error": "damaged"}
                    print(f"   📋 resume文件损坏")
        
        # 检查合并文件
        merged_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{data_type}.pt")
        if os.path.exists(merged_file):
            size_gb = os.path.getsize(merged_file) / (1024**3)
            type_status["merged_file"] = {"exists": True, "size_gb": size_gb}
            print(f"   📄 合并文件: {size_gb:.2f} GB")
        else:
            type_status["merged_file"] = {"exists": False}
        
        # 判断处理阶段
        if type_status["merged_file"]["exists"]:
            type_status["processing_stage"] = "completed"
        elif type_status["chunks"]["count"] > 0:
            if type_status["partials"]["count"] > 0:
                type_status["processing_stage"] = "processing"
            else:
                type_status["processing_stage"] = "chunks_ready"
        elif type_status["original_file"]["exists"]:
            type_status["processing_stage"] = "ready_to_process"
        else:
            type_status["processing_stage"] = "no_data"
        
        stage_names = {
            "completed": "✅ 已完成",
            "chunks_ready": "📦 chunk已就绪，待合并",
            "processing": "🔄 正在处理",
            "ready_to_process": "⏳ 待处理",
            "no_data": "❌ 无数据"
        }
        
        print(f"   🎯 处理阶段: {stage_names.get(type_status['processing_stage'], '未知')}")
        
        status[data_type] = type_status
    
    return status

def test_data_compatibility() -> bool:
    """测试数据兼容性"""
    print("🔍 测试数据兼容性...")
    
    try:
        # 创建测试词汇表
        vocab = Vocabulary("test")
        vocab.addSentence("hello world test")
        
        # 创建测试对话
        test_dialogue = [
            {"text": "hello world"},
            {"text": "how are you"},
            {"text": "I am fine"}
        ]
        
        # 测试处理函数
        from .prepare_binary_data import process_dialogue_to_tensors
        result = process_dialogue_to_tensors(test_dialogue, vocab)
        
        if result is None:
            print("❌ 处理函数返回None")
            return False
        
        x_ref, steps_data = result
        print(f"✅ 处理成功: x_ref={x_ref.shape}, steps={len(steps_data)}")
        
        # 检查数据结构
        for i, (x_t, target, gate) in enumerate(steps_data):
            x_t_info = f"{x_t.shape}" if x_t is not None else "None"
            print(f"   步骤{i}: x_t={x_t_info}, target={target.shape}, gate={gate.item()}")
        
        # 测试批处理兼容性
        from src.dataset import binary_collate_fn
        batch = [result, result]
        
        try:
            x_ref_batch, steps_batch = binary_collate_fn(batch)
            print(f"✅ 批处理兼容: x_ref={x_ref_batch.shape}, steps={len(steps_batch)}")
            return True
        except Exception as e:
            print(f"❌ 批处理失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 兼容性测试失败: {e}")
        traceback.print_exc()
        return False

def benchmark_chunk_loading(data_type: str = "valid", num_chunks: int = 5) -> Dict:
    """性能测试：chunk加载速度"""
    print(f"⚡ 性能测试: {data_type} chunk加载速度...")
    
    chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
    if not os.path.exists(chunk_dir):
        print(f"❌ 目录不存在: {chunk_dir}")
        return {"error": "directory_not_found"}
    
    chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".pt")]
    chunk_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if not chunk_files:
        print(f"❌ 无chunk文件")
        return {"error": "no_chunks"}
    
    test_files = chunk_files[:num_chunks]
    results = []
    
    print(f"📊 测试 {len(test_files)} 个chunk文件...")
    
    for chunk_file in test_files:
        chunk_path = os.path.join(chunk_dir, chunk_file)
        
        # 文件信息
        file_size_mb = os.path.getsize(chunk_path) / (1024*1024)
        
        # 加载测试
        start_time = time.time()
        try:
            data = torch.load(chunk_path, weights_only=True)
            load_time = time.time() - start_time
            dialogue_count = len(data)
            
            # 计算速度指标
            mb_per_sec = file_size_mb / load_time
            dialogues_per_sec = dialogue_count / load_time
            
            result = {
                "file": chunk_file,
                "size_mb": file_size_mb,
                "load_time": load_time,
                "dialogue_count": dialogue_count,
                "mb_per_sec": mb_per_sec,
                "dialogues_per_sec": dialogues_per_sec,
                "success": True
            }
            
            print(f"   📄 {chunk_file}: {load_time:.2f}s, {mb_per_sec:.1f} MB/s, {dialogues_per_sec:.0f} 对话/s")
            
            del data
            gc.collect()
            
        except Exception as e:
            result = {
                "file": chunk_file,
                "size_mb": file_size_mb,
                "error": str(e),
                "success": False
            }
            print(f"   ❌ {chunk_file}: 加载失败 - {e}")
        
        results.append(result)
    
    # 统计
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        avg_load_time = sum(r["load_time"] for r in successful_results) / len(successful_results)
        avg_mb_per_sec = sum(r["mb_per_sec"] for r in successful_results) / len(successful_results)
        avg_dialogues_per_sec = sum(r["dialogues_per_sec"] for r in successful_results) / len(successful_results)
        
        print(f"\n📊 平均性能:")
        print(f"   加载时间: {avg_load_time:.2f}s")
        print(f"   加载速度: {avg_mb_per_sec:.1f} MB/s")
        print(f"   处理速度: {avg_dialogues_per_sec:.0f} 对话/s")
        
        return {
            "tested_files": len(test_files),
            "successful": len(successful_results),
            "results": results,
            "averages": {
                "load_time": avg_load_time,
                "mb_per_sec": avg_mb_per_sec,
                "dialogues_per_sec": avg_dialogues_per_sec
            }
        }
    else:
        return {
            "tested_files": len(test_files),
            "successful": 0,
            "results": results,
            "error": "no_successful_loads"
        }

def main():
    """主函数 - 调试工具的命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="调试工具")
    parser.add_argument("--test-loading", action="store_true", help="测试chunk文件加载")
    parser.add_argument("--analyze-status", action="store_true", help="分析处理状态")
    parser.add_argument("--test-compatibility", action="store_true", help="测试数据兼容性")
    parser.add_argument("--benchmark", action="store_true", help="性能测试")
    parser.add_argument("--dataset", choices=["train", "valid", "test"], help="指定数据集")
    parser.add_argument("--max-test", type=int, default=3, help="最大测试文件数")
    
    args = parser.parse_args()
    
    if not any([args.test_loading, args.analyze_status, args.test_compatibility, args.benchmark]):
        # 默认执行所有测试
        args.test_loading = args.analyze_status = args.test_compatibility = True
    
    print("🔧 调试工具")
    print("=" * 40)
    
    if args.test_loading:
        test_chunk_loading(args.dataset, args.max_test)
    
    if args.analyze_status:
        analyze_processing_status()
    
    if args.test_compatibility:
        test_data_compatibility()
    
    if args.benchmark:
        benchmark_chunk_loading(args.dataset or "valid")

if __name__ == "__main__":
    main()
