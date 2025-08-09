#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多进程性能测试脚本
"""

import time
import json
import os
import sys
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.dataset import Vocabulary
from src.data_processing.prepare_binary_data import process_dialogue_to_tensors, init_worker, process_batch_optimized
from multiprocessing import Pool, cpu_count

def create_test_data(num_dialogues=1000):
    """创建测试数据"""
    test_dialogues = []
    for i in range(num_dialogues):
        dialogue = [
            {"text": f"hello world {i}"},
            {"text": f"how are you {i}"},
            {"text": f"I am fine {i}"},
            {"text": f"goodbye {i}"}
        ]
        test_dialogues.append(json.dumps(dialogue))
    return test_dialogues

def test_single_thread(test_data, vocab):
    """测试单线程性能"""
    print("测试单线程性能...")
    start_time = time.time()
    
    results = []
    for line in test_data:
        try:
            dialogue = json.loads(line)
            tensor_data = process_dialogue_to_tensors(dialogue, vocab)
            if tensor_data:
                results.append(tensor_data)
        except:
            pass
    
    end_time = time.time()
    duration = end_time - start_time
    speed = len(test_data) / duration
    
    print(f"   单线程: {len(results)} 个对话, 用时 {duration:.2f}s, 速度 {speed:.1f} it/s")
    return results, speed

def test_new_multithread(test_data, vocab, num_workers=4):
    """测试新的多线程实现"""
    print(f"测试新多线程性能 (workers={num_workers})...")
    
    start_time = time.time()
    
    # 准备词汇表状态
    vocab_state = vocab.__dict__.copy()
    
    # 分批处理
    BATCH_SIZE = max(100, len(test_data) // (num_workers * 4))
    batches = [test_data[i:i+BATCH_SIZE] for i in range(0, len(test_data), BATCH_SIZE)]
    
    with Pool(num_workers, initializer=init_worker, initargs=(vocab_state,)) as pool:
        batch_results = pool.map(process_batch_optimized, batches)
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
    
    end_time = time.time()
    duration = end_time - start_time
    speed = len(test_data) / duration
    
    print(f"   新多线程: {len(results)} 个对话, 用时 {duration:.2f}s, 速度 {speed:.1f} it/s")
    return results, speed

def test_scaling_performance():
    """测试不同worker数量的性能"""
    print("\n测试不同worker数量的性能...")
    
    # 创建测试数据
    test_data = create_test_data(2000)
    
    # 创建词汇表
    vocab = Vocabulary("test")
    for line in test_data[:100]:  # 只用前100个建立词汇表
        try:
            dialogue = json.loads(line)
            for item in dialogue:
                vocab.addSentence(item.get("text", ""))
        except:
            pass
    
    print(f"测试数据: {len(test_data)} 个对话")
    print(f"词汇表大小: {vocab.num_words}")
    print(f"可用CPU: {cpu_count()}")
    
    # 测试单线程
    _, single_speed = test_single_thread(test_data, vocab)
    
    # 测试不同worker数量
    worker_counts = [1, 2, 4, 8]
    new_speeds = []
    
    for workers in worker_counts:
        if workers <= cpu_count():
            print(f"\n--- Workers = {workers} ---")
            
            # 测试新实现
            try:
                _, new_speed = test_new_multithread(test_data, vocab, workers)
                new_speeds.append(new_speed)
            except Exception as e:
                print(f"   新多线程失败: {e}")
                new_speeds.append(0)
        else:
            new_speeds.append(0)
    
    # 性能总结
    print(f"\n性能总结:")
    print(f"{'Workers':<8} {'单线程':<12} {'新多线程':<12} {'提升倍数':<10}")
    print("-" * 50)
    
    for i, workers in enumerate(worker_counts):
        if workers <= cpu_count():
            new_speed = new_speeds[i] if i < len(new_speeds) else 0
            ratio = new_speed / single_speed if single_speed > 0 else 0
            
            print(f"{workers:<8} {single_speed:<12.1f} {new_speed:<12.1f} {ratio:<10.1f}x")
    
    return single_speed, new_speeds

def analyze_bottlenecks():
    """分析性能瓶颈"""
    print(f"\n性能瓶颈分析:")
    
    # 测试序列化开销
    vocab = Vocabulary("test")
    vocab.addSentence("hello world test")
    
    import pickle
    start_time = time.time()
    for _ in range(1000):
        pickle.dumps(vocab)
    serialize_time = time.time() - start_time
    print(f"   词汇表序列化开销: {serialize_time*1000:.2f}ms/1000次")
    
    # 测试JSON解析开销
    test_line = json.dumps([{"text": "hello world"}, {"text": "how are you"}])
    start_time = time.time()
    for _ in range(1000):
        json.loads(test_line)
    json_time = time.time() - start_time
    print(f"   JSON解析开销: {json_time*1000:.2f}ms/1000次")
    
    # 测试张量创建开销
    import torch
    start_time = time.time()
    for _ in range(1000):
        torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    tensor_time = time.time() - start_time
    print(f"   张量创建开销: {tensor_time*1000:.2f}ms/1000次")

if __name__ == "__main__":
    print("多进程性能测试开始...")
    
    try:
        # 性能测试
        single_speed, new_speeds = test_scaling_performance()
        
        # 瓶颈分析
        analyze_bottlenecks()
        
        # 结论
        print(f"\n结论:")
        if new_speeds and max(new_speeds) > 0:
            best_new = max(new_speeds)
            
            if best_new > single_speed * 1.5:
                print(f"新多线程实现显著优于单线程 ({best_new:.1f} vs {single_speed:.1f} it/s)")
            elif best_new > single_speed * 1.2:
                print(f"新多线程实现有效提升性能 ({best_new:.1f} vs {single_speed:.1f} it/s)")
            else:
                print(f"多线程提升有限，建议使用单线程")
                
            # 推荐配置
            best_workers = new_speeds.index(max(new_speeds)) + 1
            if best_workers <= len([1, 2, 4, 8]):
                actual_workers = [1, 2, 4, 8][best_workers - 1]
                print(f"推荐使用 {actual_workers} 个worker进程")
        else:
            print(f"多线程测试失败")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
