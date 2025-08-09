#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速性能测试 - 直接测试核心函数
"""

import time
import json
import sys
import os
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.dataset import Vocabulary
from src.data_processing.prepare_binary_data import process_dialogue_to_tensors

def test_core_performance():
    """测试核心处理性能"""
    print("测试核心处理性能...")
    
    # 创建词汇表
    vocab = Vocabulary("test")
    vocab.addSentence("你好 世界 测试 数据")
    vocab.addSentence("很好 再见 谢谢")
    
    # 创建测试对话
    test_dialogues = []
    for i in range(1000):
        dialogue = [
            {"text": f"你好世界 {i}"},
            {"text": f"你好吗 {i}"},
            {"text": f"我很好 {i}"},
            {"text": f"再见 {i}"}
        ]
        test_dialogues.append(dialogue)
    
    print(f"测试数据: {len(test_dialogues)} 个对话")
    print(f"词汇表大小: {vocab.num_words}")
    
    # 测试处理速度
    start_time = time.time()
    
    processed_count = 0
    for dialogue in test_dialogues:
        result = process_dialogue_to_tensors(dialogue, vocab)
        if result:
            processed_count += 1
    
    end_time = time.time()
    duration = end_time - start_time
    speed = len(test_dialogues) / duration
    
    print(f"处理结果: {processed_count}/{len(test_dialogues)} 个对话")
    print(f"处理时间: {duration:.2f}s")
    print(f"处理速度: {speed:.1f} it/s")
    
    return speed

def analyze_bottlenecks():
    """分析性能瓶颈"""
    print(f"\n性能瓶颈分析:")
    
    vocab = Vocabulary("test")
    vocab.addSentence("测试 数据")
    
    # 测试JSON解析
    test_dialogue = [{"text": "你好世界"}, {"text": "你好吗"}]
    test_json = json.dumps(test_dialogue, ensure_ascii=False)
    
    start_time = time.time()
    for _ in range(10000):
        json.loads(test_json)
    json_time = time.time() - start_time
    print(f"   JSON解析: {json_time*100:.2f}ms/1000次")
    
    # 测试张量创建
    import torch
    start_time = time.time()
    for _ in range(10000):
        torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    tensor_time = time.time() - start_time
    print(f"   张量创建: {tensor_time*100:.2f}ms/1000次")
    
    # 测试词汇表查找
    start_time = time.time()
    for _ in range(10000):
        vocab.word2index.get("测试", config.UNK_token)
    lookup_time = time.time() - start_time
    print(f"   词汇查找: {lookup_time*100:.2f}ms/1000次")

def estimate_multiprocessing_benefit():
    """估算多进程收益"""
    print(f"\n多进程收益估算:")
    
    # 基准性能
    base_speed = test_core_performance()
    
    print(f"\n理论分析:")
    print(f"   单线程基准速度: {base_speed:.1f} it/s")
    
    # 估算不同worker数的理论速度
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    print(f"   可用CPU核心: {cpu_count}")
    
    # 考虑开销的理论速度
    overheads = {
        1: 1.0,    # 无多进程开销
        2: 0.85,   # 15% 开销
        4: 0.75,   # 25% 开销  
        8: 0.65,   # 35% 开销
    }
    
    print(f"\n   理论多进程速度 (考虑开销):")
    for workers in [1, 2, 4, 8]:
        if workers <= cpu_count:
            theoretical_speed = base_speed * workers * overheads.get(workers, 0.5)
            improvement = theoretical_speed / base_speed
            print(f"     {workers} workers: {theoretical_speed:.1f} it/s ({improvement:.1f}x)")

def check_io_bottleneck():
    """检查I/O瓶颈"""
    print(f"\nI/O瓶颈检查:")
    
    import torch
    import tempfile
    
    # 创建测试数据
    test_data = []
    for _ in range(100):  # 忽略未使用的循环变量
        x_ref = torch.randint(0, 1000, (10,), dtype=torch.long)
        steps = [(torch.randint(0, 1000, (5,), dtype=torch.long), 
                 torch.randint(0, 1000, (8,), dtype=torch.long),
                 torch.tensor([1.0], dtype=torch.float)) for _ in range(10)]
        test_data.append((x_ref, steps))
    
    # 测试保存速度
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_file = f.name
    
    start_time = time.time()
    torch.save(test_data, temp_file)
    save_time = time.time() - start_time
    
    # 测试加载速度
    start_time = time.time()
    _ = torch.load(temp_file, weights_only=True)  # 忽略加载的数据
    load_time = time.time() - start_time
    
    # 清理
    os.unlink(temp_file)
    
    file_size = len(str(test_data)) / 1024 / 1024  # 估算大小
    
    print(f"   保存速度: {save_time*1000:.2f}ms ({file_size:.2f}MB)")
    print(f"   加载速度: {load_time*1000:.2f}ms")
    
    if save_time > 0.1:  # 如果保存时间超过100ms
        print(f"   I/O可能成为瓶颈")
    else:
        print(f"   I/O速度良好")

def main():
    print("快速性能测试...")
    
    try:
        # 核心性能测试
        base_speed = test_core_performance()
        
        # 瓶颈分析
        analyze_bottlenecks()
        
        # 多进程收益估算
        estimate_multiprocessing_benefit()
        
        # I/O瓶颈检查
        check_io_bottleneck()
        
        print(f"\n结论和建议:")
        
        if base_speed > 1000:
            print(f"核心处理速度良好 ({base_speed:.1f} it/s)")
            print(f"多进程可能有效，建议使用2-4个worker")
        elif base_speed > 100:
            print(f"核心处理速度中等 ({base_speed:.1f} it/s)")
            print(f"多进程收益有限，建议使用1-2个worker")
        else:
            print(f"核心处理速度较慢 ({base_speed:.1f} it/s)")
            print(f"建议优化核心算法，多进程收益不大")
        
        print(f"\n优化建议:")
        print(f"1. 如果速度 > 100 it/s，可以尝试多进程")
        print(f"2. 如果速度 < 100 it/s，优先优化单线程性能")
        print(f"3. 监控内存使用，避免内存不足")
        print(f"4. 考虑使用SSD提高I/O性能")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
