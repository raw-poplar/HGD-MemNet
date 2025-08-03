#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的性能测试
"""

import time
import json
import sys
import os
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.dataset import Vocabulary

def create_test_jsonl(filename, num_lines=1000):
    """创建测试JSONL文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(num_lines):
            dialogue = [
                {"text": f"你好世界 {i}"},
                {"text": f"你好吗 {i}"},
                {"text": f"我很好 {i}"},
                {"text": f"再见 {i}"}
            ]
            f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')

def test_processing_speed():
    """测试处理速度"""
    print("🧪 创建测试数据...")
    
    # 创建测试文件
    test_file = "test_data.jsonl"
    create_test_jsonl(test_file, 500)  # 较小的测试集
    
    # 创建词汇表
    vocab = Vocabulary("test")
    vocab.addSentence("你好 世界 测试")
    vocab.addSentence("很好 再见")
    
    print(f"📊 测试文件: {test_file}")
    print(f"📚 词汇表大小: {vocab.num_words}")
    
    # 测试不同的worker数量
    from src.prepare_binary_data import convert_to_binary
    
    results = {}
    
    for num_workers in [1, 2, 4, 8]:
        print(f"\n🔄 测试 {num_workers} workers...")
        
        # 创建输出目录
        output_dir = f"test_output_{num_workers}"
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            start_time = time.time()
            
            # 运行转换
            convert_to_binary(
                data_type="test_data",
                vocab=vocab,
                input_dir=".",  # 当前目录
                output_dir=output_dir,
                num_workers=num_workers
            )
            
            end_time = time.time()
            duration = end_time - start_time
            speed = 500 / duration  # 500个对话
            
            results[num_workers] = {
                'duration': duration,
                'speed': speed
            }
            
            print(f"   ✅ {num_workers} workers: {duration:.2f}s, {speed:.1f} it/s")
            
        except Exception as e:
            print(f"   ❌ {num_workers} workers 失败: {e}")
            results[num_workers] = {'duration': 0, 'speed': 0}
        
        # 清理输出目录
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    
    # 清理测试文件
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # 显示结果
    print(f"\n📈 性能对比:")
    print(f"{'Workers':<8} {'时间(s)':<10} {'速度(it/s)':<12} {'相对提升':<10}")
    print("-" * 45)
    
    baseline_speed = results.get(1, {}).get('speed', 1)
    
    for workers in [1, 2, 4, 8]:
        if workers in results:
            duration = results[workers]['duration']
            speed = results[workers]['speed']
            improvement = speed / baseline_speed if baseline_speed > 0 else 0
            
            print(f"{workers:<8} {duration:<10.2f} {speed:<12.1f} {improvement:<10.1f}x")
    
    return results

def analyze_optimization():
    """分析优化效果"""
    print(f"\n🔍 多线程优化分析:")
    
    print(f"📋 优化前的问题:")
    print(f"   1. 频繁的进程间通信 - 每个对话都要传输vocab对象")
    print(f"   2. 序列化开销 - vocab对象重复序列化")
    print(f"   3. 同步瓶颈 - 文件I/O在主进程串行执行")
    print(f"   4. 内存复制 - 结果数据从子进程复制回主进程")
    
    print(f"\n✅ 优化后的改进:")
    print(f"   1. 进程初始化 - vocab只在进程启动时传输一次")
    print(f"   2. 批量处理 - 减少进程间通信次数")
    print(f"   3. 动态批次大小 - 根据worker数量调整")
    print(f"   4. 异步处理 - 使用apply_async减少等待")
    
    print(f"\n💡 预期效果:")
    print(f"   - 1 worker: 与单线程相当")
    print(f"   - 2-4 workers: 应该有明显提升")
    print(f"   - 8+ workers: 可能受I/O限制")

if __name__ == "__main__":
    print("🚀 简单性能测试...")
    
    try:
        # 分析优化
        analyze_optimization()
        
        # 性能测试
        results = test_processing_speed()
        
        # 结论
        print(f"\n🎯 结论:")
        
        if results:
            speeds = [results[w]['speed'] for w in [1, 2, 4, 8] if w in results and results[w]['speed'] > 0]
            
            if len(speeds) >= 2:
                best_speed = max(speeds)
                single_speed = results.get(1, {}).get('speed', 0)
                
                if best_speed > single_speed * 1.5:
                    print(f"✅ 多线程优化有效！最佳速度提升 {best_speed/single_speed:.1f}x")
                    
                    # 找到最佳worker数
                    best_workers = None
                    for w in [1, 2, 4, 8]:
                        if w in results and results[w]['speed'] == best_speed:
                            best_workers = w
                            break
                    
                    if best_workers:
                        print(f"💡 推荐使用 {best_workers} 个worker")
                else:
                    print(f"⚠️  多线程提升有限，可能受I/O限制")
            else:
                print(f"❌ 测试数据不足")
        else:
            print(f"❌ 测试失败")
            
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
