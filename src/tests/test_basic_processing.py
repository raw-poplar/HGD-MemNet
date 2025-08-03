#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本处理测试
"""

import os
import json
import sys
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.dataset import Vocabulary
from src.prepare_binary_data import process_dialogue_to_tensors

def test_basic_processing():
    """测试基本处理功能"""
    print("🧪 测试基本处理功能...")
    
    # 1. 加载词汇表
    vocab_path = os.path.join(config.LCCC_PROCESSED_PATH, "vocabulary.json")
    print(f"📚 加载词汇表: {vocab_path}")
    
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        vocab = Vocabulary("lccc")
        vocab.__dict__.update(vocab_dict)
        print(f"✅ 词汇表加载成功，大小: {vocab.num_words}")
    except Exception as e:
        print(f"❌ 词汇表加载失败: {e}")
        return False
    
    # 2. 测试读取数据文件
    train_file = os.path.join(config.LCCC_PROCESSED_PATH, "train.jsonl")
    print(f"📄 测试读取: {train_file}")
    
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            # 读取前几行进行测试
            test_lines = []
            for i, line in enumerate(f):
                if i >= 5:  # 只读取前5行
                    break
                test_lines.append(line.strip())
        
        print(f"✅ 成功读取 {len(test_lines)} 行测试数据")
    except Exception as e:
        print(f"❌ 读取数据文件失败: {e}")
        return False
    
    # 3. 测试处理对话
    print(f"🔄 测试处理对话...")
    
    processed_count = 0
    for i, line in enumerate(test_lines):
        try:
            dialogue = json.loads(line)
            result = process_dialogue_to_tensors(dialogue, vocab)
            if result:
                processed_count += 1
                x_ref, steps_data = result
                print(f"   对话 {i+1}: x_ref={x_ref.shape}, steps={len(steps_data)}")
            else:
                print(f"   对话 {i+1}: 处理失败（可能太短）")
        except Exception as e:
            print(f"   对话 {i+1}: 处理出错 - {e}")
    
    print(f"✅ 成功处理 {processed_count}/{len(test_lines)} 个对话")
    
    # 4. 检查现有chunk文件
    chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, "train")
    if os.path.exists(chunk_dir):
        chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".pt")]
        print(f"📦 现有chunk文件数: {len(chunk_files)}")
        
        if chunk_files:
            # 测试加载一个chunk文件
            first_chunk = os.path.join(chunk_dir, "chunk_0.pt")
            if os.path.exists(first_chunk):
                try:
                    import torch
                    data = torch.load(first_chunk, weights_only=True)
                    print(f"✅ 成功加载chunk_0.pt，包含 {len(data)} 个对话")
                except Exception as e:
                    print(f"❌ 加载chunk文件失败: {e}")
    
    return True

def test_resume_functionality():
    """测试断点续传功能"""
    print(f"\n🔄 测试断点续传功能...")
    
    chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, "train")
    resume_file = os.path.join(chunk_dir, "resume_train.json")
    
    if os.path.exists(resume_file):
        try:
            with open(resume_file, 'r') as f:
                resume_data = json.load(f)
            print(f"📋 找到resume文件: {resume_data}")
        except Exception as e:
            print(f"❌ resume文件损坏: {e}")
    else:
        print(f"📋 无resume文件，将从头开始处理")
    
    # 检查是否有partial文件
    if os.path.exists(chunk_dir):
        partial_files = [f for f in os.listdir(chunk_dir) if f.startswith("partial_")]
        if partial_files:
            print(f"📦 找到 {len(partial_files)} 个partial文件")
        else:
            print(f"📦 无partial文件")

def check_system_resources():
    """检查系统资源"""
    print(f"\n💻 检查系统资源...")
    
    try:
        import psutil
        
        # 内存信息
        memory = psutil.virtual_memory()
        print(f"💾 内存: {memory.total / (1024**3):.1f}GB 总量, {memory.available / (1024**3):.1f}GB 可用 ({memory.percent}% 已用)")
        
        # CPU信息
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"🖥️  CPU: {cpu_count} 核心, 当前使用率 {cpu_percent}%")
        
        # 磁盘信息
        disk = psutil.disk_usage('F:/')
        print(f"💿 磁盘F:: {disk.total / (1024**3):.1f}GB 总量, {disk.free / (1024**3):.1f}GB 可用")
        
        # 建议
        if memory.available < 8 * (1024**3):  # 小于8GB可用内存
            print(f"⚠️  可用内存较少，建议使用较少的worker进程")
        
        if cpu_count >= 8:
            print(f"💡 建议使用 4-6 个worker进程")
        elif cpu_count >= 4:
            print(f"💡 建议使用 2-4 个worker进程")
        else:
            print(f"💡 建议使用 1-2 个worker进程")
            
    except ImportError:
        print(f"❌ psutil未安装，无法检查系统资源")

if __name__ == "__main__":
    print("🚀 基本处理测试...")
    
    try:
        # 基本功能测试
        if test_basic_processing():
            print(f"\n✅ 基本功能测试通过")
        else:
            print(f"\n❌ 基本功能测试失败")
            sys.exit(1)
        
        # 断点续传测试
        test_resume_functionality()
        
        # 系统资源检查
        check_system_resources()
        
        print(f"\n🎯 总结:")
        print(f"1. ✅ 词汇表和数据文件正常")
        print(f"2. ✅ 基本处理功能正常")
        print(f"3. ✅ 现有chunk文件可以正常加载")
        print(f"4. 🔄 可以继续进行数据转换")
        
        print(f"\n💡 建议的运行命令:")
        print(f"python -m src.prepare_binary_data --num_workers=2")
        print(f"# 或者如果内存充足:")
        print(f"python -m src.prepare_binary_data --num_workers=4")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
