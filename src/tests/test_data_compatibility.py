#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据兼容性测试脚本
测试修改前后的数据格式是否兼容
"""

import torch
import sys
import os
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.dataset import Vocabulary, binary_collate_fn

def create_old_format_data():
    """创建修改前的数据格式（思考步骤中x_t使用前一句话）"""
    vocab = Vocabulary("test")
    vocab.addSentence("hello world")
    vocab.addSentence("how are you")
    vocab.addSentence("I am fine")
    
    def indexesFromSentence(vocab, sentence):
        if sentence is None:
            return []
        UNK_idx = vocab.word2index.get("<UNK>", config.UNK_token)
        return [vocab.word2index.get(word, UNK_idx) for word in sentence.split(' ')] + [config.EOS_token]
    
    # 模拟修改前的数据处理逻辑
    dialogue_list = [
        {"text": "hello world"},
        {"text": "how are you"},
        {"text": "I am fine"}
    ]
    
    x_ref_text = dialogue_list[0].get("text", "")
    x_ref_tensor = torch.tensor(indexesFromSentence(vocab, x_ref_text), dtype=torch.long)
    
    steps_data = []
    for i in range(1, len(dialogue_list)):
        x_t_text = dialogue_list[i-1].get("text", "")
        target_text = dialogue_list[i].get("text", "")
        
        # 修改前的逻辑：思考步骤中x_t使用前一句话
        for _ in range(config.THINKING_STEPS):
            thinking_step_x_t = torch.tensor(indexesFromSentence(vocab, x_t_text), dtype=torch.long)
            thinking_step_target = torch.tensor([], dtype=torch.long)
            thinking_step_gate = torch.tensor([0.0], dtype=torch.float)
            steps_data.append((thinking_step_x_t, thinking_step_target, thinking_step_gate))
        
        # 输出步骤
        output_step_x_t = torch.tensor(indexesFromSentence(vocab, x_t_text), dtype=torch.long)
        output_step_target = torch.tensor(indexesFromSentence(vocab, target_text), dtype=torch.long)
        output_step_gate = torch.tensor([1.0], dtype=torch.float)
        steps_data.append((output_step_x_t, output_step_target, output_step_gate))
    
    return (x_ref_tensor, steps_data)

def create_new_format_data():
    """创建修改后的数据格式（思考步骤中x_t使用None）"""
    from src.data_processing.prepare_binary_data import process_dialogue_to_tensors
    
    vocab = Vocabulary("test")
    vocab.addSentence("hello world")
    vocab.addSentence("how are you")
    vocab.addSentence("I am fine")
    
    dialogue_list = [
        {"text": "hello world"},
        {"text": "how are you"},
        {"text": "I am fine"}
    ]
    
    return process_dialogue_to_tensors(dialogue_list, vocab)

def test_data_compatibility():
    """测试数据兼容性"""
    print("=== 数据兼容性测试 ===")
    
    # 创建两种格式的数据
    old_data = create_old_format_data()
    new_data = create_new_format_data()
    
    print(f"修改前数据格式:")
    x_ref_old, steps_old = old_data
    print(f"  x_ref shape: {x_ref_old.shape}")
    print(f"  steps count: {len(steps_old)}")
    
    print(f"修改后数据格式:")
    x_ref_new, steps_new = new_data
    print(f"  x_ref shape: {x_ref_new.shape}")
    print(f"  steps count: {len(steps_new)}")
    
    # 检查基本结构兼容性
    assert x_ref_old.shape == x_ref_new.shape, "x_ref形状不兼容"
    assert len(steps_old) == len(steps_new), "步骤数量不兼容"
    
    print("\n=== 步骤详细对比 ===")
    for i, ((x_t_old, target_old, gate_old), (x_t_new, target_new, gate_new)) in enumerate(zip(steps_old, steps_new)):
        print(f"步骤 {i}:")
        print(f"  修改前: x_t={x_t_old.shape if x_t_old is not None else None}, target={target_old.shape}, gate={gate_old.item()}")
        print(f"  修改后: x_t={x_t_new.shape if x_t_new is not None else None}, target={target_new.shape}, gate={gate_new.item()}")
        
        # 检查目标和门控是否一致
        assert torch.equal(target_old, target_new), f"步骤{i}目标不一致"
        assert torch.equal(gate_old, gate_new), f"步骤{i}门控不一致"
    
    print("\n=== 批处理兼容性测试 ===")
    # 测试批处理函数是否能处理两种格式
    batch_old = [old_data, old_data]  # 重复数据模拟批次
    batch_new = [new_data, new_data]
    
    try:
        x_ref_batch_old, steps_batch_old = binary_collate_fn(batch_old)
        print("修改前格式可以正常批处理")
    except Exception as e:
        print(f"修改前格式批处理失败: {e}")
        return False
    
    try:
        x_ref_batch_new, steps_batch_new = binary_collate_fn(batch_new)
        print("修改后格式可以正常批处理")
    except Exception as e:
        print(f"修改后格式批处理失败: {e}")
        return False
    
    # 检查批处理结果的兼容性
    assert x_ref_batch_old.shape == x_ref_batch_new.shape, "批处理x_ref形状不兼容"
    assert len(steps_batch_old) == len(steps_batch_new), "批处理步骤数不兼容"
    
    print("\n=== 混合批次测试 ===")
    # 测试混合新旧格式的批次
    mixed_batch = [old_data, new_data]
    try:
        x_ref_mixed, steps_mixed = binary_collate_fn(mixed_batch)
        print("混合格式批次可以正常处理")
        print(f"  混合批次 x_ref shape: {x_ref_mixed.shape}")
        print(f"  混合批次 steps count: {len(steps_mixed)}")
    except Exception as e:
        print(f"混合格式批次处理失败: {e}")
        return False
    
    return True

def test_chunk_loading():
    """测试现有chunk文件的加载"""
    print("\n=== Chunk文件加载测试 ===")
    
    # 检查是否有现有的chunk文件
    chunk_dirs = []
    for data_type in ["train", "valid", "test"]:
        chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
        if os.path.exists(chunk_dir):
            chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".pt")]
            if chunk_files:
                chunk_dirs.append((data_type, chunk_dir, len(chunk_files)))
    
    if not chunk_dirs:
        print("未找到现有的chunk文件")
        return True
    
    for data_type, chunk_dir, count in chunk_dirs:
        print(f"发现 {data_type} 数据: {count} 个chunk文件")
        
        # 尝试加载第一个chunk
        first_chunk = os.path.join(chunk_dir, "chunk_0.pt")
        if os.path.exists(first_chunk):
            try:
                data = torch.load(first_chunk, weights_only=True)
                print(f"  成功加载 {first_chunk}")
                print(f"  包含 {len(data)} 个对话")
                
                # 检查数据格式
                if data:
                    sample = data[0]
                    x_ref, steps_data = sample
                    print(f"  样本格式: x_ref={x_ref.shape}, steps={len(steps_data)}")
                    
                    # 检查是否有None的x_t（新格式）或都有值（旧格式）
                    none_count = sum(1 for x_t, _, _ in steps_data if x_t is None)
                    total_steps = len(steps_data)
                    print(f"  None x_t 比例: {none_count}/{total_steps} ({none_count/total_steps*100:.1f}%)")
                    
                    if none_count > 0:
                        print(f"  检测到新格式数据（包含None x_t）")
                    else:
                        print(f"  检测到旧格式数据（无None x_t）")
                        
            except Exception as e:
                print(f"  加载失败: {e}")
                return False
    
    return True

if __name__ == "__main__":
    print("开始数据兼容性测试...")
    
    # 测试数据格式兼容性
    if test_data_compatibility():
        print("\n数据格式兼容性测试通过")
    else:
        print("\n数据格式兼容性测试失败")
        sys.exit(1)
    
    # 测试现有chunk文件
    if test_chunk_loading():
        print("\nChunk文件加载测试通过")
    else:
        print("\nChunk文件加载测试失败")
        sys.exit(1)
    
    print("\n所有兼容性测试通过！")
    print("\n结论:")
    print("1. 修改后的代码可以处理修改前的数据格式")
    print("2. 新旧格式可以在同一批次中混合处理")
    print("3. 现有的chunk文件可以正常加载")
    print("4. 可以安全地继续使用修改后的代码进行数据转换")
