# -*- coding: utf-8 -*-
"""
HGD-MemNet架构对比测试脚本
测试有注意力机制 vs 无注意力机制的性能差异
"""

import torch
import torch.nn as nn
import config
from src.model import HGD_MemNet
import time
import numpy as np

def test_architecture_performance():
    """测试两种架构的性能对比"""
    
    # 设置测试参数
    batch_size = 4
    seq_len = 10
    vocab_size = 1000
    
    print("=" * 60)
    print("HGD-MemNet 架构对比测试")
    print("=" * 60)
    
    # 创建测试数据
    x_t = torch.randint(0, vocab_size, (batch_size, seq_len))
    x_ref = torch.randint(0, vocab_size, (batch_size, seq_len * 2))
    h_prev = torch.zeros(batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM)
    
    print(f"测试数据: batch_size={batch_size}, seq_len={seq_len}")
    print(f"模型参数: embed_dim={config.EMBEDDING_DIM}, hidden_dim={config.DYNAMIC_GROUP_HIDDEN_DIM}")
    print()
    
    # 测试1: 有注意力机制的HGD-MemNet
    print("🔍 测试1: HGD-MemNet + 注意力机制")
    model_with_attention = HGD_MemNet(
        vocab_size=vocab_size,
        embed_dim=config.EMBEDDING_DIM,
        dynamic_hidden_dim=config.DYNAMIC_GROUP_HIDDEN_DIM,
        static_hidden_dim=config.STATIC_HEAD_HIDDEN_DIM,
        use_attention=True
    )
    
    # 性能测试
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            h_next, gate_pred, output_logits = model_with_attention(x_t, x_ref, h_prev)
    attention_time = time.time() - start_time
    
    # 参数统计
    attention_params = sum(p.numel() for p in model_with_attention.parameters())
    
    print(f"  ✓ 前向传播100次耗时: {attention_time:.4f}秒")
    print(f"  ✓ 模型参数量: {attention_params:,}")
    print(f"  ✓ 输出形状: h_next={h_next.shape}, gate={gate_pred.shape}, logits={output_logits.shape}")
    print()
    
    # 测试2: 纯HGD-MemNet（无注意力）
    print("🧠 测试2: 纯HGD-MemNet架构")
    model_without_attention = HGD_MemNet(
        vocab_size=vocab_size,
        embed_dim=config.EMBEDDING_DIM,
        dynamic_hidden_dim=config.DYNAMIC_GROUP_HIDDEN_DIM,
        static_hidden_dim=config.STATIC_HEAD_HIDDEN_DIM,
        use_attention=False
    )
    
    # 性能测试
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            h_next, gate_pred, output_logits = model_without_attention(x_t, x_ref, h_prev)
    no_attention_time = time.time() - start_time
    
    # 参数统计
    no_attention_params = sum(p.numel() for p in model_without_attention.parameters())
    
    print(f"  ✓ 前向传播100次耗时: {no_attention_time:.4f}秒")
    print(f"  ✓ 模型参数量: {no_attention_params:,}")
    print(f"  ✓ 输出形状: h_next={h_next.shape}, gate={gate_pred.shape}, logits={output_logits.shape}")
    print()
    
    # 对比分析
    print("📊 对比分析")
    print("-" * 40)
    speed_improvement = (attention_time - no_attention_time) / attention_time * 100
    param_reduction = (attention_params - no_attention_params) / attention_params * 100
    
    print(f"速度提升: {speed_improvement:+.2f}% ({'更快' if speed_improvement > 0 else '更慢'})")
    print(f"参数减少: {param_reduction:+.2f}% ({abs(param_reduction):.0f}个参数)")
    print(f"内存效率: {'纯HGD-MemNet更优' if param_reduction > 0 else '混合架构更优'}")
    print()
    
    return {
        'with_attention': {
            'time': attention_time,
            'params': attention_params,
            'model': model_with_attention
        },
        'without_attention': {
            'time': no_attention_time,
            'params': no_attention_params,
            'model': model_without_attention
        }
    }

def test_thinking_mechanism():
    """测试思考机制的差异"""
    print("🧠 思考机制对比测试")
    print("-" * 40)
    
    batch_size = 2
    seq_len = 5
    vocab_size = 100
    thinking_steps = 5
    
    # 创建测试数据
    x_ref = torch.randint(0, vocab_size, (batch_size, seq_len))
    h_init = torch.zeros(batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM)
    
    # 测试两种架构的思考演化
    models = {
        '有注意力': HGD_MemNet(vocab_size, config.EMBEDDING_DIM, 
                              config.DYNAMIC_GROUP_HIDDEN_DIM, 
                              config.STATIC_HEAD_HIDDEN_DIM, True),
        '无注意力': HGD_MemNet(vocab_size, config.EMBEDDING_DIM, 
                              config.DYNAMIC_GROUP_HIDDEN_DIM, 
                              config.STATIC_HEAD_HIDDEN_DIM, False)
    }
    
    for name, model in models.items():
        print(f"\n{name}架构的思考演化:")
        h_prev = h_init.clone()
        
        for step in range(thinking_steps):
            with torch.no_grad():
                h_next, gate_pred, _ = model(None, x_ref, h_prev)  # x_t=None表示纯思考
                
                # 计算思考变化程度
                change_magnitude = torch.norm(h_next - h_prev).item()
                avg_gate = torch.mean(gate_pred).item()
                
                print(f"  步骤{step+1}: 变化幅度={change_magnitude:.4f}, 平均门控={avg_gate:.4f}")
                h_prev = h_next
    
    print()

if __name__ == "__main__":
    print("开始HGD-MemNet架构对比测试...\n")
    
    # 基础性能对比
    results = test_architecture_performance()
    
    # 思考机制对比
    test_thinking_mechanism()
    
    print("🎯 建议:")
    print("1. 如果追求纯粹的HGD-MemNet创新性，使用无注意力版本")
    print("2. 如果需要与Transformer对比，使用有注意力版本")
    print("3. 可以先用无注意力版本验证核心思考机制，再加入注意力优化")
    print("4. 建议在您的饥饿式训练实验中同时测试两种架构")
