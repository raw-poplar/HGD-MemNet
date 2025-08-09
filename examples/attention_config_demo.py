#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
注意力机制配置演示

展示如何在HGD-MemNet中使用不同的注意力配置。
"""

import os
import sys
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
from src.model import HGD_MemNet


def test_attention_configuration(attention_heads, description):
    """
    测试指定的注意力配置
    
    Args:
        attention_heads (int): 注意力头数量
        description (str): 配置描述
    """
    print(f"\n{'='*60}")
    print(f"测试配置: {description}")
    print(f"NUM_ATTENTION_HEADS = {attention_heads}")
    print(f"{'='*60}")

    # 设置注意力头数量
    original_heads = config.NUM_ATTENTION_HEADS
    config.NUM_ATTENTION_HEADS = attention_heads
    
    try:
        # 创建模型
        model = HGD_MemNet(
            vocab_size=config.VOCAB_SIZE,
            embed_dim=config.EMBEDDING_DIM,
            dynamic_hidden_dim=config.DYNAMIC_GROUP_HIDDEN_DIM,
            static_hidden_dim=config.STATIC_HEAD_HIDDEN_DIM
        )
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"模型参数统计:")
        print(f"  总参数数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        # 测试前向传播
        batch_size = 2
        seq_len = 10
        
        x_t = torch.randint(0, config.VOCAB_SIZE, (batch_size, seq_len))
        x_ref = torch.randint(0, config.VOCAB_SIZE, (batch_size, seq_len))
        h_prev = torch.zeros(batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM)
        
        # 前向传播
        with torch.no_grad():
            h_next, gate_pred, output_logits = model(x_t, x_ref, h_prev)
        
        print(f"前向传播测试:")
        print(f"  输入形状: x_t={x_t.shape}, x_ref={x_ref.shape}, h_prev={h_prev.shape}")
        print(f"  输出形状: h_next={h_next.shape}, gate={gate_pred.shape}, logits={output_logits.shape}")
        print(f"  测试结果: 成功")
        
        # 检查注意力模块类型
        attention_module = model.dynamic_group.attention
        if attention_heads == 0:
            print(f"  注意力类型: 无注意力机制（平均池化）")
        elif attention_heads == 1:
            print(f"  注意力类型: 单头注意力（Bahdanau）")
        else:
            print(f"  注意力类型: 多头注意力（{attention_heads}头）")
        
        return True
        
    except Exception as e:
        print(f"  测试失败: {e}")
        return False
    
    finally:
        # 恢复原始配置
        config.NUM_ATTENTION_HEADS = original_heads


def main():
    """主函数"""
    print("HGD-MemNet 注意力机制配置演示")
    print("本演示展示了如何配置不同类型的注意力机制")
    
    # 测试配置列表
    test_configs = [
        (0, "纯HGD-MemNet架构（无注意力机制）"),
        (1, "单头注意力机制（Bahdanau注意力）"),
        (4, "4头多头注意力机制"),
        (8, "8头多头注意力机制（默认配置）"),
        (16, "16头多头注意力机制")
    ]
    
    results = []
    
    for heads, description in test_configs:
        success = test_attention_configuration(heads, description)
        results.append((heads, description, success))
    
    # 总结结果
    print(f"\n{'='*60}")
    print("测试结果总结:")
    print(f"{'='*60}")
    
    for heads, description, success in results:
        status = "成功" if success else "失败"
        print(f"  {heads:2d}头: {description:<30} - {status}")
    
    print(f"\n配置建议:")
    print(f"  - 计算资源有限: 使用 NUM_ATTENTION_HEADS = 0")
    print(f"  - 平衡性能和效率: 使用 NUM_ATTENTION_HEADS = 1")
    print(f"  - 追求最佳性能: 使用 NUM_ATTENTION_HEADS = 8 (示例)")
    print(f"  - 大规模模型: 使用 NUM_ATTENTION_HEADS = 16 或更多")

    print(f"\n修改配置方法:")
    print(f"  在 config.py 中设置: NUM_ATTENTION_HEADS = 你想要的头数")
    print(f"  或在训练脚本中动态设置: config.NUM_ATTENTION_HEADS = 头数")


if __name__ == "__main__":
    main()
