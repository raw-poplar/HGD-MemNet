# -*- coding: utf-8 -*-
"""
多头注意力机制测试脚本
测试不同注意力头数量的HGD-MemNet性能
"""

import torch
import torch.nn as nn
import config
from src.model import HGD_MemNet
import time
import numpy as np


def test_attention_configurations():
    """测试不同注意力配置的性能"""
    # 测试配置
    batch_size = 4
    seq_len = 10
    vocab_size = 1000

    # 不同的注意力配置
    attention_configs = [
        {"heads": 0, "name": "纯HGD-MemNet (无注意力)"},
        {"heads": 1, "name": "单头Bahdanau注意力"},
        {"heads": 2, "name": "双头注意力"},
        {"heads": 4, "name": "四头注意力"},
        {"heads": 8, "name": "八头注意力"},
    ]

    print("=" * 80)
    print("HGD-MemNet 多头注意力机制对比测试")
    print("=" * 80)

    # 创建测试数据
    x_t = torch.randint(0, vocab_size, (batch_size, seq_len))
    x_ref = torch.randint(0, vocab_size, (batch_size, seq_len * 2))
    h_prev = torch.zeros(batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM)

    print(f"测试数据: batch_size={batch_size}, seq_len={seq_len}")
    print(f"模型参数: embed_dim={config.EMBEDDING_DIM}, hidden_dim={config.DYNAMIC_GROUP_HIDDEN_DIM}")
    print()

    results = []

    for i, cfg in enumerate(attention_configs):
        print(f"测试 {i+1}: {cfg['name']}")

        # 检查多头注意力的维度兼容性
        if cfg['heads'] > 1 and config.DYNAMIC_GROUP_HIDDEN_DIM % cfg['heads'] != 0:
            print(f"  跳过: hidden_dim ({config.DYNAMIC_GROUP_HIDDEN_DIM}) 不能被头数 ({cfg['heads']}) 整除")
            print()
            continue

        try:
            # 创建模型
            model = HGD_MemNet(
                vocab_size=vocab_size,
                embed_dim=config.EMBEDDING_DIM,
                dynamic_hidden_dim=config.DYNAMIC_GROUP_HIDDEN_DIM,
                static_hidden_dim=config.STATIC_HEAD_HIDDEN_DIM,
                num_attention_heads=cfg['heads'],
            )

            # 参数统计
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # 性能测试
            model.eval()
            start_time = time.time()

            with torch.no_grad():
                for _ in range(50):  # 减少测试次数以节省时间
                    h_next, gate_pred, output_logits = model(x_t, x_ref, h_prev)

            inference_time = time.time() - start_time

            # 内存使用估算
            model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32

            print(f"  前向传播50次耗时: {inference_time:.4f}秒")
            print(f"  总参数量: {total_params:,}")
            print(f"  可训练参数: {trainable_params:,}")
            print(f"  模型大小: {model_size_mb:.2f}MB")
            print(f"  输出形状: h_next={h_next.shape}, gate={gate_pred.shape}, logits={output_logits.shape}")

            # 测试思考演化
            print(f"  思考演化测试:")
            h_test = h_prev.clone()
            for step in range(3):
                with torch.no_grad():
                    h_next_test, gate_test, _ = model(None, x_ref, h_test)  # 纯思考模式
                    change = torch.norm(h_next_test - h_test).item()
                    gate_avg = torch.mean(gate_test).item()
                    print(f"    步骤{step+1}: 变化={change:.4f}, 门控={gate_avg:.4f}")

            results.append({
                "config": cfg,
                "params": total_params,
                "time": inference_time,
                "size_mb": model_size_mb,
            })

        except Exception as e:
            print(f"  错误: {str(e)}")

        print()

    # 对比分析
    if len(results) > 1:
        print("性能对比分析")
        print("-" * 60)

        baseline = results[0]  # 以第一个配置为基准

        for result in results[1:]:
            cfg = result["config"]

            # 计算相对变化
            param_change = (result["params"] - baseline["params"]) / baseline["params"] * 100
            time_change = (result["time"] - baseline["time"]) / baseline["time"] * 100
            size_change = (result["size_mb"] - baseline["size_mb"]) / baseline["size_mb"] * 100

            print(f"{cfg['name']}:")
            print(f"  参数变化: {param_change:+.1f}% ({result['params']:,})")
            print(f"  速度变化: {time_change:+.1f}% ({result['time']:.4f}s)")
            print(f"  大小变化: {size_change:+.1f}% ({result['size_mb']:.2f}MB)")
            print()

    return results


def test_attention_quality():
    """测试不同注意力机制的质量"""
    print("注意力质量测试")
    print("-" * 40)

    batch_size = 2
    seq_len = 8
    vocab_size = 100

    # 创建有意义的测试数据
    x_ref = torch.randint(1, vocab_size, (batch_size, seq_len))
    h_prev = torch.randn(batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM)

    configs = [
        {"heads": 1, "name": "单头注意力"},
        {"heads": 4, "name": "四头注意力"},
    ]

    for cfg in configs:
        if config.DYNAMIC_GROUP_HIDDEN_DIM % cfg["heads"] != 0:
            continue

        print(f"\n{cfg['name']}:")

        model = HGD_MemNet(
            vocab_size=vocab_size,
            embed_dim=config.EMBEDDING_DIM,
            dynamic_hidden_dim=config.DYNAMIC_GROUP_HIDDEN_DIM,
            static_hidden_dim=config.STATIC_HEAD_HIDDEN_DIM,
            num_attention_heads=cfg["heads"],
        )

        model.eval()
        with torch.no_grad():
            h_next, gate_pred, output_logits = model(None, x_ref, h_prev)

            # 分析输出分布
            gate_mean = torch.mean(gate_pred).item()
            gate_std = torch.std(gate_pred).item()
            output_entropy = -torch.sum(
                torch.softmax(output_logits, dim=-1)
                * torch.log_softmax(output_logits, dim=-1),
                dim=-1,
            ).mean().item()

            print(f"  门控信号: 均值={gate_mean:.4f}, 标准差={gate_std:.4f}")
            print(f"  输出熵: {output_entropy:.4f}")


if __name__ == "__main__":
    print("开始HGD-MemNet多头注意力机制测试...\n")

    # 基础性能对比
    results = test_attention_configurations()

    # 注意力质量测试
    test_attention_quality()

    print("\n使用建议:")
    print("1. NUM_ATTENTION_HEADS = 0: 纯HGD-MemNet，突出原创性")
    print("2. NUM_ATTENTION_HEADS = 1: 平衡性能和复杂度")
    print("3. NUM_ATTENTION_HEADS >= 2: 更强的表征能力，但计算开销更大")
    print("4. 确保 DYNAMIC_GROUP_HIDDEN_DIM 能被 NUM_ATTENTION_HEADS 整除")

    print("\n配置示例:")
    print("# config.py 中的设置")
    print("NUM_ATTENTION_HEADS = 0  # 纯HGD-MemNet")
    print("NUM_ATTENTION_HEADS = 1  # 单头注意力")
    print("NUM_ATTENTION_HEADS = 4  # 四头注意力 (需要hidden_dim=192能被4整除)")

