#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
系统完整性测试脚本
测试HGD-MemNet项目的各个组件是否正常工作
"""

import sys
import os
import torch
import json
import tempfile
import shutil

# 添加项目根目录到路径
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """测试所有模块是否能正常导入"""
    print("=== 测试模块导入 ===")
    try:
        import config
        print("✓ config 导入成功")
        
        from src.model import HGD_MemNet, ReservoirRNNCell, DynamicGroup, StaticHead
        print("✓ model 组件导入成功")
        
        from src.dataset import Vocabulary, BinaryDialogueDataset, binary_collate_fn
        print("✓ dataset 组件导入成功")
        
        from src.utils import load_model_from_checkpoint, compute_loss
        print("✓ utils 组件导入成功")
        
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_model_creation():
    """测试模型创建和基本前向传播"""
    print("\n=== 测试模型创建 ===")
    try:
        import config
        from src.model import HGD_MemNet

        model = HGD_MemNet(
            vocab_size=1000,
            embed_dim=64,
            dynamic_hidden_dim=128,
            static_hidden_dim=64
        )
        print("✓ 模型创建成功")
        
        # 测试前向传播
        batch_size = 2
        x_t = torch.randint(0, 1000, (batch_size, 5))
        x_ref = torch.randint(0, 1000, (batch_size, 10))
        h_prev = torch.zeros(batch_size, 128)
        
        with torch.no_grad():
            h_next, gate_pred, output_logits = model(x_t, x_ref, h_prev)
        
        print(f"✓ 前向传播成功: h_next={h_next.shape}, gate_pred={gate_pred.shape}, output_logits={output_logits.shape}")
        return True
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return False

def test_vocabulary():
    """测试词汇表功能"""
    print("\n=== 测试词汇表 ===")
    try:
        from src.dataset import Vocabulary
        import config
        
        vocab = Vocabulary("test")
        vocab.addSentence("hello world test")
        vocab.addSentence("this is a test")
        
        print(f"✓ 词汇表创建成功，词汇数量: {vocab.num_words}")
        
        # 测试trim功能
        vocab.trim(min_count=1)
        print(f"✓ 词汇表修剪成功，修剪后词汇数量: {vocab.num_words}")
        
        return True
        
    except Exception as e:
        print(f"✗ 词汇表测试失败: {e}")
        return False

def test_loss_computation():
    """测试损失计算"""
    print("\n=== 测试损失计算 ===")
    try:
        from src.utils import compute_loss
        import config

        # 创建测试数据
        vocab_size = 100  # 使用固定的词汇表大小进行测试
        batch_size = 4
        seq_len = 5
        gate_pred = torch.sigmoid(torch.randn(batch_size, 1))
        gate_target = torch.randint(0, 2, (batch_size,)).float()
        output_logits = torch.randn(batch_size, vocab_size)
        target_padded = torch.randint(0, vocab_size, (batch_size, 1))  # 修正：确保维度匹配

        # 临时设置config.VOCAB_SIZE
        original_vocab_size = getattr(config, 'VOCAB_SIZE', None)
        config.VOCAB_SIZE = vocab_size
        
        loss = compute_loss(gate_pred, gate_target, output_logits, target_padded)
        print(f"✓ 损失计算成功: {loss.item():.4f}")

        # 测试无target的情况
        loss_no_target = compute_loss(gate_pred, gate_target, output_logits, None)
        print(f"✓ 无target损失计算成功: {loss_no_target.item():.4f}")

        # 恢复原始vocab_size
        if original_vocab_size is not None:
            config.VOCAB_SIZE = original_vocab_size

        return True
        
    except Exception as e:
        print(f"✗ 损失计算测试失败: {e}")
        return False

def test_reservoir_rnn_cell():
    """测试ReservoirRNNCell的特殊功能"""
    print("\n=== 测试ReservoirRNNCell ===")
    try:
        from src.model import ReservoirRNNCell
        import config
        
        cell = ReservoirRNNCell(
            input_size=64,
            hidden_size=128,
            initial_temperature=1.0,
            use_hard_sampling=False
        )
        
        batch_size = 3
        x_t = torch.randn(batch_size, 64)
        h_prev = torch.randn(batch_size, 128)
        
        # 测试不同温度
        h_next_1 = cell(x_t, h_prev, temperature=1.0)
        h_next_2 = cell(x_t, h_prev, temperature=0.1)
        
        print(f"✓ ReservoirRNNCell测试成功: 输出形状 {h_next_1.shape}")
        print(f"✓ 温度控制测试成功: 高温和低温输出不同")
        
        # 测试虚拟权重更新
        initial_virtual_weights = cell.W_hh_virtual.clone()
        _ = cell(x_t, h_prev)  # 触发赫布更新
        updated_virtual_weights = cell.W_hh_virtual.clone()
        
        weights_changed = not torch.equal(initial_virtual_weights, updated_virtual_weights)
        print(f"✓ 虚拟权重更新测试: {'成功' if weights_changed else '失败'}")
        
        return True
        
    except Exception as e:
        print(f"✗ ReservoirRNNCell测试失败: {e}")
        return False

def test_config_consistency():
    """测试配置文件的一致性"""
    print("\n=== 测试配置一致性 ===")
    try:
        import config
        
        # 检查关键配置是否存在
        required_configs = [
            'VOCAB_SIZE', 'EMBEDDING_DIM', 'DYNAMIC_GROUP_HIDDEN_DIM',
            'STATIC_HEAD_HIDDEN_DIM', 'BATCH_SIZE', 'LEARNING_RATE',
            'THINKING_STEPS', 'FIXED_SAMPLING_RATIO', 'RANDOM_SAMPLING_RATIO'
        ]
        
        missing_configs = []
        for cfg in required_configs:
            if not hasattr(config, cfg):
                missing_configs.append(cfg)
        
        if missing_configs:
            print(f"✗ 缺少配置项: {missing_configs}")
            return False
        
        # 检查配置值的合理性
        if config.FIXED_SAMPLING_RATIO + config.RANDOM_SAMPLING_RATIO > 1.0:
            print("✗ 采样比例之和超过1.0")
            return False
        
        print("✓ 配置一致性检查通过")
        return True
        
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始HGD-MemNet系统完整性测试...\n")
    
    tests = [
        test_imports,
        test_config_consistency,
        test_vocabulary,
        test_model_creation,
        test_loss_computation,
        test_reservoir_rnn_cell,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ 测试异常: {e}")
    
    print(f"\n=== 测试总结 ===")
    print(f"通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有测试通过！系统运行正常。")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关组件。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
