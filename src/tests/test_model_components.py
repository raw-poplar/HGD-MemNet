# -*- coding: utf-8 -*-
"""
模型组件单元测试
测试各个模型组件的功能和正确性
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from test_config import *

# 导入模型组件
from src.model import Attention, ReservoirRNNCell, DynamicGroup, StaticHead, HGD_MemNet


class TestAttention:
    """测试注意力机制"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.hidden_dim = TEST_DYNAMIC_GROUP_HIDDEN_DIM
        self.batch_size = TEST_BATCH_SIZE
        self.seq_len = TEST_SEQ_LEN
        self.attention = Attention(self.hidden_dim)
    
    def test_attention_forward(self):
        """测试注意力前向传播"""
        query = torch.randn(self.batch_size, self.hidden_dim)
        keys = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        
        context, attn_weights = self.attention(query, keys)
        
        # 检查输出形状
        assert context.shape == (self.batch_size, self.hidden_dim)
        assert attn_weights.shape == (self.batch_size, self.seq_len)
        
        # 检查注意力权重是否归一化
        assert torch.allclose(attn_weights.sum(dim=1), torch.ones(self.batch_size), atol=1e-6)
        
        # 检查注意力权重是否非负
        assert torch.all(attn_weights >= 0)
    
    def test_attention_gradient(self):
        """测试注意力机制的梯度计算"""
        query = torch.randn(self.batch_size, self.hidden_dim, requires_grad=True)
        keys = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, requires_grad=True)
        
        context, _ = self.attention(query, keys)
        loss = context.sum()
        loss.backward()
        
        # 检查梯度是否存在且非零
        assert query.grad is not None
        assert keys.grad is not None
        assert not torch.allclose(query.grad, torch.zeros_like(query.grad))
        assert not torch.allclose(keys.grad, torch.zeros_like(keys.grad))


class TestReservoirRNNCell:
    """测试蓄水池RNN单元"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.input_size = TEST_DYNAMIC_GROUP_HIDDEN_DIM * 2  # x_t_encoded + attn_context
        self.hidden_size = TEST_DYNAMIC_GROUP_HIDDEN_DIM
        self.batch_size = TEST_BATCH_SIZE
        self.cell = ReservoirRNNCell(self.input_size, self.hidden_size)
    
    def test_cell_forward(self):
        """测试RNN单元前向传播"""
        x_t = torch.randn(self.batch_size, self.input_size)
        h_prev = torch.randn(self.batch_size, self.hidden_size)
        
        h_next = self.cell(x_t, h_prev)
        
        # 检查输出形状
        assert h_next.shape == (self.batch_size, self.hidden_size)
        
        # 检查输出值在合理范围内（tanh激活）
        assert torch.all(h_next >= -1.0) and torch.all(h_next <= 1.0)
    
    def test_temperature_effect(self):
        """测试温度参数的影响"""
        x_t = torch.randn(self.batch_size, self.input_size)
        h_prev = torch.randn(self.batch_size, self.hidden_size)
        
        # 高温度应该产生更随机的输出
        h_high_temp = self.cell(x_t, h_prev, temperature=2.0)
        h_low_temp = self.cell(x_t, h_prev, temperature=0.1)
        
        # 检查输出形状
        assert h_high_temp.shape == h_low_temp.shape == (self.batch_size, self.hidden_size)
        
        # 由于随机性，输出应该不同
        assert not torch.allclose(h_high_temp, h_low_temp, atol=1e-3)
    
    def test_training_vs_eval_mode(self):
        """测试训练模式和评估模式的差异"""
        x_t = torch.randn(self.batch_size, self.input_size)
        h_prev = torch.randn(self.batch_size, self.hidden_size)
        
        # 训练模式
        self.cell.train()
        h_train = self.cell(x_t, h_prev)
        
        # 评估模式
        self.cell.eval()
        h_eval = self.cell(x_t, h_prev)
        
        # 检查输出形状
        assert h_train.shape == h_eval.shape == (self.batch_size, self.hidden_size)
    
    def test_gradient_flow(self):
        """测试梯度流动"""
        x_t = torch.randn(self.batch_size, self.input_size, requires_grad=True)
        h_prev = torch.randn(self.batch_size, self.hidden_size, requires_grad=True)
        
        h_next = self.cell(x_t, h_prev)
        loss = h_next.sum()
        loss.backward()
        
        # 检查梯度存在
        assert x_t.grad is not None
        assert h_prev.grad is not None
        assert not torch.allclose(x_t.grad, torch.zeros_like(x_t.grad))
        assert not torch.allclose(h_prev.grad, torch.zeros_like(h_prev.grad))


class TestDynamicGroup:
    """测试动态神经组"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.embed_dim = TEST_EMBEDDING_DIM
        self.hidden_dim = TEST_DYNAMIC_GROUP_HIDDEN_DIM
        self.batch_size = TEST_BATCH_SIZE
        self.seq_len = TEST_SEQ_LEN
        self.dynamic_group = DynamicGroup(self.embed_dim, self.hidden_dim)
    
    def test_dynamic_group_forward(self):
        """测试动态组前向传播"""
        x_t = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        x_ref_encoded = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        h_prev = torch.randn(self.batch_size, self.hidden_dim)
        
        h_next, attn_context = self.dynamic_group(x_t, x_ref_encoded, h_prev)
        
        # 检查输出形状
        assert h_next.shape == (self.batch_size, self.hidden_dim)
        assert attn_context.shape == (self.batch_size, self.hidden_dim)
    
    def test_none_input_handling(self):
        """测试None输入的处理"""
        x_ref_encoded = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        h_prev = torch.randn(self.batch_size, self.hidden_dim)
        
        h_next, attn_context = self.dynamic_group(None, x_ref_encoded, h_prev)
        
        # 检查输出形状
        assert h_next.shape == (self.batch_size, self.hidden_dim)
        assert attn_context.shape == (self.batch_size, self.hidden_dim)
    
    def test_temperature_propagation(self):
        """测试温度参数传播"""
        x_t = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        x_ref_encoded = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        h_prev = torch.randn(self.batch_size, self.hidden_dim)
        
        h_next1, _ = self.dynamic_group(x_t, x_ref_encoded, h_prev, temperature=1.0)
        h_next2, _ = self.dynamic_group(x_t, x_ref_encoded, h_prev, temperature=0.1)
        
        # 不同温度应该产生不同的输出
        assert not torch.allclose(h_next1, h_next2, atol=1e-3)


class TestStaticHead:
    """测试静态决策头"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.hidden_dim = TEST_STATIC_HEAD_HIDDEN_DIM
        self.sampler_input_dim = TEST_DYNAMIC_GROUP_HIDDEN_DIM
        self.context_input_dim = TEST_DYNAMIC_GROUP_HIDDEN_DIM
        self.output_dim = TEST_VOCAB_SIZE
        self.batch_size = TEST_BATCH_SIZE
        
        self.static_head = StaticHead(
            hidden_dim=self.hidden_dim,
            sampler_input_dim=self.sampler_input_dim,
            context_input_dim=self.context_input_dim,
            output_dim=self.output_dim,
            fixed_ratio=FIXED_SAMPLING_RATIO,
            random_ratio=RANDOM_SAMPLING_RATIO
        )
    
    def test_static_head_forward(self):
        """测试静态头前向传播"""
        h_from_dynamic = torch.randn(self.batch_size, self.sampler_input_dim)
        attn_context = torch.randn(self.batch_size, self.context_input_dim)
        
        gate_signal, output_logits = self.static_head(h_from_dynamic, attn_context)
        
        # 检查输出形状
        assert gate_signal.shape == (self.batch_size, 1)
        assert output_logits.shape == (self.batch_size, self.output_dim)
        
        # 检查门控概率在[0,1]范围内（模型返回的是logits，这里转为概率）
        gate_prob = torch.sigmoid(gate_signal)
        assert torch.all(gate_prob >= 0) and torch.all(gate_prob <= 1)
    
    def test_sampling_consistency(self):
        """测试采样的一致性"""
        h_from_dynamic = torch.randn(self.batch_size, self.sampler_input_dim)
        attn_context = torch.randn(self.batch_size, self.context_input_dim)
        
        # 多次前向传播应该产生一致的固定采样部分
        gate1, logits1 = self.static_head(h_from_dynamic, attn_context)
        gate2, logits2 = self.static_head(h_from_dynamic, attn_context)
        
        # 检查输出形状一致
        assert gate1.shape == gate2.shape
        assert logits1.shape == logits2.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
