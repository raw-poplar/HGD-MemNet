# -*- coding: utf-8 -*-
"""
训练流程测试
测试训练相关的功能和流程
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
import numpy as np
from test_config import *

# 导入训练相关模块
from src.model import HGD_MemNet
from src.utils import compute_loss


class TestTrainingComponents:
    """测试训练组件"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.model = HGD_MemNet(
            vocab_size=TEST_VOCAB_SIZE,
            embed_dim=TEST_EMBEDDING_DIM,
            dynamic_hidden_dim=TEST_DYNAMIC_GROUP_HIDDEN_DIM,
            static_hidden_dim=TEST_STATIC_HEAD_HIDDEN_DIM
        )
        self.batch_size = TEST_BATCH_SIZE
        self.seq_len = TEST_SEQ_LEN
    
    def test_compute_loss_function(self):
        """测试损失计算函数"""
        # 创建模拟输出
        gate_pred = torch.sigmoid(torch.randn(self.batch_size, 1))
        gate_target = torch.randint(0, 2, (self.batch_size, 1)).float()
        output_logits = torch.randn(self.batch_size, TEST_VOCAB_SIZE)
        target_padded = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size,))
        
        # 计算损失
        loss = compute_loss(gate_pred, gate_target, output_logits, target_padded)
        
        # 检查损失是标量且为正
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_compute_loss_with_none_target(self):
        """测试处理None目标的损失计算"""
        gate_pred = torch.sigmoid(torch.randn(self.batch_size, 1))
        gate_target = torch.randint(0, 2, (self.batch_size, 1)).float()
        output_logits = torch.randn(self.batch_size, TEST_VOCAB_SIZE)
        
        # target_padded为None的情况
        loss = compute_loss(gate_pred, gate_target, output_logits, None)
        
        # 应该只包含门控损失
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_gradient_accumulation_simulation(self):
        """测试梯度累积模拟"""
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scaler = GradScaler()
        
        # 模拟梯度累积
        accumulation_steps = 4
        total_loss = 0
        
        for step in range(accumulation_steps):
            x_t = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
            x_ref = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
            h_prev = torch.zeros(self.batch_size, TEST_DYNAMIC_GROUP_HIDDEN_DIM)
            
            h_next, gate_pred, output_logits = self.model(x_t, x_ref, h_prev)
            
            # 创建目标
            gate_target = torch.randint(0, 2, (self.batch_size, 1)).float()
            target_padded = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size,))
            
            loss = compute_loss(gate_pred, gate_target, output_logits, target_padded)
            loss = loss / accumulation_steps  # 缩放损失
            
            scaler.scale(loss).backward()
            total_loss += loss.item()
        
        # 执行优化步骤
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        assert total_loss > 0
    
    def test_temperature_annealing(self):
        """测试温度退火"""
        initial_temp = INITIAL_TEMPERATURE
        decay_rate = TEMPERATURE_DECAY
        min_temp = MIN_TEMPERATURE
        
        temperatures = []
        current_temp = initial_temp
        
        for step in range(10):
            current_temp = max(initial_temp * (decay_rate ** step), min_temp)
            temperatures.append(current_temp)
        
        # 检查温度递减
        for i in range(1, len(temperatures)):
            assert temperatures[i] <= temperatures[i-1]
        
        # 检查最终温度不低于最小值
        assert temperatures[-1] >= min_temp
    
    def test_learning_rate_scheduling(self):
        """测试学习率调度"""
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        
        initial_lr = optimizer.param_groups[0]['lr']
        lrs = [initial_lr]
        
        for epoch in range(5):
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
        
        # 检查学习率在指定步骤后下降
        assert lrs[2] < lrs[0]  # 第2步后应该下降
        assert lrs[4] < lrs[2]  # 第4步后应该再次下降


class TestTrainingStability:
    """测试训练稳定性"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.model = HGD_MemNet(
            vocab_size=TEST_VOCAB_SIZE,
            embed_dim=TEST_EMBEDDING_DIM,
            dynamic_hidden_dim=TEST_DYNAMIC_GROUP_HIDDEN_DIM,
            static_hidden_dim=TEST_STATIC_HEAD_HIDDEN_DIM
        )
        self.batch_size = TEST_BATCH_SIZE
        self.seq_len = TEST_SEQ_LEN
    
    def test_gradient_clipping(self):
        """测试梯度裁剪"""
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # 创建可能导致梯度爆炸的输入
        x_t = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
        x_ref = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
        h_prev = torch.zeros(self.batch_size, TEST_DYNAMIC_GROUP_HIDDEN_DIM)
        
        h_next, gate_pred, output_logits = self.model(x_t, x_ref, h_prev)
        
        # 创建大的损失
        gate_target = torch.ones_like(gate_pred)
        target_padded = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size,))
        loss = compute_loss(gate_pred, gate_target, output_logits, target_padded) * 100
        
        loss.backward()
        
        # 计算梯度范数
        grad_norm_before = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
        
        # 应用梯度裁剪
        max_norm = 1.0
        grad_norm_after = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        
        # 检查梯度被正确裁剪
        assert grad_norm_after <= max_norm + 1e-6
    
    def test_nan_gradient_detection(self):
        """测试NaN梯度检测"""
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        x_t = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
        x_ref = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
        h_prev = torch.zeros(self.batch_size, TEST_DYNAMIC_GROUP_HIDDEN_DIM)
        
        h_next, gate_pred, output_logits = self.model(x_t, x_ref, h_prev)
        
        gate_target = torch.randint(0, 2, (self.batch_size, 1)).float()
        target_padded = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size,))
        loss = compute_loss(gate_pred, gate_target, output_logits, target_padded)
        
        loss.backward()
        
        # 检查是否有NaN梯度
        has_nan_grad = False
        for param in self.model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break
        
        assert not has_nan_grad, "Model has NaN gradients"
    
    def test_training_convergence_simulation(self):
        """测试训练收敛模拟"""
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        losses = []
        
        # 使用固定的简单数据进行多次训练
        x_t = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
        x_ref = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
        h_prev = torch.zeros(self.batch_size, TEST_DYNAMIC_GROUP_HIDDEN_DIM)
        gate_target = torch.ones(self.batch_size, 1) * 0.5
        target_padded = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size,))
        
        for step in range(20):
            optimizer.zero_grad()
            
            h_next, gate_pred, output_logits = self.model(x_t, x_ref, h_prev)
            loss = compute_loss(gate_pred, gate_target, output_logits, target_padded)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # 检查损失是否有下降趋势
        early_avg = np.mean(losses[:5])
        late_avg = np.mean(losses[-5:])
        
        assert late_avg < early_avg, f"Loss did not decrease: {early_avg:.4f} -> {late_avg:.4f}"


class TestModelModes:
    """测试模型不同模式"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.model = HGD_MemNet(
            vocab_size=TEST_VOCAB_SIZE,
            embed_dim=TEST_EMBEDDING_DIM,
            dynamic_hidden_dim=TEST_DYNAMIC_GROUP_HIDDEN_DIM,
            static_hidden_dim=TEST_STATIC_HEAD_HIDDEN_DIM
        )
        self.batch_size = TEST_BATCH_SIZE
        self.seq_len = TEST_SEQ_LEN
    
    def test_hard_vs_soft_sampling(self):
        """测试硬采样vs软采样"""
        x_t = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
        x_ref = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
        h_prev = torch.zeros(self.batch_size, TEST_DYNAMIC_GROUP_HIDDEN_DIM)
        
        # 软采样（训练模式）
        self.model.train()
        self.model.dynamic_group.core_rnn.use_hard_sampling = False
        h_soft, _, _ = self.model(x_t, x_ref, h_prev)
        
        # 硬采样（评估模式）
        self.model.eval()
        self.model.dynamic_group.core_rnn.use_hard_sampling = True
        with torch.no_grad():
            h_hard, _, _ = self.model(x_t, x_ref, h_prev)
        
        # 检查输出形状一致
        assert h_soft.shape == h_hard.shape
        
        # 由于采样方式不同，输出应该有差异
        assert not torch.allclose(h_soft, h_hard, atol=1e-3)
    
    def test_temperature_effects_in_training(self):
        """测试训练中温度效果"""
        x_t = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
        x_ref = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
        h_prev = torch.zeros(self.batch_size, TEST_DYNAMIC_GROUP_HIDDEN_DIM)
        
        self.model.train()
        
        # 高温度
        h_high, _, _ = self.model(x_t, x_ref, h_prev, temperature=2.0)
        
        # 低温度
        h_low, _, _ = self.model(x_t, x_ref, h_prev, temperature=0.1)
        
        # 不同温度应该产生不同的隐藏状态
        assert not torch.allclose(h_high, h_low, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
