# -*- coding: utf-8 -*-
"""
性能测试
测试模型的性能、内存使用和计算效率
"""

import pytest
import torch
import time
import psutil
import os
import numpy as np
from test_config import *

# 导入模型
from src.model import HGD_MemNet, ReservoirRNNCell, DynamicGroup, StaticHead


class TestModelPerformance:
    """测试模型性能"""
    
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
    
    def test_forward_pass_speed(self):
        """测试前向传播速度"""
        x_t = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
        x_ref = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
        h_prev = torch.zeros(self.batch_size, TEST_DYNAMIC_GROUP_HIDDEN_DIM)
        
        # 预热
        for _ in range(5):
            _ = self.model(x_t, x_ref, h_prev)
        
        # 测试速度
        num_runs = 20
        start_time = time.time()
        
        for _ in range(num_runs):
            h_next, gate_pred, output_logits = self.model(x_t, x_ref, h_prev)
            h_prev = h_next.detach()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        print(f"Average forward pass time: {avg_time*1000:.2f}ms")
        
        # 对于测试配置，前向传播应该很快
        assert avg_time < 0.1, f"Forward pass too slow: {avg_time:.3f}s"
    
    def test_memory_usage(self):
        """测试内存使用"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        x_t = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
        x_ref = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
        h_prev = torch.zeros(self.batch_size, TEST_DYNAMIC_GROUP_HIDDEN_DIM)
        
        # 多次前向传播
        for _ in range(50):
            h_next, gate_pred, output_logits = self.model(x_t, x_ref, h_prev)
            h_prev = h_next.detach()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase: {memory_increase:.2f}MB")
        
        # 内存增长应该在合理范围内
        assert memory_increase < 50, f"Memory usage too high: {memory_increase:.2f}MB"
    
    def test_batch_size_scaling(self):
        """测试批次大小扩展性"""
        batch_sizes = [1, 2, 4, 8]
        times = []
        
        for batch_size in batch_sizes:
            x_t = torch.randint(0, TEST_VOCAB_SIZE, (batch_size, self.seq_len))
            x_ref = torch.randint(0, TEST_VOCAB_SIZE, (batch_size, self.seq_len))
            h_prev = torch.zeros(batch_size, TEST_DYNAMIC_GROUP_HIDDEN_DIM)
            
            # 预热
            _ = self.model(x_t, x_ref, h_prev)
            
            # 测试时间
            start_time = time.time()
            for _ in range(10):
                _ = self.model(x_t, x_ref, h_prev)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            times.append(avg_time)
            print(f"Batch size {batch_size}: {avg_time*1000:.2f}ms")
        
        # 检查时间复杂度是否合理（不应该超线性增长太多）
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1]
            batch_ratio = batch_sizes[i] / batch_sizes[i-1]
            # 时间增长不应该超过批次大小增长的1.5倍
            assert ratio <= batch_ratio * 1.5, f"Poor scaling: {ratio:.2f}x time for {batch_ratio}x batch"
    
    def test_sequence_length_scaling(self):
        """测试序列长度扩展性"""
        seq_lengths = [5, 10, 20, 40]
        times = []
        
        for seq_len in seq_lengths:
            x_t = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, seq_len))
            x_ref = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, seq_len))
            h_prev = torch.zeros(self.batch_size, TEST_DYNAMIC_GROUP_HIDDEN_DIM)
            
            # 预热
            _ = self.model(x_t, x_ref, h_prev)
            
            # 测试时间
            start_time = time.time()
            for _ in range(10):
                _ = self.model(x_t, x_ref, h_prev)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            times.append(avg_time)
            print(f"Sequence length {seq_len}: {avg_time*1000:.2f}ms")
        
        # 检查时间复杂度（注意力机制应该是O(n^2)）
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1]
            seq_ratio = seq_lengths[i] / seq_lengths[i-1]
            # 由于注意力机制，时间可能会二次增长，但不应该过度
            assert ratio <= seq_ratio * seq_ratio * 1.2, f"Poor sequence scaling: {ratio:.2f}x time for {seq_ratio}x length"


class TestComponentPerformance:
    """测试组件性能"""
    
    def test_reservoir_rnn_cell_performance(self):
        """测试蓄水池RNN单元性能"""
        input_size = TEST_DYNAMIC_GROUP_HIDDEN_DIM * 2
        hidden_size = TEST_DYNAMIC_GROUP_HIDDEN_DIM
        batch_size = TEST_BATCH_SIZE
        
        cell = ReservoirRNNCell(input_size, hidden_size)
        
        x_t = torch.randn(batch_size, input_size)
        h_prev = torch.randn(batch_size, hidden_size)
        
        # 预热
        for _ in range(5):
            _ = cell(x_t, h_prev)
        
        # 测试速度
        num_runs = 100
        start_time = time.time()
        
        for _ in range(num_runs):
            h_next = cell(x_t, h_prev)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        print(f"ReservoirRNNCell average time: {avg_time*1000:.2f}ms")
        
        # RNN单元应该很快
        assert avg_time < 0.01, f"RNN cell too slow: {avg_time:.4f}s"
    
    def test_attention_performance(self):
        """测试注意力机制性能"""
        from src.model import Attention
        
        hidden_dim = TEST_DYNAMIC_GROUP_HIDDEN_DIM
        batch_size = TEST_BATCH_SIZE
        seq_len = TEST_SEQ_LEN
        
        attention = Attention(hidden_dim)
        
        query = torch.randn(batch_size, hidden_dim)
        keys = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 预热
        for _ in range(5):
            _ = attention(query, keys)
        
        # 测试速度
        num_runs = 50
        start_time = time.time()
        
        for _ in range(num_runs):
            context, attn_weights = attention(query, keys)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        print(f"Attention average time: {avg_time*1000:.2f}ms")
        
        # 注意力机制应该相对快速
        assert avg_time < 0.05, f"Attention too slow: {avg_time:.4f}s"
    
    def test_static_head_performance(self):
        """测试静态头性能"""
        hidden_dim = TEST_STATIC_HEAD_HIDDEN_DIM
        sampler_input_dim = TEST_DYNAMIC_GROUP_HIDDEN_DIM
        context_input_dim = TEST_DYNAMIC_GROUP_HIDDEN_DIM
        output_dim = TEST_VOCAB_SIZE
        batch_size = TEST_BATCH_SIZE
        
        static_head = StaticHead(
            hidden_dim=hidden_dim,
            sampler_input_dim=sampler_input_dim,
            context_input_dim=context_input_dim,
            output_dim=output_dim,
            fixed_ratio=FIXED_SAMPLING_RATIO,
            random_ratio=RANDOM_SAMPLING_RATIO
        )
        
        h_from_dynamic = torch.randn(batch_size, sampler_input_dim)
        attn_context = torch.randn(batch_size, context_input_dim)
        
        # 预热
        for _ in range(5):
            _ = static_head(h_from_dynamic, attn_context)
        
        # 测试速度
        num_runs = 50
        start_time = time.time()
        
        for _ in range(num_runs):
            gate_signal, output_logits = static_head(h_from_dynamic, attn_context)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        print(f"StaticHead average time: {avg_time*1000:.2f}ms")
        
        # 静态头应该很快
        assert avg_time < 0.02, f"StaticHead too slow: {avg_time:.4f}s"


class TestGPUPerformance:
    """测试GPU性能（如果可用）"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_vs_cpu_performance(self):
        """测试GPU vs CPU性能"""
        model_cpu = HGD_MemNet(
            vocab_size=TEST_VOCAB_SIZE,
            embed_dim=TEST_EMBEDDING_DIM,
            dynamic_hidden_dim=TEST_DYNAMIC_GROUP_HIDDEN_DIM,
            static_hidden_dim=TEST_STATIC_HEAD_HIDDEN_DIM
        )
        
        model_gpu = HGD_MemNet(
            vocab_size=TEST_VOCAB_SIZE,
            embed_dim=TEST_EMBEDDING_DIM,
            dynamic_hidden_dim=TEST_DYNAMIC_GROUP_HIDDEN_DIM,
            static_hidden_dim=TEST_STATIC_HEAD_HIDDEN_DIM
        ).cuda()
        
        batch_size = 8  # 更大的批次以突出GPU优势
        seq_len = 20
        
        x_t_cpu = torch.randint(0, TEST_VOCAB_SIZE, (batch_size, seq_len))
        x_ref_cpu = torch.randint(0, TEST_VOCAB_SIZE, (batch_size, seq_len))
        h_prev_cpu = torch.zeros(batch_size, TEST_DYNAMIC_GROUP_HIDDEN_DIM)
        
        x_t_gpu = x_t_cpu.cuda()
        x_ref_gpu = x_ref_cpu.cuda()
        h_prev_gpu = h_prev_cpu.cuda()
        
        # 预热
        for _ in range(5):
            _ = model_cpu(x_t_cpu, x_ref_cpu, h_prev_cpu)
            _ = model_gpu(x_t_gpu, x_ref_gpu, h_prev_gpu)
        
        # 测试CPU
        num_runs = 10
        start_time = time.time()
        for _ in range(num_runs):
            _ = model_cpu(x_t_cpu, x_ref_cpu, h_prev_cpu)
        cpu_time = (time.time() - start_time) / num_runs
        
        # 测试GPU
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            _ = model_gpu(x_t_gpu, x_ref_gpu, h_prev_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start_time) / num_runs
        
        print(f"CPU time: {cpu_time*1000:.2f}ms")
        print(f"GPU time: {gpu_time*1000:.2f}ms")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
        # GPU应该更快（至少对于较大的模型）
        # 注意：对于很小的模型，GPU可能由于开销而更慢
        if batch_size >= 4:
            assert gpu_time <= cpu_time * 1.5, "GPU not providing expected performance"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_usage(self):
        """测试GPU内存使用"""
        model = HGD_MemNet(
            vocab_size=TEST_VOCAB_SIZE,
            embed_dim=TEST_EMBEDDING_DIM,
            dynamic_hidden_dim=TEST_DYNAMIC_GROUP_HIDDEN_DIM,
            static_hidden_dim=TEST_STATIC_HEAD_HIDDEN_DIM
        ).cuda()
        
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        batch_size = 8
        seq_len = 20
        
        x_t = torch.randint(0, TEST_VOCAB_SIZE, (batch_size, seq_len)).cuda()
        x_ref = torch.randint(0, TEST_VOCAB_SIZE, (batch_size, seq_len)).cuda()
        h_prev = torch.zeros(batch_size, TEST_DYNAMIC_GROUP_HIDDEN_DIM).cuda()
        
        # 多次前向传播
        for _ in range(20):
            h_next, gate_pred, output_logits = model(x_t, x_ref, h_prev)
            h_prev = h_next.detach()
        
        final_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"GPU memory increase: {memory_increase:.2f}MB")
        
        # GPU内存使用应该在合理范围内
        assert memory_increase < 100, f"GPU memory usage too high: {memory_increase:.2f}MB"


class TestNumericalStability:
    """测试数值稳定性"""
    
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
    
    def test_gradient_stability(self):
        """测试梯度稳定性"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        x_t = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
        x_ref = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size, self.seq_len))
        h_prev = torch.zeros(self.batch_size, TEST_DYNAMIC_GROUP_HIDDEN_DIM)
        
        gradient_norms = []
        
        for step in range(10):
            optimizer.zero_grad()
            
            h_next, gate_pred, output_logits = self.model(x_t, x_ref, h_prev)
            
            # 计算损失
            gate_target = torch.rand(self.batch_size, 1)
            target = torch.randint(0, TEST_VOCAB_SIZE, (self.batch_size,))
            
            gate_loss = torch.nn.BCELoss()(gate_pred, gate_target)
            output_loss = torch.nn.CrossEntropyLoss()(output_logits, target)
            loss = gate_loss + output_loss
            
            loss.backward()
            
            # 计算梯度范数
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
            gradient_norms.append(grad_norm.item())
            
            optimizer.step()
            h_prev = h_next.detach()
        
        # 检查梯度范数的稳定性
        grad_std = np.std(gradient_norms)
        grad_mean = np.mean(gradient_norms)
        
        print(f"Gradient norm mean: {grad_mean:.4f}, std: {grad_std:.4f}")
        
        # 梯度应该相对稳定
        assert grad_std / grad_mean < 2.0, f"Gradient instability: std/mean = {grad_std/grad_mean:.2f}"
        
        # 梯度不应该爆炸
        assert max(gradient_norms) < 100, f"Gradient explosion: max norm = {max(gradient_norms):.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
