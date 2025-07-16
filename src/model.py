# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

# --- 注意力模块 (无变化) ---
class Attention(nn.Module):
    """
    标准的 Bahdanau 注意力机制
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1)

    def forward(self, query, keys):
        """
        Args:
            query (torch.Tensor): 上一步的解码器隐藏状态 (h_prev), shape: (batch_size, hidden_dim)
            keys (torch.Tensor): 编码器的所有输出 (x_ref_encoded), shape: (batch_size, seq_len, hidden_dim)
        
        Returns:
            context (torch.Tensor): 上下文向量, shape: (batch_size, hidden_dim)
            attn_weights (torch.Tensor): 注意力权重, shape: (batch_size, seq_len)
        """
        # query shape: (batch_size, 1, hidden_dim)
        # keys shape: (batch_size, seq_len, hidden_dim)
        scores = self.Va(torch.tanh(self.Wa(query.unsqueeze(1)) + self.Ua(keys)))
        scores = scores.squeeze(2) # (batch_size, seq_len)
        
        attn_weights = F.softmax(scores, dim=1)
        
        # context = bmm(attn_weights.unsqueeze(1), keys)
        # context shape: (batch_size, 1, hidden_dim)
        context = torch.bmm(attn_weights.unsqueeze(1), keys)
        context = context.squeeze(1) # (batch_size, hidden_dim)
        
        return context, attn_weights


# --- 新: 可训练的“蓄水池”式RNN单元 ---
class ReservoirRNNCell(nn.Module):
    """
    一个简单的可训练RNN单元，用于模拟“两两相连”的神经元动态。
    这取代了原版的GRU单元。
    """
    def __init__(self, input_size, hidden_size):
        super(ReservoirRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x_t, h_prev):
        """
        Args:
            x_t (torch.Tensor): 当前时间步的输入, shape: (batch_size, input_size)
            h_prev (torch.Tensor): 上一步的隐藏状态, shape: (batch_size, hidden_size)
        Returns:
            h_next (torch.Tensor): 当前步的输出隐藏状态, shape: (batch_size, hidden_size)
        """
        h_next = torch.tanh(self.W_ih(x_t) + self.W_hh(h_prev))
        return h_next


# --- 动态神经组 (核心演化模块) ---
class DynamicGroup(nn.Module):
    """
    模型的核心，一个动态演化的神经组。
    核心RNN被替换为ReservoirRNNCell以匹配描述。
    """
    def __init__(self, embed_dim, hidden_dim):
        super(DynamicGroup, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 编码器: 用于处理参照输入 x_ref (保持不变)
        self.encoder_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
        # 注意力模块 (保持不变)
        self.attention = Attention(hidden_dim)

        # 核心演化RNN: 被替换为新的可训练“蓄水池”单元
        self.core_rnn = ReservoirRNNCell(embed_dim + hidden_dim, hidden_dim)

    def forward(self, x_t, x_ref_encoded, h_prev):
        """
        Args:
            x_t (torch.Tensor): 当前时间步的输入, shape: (batch_size, seq_len, embed_dim)
            x_ref_encoded (torch.Tensor): 经过编码的参照输入, shape: (batch_size, ref_seq_len, hidden_dim)
            h_prev (torch.Tensor): 上一步的隐藏状态, shape: (batch_size, hidden_dim)
            
        Returns:
            h_next (torch.Tensor): 当前步的输出隐藏状态, shape: (batch_size, hidden_dim)
            attn_context (torch.Tensor): 注意力上下文向量, shape: (batch_size, hidden_dim)
        """
        # 如果当前步没有输入，我们假设输入是一个零向量
        if x_t is None:
            # batch_size, 1, embed_dim
            # 我们需要一个单步的输入
            x_t_input = torch.zeros(h_prev.size(0), 1, self.core_rnn.input_size - self.hidden_dim, device=h_prev.device)
        else:
            # 在这个简化版本中，我们只取x_t的第一个时间步的嵌入作为代表
            # 这是一个简化，更复杂的模型可能会对x_t也进行编码
            x_t_input = x_t[:, 0:1, :] # (batch_size, 1, embed_dim)

        # 1. 使用注意力机制计算上下文向量
        # h_prev 是 query, x_ref_encoded 是 keys
        attn_context, attn_weights = self.attention(h_prev, x_ref_encoded)
        
        # 2. 将注意力上下文向量和当前输入拼接
        # attn_context shape: (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)
        # rnn_input shape: (batch_size, 1, embed_dim + hidden_dim)
        rnn_input = torch.cat((x_t_input, attn_context.unsqueeze(1)), dim=2)

        # 3. 将拼接后的向量输入到核心RNN
        # h_prev shape: (batch_size, hidden_dim) for our cell
        # rnn_input is squeezed to (batch_size, embed_dim + hidden_dim)
        h_next = self.core_rnn(rnn_input.squeeze(1), h_prev)

        return h_next, attn_context


# --- 静态网络对 (决策模块) ---
class StaticHead(nn.Module):
    """
    静态网络对 (决策前端)
    根据描述重构：从动态组的输出中进行固定和随机采样。
    """
    def __init__(self, hidden_dim, sampler_input_dim, context_input_dim, output_dim, fixed_ratio, random_ratio):
        super(StaticHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 计算采样数量
        self.num_fixed = int(sampler_input_dim * fixed_ratio)
        self.num_random = int(sampler_input_dim * random_ratio)
        self.sampler_output_dim = self.num_fixed + self.num_random
        
        # 确定用于随机采样的神经元池的维度
        self.random_pool_dim = sampler_input_dim - self.num_fixed
        if self.random_pool_dim < self.num_random:
            raise ValueError("Random sampling pool is smaller than the number of items to sample.")

        # 拼接后的维度：采样状态 + 注意力上下文
        concatenated_dim = self.sampler_output_dim + context_input_dim

        # 1. 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(concatenated_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 2. 输出网络
        self.output_network = nn.Sequential(
            nn.Linear(concatenated_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, h_from_dynamic, attn_context):
        """
        :param h_from_dynamic: 从动态组输出的完整隐藏状态 (batch_size, sampler_input_dim)
        :param attn_context: 从注意力机制输出的上下文向量 (batch_size, context_input_dim)
        :return: gate_signal (batch_size, 1), output_logits (batch_size, output_dim)
        """
        # 1. 固定采样 (从前N个神经元)
        fixed_sample = h_from_dynamic[:, :self.num_fixed]

        # 2. 随机采样 (从剩余的神经元池中)
        random_pool = h_from_dynamic[:, self.num_fixed:]
        # 为整个批次生成一套随机索引
        rand_indices = torch.randperm(self.random_pool_dim, device=h_from_dynamic.device)[:self.num_random]
        random_sample = random_pool[:, rand_indices]

        # 3. 拼接采样结果
        sampled_state = torch.cat((fixed_sample, random_sample), dim=1)
        
        # 4. 将采样状态和注意力上下文拼接起来
        combined_input = torch.cat((sampled_state, attn_context), dim=1)

        gate_signal = self.gate_network(combined_input)
        output_logits = self.output_network(combined_input)

        return gate_signal, output_logits


class HGD_MemNet(nn.Module):
    """
    分层门控动态记忆网络 (Hierarchical Gated-Dynamic Memory Network)
    整合了所有模块
    """
    def __init__(self, vocab_size, embed_dim, dynamic_hidden_dim, static_hidden_dim):
        super(HGD_MemNet, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        self.dynamic_group = DynamicGroup(embed_dim, dynamic_hidden_dim)
        
        self.static_head = StaticHead(
            hidden_dim=static_hidden_dim,
            sampler_input_dim=dynamic_hidden_dim,  # 从DynamicGroup的完整输出中采样
            context_input_dim=dynamic_hidden_dim,  # 注意力上下文的维度
            output_dim=vocab_size,
            fixed_ratio=config.FIXED_SAMPLING_RATIO,
            random_ratio=config.RANDOM_SAMPLING_RATIO
        )

    def forward(self, x_t, x_ref, h_prev):
        """
        一次完整的思考步骤
        
        Args:
            x_t (torch.Tensor or None): 当前步的输入张量, shape: (batch, seq_len)
            x_ref (torch.Tensor): 参照输入张量, shape: (batch, ref_seq_len)
            h_prev (torch.Tensor): 上一步的隐藏状态, shape: (batch, dynamic_hidden_dim)

        Returns:
            h_next (torch.Tensor): 下一步的隐藏状态
            gate_pred (torch.Tensor): 门控预测值
            output_logits (torch.Tensor): 输出预测的logits
        """
        # 1. 嵌入输入
        # 如果 x_t 存在，进行嵌入；否则为 None
        x_t_embedded = self.embed(x_t) if x_t is not None else None
        x_ref_embedded = self.embed(x_ref)

        # 2. 编码参照输入 x_ref 来为注意力机制和上下文向量做准备
        # encoder_outputs shape: (batch_size, ref_seq_len, dynamic_hidden_dim)
        encoder_outputs, _ = self.dynamic_group.encoder_rnn(x_ref_embedded)
        
        # 3. 通过动态组进行一步演化
        # 传入编码后的x_ref作为key, 获得下一步的隐藏状态和该步的注意力上下文
        h_next, attn_context = self.dynamic_group(x_t_embedded, encoder_outputs, h_prev)

        # 4. 将新的隐藏状态和动态上下文向量输入到静态决策头
        gate_pred, output_logits = self.static_head(h_next, attn_context)
        
        return h_next, gate_pred, output_logits


# 这是一个简单的测试，确保我们的类能正常工作
if __name__ == '__main__':
    # --- 测试 DynamicGroup (已有代码) ---
    print("--- 测试 DynamicGroup (新版) ---")
    batch_size = 4  # 使用较小的批处理大小进行测试
    input_dim = config.EMBEDDING_DIM
    hidden_dim = config.DYNAMIC_GROUP_HIDDEN_DIM

    dynamic_group = DynamicGroup(input_dim, hidden_dim)
    x_t_embed = torch.randn(batch_size, 1, input_dim) 
    x_ref_encoded = torch.randn(batch_size, 10, hidden_dim)
    h_prev = torch.randn(batch_size, hidden_dim)
    
    h_next, attn_context = dynamic_group(x_t_embed, x_ref_encoded, h_prev)
    print(f"输出 h_next 的形状: {h_next.shape}")
    print(f"输出 attn_context 的形状: {attn_context.shape}\n")
    assert h_next.shape == (batch_size, hidden_dim)
    assert attn_context.shape == (batch_size, hidden_dim)
    print("DynamicGroup 维度检查通过！\n")

    # --- 测试 StaticHead ---
    print("--- 测试 StaticHead (新版采样) ---")
    output_dim = config.VOCAB_SIZE
    static_head_hidden_dim = config.STATIC_HEAD_HIDDEN_DIM
    
    static_head = StaticHead(
        hidden_dim=static_head_hidden_dim,
        sampler_input_dim=hidden_dim,
        context_input_dim=hidden_dim,
        output_dim=output_dim,
        fixed_ratio=config.FIXED_SAMPLING_RATIO,
        random_ratio=config.RANDOM_SAMPLING_RATIO
    )
    print("静态决策头 (StaticHead) 已实例化:")
    print(static_head)

    # 创建虚拟的采样状态和参照输入
    h_from_dynamic_test = torch.randn(batch_size, hidden_dim)
    attn_context_test = torch.randn(batch_size, hidden_dim)

    print(f"\n输入 h_from_dynamic 的形状: {h_from_dynamic_test.shape}")
    print(f"输入 attn_context 的形状: {attn_context_test.shape}")

    # 执行一次前向传播
    gate_signal, output_logits = static_head(h_from_dynamic_test, attn_context_test)

    print(f"\n输出 gate_signal 的形状: {gate_signal.shape}")
    print(f"输出 output_logits 的形状: {output_logits.shape}")

    # 验证输出维度是否正确
    assert gate_signal.shape == (batch_size, 1)
    assert output_logits.shape == (batch_size, output_dim)
    print("\nStaticHead 维度检查通过！\n")
    
    # --- 测试 HGD_MemNet ---
    print("\n--- 测试 HGD_MemNet (完整模型, 新版) ---")
    model = HGD_MemNet(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBEDDING_DIM,
        dynamic_hidden_dim=config.DYNAMIC_GROUP_HIDDEN_DIM,
        static_hidden_dim=config.STATIC_HEAD_HIDDEN_DIM
    ).to("cpu")
    
    print("\n模型已实例化:")
    # print(model) # 模型结构较大，可以选择不打印

    # 准备虚拟输入
    x_t_test = torch.randint(0, config.VOCAB_SIZE, (batch_size, 5)) 
    x_ref_test = torch.randint(0, config.VOCAB_SIZE, (batch_size, 15))
    h_prev_test = torch.randn(batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM)
    
    # 执行前向传播
    h_next_test, gate_pred_test, output_logits_test = model(x_t_test, x_ref_test, h_prev_test)
    
    # 打印输出形状
    print(f"\n模型输出 h_next 的形状: {h_next_test.shape}")
    print(f"模型输出 gate_pred 的形状: {gate_pred_test.shape}")
    print(f"模型输出 output_logits 的形状: {output_logits_test.shape}")

    # 断言检查
    assert h_next_test.shape == (batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM)
    assert gate_pred_test.shape == (batch_size, 1)
    assert output_logits_test.shape == (batch_size, config.VOCAB_SIZE)
    print("\nHGD_MemNet 维度检查通过！") 
