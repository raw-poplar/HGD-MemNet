# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

# --- 注意力模块 ---
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


# --- 动态神经组 (核心演化模块) ---
class DynamicGroup(nn.Module):
    """
    模型的核心，一个动态演化的神经组，现在加入了注意力机制
    """
    def __init__(self, embed_dim, hidden_dim):
        super(DynamicGroup, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 编码器: 用于处理参照输入 x_ref
        self.encoder_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
        # 注意力模块
        self.attention = Attention(hidden_dim)

        # 核心演化RNN: 输入维度现在是 embed_dim (x_t) + hidden_dim (attention_context)
        self.core_rnn = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)

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
        # h_prev shape: (1, batch_size, hidden_dim) for GRU
        _, h_next = self.core_rnn(rnn_input, h_prev.unsqueeze(0))

        return h_next.squeeze(0), attn_context


# --- 静态网络对 (决策模块) ---
class StaticHead(nn.Module):
    """
    静态网络对 (决策前端)
    包含一个门控网络和一个输出网络。
    """
    def __init__(self, hidden_dim, sampler_dim, input_dim, output_dim):
        super(StaticHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.sampler_dim = sampler_dim # 从动态组采样得到的维度
        self.input_dim = input_dim # 原始外部输入的维度
        self.output_dim = output_dim # 输出维度，即词汇表大小

        # 两个网络接收的输入是拼接后的向量
        concatenated_dim = sampler_dim + input_dim

        # 1. 门控网络
        # MLP: Linear -> ReLU -> Linear -> Sigmoid
        self.gate_network = nn.Sequential(
            nn.Linear(concatenated_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # 输出一个0~1之间的值作为门控信号
        )

        # 2. 输出网络
        # MLP: Linear -> ReLU -> Linear
        self.output_network = nn.Sequential(
            nn.Linear(concatenated_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) # 输出词汇表大小的logits
        )

    def forward(self, sampled_state, x_ref):
        """
        :param sampled_state: 从动态组采样得到的内部状态 (batch_size, sampler_dim)
        :param x_ref: 原始的外部输入，作为参照 (batch_size, input_dim)
        :return: gate_signal (batch_size, 1), output_logits (batch_size, output_dim)
        """
        # 将采样状态和参照输入拼接起来
        combined_input = torch.cat((sampled_state, x_ref), dim=1)

        gate_signal = self.gate_network(combined_input)
        output_logits = self.output_network(combined_input)

        return gate_signal, output_logits

class HGD_MemNet(nn.Module):
    """
    分层门控动态记忆网络 (Hierarchical Gated-Dynamic Memory Network)
    整合了所有模块
    """
    def __init__(self, vocab_size, embed_dim, context_dim, dynamic_hidden_dim, static_hidden_dim):
        super(HGD_MemNet, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # 动态组现在需要 embed_dim 和 dynamic_hidden_dim
        self.dynamic_group = DynamicGroup(embed_dim, dynamic_hidden_dim)
        
        self.static_head = StaticHead(
            hidden_dim=static_hidden_dim,   # 静态网络内部维度
            sampler_dim=dynamic_hidden_dim,  # 来自DynamicGroup的输出 (h_next)
            input_dim=dynamic_hidden_dim,    # 参照输入的上下文 (attn_context)
            output_dim=vocab_size
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
    print("--- 测试 DynamicGroup ---")
    batch_size = config.BATCH_SIZE
    input_dim = config.EMBEDDING_DIM
    context_dim = config.CONTEXT_VECTOR_DIM
    hidden_dim = config.DYNAMIC_GROUP_HIDDEN_DIM

    dynamic_group = DynamicGroup(input_dim, hidden_dim)
    x_t = torch.randn(batch_size, 1, input_dim) # 修改为 (batch_size, 1, input_dim)
    x_ref_encoded = torch.randn(batch_size, 10, hidden_dim) # 假设一个较长的序列
    h_prev = torch.randn(batch_size, hidden_dim)
    h_next, attn_context = dynamic_group(x_t, x_ref_encoded, h_prev)
    print(f"输出 h_next 的形状: {h_next.shape}")
    print(f"输出 attn_context 的形状: {attn_context.shape}\n")
    assert h_next.shape == (batch_size, hidden_dim)
    
    # --- 测试 StaticHead ---
    print("--- 测试 StaticHead ---")
    sampler_dim = hidden_dim # 在新版中，采样维度就是隐藏层维度
    output_dim = config.VOCAB_SIZE
    static_head_hidden_dim = config.STATIC_HEAD_HIDDEN_DIM
    
    # 注意：这里的input_dim现在是注意力上下文的维度，也等于hidden_dim
    static_head = StaticHead(static_head_hidden_dim, sampler_dim, hidden_dim, output_dim)
    print("静态决策头 (StaticHead) 已实例化:")
    print(static_head)

    # 创建虚拟的采样状态和参照输入
    sampled_state = torch.randn(batch_size, sampler_dim)
    x_ref = torch.randn(batch_size, hidden_dim) # x_ref现在是attn_context

    print(f"\n输入 sampled_state 的形状: {sampled_state.shape}")
    print(f"输入 x_ref (attn_context) 的形状: {x_ref.shape}")

    # 执行一次前向传播
    gate_signal, output_logits = static_head(sampled_state, x_ref)

    print(f"\n输出 gate_signal 的形状: {gate_signal.shape}")
    print(f"输出 output_logits 的形状: {output_logits.shape}")

    # 验证输出维度是否正确
    assert gate_signal.shape == (batch_size, 1)
    assert output_logits.shape == (batch_size, output_dim)
    print("\n维度检查通过！") 
    
    # --- 测试 HGD_MemNet ---
    print("\n--- 测试 HGD_MemNet (完整模型) ---")
    model = HGD_MemNet(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBEDDING_DIM,
        context_dim=config.CONTEXT_VECTOR_DIM,
        dynamic_hidden_dim=config.DYNAMIC_GROUP_HIDDEN_DIM,
        static_hidden_dim=config.STATIC_HEAD_HIDDEN_DIM
    ).to("cpu")
    
    print("\n模型已实例化:")
    print(model)

    # 准备虚拟输入
    # (batch, seq_len)
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