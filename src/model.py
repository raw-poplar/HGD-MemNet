# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init  # 新增 for xavier
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
    def __init__(self, input_size, hidden_size, initial_temperature=1.0, use_hard_sampling=False):
        """
        初始化 ReservoirRNNCell：高级版带有温度采样的可训练RNN单元，模拟概率驱动的数据交换。
        
        新增：initial_temperature - 初始温度参数。
        
        Args:
            input_size (int): 输入维度。
            hidden_size (int): 隐藏状态维度（神经元数量）。
            initial_temperature (float): 初始温度参数，控制采样随机性（>1: 更随机；<1: 更确定）。默认1.0（标准softmax）。
            use_hard_sampling (bool): 是否在 forward 中使用硬采样。默认False（软采样）。
        """
        super(ReservoirRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))  # 改为可学习参数，支持退火
        self.use_hard_sampling = use_hard_sampling
        self.W_ih = nn.Linear(input_size, hidden_size, bias=True)
        # 可训练权重矩阵：(hidden_out, hidden_in)，每个行代表一个输出神经元的出边权重到输入神经元。
        self.W_hh_fixed = nn.Parameter(torch.empty(hidden_size, hidden_size), requires_grad=False)
        self.register_buffer('W_hh_virtual', torch.zeros(hidden_size, hidden_size)) # 虚拟权重，作为缓冲区
        init.xavier_uniform_(self.W_hh_fixed)
    
    def forward(self, x_t, h_prev, temperature=None):
        """
        前向传播：高级版使用温度调整的 Gumbel-Softmax 采样更新隐藏状态。
        
        高级特性：
        - 支持动态温度（通过可选参数传入，覆盖 self.temperature）。
        - 可切换硬/软采样：硬采样 (hard=True) 生成离散 one-hot 向量（适合推理）；软采样 (hard=False) 生成连续近似（适合训练，反向传播友好）。
        - 数学原理：Gumbel-Softmax 是对分类分布的连续松弛，使用 Gumbel 噪声模拟采样过程。公式：y_i = softmax((log π_i + g_i) / τ)，其中 g_i ~ Gumbel(0,1)，τ 是温度。
        - 影响：高温τ增加熵（多样性），低温τ减少熵（确定性）。如果τ→0，退化为 argmax（需小心除零）。
        - 边缘处理：如果 temperature <=0，使用 argmax 作为近似硬采样。
        - 警告：高温 (tau >1) 可能增加训练方差，导致不稳定；建议从小值开始实验。
        
        逻辑：
        1. 计算输入贡献 (不变)。
        2. 准备 logits - 应用温度缩放 (1, hidden_out, hidden_in)。
        3. 使用 Gumbel-Softmax 进行可微采样（重复到批次维度）。
        4. 计算每个输出神经元的贡献：采样权重与 h_prev 的加权和（使用 einsum 高效实现）。
        5. 组合输入贡献并应用 tanh 激活生成 h_next。
        
        这实现了概率数据交换：高温增加低概率连接的选中率，促进探索；低温偏向高概率连接，确保稳定性。
        Gumbel-Softmax 确保采样过程可微，便于端到端训练。
        
        Args:
            x_t (torch.Tensor): 当前输入，shape: (batch_size, input_size)
            h_prev (torch.Tensor): 上一步隐藏状态，shape: (batch_size, hidden_size)
            temperature (float, optional): 覆盖 self.temperature 的动态温度。默认None（使用 self.temperature）。
        
        Returns:
            h_next (torch.Tensor): 新隐藏状态，shape: (batch_size, hidden_size)
        """
        batch_size = h_prev.size(0)
        
        # 使用动态温度，如果提供；否则使用实例温度
        tau = temperature if temperature is not None else self.temperature.item()  # 支持可学习温度
        if tau <= 0:
            # 边缘案例：低温极限，使用 argmax 作为硬采样近似
            hard = True
            tau = 1e-5  # 避免除零，小正值
        else:
            hard = self.use_hard_sampling
        if self.training:
            hard = False  # 训练时强制软采样，确保梯度平滑
        
        # 步骤1: 计算输入到隐藏的贡献 (batch, hidden)
        input_contrib = self.W_ih(x_t)
        
        # 步骤2: 结合固定权重和虚拟权重，并准备 logits
        W_hh_effective = self.W_hh_fixed + self.W_hh_virtual
        logits = (W_hh_effective / tau).unsqueeze(0)
        
        # 步骤3: 矢量化 Gumbel-Softmax 采样 - 重复到批次维度 (batch, hidden_out, hidden_in)
        # 公式: y_i = softmax((log π_i + g_i) / τ), 其中 g_i ~ Gumbel(0,1), τ 是温度
        gumbel_samples = F.gumbel_softmax(logits.repeat(batch_size, 1, 1), tau=tau, hard=hard, dim=2)
        
        # 步骤4: 计算贡献 - einsum 高效矩阵乘法：sum over input neurons (batch, hidden_out)
        if self.hidden_size > 512:
            contrib = torch.matmul(gumbel_samples, h_prev.unsqueeze(2)).squeeze(2)  # 替代以加速大矩阵
        else:
            contrib = torch.einsum('boh,bh->bo', gumbel_samples, h_prev)
        
        # 步骤5: 组合并激活
        h_next = torch.tanh(input_contrib + contrib)

        # --- 新增: 实时赫布更新虚拟权重 ---
        with torch.no_grad():
            # 1. 衰减旧的虚拟权重
            self.W_hh_virtual.mul_(1 - config.FAST_WEIGHT_DECAY)
            
            # 2. 计算赫布更新增量 (外积)
            # h_next是输出激活，h_prev是输入激活
            # (batch, out_features) x (batch, in_features) -> (batch, out_features, in_features)
            hebbian_update = torch.bmm(h_next.unsqueeze(2), h_prev.unsqueeze(1))
            
            # 3. 应用更新 (取批次均值)
            self.W_hh_virtual.add_(hebbian_update.mean(dim=0) * config.FAST_WEIGHT_LR)

        
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
        self.x_t_encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)  # 新增：x_t的独立GRU编码器
        
        # 注意力模块 (保持不变)
        self.attention = Attention(hidden_dim)

        # 核心演化RNN: 被替换为新的可训练“蓄水池”单元
        self.core_rnn = ReservoirRNNCell(hidden_dim + hidden_dim, hidden_dim)  # 更新输入大小：x_t_encoded + attn_context

        self.norm = nn.LayerNorm(hidden_dim)  # 新增：LayerNorm for RNN输出

    def forward(self, x_t, x_ref_encoded, h_prev, temperature=None):
        """
        Args:
            x_t (torch.Tensor): 当前时间步的输入, shape: (batch_size, seq_len, embed_dim)
            x_ref_encoded (torch.Tensor): 经过编码的参照输入, shape: (batch_size, ref_seq_len, hidden_dim)
            h_prev (torch.Tensor): 上一步的隐藏状态, shape: (batch_size, hidden_dim)
            temperature (float, optional): 用于Gumbel-Softmax的温度参数. Defaults to None.
            
        Returns:
            h_next (torch.Tensor): 当前步的输出隐藏状态, shape: (batch_size, hidden_dim)
            attn_context (torch.Tensor): 注意力上下文向量, shape: (batch_size, hidden_dim)
        """
        # 如果当前步没有输入，我们假设输入是一个零向量
        if x_t is None:
            x_t_encoded = torch.zeros(h_prev.size(0), self.hidden_dim, device=h_prev.device)
        else:
            _, x_t_encoded = self.x_t_encoder(x_t)  # 使用GRU编码x_t，获取最后隐藏状态 (1, batch, hidden) -> squeeze to (batch, hidden)
            x_t_encoded = x_t_encoded.squeeze(0)
        
        # 1. 使用注意力机制计算上下文向量
        # h_prev 是 query, x_ref_encoded 是 keys
        attn_context, attn_weights = self.attention(h_prev, x_ref_encoded)
        
        # 2. 将注意力上下文向量和当前输入拼接
        rnn_input = torch.cat((x_t_encoded, attn_context), dim=1)  # (batch, hidden + hidden)

        # 3. 将拼接后的向量输入到核心RNN
        # h_prev shape: (batch_size, hidden_dim) for our cell
        # rnn_input is squeezed to (batch_size, embed_dim + hidden_dim)
        h_next = self.core_rnn(rnn_input, h_prev, temperature=temperature)  # 更新：传入编码后的x_t_encoded和温度参数

        h_next = self.norm(h_next)  # 应用LayerNorm

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

        self.random_sampler = nn.Linear(self.random_pool_dim, self.random_pool_dim)  # 修正: 输出维度应为池大小

        # 拼接后的维度：采样状态 + 注意力上下文
        concatenated_dim = self.sampler_output_dim + context_input_dim

        # 1. 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(concatenated_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 新增：防止过拟合
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 2. 输出网络
        self.output_network = nn.Sequential(
            nn.Linear(concatenated_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 新增：防止过拟合
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

        sampling_logits = self.random_sampler(random_pool)  # 生成采样logits (batch, random_pool_dim)
        sampling_probs = F.softmax(sampling_logits, dim=1)  # 转换为概率

        # 修正：采样 self.num_random 个索引
        rand_indices = torch.multinomial(sampling_probs, self.num_random, replacement=False)  # (batch, num_random)

        # 修正：使用 gather 从池中提取样本
        random_sample = torch.gather(random_pool, 1, rand_indices)  # (batch, num_random)

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

    def forward(self, x_t, x_ref, h_prev, temperature=None):
        """
        一次完整的思考步骤

        Args:
            x_t (torch.Tensor or None): 当前步的输入张量, shape: (batch, seq_len)
            x_ref (torch.Tensor): 参照输入张量, shape: (batch, ref_seq_len)
            h_prev (torch.Tensor): 上一步的隐藏状态, shape: (batch, dynamic_hidden_dim)
            temperature (float, optional): 传递给核心RNN的温度参数. Defaults to None.

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
        h_next, attn_context = self.dynamic_group(x_t_embedded, encoder_outputs, h_prev, temperature=temperature)

        # 4. 将新的隐藏状态和动态上下文向量输入到静态决策头
        gate_pred, output_logits = self.static_head(h_next, attn_context)
        
        return h_next, gate_pred, output_logits

    def reset_virtual_weights(self):
        """
        [新增方法] 重置核心RNN单元中的虚拟权重。
        在处理每个新的、独立的对话样本（或批次）之前调用此方法，
        以防止短期记忆（虚拟权重）从一个样本泄露到下一个样本，确保训练的纯净性。
        """
        self.dynamic_group.core_rnn.W_hh_virtual.fill_(0)



# 这是一个简单的测试，确保我们的类能正常工作
if __name__ == '__main__':
    # --- 测试 DynamicGroup (已有代码) ---
    print("--- 测试 DynamicGroup (新版) ---")
    batch_size = 4  # 使用较小的批处理大小进行测试
    input_dim = config.EMBEDDING_DIM
    hidden_dim = config.DYNAMIC_GROUP_HIDDEN_DIM

    dynamic_group = DynamicGroup(input_dim, hidden_dim)
    x_t_embed = torch.randn(batch_size, 5, input_dim)  # 更新：测试多步x_t
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

    # 新: 测试高级RNN cell
    rnn_cell = ReservoirRNNCell(hidden_dim + hidden_dim, hidden_dim, initial_temperature=1.5, use_hard_sampling=True)  # 更新输入大小
    h_next_test = rnn_cell(x_t_embed.squeeze(1), h_prev, temperature=0.5)  # 测试动态温度
    print(f'高级RNN cell 输出形状: {h_next_test.shape}')
    assert h_next_test.shape == (batch_size, hidden_dim)
    print('高级RNN cell 测试通过！')
