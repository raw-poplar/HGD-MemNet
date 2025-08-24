# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init  # 新增 for xavier
import config

"""
模块概览：HGD-MemNet 主要组件
- Attention / MultiHeadAttention: 对编码的参照序列 x_ref 执行注意力聚合，支持 padding mask。
- DynamicGroup: 编码 x_t 与 x_ref，利用注意力/池化得到上下文，并交给核心 RNN（ReservoirRNNCell）演化隐藏状态。
- ReservoirRNNCell: 通过 Gumbel-Softmax 对隐藏单元的入连接进行可微选择（训练软采样、推理可硬采样）；
  同时在训练时积累 Hebbian/usage 统计以驱动剪枝与再生长。
- StaticHead: 从动态组隐藏状态中做“固定+随机”子采样（训练可用近似可微 Top-k），与上下文拼接后输出门控与 logits。

注意：
- 模型会基于 config.PAD_token 自动构造注意力 mask，避免 PAD 影响注意力分布；
- 训练循环内注入温度退火与（可选）周期性剪枝/再生长，详见 src/train.py。
"""


# --- 注意力模块 ---
class Attention(nn.Module):
    """
    标准的 Bahdanau 注意力机制
    - 输入：
    - query: (B, H) 当前步的查询（通常是上一步隐藏状态 h_prev）
    - keys:  (B, L, H) 编码器输出序列（作为注意力的 key/value）
    - mask:  (B, L) 可选，True 表示有效 token，False 表示 padding；若提供将避免将 padding 纳入 softmax
    - 输出：
    - context: (B, H) 按注意力权重加权后的上下文向量
    - attn_weights: (B, L) 每个时间步的注意力分布
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1)

    def forward(self, query, keys, mask=None):
        """
        Args:
            query (torch.Tensor): 上一步的解码器隐藏状态 (h_prev), shape: (batch_size, hidden_dim)
            keys (torch.Tensor): 编码器的所有输出 (x_ref_encoded), shape: (batch_size, seq_len, hidden_dim)
            mask (torch.BoolTensor, optional): 有效位置为 True 的 padding 掩码，shape: (batch_size, seq_len)

        Returns:
            context (torch.Tensor): 上下文向量, shape: (batch_size, hidden_dim)
            attn_weights (torch.Tensor): 注意力权重, shape: (batch_size, seq_len)
        """
        # query shape: (batch_size, 1, hidden_dim)
        # keys shape: (batch_size, seq_len, hidden_dim)
        scores = self.Va(torch.tanh(self.Wa(query.unsqueeze(1)) + self.Ua(keys)))
        scores = scores.squeeze(2)  # (batch_size, seq_len)

        if mask is not None:
            # 将被mask的位置设为 -inf，softmax 后权重为0
            scores = scores.masked_fill(~mask, float('-inf'))

        # 数值稳定性处理：softmax 前后均做保护
        attn_weights = F.softmax(scores, dim=1)

        # 若出现 NaN/Inf（如整行均为 padding），回退到安全分布
        if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
            if mask is not None:
                # 对每个样本，若有效位置和为0，则使用均匀分布；否则按mask均匀分布
                with torch.no_grad():
                    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
                    safe = mask.float() / valid_counts
                attn_weights = safe
            else:
                attn_weights = torch.ones_like(scores) / max(1, scores.size(1))

        # context shape: (batch_size, 1, hidden_dim)
        context = torch.bmm(attn_weights.unsqueeze(1), keys)
        context = context.squeeze(1)  # (batch_size, hidden_dim)

        return context, attn_weights


# --- 多头注意力模块 ---
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制（单步 query, 序列 keys）
    - 输入：query: (B, H)、keys: (B, L, H)、mask: (B, L)
    - 实现：标准缩放点积注意力 + 残差 + LayerNorm；支持温度与 dropout。
    - 输出：context: (B, H)，attn_weights: (B, num_heads, L)
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.1, use_bias=True, temperature=1.0):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature

        # 线性投影层
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, keys, mask=None):
        """
        Args:
            query (torch.Tensor): shape: (batch_size, hidden_dim)
            keys (torch.Tensor): shape: (batch_size, seq_len, hidden_dim)
            mask (torch.BoolTensor, optional): 有效位置为 True 的 padding 掩码，shape: (batch_size, seq_len)

        Returns:
            context (torch.Tensor): shape: (batch_size, hidden_dim)
            attn_weights (torch.Tensor): shape: (batch_size, num_heads, seq_len)
        """
        batch_size, seq_len, _ = keys.shape

        # 扩展query维度以匹配keys
        query = query.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # 线性投影
        Q = self.W_q(query)  # (batch_size, 1, hidden_dim)
        K = self.W_k(keys)   # (batch_size, seq_len, hidden_dim)
        V = self.W_v(keys)   # (batch_size, seq_len, hidden_dim)

        # 重塑为多头格式
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, 1, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5 * max(float(self.temperature), 1e-6))  # (batch_size, num_heads, 1, seq_len)

        if mask is not None:
            # 扩展 mask 以匹配多头形状: (B, 1, 1, L)
            mask_exp = mask.unsqueeze(1).unsqueeze(1)  # bool
            scores = scores.masked_fill(~mask_exp, float('-inf'))

        # 数值稳定性：softmax 之前处理掩码
        attn_weights = F.softmax(scores, dim=-1)

        # 检查并修复NaN或无效值
        if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
            if mask is not None:
                with torch.no_grad():
                    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
                    safe = mask.float() / valid_counts
                attn_weights = safe.unsqueeze(1)  # (B,1,L) -> 广播到 (B, num_heads, 1, L) 时将在 matmul 前使用
            else:
                attn_weights = torch.ones_like(attn_weights) / attn_weights.size(-1)

        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        context = torch.matmul(attn_weights, V)  # (batch_size, num_heads, 1, head_dim)

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_dim)  # (batch_size, 1, hidden_dim)
        context = context.squeeze(1)  # (batch_size, hidden_dim)

        # 输出投影
        context = self.W_o(context)

        # 残差连接和层归一化
        context = self.layer_norm(context + query.squeeze(1))

        return context, attn_weights.squeeze(2)  # (batch_size, hidden_dim), (batch_size, num_heads, seq_len)


# --- 灵活的注意力工厂 ---
class AttentionFactory:
    """
    注意力机制工厂类，根据配置创建不同类型的注意力机制
    """
    @staticmethod
    def create_attention(hidden_dim, num_heads=1, attention_type="bahdanau", **kwargs):
        """
        创建注意力机制

        Args:
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数量 (0: 无注意力, 1: 单头, >=2: 多头)
            attention_type: 注意力类型
            **kwargs: 其他参数

        Returns:
            注意力模块或None
        """
        if num_heads == 0:
            return None
        elif num_heads == 1:
            if attention_type == "bahdanau":
                return Attention(hidden_dim)
            else:
                # 单头也可以使用多头注意力实现
                return MultiHeadAttention(hidden_dim, 1, **kwargs)
        else:
            return MultiHeadAttention(hidden_dim, num_heads, **kwargs)


# --- 新: 可训练的“蓄水池”式RNN单元 ---
class ReservoirRNNCell(nn.Module):
    """
    一个“蓄水池”式、可训练的 RNN 单元：
    - 通过 Gumbel-Softmax 在每个输出神经元行上对输入隐藏单元进行“软选择/硬采样”；
    - 支持温度 τ（可学习或外部传入）控制探索-利用；
    - 在训练态以 EMA 方式积累 Hebbian 与使用频度统计，供剪枝/再生长策略使用。
    """
    def __init__(self, input_size, hidden_size, initial_temperature=1.0, use_hard_sampling=False):
        """
        Args:
            input_size (int): 输入维度。
            hidden_size (int): 隐藏状态维度（神经元数量）。
            initial_temperature (float): 初始温度参数，控制采样随机性（>1: 更随机；<1: 更确定）。默认1.0。
            use_hard_sampling (bool): 是否在 forward 中使用硬采样。默认 False（训练建议软采样）。
        """
        super(ReservoirRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))  # 可学习温度（也可被 forward 参数覆盖）
        self.use_hard_sampling = use_hard_sampling
        self.W_ih = nn.Linear(input_size, hidden_size, bias=True)
        # 可训练的“隐藏-隐藏”权重矩阵：(hidden_out, hidden_in)，行表示从所有输入隐藏单元聚合到该输出单元的权重。
        self.W_hh_matrix = nn.Parameter(torch.empty(hidden_size, hidden_size))
        init.xavier_uniform_(self.W_hh_matrix)  # Xavier 初始化提高稳定性
        # 剪枝与再生长相关：连接掩码与统计缓冲（赫布分数/使用频度）
        self.register_buffer("hh_mask", torch.ones(hidden_size, hidden_size))
        self.register_buffer("hebb_score", torch.zeros(hidden_size, hidden_size))
        self.register_buffer("usage_score", torch.zeros(hidden_size, hidden_size))  # 记录 gumbel 选择概率的 EMA

    def forward(self, x_t, h_prev, temperature=None):
        """
        使用 Gumbel-Softmax 的可微“连接选择”来进行隐藏状态更新。
        - 训练时默认软采样（hard=False），以获得平滑梯度；
        - 推理/验证可切到硬采样（hard=True）以获得更确定的行为。

        数学背景（每个输出单元 o 对输入隐藏单元 h 的选择）：
        y = softmax((log π + gumbel()) / τ)
        h_o = Σ_h y_{o,h} * h_prev_h
        其中 τ 为温度；τ↓ → 更接近 argmax；τ↑ → 更平滑/高熵。
        """
        batch_size = h_prev.size(0)

        # 使用动态温度（外部优先），并对 τ 做下限裁剪以避免除零
        if temperature is not None:
            tau_tensor = torch.as_tensor(temperature, device=h_prev.device, dtype=self.W_hh_matrix.dtype)
        else:
            tau_tensor = self.temperature
        tau_tensor = torch.clamp(tau_tensor, min=max(1e-5, getattr(config, 'CELL_MIN_TAU', 1e-3)))

        # 训练态强制软采样，有助于稳定反传；推理可允许硬采样
        hard = False if self.training else self.use_hard_sampling

        # 1) 输入到隐藏的贡献
        input_contrib = self.W_ih(x_t)

        # 2) 构造 logits（将 τ 融入缩放），并为批量扩展：shape (B, H_out, H_in)
        effective = (self.W_hh_matrix * self.hh_mask) / tau_tensor
        logits = effective.unsqueeze(0).repeat(batch_size, 1, 1)

        # 3) 基于 Gumbel-Softmax 的“连接选择”权重
        gumbel_samples = F.gumbel_softmax(logits, tau=1.0, hard=hard, dim=2)

        # 数值保护：避免 NaN/Inf
        if torch.isnan(gumbel_samples).any() or torch.isinf(gumbel_samples).any():
            gumbel_samples = torch.softmax(logits, dim=2)

        # 监控：保存最近一次 gumbel 选择的一些统计
        try:
            # 每行（输出单元）的最大选择概率均值、稀疏度估计
            row_max = gumbel_samples.max(dim=2).values.mean().detach()
            avg_entropy = (-(gumbel_samples.clamp_min(1e-8) * gumbel_samples.clamp_min(1e-8).log()).sum(dim=2).mean()).detach()
            self.last_selection_stats = {
                'avg_row_max_prob': float(row_max.item()),
                'avg_row_entropy': float(avg_entropy.item()),
            }
        except Exception:
            self.last_selection_stats = None

        # 4) 聚合上一步隐藏状态：h_prev ∈ (B, H_in)
        if self.hidden_size > 512:
            contrib = torch.matmul(gumbel_samples, h_prev.unsqueeze(2)).squeeze(2)
        else:
            contrib = torch.einsum('boh,bh->bo', gumbel_samples, h_prev)

        # 5) 激活得到 h_next，并积累 Hebbian 统计
        h_preact = input_contrib + contrib
        h_next = torch.tanh(h_preact)

        if self.training:
            with torch.no_grad():
                pre = h_prev.abs()                  # (B, H_in)
                post = h_next.abs()                 # (B, H_out)
                hebb_batch = torch.einsum('bo,bh->oh', post, pre) / max(1, h_prev.size(0))
                usage_batch = gumbel_samples.mean(dim=0)
                beta = getattr(config, 'HEBB_EMA_BETA', 0.9)
                self.hebb_score.mul_(beta).add_((1.0 - beta) * hebb_batch)
                self.usage_score.mul_(beta).add_((1.0 - beta) * usage_batch)

        return h_next

    def prune_by_magnitude(self, sparsity_step=0.05, min_keep=4):
        """按幅值剪枝（每次新增剪除一定比例），保留每行至少 min_keep 个连接。"""
        with torch.no_grad():
            W_eff = (self.W_hh_matrix * self.hh_mask).abs()
            # 只在当前激活的连接里做排序
            active_vals = W_eff[self.hh_mask > 0]
            if active_vals.numel() == 0:
                return
            k = max(1, int(active_vals.numel() * sparsity_step))
            thresh = torch.topk(active_vals, k=k, largest=False).values.max()
            # 先标记剪枝
            new_mask = self.hh_mask.clone()
            new_mask[(W_eff <= thresh) & (self.hh_mask > 0)] = 0.0
            # 行保底
            row_nz = new_mask.sum(dim=1)
            need = (row_nz < min_keep).nonzero(as_tuple=True)[0]
            if need.numel() > 0:
                for i in need.tolist():
                    # 在该行按幅值选择 top min_keep
                    topk_idx = torch.topk(W_eff[i], k=min_keep).indices
                    new_mask[i].zero_()
                    new_mask[i, topk_idx] = 1.0
            self.hh_mask.copy_(new_mask)

    def regrow_by_hebb(self, per_row=1, init_std=1e-3):
        """按组合评分在被剪枝位置里每行再生长若干连接：score = norm(hebb) × norm(usage)。"""
        if per_row <= 0:
            return
        with torch.no_grad():
            pruned = (self.hh_mask == 0)
            if pruned.sum() == 0:
                return
            # 行内 min-max 规范化，避免尺度偏置
            eps = 1e-8
            hebb = self.hebb_score.clone()
            usage = self.usage_score.clone()
            # 对每行做 (x - min)/(max - min + eps)
            h_min, _ = hebb.min(dim=1, keepdim=True)
            h_max, _ = hebb.max(dim=1, keepdim=True)
            u_min, _ = usage.min(dim=1, keepdim=True)
            u_max, _ = usage.max(dim=1, keepdim=True)
            hebb_n = (hebb - h_min) / (h_max - h_min + eps)
            usage_n = (usage - u_min) / (u_max - u_min + eps)
            score = hebb_n * usage_n
            # 只在 pruned 集合上挑分数最高的 per_row 个
            score[~pruned] = -1e9
            topk_vals, topk_idx = torch.topk(score, k=per_row, dim=1)
            self.hh_mask.scatter_(1, topk_idx, 1.0)
            # 初始化新生权重为小值
            noise = torch.randn_like(topk_vals) * init_std
            self.W_hh_matrix.scatter_(1, topk_idx, noise)




# --- 动态神经组 (核心演化模块) ---
class DynamicGroup(nn.Module):
    """
    模型的核心：编码 x_ref、根据 h_prev 与（可选）注意力得到上下文，再与编码后的 x_t 拼接交给核心 RNN。
    - 支持多种注意力机制；无注意力时退化为对历史的 mask 平均池化。
    - 核心 RNN 使用 ReservoirRNNCell（可微采样 + 稀疏演化接口）。
    """
    def __init__(self, embed_dim, hidden_dim, num_attention_heads=1, attention_config=None):
        super(DynamicGroup, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads

        # 编码器: 用于处理参照输入 x_ref (保持不变)
        self.encoder_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.x_t_encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)  # 新增：x_t的独立GRU编码器

        # 灵活的注意力机制
        if attention_config is None:
            attention_config = {}

        self.attention = AttentionFactory.create_attention(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            attention_type=attention_config.get('type', 'bahdanau'),
            dropout=attention_config.get('dropout', 0.1),
            use_bias=attention_config.get('use_bias', True),
            temperature=attention_config.get('temperature', 1.0)
        )

        # 根据是否有注意力机制决定输入大小
        if self.attention is not None:
            core_input_size = hidden_dim + hidden_dim  # x_t_encoded + attn_context
        else:
            # 无注意力版本：使用历史信息的全局表示
            self.history_projector = nn.Linear(hidden_dim, hidden_dim)
            core_input_size = hidden_dim + hidden_dim  # x_t_encoded + history_context

        # 动态反馈投影（可选）
        self.use_feedback = getattr(config, 'USE_DYNAMIC_FEEDBACK', False)
        if self.use_feedback:
            self.feedback_proj = nn.Sequential(
                nn.Linear(6, getattr(config, 'FEEDBACK_EMBED_DIM', 32)),  # 预留6维反馈: [th, gap, gate_p, top1, min_done, t_norm]
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            core_input_size += getattr(config, 'FEEDBACK_EMBED_DIM', 32)

        # 核心演化RNN: 被替换为新的可训练“蓄水池”单元
        self.core_rnn = ReservoirRNNCell(core_input_size, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)  # 新增：LayerNorm for RNN输出

    def forward(self, x_t, x_ref_encoded, h_prev, temperature=None, x_ref_mask=None, feedback_vec=None):
        """
        Args:
            x_t (torch.Tensor): 当前时间步的输入, shape: (batch_size, seq_len, embed_dim)
            x_ref_encoded (torch.Tensor): 经过编码的参照输入, shape: (batch_size, ref_seq_len, hidden_dim)
            h_prev (torch.Tensor): 上一步的隐藏状态, shape: (batch_size, hidden_dim)
            temperature (float, optional): 温度参数，传递给核心RNN
            x_ref_mask (torch.BoolTensor, optional): 参照序列的有效位置掩码，True 表示有效 token，shape: (batch_size, ref_seq_len)

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

        # 1. 计算上下文向量
        if self.attention is not None:
            # 使用注意力机制计算上下文向量
            # h_prev 是 query, x_ref_encoded 是 keys
            context, _ = self.attention(h_prev, x_ref_encoded, mask=x_ref_mask)  # 忽略注意力权重
        else:
            # 无注意力版本：使用历史信息的全局表示
            # 简单的平均池化 + 线性投影
            if x_ref_mask is not None:
                # 避免将 padding 位计入平均
                lengths = x_ref_mask.sum(dim=1).clamp(min=1).unsqueeze(1)  # (B,1)
                history_pooled = (x_ref_encoded * x_ref_mask.unsqueeze(-1)).sum(dim=1) / lengths
            else:
                history_pooled = torch.mean(x_ref_encoded, dim=1)  # (batch_size, hidden_dim)
            context = self.history_projector(history_pooled)  # (batch_size, hidden_dim)

        # 2. 将上下文向量和当前输入拼接（可选拼入反馈）
        rnn_input = torch.cat((x_t_encoded, context), dim=1)  # (batch, hidden + hidden)
        if self.use_feedback:
            try:
                if feedback_vec is not None:
                    fb = feedback_vec.detach() if isinstance(feedback_vec, torch.Tensor) else torch.as_tensor(feedback_vec, device=h_prev.device, dtype=x_t_encoded.dtype)
                    if fb.dim() == 1:
                        fb = fb.unsqueeze(0).expand(x_t_encoded.size(0), -1)
                    fb_emb = self.feedback_proj(fb)
                else:
                    # 无反馈时拼接零嵌入，保证维度一致
                    fb_emb_dim = getattr(config, 'FEEDBACK_EMBED_DIM', 32)
                    fb_emb = torch.zeros(x_t_encoded.size(0), fb_emb_dim, device=h_prev.device, dtype=x_t_encoded.dtype)
                rnn_input = torch.cat([rnn_input, fb_emb], dim=1)
            except Exception:
                # 出错则退化为不使用反馈
                pass

        # 3. 将拼接后的向量输入到核心RNN
        # h_prev shape: (batch_size, hidden_dim) for our cell
        # rnn_input is squeezed to (batch_size, embed_dim + hidden_dim)
        h_next = self.core_rnn(rnn_input, h_prev, temperature)  # 更新：传入编码后的x_t_encoded和温度

        h_next = self.norm(h_next)  # 应用LayerNorm

        return h_next, context


# --- 静态网络对 (决策模块) ---
class StaticHead(nn.Module):
    """
    静态网络对（决策前端）：从动态组隐藏状态中抽取“固定 + 随机”子集，再与上下文拼接，输出门控与分类 logits。
    - 固定采样：取前 num_fixed 个维度（可视作保留主干单元）。
    - 随机采样：在剩余维度中按学习到的 logits 选择 top‑k（训练可用近似可微 Top‑k，推理用硬采样）。
    - 评分增强：可选将注意力上下文拼入评分器（use_contextual_sampler），提升“在何种上下文下采哪类单元”的表达。
    """
    def __init__(self, hidden_dim, sampler_input_dim, context_input_dim, output_dim, fixed_ratio, random_ratio, use_soft_topk_training=True, use_contextual_sampler=True):
        super(StaticHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_soft_topk_training = use_soft_topk_training
        self.use_contextual_sampler = use_contextual_sampler

        # 计算采样数量
        self.num_fixed = int(sampler_input_dim * fixed_ratio)
        self.num_random = int(sampler_input_dim * random_ratio)
        self.sampler_output_dim = self.num_fixed + self.num_random

        # 确定用于随机采样的神经元池的维度
        self.random_pool_dim = sampler_input_dim - self.num_fixed
        if self.random_pool_dim < self.num_random:
            raise ValueError("Random sampling pool is smaller than the number of items to sample.")

        # 采样评分器：可选择上下文增强
        if self.use_contextual_sampler:
            # 使用上下文 (context) + 剩余状态作为评分输入
            self.sampler_scorer = nn.Sequential(
                nn.Linear(self.random_pool_dim + context_input_dim, max(64, self.random_pool_dim // 2)),
                nn.ReLU(),
                nn.Linear(max(64, self.random_pool_dim // 2), self.random_pool_dim)
            )
        else:
            self.random_sampler = nn.Linear(self.random_pool_dim, self.random_pool_dim)

        # 拼接后的维度：采样状态 + 注意力上下文
        concatenated_dim = self.sampler_output_dim + context_input_dim

        # 1. 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(concatenated_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 新增：防止过拟合
            nn.Linear(hidden_dim, 1),
        )  # 输出为logits，损失中使用 BCEWithLogitsLoss
        # 门控logits归一化，避免过大：可选 LayerNorm
        self.gate_norm = nn.LayerNorm(1, elementwise_affine=False)

        # 2. 输出网络
        self.output_network = nn.Sequential(
            nn.Linear(concatenated_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 新增：防止过拟合
            nn.Linear(hidden_dim, output_dim)
        )

        # 监控：最近一次采样统计（由 forward 填充）
        self.last_sampling_stats = None

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

        # 评分器：可上下文增强
        if getattr(self, 'sampler_scorer', None) is not None:
            scorer_in = torch.cat([random_pool, attn_context], dim=1)
            sampling_logits = self.sampler_scorer(scorer_in)
        else:
            sampling_logits = self.random_sampler(random_pool)

        # 近似可微的Top-k选择：使用Gumbel-Softmax扰动后取top-k索引（推理可用硬采样）
        if self.training and self.use_soft_topk_training:
            # 训练时：使用Gumbel噪声 + softmax 权重化的“软Top-k”聚合
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(sampling_logits).clamp(min=1e-9)))
            perturbed = sampling_logits + gumbel_noise
            weights = F.softmax(perturbed, dim=1)  # (batch, random_pool_dim)
            # 选出top-k权重并进行加权求和（保持维度 batch x num_random）
            topk_vals, topk_idx = torch.topk(weights, k=self.num_random, dim=1)
            # 规范化top-k权重
            topk_weights = topk_vals / (topk_vals.sum(dim=1, keepdim=True) + 1e-9)
            # 从random_pool中gather对应值，然后按权重线性组合（仍输出 batch x num_random）
            topk_samples = torch.gather(random_pool, 1, topk_idx)
            # 为了与原先 concat 后的维度兼容，我们保留这 num_random 个通道（权重仅用于梯度引导）
            random_sample = topk_samples * topk_weights

            # 监控：采样统计
            try:
                avg_topk_weight = topk_weights.mean().detach()
                max_per_sample = topk_vals.max(dim=1).values.mean().detach()
                # 批级覆盖率：本批 top-k 索引的全体去重比例
                coverage = (torch.unique(topk_idx).numel() / max(1, self.random_pool_dim))
                self.last_sampling_stats = {
                    'avg_topk_weight': float(avg_topk_weight.item()),
                    'avg_topk_max': float(max_per_sample.item()),
                    'coverage_ratio': float(coverage),
                }
            except Exception:
                self.last_sampling_stats = None
        else:
            # 推理时：使用多项式抽样（硬采样）
            sampling_probs = F.softmax(sampling_logits, dim=1)
            sampling_probs = torch.clamp(sampling_probs, min=1e-8, max=1.0)
            sampling_probs = sampling_probs / sampling_probs.sum(dim=1, keepdim=True)
            rand_indices = torch.multinomial(sampling_probs, self.num_random, replacement=False)
            random_sample = torch.gather(random_pool, 1, rand_indices)

            # 监控：采样统计（推理）
            try:
                coverage = (torch.unique(rand_indices).numel() / max(1, self.random_pool_dim))
                self.last_sampling_stats = {
                    'avg_topk_weight': None,
                    'avg_topk_max': None,
                    'coverage_ratio': float(coverage),
                }
            except Exception:
                self.last_sampling_stats = None

        # 3. 拼接采样结果
        sampled_state = torch.cat((fixed_sample, random_sample), dim=1)

        # 4. 将采样状态和注意力上下文拼接起来
        combined_input = torch.cat((sampled_state, attn_context), dim=1)

        gate_signal = self.gate_network(combined_input)
        try:
            if getattr(config, 'GATE_LOGIT_NORM', True):
                gate_signal = self.gate_norm(gate_signal)
        except Exception:
            pass
        output_logits = self.output_network(combined_input)

        return gate_signal, output_logits


class HGD_MemNet(nn.Module):
    """
    分层门控动态记忆网络 (Hierarchical Gated-Dynamic Memory Network)
    整合了所有模块
    """
    def __init__(self, vocab_size, embed_dim, dynamic_hidden_dim, static_hidden_dim,
                 num_attention_heads=None, attention_config=None):
        super(HGD_MemNet, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # 如果没有指定注意力头数，从config中获取
        if num_attention_heads is None:
            num_attention_heads = config.NUM_ATTENTION_HEADS

        # 构建注意力配置
        if attention_config is None:
            attention_config = {
                'type': config.ATTENTION_TYPE,
                'dropout': config.ATTENTION_DROPOUT,
                'use_bias': config.USE_ATTENTION_BIAS,
                'temperature': config.ATTENTION_TEMPERATURE
            }

        self.dynamic_group = DynamicGroup(
            embed_dim,
            dynamic_hidden_dim,
            num_attention_heads=num_attention_heads,
            attention_config=attention_config
        )

        self.static_head = StaticHead(
            hidden_dim=static_hidden_dim,
            sampler_input_dim=dynamic_hidden_dim,
            context_input_dim=dynamic_hidden_dim,
            output_dim=vocab_size,
            fixed_ratio=config.FIXED_SAMPLING_RATIO,
            random_ratio=config.RANDOM_SAMPLING_RATIO,
            use_soft_topk_training=getattr(config, 'USE_SOFT_TOPK_TRAINING', True),
            use_contextual_sampler=getattr(config, 'USE_CONTEXTUAL_SAMPLER', True)
        )

        # 序列级 CE（可选）：简单 Teacher-Forcing 解码器
        self.use_sequence_ce = getattr(config, 'USE_SEQUENCE_CE', False)
        if self.use_sequence_ce:
            self.seq_decoder = nn.GRU(embed_dim, dynamic_hidden_dim, batch_first=True)
            self.seq_out = nn.Linear(dynamic_hidden_dim, vocab_size)

    def forward(self, x_t, x_ref, h_prev, temperature=None, control=None, feedback_vec=None, x_ref_encoded=None, x_ref_mask=None):
        """
        一次完整的思考步骤

        Args:
            x_t (torch.Tensor or None): 当前步的输入张量, shape: (batch, seq_len)
            x_ref (torch.Tensor): 参照输入张量, shape: (batch, ref_seq_len)
            h_prev (torch.Tensor): 上一步的隐藏状态, shape: (batch, dynamic_hidden_dim)
            temperature (float, optional): 温度参数，传递给动态组的核心RNN
            control (torch.Tensor, optional): (batch, C) 控制向量，如 [t_norm, remain_norm, min_done, target_speak_ratio]
            feedback_vec (torch.Tensor, optional): (batch, 6) 动态反馈（阈值与上一步门控/输出摘要），仅用于下一步演化

        Returns:
            h_next (torch.Tensor): 下一步的隐藏状态
            gate_pred (torch.Tensor): 门控预测值
            output_logits (torch.Tensor): 输出预测的logits
        """
        # 1. 嵌入输入
        # 如果 x_t 存在，进行嵌入；否则为 None
        x_t_embedded = self.embed(x_t) if x_t is not None else None
        if x_ref_encoded is None:
            # 正常路径：需要嵌入并编码 x_ref
            x_ref_embedded = self.embed(x_ref)
            # 1.1 构建 padding 掩码：True=有效，False=padding
            x_ref_mask = (x_ref != config.PAD_token) if x_ref is not None else None
            # 2. 编码参照输入 x_ref 来为注意力机制和上下文向量做准备
            # encoder_outputs shape: (batch_size, ref_seq_len, dynamic_hidden_dim)
            encoder_outputs, _ = self.dynamic_group.encoder_rnn(x_ref_embedded)
        else:
            # 复用外部预编码结果（典型用于验证/测试加速）
            encoder_outputs = x_ref_encoded

        # 3. 通过动态组进行一步演化（可选传入反馈）
        h_next, attn_context = self.dynamic_group(
            x_t_embedded, encoder_outputs, h_prev, temperature, x_ref_mask=x_ref_mask, feedback_vec=feedback_vec
        )

        # 4. 将新的隐藏状态和动态上下文向量输入到静态决策头
        gate_pred, output_logits = self.static_head(h_next, attn_context)
        # 可选：门控logits温度，降低饱和（>1 更保守）
        try:
            g_temp = float(getattr(config, 'GATE_LOGIT_TEMPERATURE', 1.0) or 1.0)
            if g_temp > 0 and g_temp != 1.0:
                gate_pred = gate_pred / g_temp
        except Exception:
            pass

        # 4.1 若启用序列级 CE，则用简单 GRU 解码器进行 Teacher-Forcing，输出 (B, L, V)
        seq_logits = None
        if self.use_sequence_ce and (x_t_embedded is not None):
            # 使用 x_t 的嵌入作为 Teacher-Forcing 输入
            dec_h0 = h_next.unsqueeze(0)  # (1, B, H)
            dec_out, _ = self.seq_decoder(x_t_embedded, dec_h0)  # (B, L, H)
            seq_logits = self.seq_out(dec_out)         # (B, L, V)

        # 5. 可选：基于时间进度的门控“时间偏置”与微调
        #  - 目标：t < MIN 时抑制发声，t >= MIN 后随时间递增，加速靠近 MAX 时的发声倾向
        if control is not None:
            try:
                # control = [t_norm, remain_norm, min_done, budget]
                t_norm = control[:, 0:1]
                remain_norm = control[:, 1:2]
                min_done_flag = control[:, 2:3]
                # 时间偏置（作用在 logits 上）：bias = strength * (min_weight*(min_done-0.5) + (t_norm^gamma))
                if getattr(config, 'GATE_TIME_BIAS_ENABLE', True):
                    strength = float(getattr(config, 'GATE_TIME_BIAS_STRENGTH', 2.0) or 0.0)
                    gamma = float(getattr(config, 'GATE_TIME_BIAS_GAMMA', 2.0) or 1.0)
                    min_weight = float(getattr(config, 'GATE_TIME_BIAS_MIN_WEIGHT', 1.0) or 0.0)
                    if strength != 0.0:
                        p = torch.clamp(torch.nan_to_num(t_norm, nan=0.0), 0.0, 1.0)
                        md = torch.clamp(torch.nan_to_num(min_done_flag, nan=0.0), 0.0, 1.0)
                        time_term = torch.pow(p, gamma)
                        min_term = (md - 0.5) * min_weight
                        bias = strength * (time_term + min_term)
                        gate_pred = gate_pred + bias
                        # 偏置加入后对 logits 做幅度裁剪，避免数值饱和到 0/1 概率
                        try:
                            clamp_v = float(getattr(config, 'GATE_POST_BIAS_CLAMP', 0.0) or 0.0)
                            if clamp_v and clamp_v > 0:
                                gate_pred = torch.clamp(gate_pred, -clamp_v, clamp_v)
                        except Exception:
                            pass

                # 额外：Sigmoid 型时间引导（将 gate 概率往目标时间曲线做凸组合）
                try:
                    guide_alpha = float(getattr(config, 'GATE_TIME_GUIDE_ALPHA', 0.0) or 0.0)
                    if guide_alpha > 0:
                        # 计算当前归一化步位置 p，并构建分段 sigmoid 目标曲线
                        # p_min: MIN/denom, p_max: MAX/denom, p_safe: SAFETY/denom
                        denom = torch.clamp(remain_norm + t_norm, min=1e-6)  # 之前构造的 denom: cap+extra
                        p = torch.clamp(t_norm, 0.0, 1.0)
                        # 标量配置
                        k1 = float(getattr(config, 'GATE_PRE_MIN_K', 6.0))
                        k2 = float(getattr(config, 'GATE_MID_K', 12.0))
                        k3 = float(getattr(config, 'GATE_POST_MAX_K', 6.0))
                        mid_lo = float(getattr(config, 'GATE_MID_LOW', 0.1))
                        mid_hi = float(getattr(config, 'GATE_MID_HIGH', 0.8))
                        MIN_T = float(getattr(config, 'MIN_THINKING_STEPS', 0) or 0)
                        MAX_T = float(getattr(config, 'MAX_THINKING_STEPS', 0) or 0)
                        SAFE_T = float(getattr(config, 'SAFETY_MAX_THINKING_STEPS', 0) or 0)
                        # 将阈值映射到 [0,1] 空间，避免除0
                        p_min = torch.clamp(torch.tensor(MIN_T, device=gate_pred.device) / torch.tensor((MAX_T if MAX_T>0 else MIN_T)+ (getattr(config,'GATE_TIME_RAMP_EXTRA_STEPS',0) or 0), device=gate_pred.device).clamp(min=1.0), 0.0, 1.0)
                        p_max = torch.clamp(torch.tensor(MAX_T if MAX_T>0 else MIN_T, device=gate_pred.device) / torch.tensor((MAX_T if MAX_T>0 else MIN_T)+ (getattr(config,'GATE_TIME_RAMP_EXTRA_STEPS',0) or 0), device=gate_pred.device).clamp(min=1.0), 0.0, 1.0)
                        p_safe = torch.clamp(torch.tensor(SAFE_T if SAFE_T>0 else MAX_T, device=gate_pred.device) / torch.tensor((MAX_T if MAX_T>0 else SAFE_T)+ (getattr(config,'GATE_TIME_RAMP_EXTRA_STEPS',0) or 0), device=gate_pred.device).clamp(min=1.0), 0.0, 1.0)

                        def _sigmoid(x, k, x0):
                            return 1.0 / (1.0 + torch.exp(-k * (x - x0)))

                        # 三段式：
                        # 1) [0, p_min): 缓慢上升到 mid_lo
                        pre = mid_lo * _sigmoid(p, k1, p_min * 0.9)
                        # 2) [p_min, p_max]: 快速从 mid_lo 上升到 mid_hi
                        mid = mid_lo + (mid_hi - mid_lo) * _sigmoid(p, k2, (p_min + p_max) * 0.5)
                        # 3) (p_max, p_safe]: 缓慢从 mid_hi 上升到 1.0
                        post = mid_hi + (1.0 - mid_hi) * _sigmoid(p, k3, (p_max + p_safe) * 0.5)
                        g_target = torch.where(p < p_min, pre, torch.where(p <= p_max, mid, post))
                        # 安全步及以后目标为 1.0（作为引导）
                        g_target = torch.where(p >= p_safe, torch.ones_like(g_target), g_target)
                        # 将 gate_pred 的概率与目标做凸组合
                        g_prob = torch.sigmoid(gate_pred)
                        # 分段引导权重：MIN前弱引导，MIN~MAX 强引导，MAX~SAFETY 中等引导
                        base = guide_alpha
                        w_pre = base * float(getattr(config, 'GATE_GUIDE_ALPHA_PRE', 0.5))
                        w_mid = base * float(getattr(config, 'GATE_GUIDE_ALPHA_MID', 1.0))
                        w_post = base * float(getattr(config, 'GATE_GUIDE_ALPHA_POST', 0.6))
                        w = torch.where(p < p_min, torch.full_like(g_prob, w_pre), torch.where(p <= p_max, torch.full_like(g_prob, w_mid), torch.full_like(g_prob, w_post)))
                        g_prob = (1 - w) * g_prob + w * g_target
                        # 回到 logits 空间（数值安全）
                        g_prob = torch.clamp(g_prob, 1e-6, 1 - 1e-6)
                        gate_pred = torch.log(g_prob) - torch.log1p(-g_prob)

                        # 安全兜底：到达 SAFETY_MAX_THINKING_STEPS 时将 logits 直接推到大正值，确保 gate_p≈1
                        try:
                            force_logit = float(getattr(config, 'GATE_SAFETY_FORCE_LOGIT', 0.0) or 0.0)
                            if force_logit > 0:
                                gate_pred = torch.where(p >= p_safe, torch.full_like(gate_pred, force_logit), gate_pred)
                        except Exception:
                            pass
                except Exception:
                    pass

                # 兼容旧的概率级微调（可选）
                alpha = getattr(config, 'CONTROL_GATE_ALPHA', 0.0)
                if alpha and alpha > 0:
                    c = torch.sigmoid(torch.nan_to_num(control, nan=0.0))
                    if c.dim() == 2 and c.size(0) == gate_pred.size(0):
                        c_val = c.mean(dim=1, keepdim=True)
                    else:
                        c_val = torch.sigmoid(torch.tensor(0.0, device=gate_pred.device)).expand_as(gate_pred)
                    gate_pred = (1 - alpha) * gate_pred + alpha * c_val
            except Exception:
                pass

        # 返回序列级 logits（若启用）以便 compute_loss 使用
        if self.use_sequence_ce:
            return h_next, gate_pred, (seq_logits if seq_logits is not None else output_logits)
        return h_next, gate_pred, output_logits


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
    dummy_input = torch.randn(batch_size, hidden_dim + hidden_dim)
    h_next_test = rnn_cell(dummy_input, h_prev, temperature=0.5)  # 测试动态温度
    print(f'高级RNN cell 输出形状: {h_next_test.shape}')
    assert h_next_test.shape == (batch_size, hidden_dim)
    print('高级RNN cell 测试通过！')