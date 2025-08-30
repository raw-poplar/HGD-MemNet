# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

"""
上下文提供者（替代内嵌注意力）
- 提供统一接口：BaseContextProvider
- 现有实现：Attention（Bahdanau）、MultiHeadAttention、NoContext、AvgPoolContext
- 工厂：AttentionFactory（与旧接口保持名义兼容）

说明：为兼容现有测试，保留类名 Attention/MultiHeadAttention，并在 src/model.py 中可直接导入这些符号。
"""


class BaseContextProvider(nn.Module):
    def __init__(self, hidden_dim: int):
        super(BaseContextProvider, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor | None = None):
        raise NotImplementedError


class Attention(BaseContextProvider):
    """
    Bahdanau 注意力（保持旧名以兼容测试）
    - 输入：query: (B, H), keys: (B, L, H), mask: (B, L) True=valid
    - 输出：context: (B, H), attn_weights: (B, L)
    """
    def __init__(self, hidden_dim: int):
        super(Attention, self).__init__(hidden_dim)
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Ua = nn.Linear(hidden_dim, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1)

    def forward(self, query, keys, mask: torch.BoolTensor | None = None):
        scores = self.Va(torch.tanh(self.Wa(query.unsqueeze(1)) + self.Ua(keys))).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=1)
        if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
            if mask is not None:
                with torch.no_grad():
                    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
                    safe = mask.float() / valid_counts
                attn_weights = safe
            else:
                attn_weights = torch.ones_like(scores) / max(1, scores.size(1))
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)
        return context, attn_weights


class MultiHeadAttention(BaseContextProvider):
    """
    多头注意力（单步 query, 序列 keys）
    - 输入：query: (B, H)、keys: (B, L, H)、mask: (B, L)
    - 输出：context: (B, H), attn_weights: (B, num_heads, L)
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = True, temperature: float = 1.0):
        super(MultiHeadAttention, self).__init__(hidden_dim)
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, keys, mask: torch.BoolTensor | None = None):
        batch_size, seq_len, _ = keys.shape
        query = query.unsqueeze(1)
        Q = self.W_q(query)
        K = self.W_k(keys)
        V = self.W_v(keys)
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5 * max(float(self.temperature), 1e-6))
        if mask is not None:
            mask_exp = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(~mask_exp, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
            if mask is not None:
                with torch.no_grad():
                    valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
                    safe = mask.float() / valid_counts
                attn_weights = safe.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, 1, -1)
            else:
                attn_weights = torch.ones_like(attn_weights) / attn_weights.size(-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_dim).squeeze(1)
        context = self.W_o(context)
        context = self.layer_norm(context + query.squeeze(1))
        return context, attn_weights.squeeze(2)


class NoContext(BaseContextProvider):
    """不使用上下文，返回零向量与空权重。"""
    def __init__(self, hidden_dim: int):
        super(NoContext, self).__init__(hidden_dim)

    def forward(self, query, keys, mask: torch.BoolTensor | None = None):
        batch = query.size(0)
        device = query.device
        context = torch.zeros(batch, self.hidden_dim, device=device, dtype=query.dtype)
        attn = None
        return context, attn


class AvgPoolContext(BaseContextProvider):
    """对 keys 做 mask 平均池化作为上下文。"""
    def __init__(self, hidden_dim: int):
        super(AvgPoolContext, self).__init__(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, keys, mask: torch.BoolTensor | None = None):
        if mask is not None:
            lengths = mask.sum(dim=1).clamp(min=1).unsqueeze(1)
            pooled = (keys * mask.unsqueeze(-1)).sum(dim=1) / lengths
        else:
            pooled = torch.mean(keys, dim=1)
        context = self.proj(pooled)
        return context, None


class AttentionFactory:
    """
    注意力/上下文工厂（与旧版接口兼容）
    - num_heads == 0: 返回 None（外部使用 AvgPoolContext 或 NoContext）
    - num_heads == 1: 返回 Attention（Bahdanau）或强制单头 MultiHeadAttention
    - num_heads >= 2: 返回 MultiHeadAttention
    """
    @staticmethod
    def create_attention(hidden_dim, num_heads=1, attention_type="bahdanau", **kwargs):
        if num_heads == 0:
            return None
        elif num_heads == 1:
            if attention_type == "bahdanau":
                return Attention(hidden_dim)
            else:
                return MultiHeadAttention(hidden_dim, 1, **kwargs)
        else:
            return MultiHeadAttention(hidden_dim, num_heads, **kwargs)


def get_context_provider(hidden_dim: int, num_heads: int, attention_type: str = "bahdanau", **kwargs) -> BaseContextProvider | None:
    """统一入口，便于未来扩展不同上下文类型。"""
    return AttentionFactory.create_attention(hidden_dim=hidden_dim, num_heads=num_heads, attention_type=attention_type, **kwargs)


__all__ = [
    "BaseContextProvider",
    "Attention",
    "MultiHeadAttention",
    "NoContext",
    "AvgPoolContext",
    "AttentionFactory",
    "get_context_provider",
]


