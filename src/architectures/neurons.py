# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import config
try:
    import src.arch_config as arch_config
except Exception:
    arch_config = None

"""
可插拔神经元架构与组合蓄水池细胞
- BaseNeuronKernel: 逐通道更新核接口
- 现有内核：TanhKernel、GRU1Kernel、LeakyIntegratorKernel
- CompositeReservoirCell: 复用原 ReservoirRNNCell 的连接选择与统计机制，
  仅将 "非线性/更新公式" 委托给按片段切分的内核，支持在同一隐藏状态中混合多种神经元类型。
"""


# ------------------------------
# 内核注册表
# ------------------------------
NEURON_REGISTRY: dict[str, type[nn.Module]] = {}


def register_neuron(name: str):
    def _decorator(cls):
        NEURON_REGISTRY[name.lower()] = cls
        return cls
    return _decorator


class BaseNeuronKernel(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super(BaseNeuronKernel, self).__init__()
        self.dim = int(dim)

    def forward(self, preact: torch.Tensor, h_prev: torch.Tensor, x_in: torch.Tensor | None = None) -> torch.Tensor:
        """逐通道更新：输入与输出均为 (B, dim)。"""
        raise NotImplementedError


@register_neuron("tanh")
class TanhKernel(BaseNeuronKernel):
    def __init__(self, dim: int, **kwargs):
        super(TanhKernel, self).__init__(dim, **kwargs)

    def forward(self, preact: torch.Tensor, h_prev: torch.Tensor, x_in: torch.Tensor | None = None) -> torch.Tensor:
        return torch.tanh(preact)


@register_neuron("gru1")
class GRU1Kernel(BaseNeuronKernel):
    """
    轻量逐通道门控（非标准GRU，仅以逐通道门 z 融合）：
        z = sigmoid(a*preact + b*h_prev + c)
        h' = (1-z)*h_prev + z*tanh(preact)
    参数量：3*dim
    """
    def __init__(self, dim: int, gate_bias: float = 1.0, **kwargs):
        super(GRU1Kernel, self).__init__(dim, **kwargs)
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.c = nn.Parameter(torch.full((dim,), float(gate_bias)))

    def forward(self, preact: torch.Tensor, h_prev: torch.Tensor, x_in: torch.Tensor | None = None) -> torch.Tensor:
        z = torch.sigmoid(preact * self.a + h_prev * self.b + self.c)
        cand = torch.tanh(preact)
        return (1.0 - z) * h_prev + z * cand


@register_neuron("leaky")
class LeakyIntegratorKernel(BaseNeuronKernel):
    """
    漏斗整合核：
        alpha = sigmoid(p)
        h' = (1-alpha) * h_prev + alpha * tanh(preact)
    参数量：1*dim
    """
    def __init__(self, dim: int, init_alpha: float = 0.5, **kwargs):
        super(LeakyIntegratorKernel, self).__init__(dim, **kwargs)
        init_p = torch.logit(torch.tensor(float(init_alpha)).clamp(1e-4, 1-1e-4))
        self.p = nn.Parameter(torch.full((dim,), float(init_p)))

    def forward(self, preact: torch.Tensor, h_prev: torch.Tensor, x_in: torch.Tensor | None = None) -> torch.Tensor:
        alpha = torch.sigmoid(self.p)
        return (1.0 - alpha) * h_prev + alpha * torch.tanh(preact)


def build_kernels_and_slices(hidden_size: int, neuron_plan: list[dict]) -> list[tuple[slice, BaseNeuronKernel]]:
    """根据 NEURON_PLAN 生成 (切片, 内核) 列表。"""
    kernels: list[tuple[slice, BaseNeuronKernel]] = []
    start = 0
    for item in neuron_plan:
        typ = str(item.get("type", "tanh")).lower()
        count = int(item.get("count", 0))
        params = dict(item.get("params", {}))
        if count <= 0:
            continue
        cls = NEURON_REGISTRY.get(typ)
        if cls is None:
            raise ValueError(f"Unknown neuron kernel type: {typ}")
        sl = slice(start, start + count)
        kernels.append((sl, cls(count, **params)))
        start += count
    if start != hidden_size:
        raise ValueError(f"Sum of counts in NEURON_PLAN ({start}) != hidden_size ({hidden_size})")
    return kernels


class CompositeReservoirCell(nn.Module):
    """
    组合式“蓄水池”RNN单元：
    - 复用可微连接选择（Gumbel-Softmax）、统计、剪枝/再生长；
    - 将逐通道更新交给按片段切分的内核执行；
    - 接口与原 ReservoirRNNCell 保持一致，便于动态组直接替换。
    """
    def __init__(self, input_size: int, hidden_size: int, neuron_plan: list[dict] | None = None,
                 initial_temperature: float = 1.0, use_hard_sampling: bool = False,
                 async_groups: int = 1, async_shuffle: bool = False):
        super(CompositeReservoirCell, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.temperature = nn.Parameter(torch.tensor(float(initial_temperature)))
        self.use_hard_sampling = bool(use_hard_sampling)
        self.async_groups = max(1, int(async_groups))
        self.async_shuffle = bool(async_shuffle)

        self.W_ih = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.W_hh_matrix = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        init.xavier_uniform_(self.W_hh_matrix)

        self.register_buffer("hh_mask", torch.ones(self.hidden_size, self.hidden_size))
        self.register_buffer("hebb_score", torch.zeros(self.hidden_size, self.hidden_size))
        self.register_buffer("usage_score", torch.zeros(self.hidden_size, self.hidden_size))

        # 解析内核计划
        if neuron_plan is None and arch_config is not None and hasattr(arch_config, "resolve_neuron_plan"):
            neuron_plan = arch_config.resolve_neuron_plan(self.hidden_size)
        elif neuron_plan is None:
            neuron_plan = [{"type": "tanh", "count": self.hidden_size}]
        self.kernels_and_slices = build_kernels_and_slices(self.hidden_size, neuron_plan)
        # 将各内核注册为子模块
        for i, (sl, kernel) in enumerate(self.kernels_and_slices):
            self.add_module(f"kernel_{i}_{kernel.__class__.__name__}", kernel)

        self.last_selection_stats = None

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, temperature: float | None = None) -> torch.Tensor:
        batch_size = h_prev.size(0)
        if temperature is not None:
            tau_tensor = torch.as_tensor(temperature, device=h_prev.device, dtype=self.W_hh_matrix.dtype)
        else:
            tau_tensor = self.temperature
        tau_tensor = torch.clamp(tau_tensor, min=max(1e-5, getattr(config, 'CELL_MIN_TAU', 1e-3)))

        hard = False if self.training else self.use_hard_sampling

        input_contrib = self.W_ih(x_t)
        effective = (self.W_hh_matrix * self.hh_mask) / tau_tensor
        logits = effective.unsqueeze(0).repeat(batch_size, 1, 1)
        gumbel_samples = F.gumbel_softmax(logits, tau=1.0, hard=hard, dim=2)
        if torch.isnan(gumbel_samples).any() or torch.isinf(gumbel_samples).any():
            gumbel_samples = torch.softmax(logits, dim=2)

        try:
            row_max = gumbel_samples.max(dim=2).values.mean().detach()
            avg_entropy = (-(gumbel_samples.clamp_min(1e-8) * gumbel_samples.clamp_min(1e-8).log()).sum(dim=2).mean()).detach()
            self.last_selection_stats = {
                'avg_row_max_prob': float(row_max.item()),
                'avg_row_entropy': float(avg_entropy.item()),
            }
        except Exception:
            self.last_selection_stats = None

        # 同步更新（当前实现）：
        if self.hidden_size > 512:
            contrib = torch.matmul(gumbel_samples, h_prev.unsqueeze(2)).squeeze(2)
        else:
            contrib = torch.einsum('boh,bh->bo', gumbel_samples, h_prev)
        preact = input_contrib + contrib

        # 逐片段应用不同内核
        h_next = torch.empty_like(h_prev)
        for sl, kernel in self.kernels_and_slices:
            pre = preact[:, sl]
            hp = h_prev[:, sl]
            h_next[:, sl] = kernel(pre, hp, None)

        if self.training:
            with torch.no_grad():
                pre_abs = h_prev.abs()
                post_abs = h_next.abs()
                hebb_batch = torch.einsum('bo,bh->oh', post_abs, pre_abs) / max(1, h_prev.size(0))
                usage_batch = gumbel_samples.mean(dim=0)
                beta = getattr(config, 'HEBB_EMA_BETA', 0.9)
                self.hebb_score.mul_(beta).add_((1.0 - beta) * hebb_batch)
                self.usage_score.mul_(beta).add_((1.0 - beta) * usage_batch)

        return h_next

    # 与原单元相同的剪枝/再生长接口
    def prune_by_magnitude(self, sparsity_step: float = 0.05, min_keep: int = 4):
        with torch.no_grad():
            W_eff = (self.W_hh_matrix * self.hh_mask).abs()
            active_vals = W_eff[self.hh_mask > 0]
            if active_vals.numel() == 0:
                return
            k = max(1, int(active_vals.numel() * sparsity_step))
            thresh = torch.topk(active_vals, k=k, largest=False).values.max()
            new_mask = self.hh_mask.clone()
            new_mask[(W_eff <= thresh) & (self.hh_mask > 0)] = 0.0
            row_nz = new_mask.sum(dim=1)
            need = (row_nz < min_keep).nonzero(as_tuple=True)[0]
            if need.numel() > 0:
                for i in need.tolist():
                    topk_idx = torch.topk(W_eff[i], k=min_keep).indices
                    new_mask[i].zero_()
                    new_mask[i, topk_idx] = 1.0
            self.hh_mask.copy_(new_mask)

    def regrow_by_hebb(self, per_row: int = 1, init_std: float = 1e-3):
        if per_row <= 0:
            return
        with torch.no_grad():
            pruned = (self.hh_mask == 0)
            if pruned.sum() == 0:
                return
            eps = 1e-8
            hebb = self.hebb_score.clone()
            usage = self.usage_score.clone()
            h_min, _ = hebb.min(dim=1, keepdim=True)
            h_max, _ = hebb.max(dim=1, keepdim=True)
            u_min, _ = usage.min(dim=1, keepdim=True)
            u_max, _ = usage.max(dim=1, keepdim=True)
            hebb_n = (hebb - h_min) / (h_max - h_min + eps)
            usage_n = (usage - u_min) / (u_max - u_min + eps)
            score = hebb_n * usage_n
            score[~pruned] = -1e9
            topk_vals, topk_idx = torch.topk(score, k=per_row, dim=1)
            self.hh_mask.scatter_(1, topk_idx, 1.0)
            noise = torch.randn_like(topk_vals) * init_std
            self.W_hh_matrix.scatter_(1, topk_idx, noise)


__all__ = [
    "BaseNeuronKernel",
    "TanhKernel",
    "GRU1Kernel",
    "LeakyIntegratorKernel",
    "NEURON_REGISTRY",
    "register_neuron",
    "build_kernels_and_slices",
    "CompositeReservoirCell",
]


