# -*- coding: utf-8 -*-
"""
架构配置：上下文与神经元计划
用法：
- CONTEXT 定义上下文/注意力选择与参数
- NEURON_PLAN 定义动态隐藏状态中不同神经元类型所占的数量或比例

注意：
- 若同时给出 count 与 ratio，优先使用 count。
- ratio 会按 hidden_size 分配并四舍五入，最后一个条目自动补齐差额。
"""

from __future__ import annotations
from typing import List, Dict, Any
import math
import config


# ------------------------------
# 上下文/注意力配置
# ------------------------------
CONTEXT: Dict[str, Any] = {
    # type: "none" | "bahdanau" | "multi_head"
    "type": getattr(config, "ATTENTION_TYPE", "bahdanau"),
    # heads: 0=无上下文（模型内部会回退为历史平均池化），1=Bahdanau，>=2=多头
    "heads": getattr(config, "NUM_ATTENTION_HEADS", 0),
    "dropout": getattr(config, "ATTENTION_DROPOUT", 0.1),
    "use_bias": getattr(config, "USE_ATTENTION_BIAS", True),
    "temperature": getattr(config, "ATTENTION_TEMPERATURE", 1.0),
}


# ------------------------------
# 动态神经元计划
# 示例：混合 3 种内核（按比例）
# ------------------------------
NEURON_PLAN: List[Dict[str, Any]] = [
    {"type": "tanh", "ratio": 0.6},
    {"type": "gru1", "ratio": 0.3, "params": {"gate_bias": 1.0}},
    {"type": "leaky", "ratio": 0.1, "params": {"init_alpha": 0.5}},
]


def resolve_neuron_plan(hidden_size: int, plan: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    """
    将带有 ratio 的计划解析为确定的 count，并确保总数恰为 hidden_size。
    """
    if plan is None:
        plan = NEURON_PLAN

    # 若任何项含有 count，则直接验证总和；否则按 ratio 分配
    has_count = any("count" in item for item in plan)
    if has_count:
        counts = [int(item.get("count", 0)) for item in plan]
        total = sum(counts)
        if total != hidden_size:
            raise ValueError(f"Sum of counts in NEURON_PLAN ({total}) != hidden_size ({hidden_size})")
        return [
            {
                "type": item.get("type", "tanh"),
                "count": int(item.get("count", 0)),
                "params": dict(item.get("params", {})),
            }
            for item in plan
            if int(item.get("count", 0)) > 0
        ]

    # ratio 路径：
    ratios = [float(item.get("ratio", 0.0)) for item in plan]
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        # 兜底：全部用 tanh
        return [{"type": "tanh", "count": hidden_size}]
    # 归一化
    ratios = [r / ratio_sum for r in ratios]
    # 先按四舍五入分配
    counts = [int(round(hidden_size * r)) for r in ratios]
    diff = hidden_size - sum(counts)
    # 调整到精确 hidden_size：优先调整最后一个非零项
    if diff != 0:
        # 找到可调整的索引
        indices = [i for i, c in enumerate(counts) if c > 0] or [len(counts) - 1]
        idx = indices[-1]
        counts[idx] += diff
        if counts[idx] < 0:
            # 极端情况下重新分配
            counts = [0 for _ in counts]
            counts[-1] = hidden_size

    resolved = []
    for item, c in zip(plan, counts):
        if c <= 0:
            continue
        resolved.append({
            "type": item.get("type", "tanh"),
            "count": int(c),
            "params": dict(item.get("params", {})),
        })
    # 再次校验
    if sum(x["count"] for x in resolved) != hidden_size:
        raise AssertionError("Resolved counts do not sum to hidden_size")
    return resolved


# 用于测试发现已注册内核
ALL_NEURON_TYPES = ["tanh", "gru1", "leaky"]


