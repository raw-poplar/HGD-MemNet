# -*- coding: utf-8 -*-
"""
测试可插拔神经元架构：确保所有已实现的内核都能在组合细胞中正确工作。
运行：pytest -q src/tests/test_neuron_architectures.py
"""

import pytest
import torch
import config
from src.architectures.neurons import CompositeReservoirCell, NEURON_REGISTRY
import src.arch_config as arch_config
from src.model import DynamicGroup


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_all_registered_kernels_forward():
    hidden_size = min(64, config.DYNAMIC_GROUP_HIDDEN_DIM)
    input_size = hidden_size * 2
    batch_size = 4
    device = _device()

    for name in sorted(NEURON_REGISTRY.keys()):
        plan = [{"type": name, "count": hidden_size}]
        cell = CompositeReservoirCell(input_size, hidden_size, neuron_plan=plan).to(device)
        cell.train()
        x = torch.randn(batch_size, input_size, device=device)
        h_prev = torch.randn(batch_size, hidden_size, device=device)
        h_next = cell(x, h_prev, temperature=1.0)
        assert h_next.shape == (batch_size, hidden_size)
        assert torch.all(h_next <= 1.0 + 1e-5) and torch.all(h_next >= -1.0 - 1e-5)


def test_mixed_plan_counts_and_shapes():
    hidden_size = min(96, config.DYNAMIC_GROUP_HIDDEN_DIM)
    input_size = hidden_size * 2
    batch_size = 2
    device = _device()

    # 使用 arch_config 中的默认计划（ratio），解析为 count
    resolved = arch_config.resolve_neuron_plan(hidden_size)
    assert sum(item["count"] for item in resolved) == hidden_size

    cell = CompositeReservoirCell(input_size, hidden_size, neuron_plan=resolved).to(device)
    x = torch.randn(batch_size, input_size, device=device)
    h_prev = torch.randn(batch_size, hidden_size, device=device)
    with torch.no_grad():
        h_next = cell(x, h_prev, temperature=0.8)
    assert h_next.shape == (batch_size, hidden_size)


def test_dynamic_group_uses_composite_cell():
    dg = DynamicGroup(config.EMBEDDING_DIM, config.DYNAMIC_GROUP_HIDDEN_DIM)
    from src.architectures.neurons import CompositeReservoirCell as _C
    assert isinstance(dg.core_rnn, _C)


