# -*- coding: utf-8 -*-
"""
快速单测：验证 ReservoirRNNCell 的剪枝 + 再生长（赫布×使用频度 组合评分）行为。
运行：pytest -q src/tests/test_sparsity_regrowth.py
"""

import torch
import pytest
import os, sys

# 确保仓库根目录在 sys.path 中（便于导入根目录下的 config.py 和 src 包）
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from src.model import ReservoirRNNCell


def _run_cell_steps(cell: ReservoirRNNCell, steps: int = 5, batch_size: int = 4, input_size: int = 8):
    """以训练模式运行若干步，以便累计 usage_score 与 hebb_score。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cell.train()
    cell.to(device)

    h_prev = torch.zeros(batch_size, cell.hidden_size, device=device)
    for _ in range(steps):
        x_t = torch.randn(batch_size, input_size, device=device)
        # 适度温度，避免退化到 argmax
        h_prev = cell(x_t, h_prev, temperature=1.0)
    return cell


def test_prune_and_regrow_combined_score():
    # 构造一个小尺寸的单元，保证测试快速
    input_size = 8
    hidden_size = 16
    cell = ReservoirRNNCell(input_size, hidden_size, initial_temperature=1.0, use_hard_sampling=False)

    # 跑几步以累计 usage/hebb 的 EMA
    _run_cell_steps(cell, steps=6, batch_size=4, input_size=input_size)

    # usage / hebb 应该有非零统计
    assert torch.isfinite(cell.usage_score).all()
    assert torch.isfinite(cell.hebb_score).all()
    assert cell.usage_score.abs().sum().item() > 0.0
    assert cell.hebb_score.abs().sum().item() > 0.0

    # 初始掩码应全为1
    total_edges = hidden_size * hidden_size
    mask_before = cell.hh_mask.clone()
    assert int(mask_before.sum().item()) == total_edges

    # 执行一次小比例剪枝（行保底）
    cell.prune_by_magnitude(sparsity_step=0.2, min_keep=2)
    mask_after_prune = cell.hh_mask.clone()
    nnz_after_prune = int(mask_after_prune.sum().item())
    assert nnz_after_prune < total_edges  # 确实剪掉了一部分

    # 行保底：每行至少 min_keep 个连接
    per_row_nz = mask_after_prune.sum(dim=1)
    assert int(per_row_nz.min().item()) >= 2

    # 记录被剪位置
    pruned_positions = (mask_after_prune == 0).clone()

    # 基于“赫布×使用频度”组合评分再生长
    cell.regrow_by_hebb(per_row=1, init_std=1e-3)
    mask_after_regrow = cell.hh_mask.clone()

    # 新增的连接应来自曾被剪掉的位置
    newly_on = (mask_after_regrow == 1) & (mask_after_prune == 0)
    num_new = int(newly_on.sum().item())
    assert num_new >= 1  # 至少有一条被复活
    assert int((newly_on & pruned_positions).sum().item()) == num_new

    # 复活后非零数应不小于剪枝后
    assert int(mask_after_regrow.sum().item()) >= nnz_after_prune


if __name__ == "__main__":
    # 允许直接运行本文件进行快速验证
    pytest.main([__file__, "-q"]) 

