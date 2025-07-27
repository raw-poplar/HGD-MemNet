import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
import torch
from src.model import HGD_MemNet

def test_model_forward():
    model = HGD_MemNet(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBEDDING_DIM,
        dynamic_hidden_dim=config.DYNAMIC_GROUP_HIDDEN_DIM,
        static_hidden_dim=config.STATIC_HEAD_HIDDEN_DIM
    )
    batch_size = 2
    x_t = torch.randint(0, config.VOCAB_SIZE, (batch_size, 5))
    x_ref = torch.randint(0, config.VOCAB_SIZE, (batch_size, 10))
    h_prev = torch.zeros(batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM)
    h_next, gate, logits = model(x_t, x_ref, h_prev)
    assert h_next.shape == (batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM)
    assert gate.shape == (batch_size, 1)
    assert logits.shape == (batch_size, config.VOCAB_SIZE)

def test_temperature():
    model = HGD_MemNet(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBEDDING_DIM,
        dynamic_hidden_dim=config.DYNAMIC_GROUP_HIDDEN_DIM,
        static_hidden_dim=config.STATIC_HEAD_HIDDEN_DIM
    )
    batch_size = 2
    x_t = torch.randint(0, config.VOCAB_SIZE, (batch_size, 5))
    x_ref = torch.randint(0, config.VOCAB_SIZE, (batch_size, 10))
    h_prev = torch.zeros(batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM)
    h_next, gate, logits = model(x_t, x_ref, h_prev)
    assert h_next.shape == (batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM)
    assert gate.shape == (batch_size, 1)
    assert logits.shape == (batch_size, config.VOCAB_SIZE)
    # 注意：主模型 forward 不直接支持 temperature；如果需要测试温度，请针对内部组件如 ReservoirRNNCell 添加特定测试 