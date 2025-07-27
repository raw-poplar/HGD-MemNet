import pytest
import torch
import config
from ..model import HGD_MemNet

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
    h_next, gate, logits = model(x_t, x_ref, h_prev, temperature=1.0)
    assert h_next.shape == (batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM)
    assert gate.shape == (batch_size, 1)
    assert logits.shape == (batch_size, config.VOCAB_SIZE)

def test_temperature():
    # 测试温度影响（简单检查无错误）
    model = HGD_MemNet(...)  # 简化
    # ... 类似以上 