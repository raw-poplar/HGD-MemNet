# -*- coding: utf-8 -*-
"""
对训练流程的冒烟测试 (Smoke Test)
"""

import pytest
import torch
import os
import json
from src import train

@pytest.fixture
def setup_test_environment(tmp_path, monkeypatch):
    """
    一个用于测试的Pytest夹具，功能是搭建一个完整的、隔离的测试环境。
    它会执行以下操作:
    1. 创建临时的文件夹用于存放数据和模型。
    2. 生成一个迷你的虚拟词汇表。
    3. 创建一个包含单一样本的虚拟训练数据块。
    4. 使用 monkeypatch 来动态修改config模块中的参数，使其指向临时目录并使用超小型模型配置。
    """
    # --- 1. 创建所有需要的临时目录 ---
    processed_path = tmp_path / "lccc_processed"
    train_path = processed_path / "train"
    valid_path = processed_path / "valid"
    checkpoint_path = tmp_path / "checkpoints"
    best_model_path = tmp_path / "best_model"
    
    for p in [processed_path, train_path, valid_path, checkpoint_path, best_model_path]:
        p.mkdir()

    # --- 2. 创建虚拟数据 ---
    # 虚拟词汇表
    vocab_data = {
        "word2index": {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "你": 4, "好": 5},
        "index2word": ["<pad>", "<s>", "</s>", "<unk>", "你", "好"],
        "num_words": 6
    }
    with open(processed_path / "vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(vocab_data, f)
        
    # 虚拟训练数据块
    # 数据格式为: (参照上下文, [ (当前输入, 目标输出, 门控目标), ... ])
    x_ref = torch.tensor([4, 5]) # "你好"
    x_t = torch.tensor([4])      # "你"
    y_t = torch.tensor([5])      # "好"
    g_t = torch.tensor([1.0])    # 门控目标为1 (生成回应)
    
    # 一个批次，只包含一个样本
    dummy_chunk = [(x_ref, [(x_t, y_t, g_t)])]
    torch.save(dummy_chunk, train_path / "chunk_0.pt")
    
    # 创建一个空的验证文件块，以防止验证函数报错
    torch.save([], valid_path / "chunk_0.pt")

    # --- 3. 使用Monkeypatch动态修改config模块的属性 ---
    # 将训练脚本中引用的config变量全部指向我们的临时测试环境
    monkeypatch.setattr(train.config, 'LCCC_PROCESSED_PATH', str(processed_path))
    monkeypatch.setattr(train.config, 'VOCAB_SIZE', 6)
    monkeypatch.setattr(train.config, 'EMBEDDING_DIM', 4)
    monkeypatch.setattr(train.config, 'DYNAMIC_GROUP_HIDDEN_DIM', 4)
    monkeypatch.setattr(train.config, 'STATIC_HEAD_HIDDEN_DIM', 4)
    monkeypatch.setattr(train.config, 'BATCH_SIZE', 1)
    monkeypatch.setattr(train.config, 'NUM_EPOCHS', 1)
    # 设置一个很大的步数，确保在单一样本的测试中不会触发验证和保存
    monkeypatch.setattr(train.config, 'VALIDATE_EVERY_N_STEPS', 999)
    monkeypatch.setattr(train.config, 'SAVE_CHECKPOINT_EVERY_N_BATCHES', 999)
    monkeypatch.setattr(train.config, 'CHECKPOINT_DIR', str(checkpoint_path))
    monkeypatch.setattr(train.config, 'BEST_MODEL_DIR', str(best_model_path))
    monkeypatch.setattr(train.config, 'FIXED_SAMPLING_RATIO', 0.5)
    monkeypatch.setattr(train.config, 'RANDOM_SAMPLING_RATIO', 0.5)

def test_train_smoke(setup_test_environment):
    """
    对主训练函数 train_model 的冒烟测试。
    
    该测试的核心目的是验证训练流程可以被成功调用，并能在最小化的虚拟数据上
    无异常地完成一个训练步骤。它依赖 `setup_test_environment` 夹具来保证测试环境的纯净。
    """
    try:
        # 运行训练主函数，它将使用被我们动态修改过的配置
        train.train_model()
    except Exception as e:
        # 如果在训练过程中出现任何异常，测试失败
        pytest.fail(f"训练流程 'train.train_model' 抛出异常: {e}")
