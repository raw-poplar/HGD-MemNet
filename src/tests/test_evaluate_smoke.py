# -*- coding: utf-8 -*-
"""
对评估流程的冒烟测试 (Smoke Test)
"""

import pytest
import torch
import os
import json
from src import evaluation

# 注意：这个夹具与 test_training_smoke.py 中的非常相似，
# 在一个真实的、大型的项目中，通常会把这种共享的夹具提取到顶层的 conftest.py 文件中。
# 为了保持本示例的清晰和独立，我们在这里暂时重复定义。
@pytest.fixture
def setup_evaluate_environment(tmp_path, monkeypatch):
    """
    搭建一个用于评估测试的隔离环境。
    它会创建虚拟的数据、词汇表和一个预先训练好的、极简的模型。
    """
    # --- 1. 创建临时目录 ---
    processed_path = tmp_path / "lccc_processed"
    valid_path = processed_path / "valid"
    best_model_path = tmp_path / "best_model"
    
    for p in [processed_path, valid_path, best_model_path]:
        p.mkdir()

    # --- 2. 创建虚拟数据 ---
    vocab_data = {
        "name": "test_vocab",
        "word2index": {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "你": 4, "好": 5},
        "index2word": ["<pad>", "<s>", "</s>", "<unk>", "你", "好"],
        "num_words": 6
    }
    with open(processed_path / "vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(vocab_data, f)
        
    # 创建一个虚拟的验证数据块
    x_ref = torch.tensor([[4, 5]]) # "你好"
    x_t = torch.tensor([[4]])      # "你"
    y_t = torch.tensor([[5]])      # "好"
    g_t = torch.tensor([[1.0]])
    dummy_chunk = [(x_ref, [(x_t, y_t, g_t)])]
    torch.save(dummy_chunk, valid_path / "chunk_0.pt")

    # --- 3. 创建一个虚拟的“最佳模型”文件 ---
    # 实例化一个超小模型
    from src.model import HGD_MemNet
    mini_model = HGD_MemNet(
        vocab_size=6, 
        embed_dim=4, 
        dynamic_hidden_dim=4, 
        static_hidden_dim=4
    )
    # 保存它的状态字典
    torch.save({'model_state_dict': mini_model.state_dict()}, best_model_path / "best_model.pth")

    # --- 4. 使用Monkeypatch动态修改config模块的属性 ---
    from src import evaluation
    monkeypatch.setattr(evaluation.config, 'VOCAB_SIZE', 6)
    monkeypatch.setattr(evaluation.config, 'EMBEDDING_DIM', 4)
    monkeypatch.setattr(evaluation.config, 'DYNAMIC_GROUP_HIDDEN_DIM', 4)
    monkeypatch.setattr(evaluation.config, 'STATIC_HEAD_HIDDEN_DIM', 4)
    monkeypatch.setattr(evaluation.config, 'BATCH_SIZE', 1)

    return {
        "valid_path": valid_path,
        "checkpoint_path": best_model_path / "best_model.pth"
    }

def test_evaluate_smoke(setup_evaluate_environment, capsys):
    """
    对主评估函数 evaluate_model 的冒烟测试。
    
    验证评估流程可以被成功调用，并在虚拟数据和模型上无异常地完成。
    同时，使用 capsys 夹具来捕获标准输出，验证评估结果的打印格式是否正确。
    """
    try:
        # 创建一个模拟的 argparse.Namespace 对象
        args = evaluation.argparse.Namespace(
            checkpoint=str(setup_evaluate_environment["checkpoint_path"]),
            data_dir=str(setup_evaluate_environment["valid_path"]),
            batch_size=1,
            lang="zh"
        )
        
        # 运行评估主函数
        evaluation.evaluate_model(args)
        
        # 捕获打印到控制台的输出
        captured = capsys.readouterr()
        
        # 验证输出中是否包含了预期的评估结果信息
        # 基于 evaluation.py 的实际输出进行调整
        assert "--- 开始评估 ---" in captured.out
        assert "已生成 1 条预测。" in captured.out
        assert "--- 评估结果 ---" in captured.out
        
    except Exception as e:
        pytest.fail(f"评估流程 'evaluate.evaluate_model' 抛出异常: {e}")
