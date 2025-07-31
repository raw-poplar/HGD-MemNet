import pytest
from unittest.mock import patch
import torch
import os
import json
from src import chat
import config

@pytest.fixture
def setup_chat_environment(tmp_path, monkeypatch):
    """一个用于聊天测试的夹具"""
    processed_path = tmp_path / "lccc_processed"
    processed_path.mkdir()
    best_model_path = tmp_path / "best_model"
    best_model_path.mkdir()

    vocab_data = {
        "name": "test_vocab",
        "word2index": {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "你": 4, "好": 5},
        "index2word": { "0": "<pad>", "1": "<s>", "2": "</s>", "3": "<unk>", "4": "你", "5": "好" },
        "num_words": 6
    }
    vocab_path = processed_path / "vocabulary.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f)

    from src.model import HGD_MemNet
    mini_model = HGD_MemNet(
        vocab_size=6, embed_dim=4, dynamic_hidden_dim=4, static_hidden_dim=4
    )
    model_path = best_model_path / "checkpoint_0.pth"
    torch.save({'model_state_dict': mini_model.state_dict()}, model_path)

    monkeypatch.setattr(config, 'LCCC_PROCESSED_PATH', str(processed_path))
    monkeypatch.setattr(config, 'BEST_MODEL_DIR', str(best_model_path))
    monkeypatch.setattr(config, 'VOCAB_SIZE', 6)
    monkeypatch.setattr(config, 'EMBEDDING_DIM', 4)
    monkeypatch.setattr(config, 'DYNAMIC_GROUP_HIDDEN_DIM', 4)
    monkeypatch.setattr(config, 'STATIC_HEAD_HIDDEN_DIM', 4)

def test_chat_response_smoke(setup_chat_environment):
    """对核心对话函数 generate_response 的冒烟测试"""
    try:
        model = chat.load_model_for_chat()
        vocab = chat.load_vocabulary()
        assert model is not None, "模型加载失败"
        assert vocab is not None, "词汇表加载失败"

        user_input = "你好"
        history = []
        h_prev = torch.zeros(1, config.DYNAMIC_GROUP_HIDDEN_DIM)

        response, new_history, h_next = chat.generate_response(model, vocab, user_input, history, h_prev)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert len(new_history) == 2
        assert new_history[0] == user_input
        assert new_history[1] == response

    except Exception as e:
        pytest.fail(f"交互对话流程 'generate_response' 抛出异常: {e}")

@patch('builtins.input', side_effect=['你好', 'exit'])
def test_chat_main_loop_smoke(mock_input, setup_chat_environment, capsys):
    """对主聊天循环 `main` 的冒烟测试"""
    try:
        chat.main()
        captured = capsys.readouterr()
        assert "模型加载成功" in captured.out
        assert "您:" in captured.out
        assert "模型:" in captured.out
        assert "再见!" in captured.out

    except SystemExit:
        pass
    except Exception as e:
        pytest.fail(f"主聊天循环 'main' 抛出异常: {e}")
