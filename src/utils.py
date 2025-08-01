import torch
import os
import glob
import config
from .model import HGD_MemNet

def load_model_from_checkpoint(directory, device):
    """从指定目录加载最新的模型检查点。"""
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在，无法加载检查点。")
        return None
    checkpoints = glob.glob(os.path.join(directory, "checkpoint_*.pth"))
    if not checkpoints:
        print("未找到检查点文件。")
        return None
    latest_checkpoint_path = max(checkpoints, key=os.path.getmtime)
    print(f"正在从 {latest_checkpoint_path} 加载模型...")
    checkpoint = torch.load(latest_checkpoint_path, map_location=device)
    model = HGD_MemNet(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBEDDING_DIM,
        dynamic_hidden_dim=config.DYNAMIC_GROUP_HIDDEN_DIM,
        static_hidden_dim=config.STATIC_HEAD_HIDDEN_DIM
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("模型加载完成。")
    return model

def compute_loss(gate_pred, gate_target, output_logits, target_padded):
    """计算单个步骤的损失。"""
    import torch.nn as nn

    # 修正：确保gate_pred在有效范围内，避免log(0)或log(1)
    gate_pred_clamped = torch.clamp(gate_pred, min=1e-7, max=1-1e-7)
    gate_pred_logits = torch.log(gate_pred_clamped / (1.0 - gate_pred_clamped))

    # 修正：确保gate_target的维度匹配
    if gate_target.dim() == 1:
        gate_target = gate_target.unsqueeze(1)

    loss = nn.BCEWithLogitsLoss()(gate_pred_logits, gate_target)

    if target_padded is not None and target_padded.numel() > 0:
        output_loss = nn.CrossEntropyLoss(ignore_index=config.PAD_token)(
            output_logits.view(-1, config.VOCAB_SIZE),
            target_padded.view(-1)
        )
        if not torch.isnan(output_loss):
            loss += output_loss
    return loss