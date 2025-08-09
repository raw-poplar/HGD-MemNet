import torch
import os
import glob
import random
import numpy as np
import config
from .model import HGD_MemNet
from typing import Optional

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



def compute_loss(
    gate_pred,
    gate_target,
    output_logits,
    target_padded,
    *,
    gate_entropy_weight: float = None,
    target_speak_ratio: float = None,
    h_prev: Optional[torch.Tensor] = None,
    h_next: Optional[torch.Tensor] = None,
    think_loss_weight: float = None,
    info_nce_tau: float = None,
):
    """计算单个步骤的损失（向后兼容）。
    新增：
      - 门控熵正则（gate_entropy_weight）
      - 发声预算约束（target_speak_ratio）
      - 思考信息量损失（InfoNCE，对 h_next vs h_prev）
    """
    import torch.nn as nn
    import torch.nn.functional as F

    loss = 0.0

    # 门控损失（若提供标签）
    if gate_target is not None:
        gate_loss = nn.BCELoss()(gate_pred, gate_target)
        loss = loss + gate_loss

    # 输出语言损失
    if target_padded is not None:
        output_loss = nn.CrossEntropyLoss(ignore_index=config.PAD_token)(output_logits, target_padded)
        if not torch.isnan(output_loss):
            loss = loss + output_loss

    # 门控熵正则（鼓励适度不确定性/探索）
    ge_w = gate_entropy_weight if gate_entropy_weight is not None else getattr(config, 'GATE_ENTROPY_WEIGHT', 0.0)
    if ge_w and ge_w > 0:
        p = torch.clamp(gate_pred, 1e-6, 1-1e-6)
        entropy = -(p*torch.log(p) + (1-p)*torch.log(1-p)).mean()
        loss = loss + ge_w * entropy

    # 发声预算约束（控制平均发声比例/思考步）
    tsr = target_speak_ratio if target_speak_ratio is not None else getattr(config, 'TARGET_SPEAK_RATIO', None)
    if tsr is not None:
        budget_loss = (gate_pred.mean() - float(tsr)).abs()
        loss = loss + 0.1 * budget_loss

    # 思考信息量损失（InfoNCE：拉近 h_next 与“正样本”（自身/未来），远离“负样本”（其他样本））
    tlw = think_loss_weight if think_loss_weight is not None else getattr(config, 'THINK_LOSS_WEIGHT', 0.0)
    if tlw and tlw > 0 and h_prev is not None and h_next is not None:
        z_q = F.normalize(h_next, dim=-1)
        z_k = F.normalize(h_prev.detach(), dim=-1)  # detach 防止梯度泄回前一状态
        logits = torch.matmul(z_q, z_k.t())  # (B, B)
        tau = info_nce_tau if info_nce_tau is not None else getattr(config, 'THINK_INFO_TAU', 0.1)
        logits = logits / max(tau, 1e-6)
        labels = torch.arange(logits.size(0), device=logits.device)
        info_nce = nn.CrossEntropyLoss()(logits, labels)
        loss = loss + tlw * info_nce

    return loss


def set_seed(seed: Optional[int] = None, deterministic: bool = False):
    """统一的随机种子设置，支持确定性后端选项。"""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 确定性（会牺牲速度）
    try:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = not deterministic
        cudnn.deterministic = deterministic
    except Exception:
        pass

