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
    return_components: bool = False,
):
    """计算单个步骤的损失并可选返回分项。
    分项包含：
      - gate_bce: 门控 BCEWithLogitsLoss
      - token_ce: 词预测交叉熵（首 token 或序列展平）
      - gate_entropy_reg: 门控熵正则项（若启用）
      - budget_reg: 发声预算约束（若配置了 target_speak_ratio）
      - think_nce: 思考信息量损失 InfoNCE（若启用）
    """
    import torch.nn as nn
    import torch.nn.functional as F

    # 标量分项（Tensor，便于与总损失同设备）
    # 设备推断：优先 gate_pred，其次 output_logits，最后 CPU（避免 None 时发生跨设备相加错误）
    if isinstance(gate_pred, torch.Tensor):
        device = gate_pred.device
    elif isinstance(output_logits, torch.Tensor):
        device = output_logits.device
    else:
        device = torch.device('cpu')
    zero = torch.tensor(0.0, device=device)
    gate_bce = zero
    token_ce = zero
    gate_entropy_reg = zero
    budget_reg = zero
    think_nce = zero

    loss = zero.clone()

    # 1) 门控损失（若提供标签）
    if gate_target is not None:
        gp_logits = gate_pred.view(-1).float()
        gt = gate_target.view(-1).to(gp_logits.dtype)
        # 处理类别不均衡：正样本（发声）通常远少于负样本（思考）
        pos_w_cfg = getattr(config, 'GATE_POS_WEIGHT', None)
        if pos_w_cfg is not None:
            try:
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_w_cfg), device=gp_logits.device, dtype=gp_logits.dtype))
            except Exception:
                criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        gate_bce = criterion(gp_logits, gt)
        loss = loss + gate_bce

    # 2) 输出语言损失
    if target_padded is not None and output_logits is not None:
        ce = nn.CrossEntropyLoss(ignore_index=config.PAD_token)
        if output_logits.dim() == 2:
            if target_padded.dim() == 2:
                # 选择每个样本中第一个非 PAD 的目标作为监督；若整行均为 PAD，则设置为 PAD（将被 ignore）
                nonpad_mask = (target_padded != config.PAD_token)
                any_valid = nonpad_mask.any(dim=1)
                # argmax 在全 False 时返回 0；我们后续会把这些样本的标签强制为 PAD
                first_idx = nonpad_mask.float().argmax(dim=1)
                batch_indices = torch.arange(target_padded.size(0), device=target_padded.device)
                labels = target_padded[batch_indices, first_idx]
                if (~any_valid).any():
                    labels = labels.clone()
                    labels[~any_valid] = config.PAD_token
            else:
                labels = target_padded
            token_ce = ce(output_logits, labels.long())
        elif output_logits.dim() == 3:
            B, L, V = output_logits.size()
            # 对齐序列长度，防止 logits 长度与标签长度不一致
            if target_padded.dim() == 2:
                L_t = target_padded.size(1)
                L_eff = min(L, L_t)
                logits_eff = output_logits[:, :L_eff, :]
                labels_eff = target_padded[:, :L_eff]
                logits_flat = logits_eff.reshape(B * L_eff, V)
                labels_flat = labels_eff.reshape(B * L_eff)
            else:
                # 单标签广播到时间维（保持与 logits 的时间维一致）
                L_eff = L
                logits_flat = output_logits.reshape(B * L_eff, V)
                labels_flat = target_padded.view(B, 1).expand(B, L_eff).reshape(B * L_eff)
            token_ce = ce(logits_flat, labels_flat.long())

            # 调试：详细打印 CE 计算过程 + 简易准确率/标签概率
            if getattr(config, 'DEBUG_TOKEN_CE', False):
                try:
                    pad_token = getattr(config, 'PAD_token', 0)
                    valid_mask = (labels_flat != pad_token)
                    valid_count = int(valid_mask.sum().item())
                    unique_labels = torch.unique(labels_flat).tolist()
                    print(f"[debug/ce_detail] labels_flat_shape={labels_flat.shape} valid_count={valid_count} unique_labels={unique_labels[:10]} token_ce={float(token_ce.item()):.6f}")
                    if valid_count > 0:
                        # 检查有效标签的范围
                        valid_labels = labels_flat[valid_mask]
                        min_label, max_label = int(valid_labels.min().item()), int(valid_labels.max().item())
                        vocab_size = logits_flat.size(-1)
                        print(f"[debug/ce_detail] valid_label_range=[{min_label}, {max_label}] vocab_size={vocab_size} out_of_range={max_label >= vocab_size}")
                        # 简易准确率
                        with torch.no_grad():
                            preds = torch.argmax(logits_flat[valid_mask], dim=-1)
                            acc = float((preds == valid_labels).float().mean().item())
                            # 标签概率
                            log_probs = torch.log_softmax(logits_flat[valid_mask], dim=-1)
                            true_logp = log_probs[torch.arange(valid_labels.size(0), device=log_probs.device), valid_labels]
                            true_p = torch.exp(true_logp).mean().item()
                            print(f"[debug/ce_acc] acc={acc:.4f} true_p_mean={true_p:.4f}")
                except Exception as e:
                    print(f"[debug/ce_detail] error: {e}")
        else:
            token_ce = zero
        if not torch.isnan(token_ce):
            loss = loss + token_ce

    # 3) 门控熵正则
    ge_w = gate_entropy_weight if gate_entropy_weight is not None else getattr(config, 'GATE_ENTROPY_WEIGHT', 0.0)
    if ge_w and ge_w > 0:
        p = torch.sigmoid(gate_pred)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        gate_entropy_reg = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()
        loss = loss + ge_w * gate_entropy_reg

    # 4) 发声预算约束
    tsr = target_speak_ratio if target_speak_ratio is not None else getattr(config, 'TARGET_SPEAK_RATIO', None)
    if tsr is not None:
        budget_reg = (torch.sigmoid(gate_pred).mean() - float(tsr)).abs()
        loss = loss + 0.1 * budget_reg

    # 5) 思考信息量损失（InfoNCE）
    tlw = think_loss_weight if think_loss_weight is not None else getattr(config, 'THINK_LOSS_WEIGHT', 0.0)
    if tlw and tlw > 0 and h_prev is not None and h_next is not None:
        z_q = F.normalize(h_next, dim=-1)
        z_k = F.normalize(h_prev.detach(), dim=-1)
        logits = torch.matmul(z_q, z_k.t())
        tau = info_nce_tau if info_nce_tau is not None else getattr(config, 'THINK_INFO_TAU', 0.1)
        logits = logits / max(tau, 1e-6)
        labels = torch.arange(logits.size(0), device=logits.device)
        think_nce = nn.CrossEntropyLoss()(logits, labels)
        loss = loss + tlw * think_nce

    if return_components:
        comps = {
            'gate_bce': float(gate_bce.detach().item()),
            'token_ce': float(token_ce.detach().item()) if token_ce is not None else None,
            'gate_entropy_reg': float(gate_entropy_reg.detach().item()) if ge_w and ge_w > 0 else 0.0,
            'budget_reg': float(budget_reg.detach().item()) if tsr is not None else 0.0,
            'think_nce': float(think_nce.detach().item()) if (tlw and tlw > 0) else 0.0,
        }
        return loss, comps

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

