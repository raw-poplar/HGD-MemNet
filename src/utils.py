import torch
import os
import glob
import random
import numpy as np
import config
from .model import HGD_MemNet
from typing import Optional

# 告警节流（避免刷屏）
_WARN_COUNTS = {}
def _warn_throttled(tag: str, message: str):
    try:
        every_n = int(getattr(config, 'NUMERIC_WARNINGS_EVERY_N', 100) or 0)
    except Exception:
        every_n = 100
    if every_n <= 0:
        return
    c = _WARN_COUNTS.get(tag, 0) + 1
    _WARN_COUNTS[tag] = c
    if c % every_n == 0:
        print(message)

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

     # 在任意使用前，对 gate_pred 进行一次全局“去 NaN/Inf”净化，供后续所有分项复用
    logits_sane = None
    if gate_pred is not None:
        try:
            logits_sane = torch.nan_to_num(gate_pred, nan=0.0, posinf=10.0, neginf=-10.0)
        except Exception:
            logits_sane = gate_pred

    # 1) 门控损失（若提供标签）
    if gate_target is not None and gate_pred is not None:
        gp_logits = (logits_sane if logits_sane is not None else gate_pred).view(-1).float()
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

        # 检查gate_bce是否为NaN
        if torch.isnan(gate_bce) or torch.isinf(gate_bce):
            print(f"[WARNING] gate_bce is NaN/Inf, replacing with 0.0")
            gate_bce = zero.clone()

        loss = loss + gate_bce

    # 2) 输出语言损失
    if target_padded is not None and output_logits is not None:
        smoothing = float(getattr(config, 'LABEL_SMOOTHING', 0.0) or 0.0)
        ce = nn.CrossEntropyLoss(ignore_index=config.PAD_token, label_smoothing=smoothing)
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
        else:
            token_ce = zero

        # 数值保护：若 token_ce 为 NaN/Inf 则置零（节流打印）
        if torch.isnan(token_ce) or torch.isinf(token_ce):
            _warn_throttled('token_ce_nan', "[WARNING] token_ce is NaN/Inf, replacing with 0.0")
            token_ce = zero

        loss = loss + token_ce

    # 3) 门控熵正则
    ge_w = gate_entropy_weight if gate_entropy_weight is not None else getattr(config, 'GATE_ENTROPY_WEIGHT', 0.0)
    if ge_w and ge_w > 0 and gate_pred is not None:
        gp_for_entropy = logits_sane if logits_sane is not None else gate_pred
        p = torch.sigmoid(gp_for_entropy)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        gate_entropy_reg = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()
        if torch.isnan(gate_entropy_reg) or torch.isinf(gate_entropy_reg):
            _warn_throttled('gate_entropy_nan', "[WARNING] gate_entropy_reg is NaN/Inf, replacing with 0.0")
            gate_entropy_reg = zero
        loss = loss + ge_w * gate_entropy_reg

    # 4) 发声预算约束
    tsr = target_speak_ratio if target_speak_ratio is not None else getattr(config, 'TARGET_SPEAK_RATIO', None)
    if tsr is not None and gate_pred is not None:
        gp_for_budget = logits_sane if logits_sane is not None else gate_pred
        budget_reg = (torch.sigmoid(gp_for_budget).mean() - float(tsr)).abs()
        if torch.isnan(budget_reg) or torch.isinf(budget_reg):
            _warn_throttled('budget_reg_nan', "[WARNING] budget_reg is NaN/Inf, replacing with 0.0")
            budget_reg = zero
        loss = loss + 0.1 * budget_reg

    # 5) 思考信息量损失（InfoNCE）
    tlw = think_loss_weight if think_loss_weight is not None else getattr(config, 'THINK_LOSS_WEIGHT', 0.0)
    if tlw and tlw > 0 and h_prev is not None and h_next is not None:
        z_q = F.normalize(h_next, dim=-1)
        z_k = F.normalize(h_prev.detach(), dim=-1)

        # 若出现 NaN，进行兜底
        if torch.isnan(z_q).any() or torch.isinf(z_q).any():
            _warn_throttled('zq_nan', "[WARNING] z_q contains NaN/Inf, applying nan_to_num")
            z_q = torch.nan_to_num(z_q)
        if torch.isnan(z_k).any() or torch.isinf(z_k).any():
            _warn_throttled('zk_nan', "[WARNING] z_k contains NaN/Inf, applying nan_to_num")
            z_k = torch.nan_to_num(z_k)

        logits = torch.matmul(z_q, z_k.t())
        tau = info_nce_tau if info_nce_tau is not None else getattr(config, 'THINK_INFO_TAU', 0.1)
        logits = logits / max(tau, 1e-6)
        labels = torch.arange(logits.size(0), device=logits.device)
        think_nce = nn.CrossEntropyLoss()(logits, labels)
        if torch.isnan(think_nce) or torch.isinf(think_nce):
            _warn_throttled('think_nce_nan', "[WARNING] think_nce is NaN/Inf, replacing with 0.0")
            think_nce = zero
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

