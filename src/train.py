# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import os
import glob
from tqdm import tqdm
from torch.amp import GradScaler

# 新增：性能优化
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # 启用 cudnn 基准测试以加速

import config
from .model import HGD_MemNet
from .dataset import (
    Vocabulary,
    BinaryDialogueDataset,
    binary_collate_fn,
)
from .utils import compute_loss, set_seed  # 新增：导入工具函数与随机种子
from .utils import _WARN_COUNTS  # 引入节流计数（仅用于统计/复用，不直接打印）
import logging  # 新增：日志记录
import gc
from concurrent.futures import ThreadPoolExecutor
try:
    from . import arch_config as _arch_cfg
except Exception:
    _arch_cfg = None

# 线程/队列（内存流式）
import threading
from queue import Queue
# 设置日志
logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 可选：确定性设置（由 config 决定）
if getattr(config, 'DETERMINISTIC', False):
    cudnn.benchmark = False
    cudnn.deterministic = True
# --- 调试辅助：ID序列解码为token列表 ---
def _decode_ids_to_tokens(ids_tensor, vocab, max_len: int = 100):
    try:
        ids = ids_tensor.detach().cpu().tolist()
    except Exception:
        try:
            ids = list(ids_tensor)
        except Exception:
            return []
    toks = []
    for i, idx in enumerate(ids):
        if i >= max_len:
            toks.append("<...>")
            break
        if idx == getattr(config, 'PAD_token', 0):
            continue
        toks.append(vocab.index2word.get(idx, f"<UNK:{idx}>"))
        if idx == getattr(config, 'EOS_token', 2):
            break
    return toks

# 从 steps_data 中提取“最终目标序列”（最后一个非 None 的 y_t）
def _extract_final_target_from_steps(steps_data, sample_idx: int = 0):
    final_y = None
    for _x_t, y_t, _g_t in steps_data:
        if y_t is not None:
            # 记录最新的非 None 目标
            final_y = y_t
    if final_y is None:
        return None
    try:
        return final_y[sample_idx]
    except Exception:
        return None


def train_batch_stepwise(x_ref_padded, steps_data, model, optimizer, scaler, global_total_steps: int = 0):
    """
    以“内部思考步”为核心的训练过程：
    - 对同一个样本序列，逐步推进 t=1..T；每步都计算损失并反传；
    - 在步内衰减学习率（INNER_STEP_LR_DECAY），并对 ReservoirRNNCell 施加温度退火；
    - 可选启用基于门控阈值的早停（USE_GATED_MULTISTEP）。

    返回：
        total_batch_loss: 本 batch 所有内部步损失之和（标量）
        steps_updated: 实际反向更新的步数
        gate_mean: 门控均值（用于监控）
        gate_entropy: 门控熵（用于监控/正则）
        cap_hit: 是否由最大步 cap 截断（0/1）
    """
    model.train()
    x_ref_padded = x_ref_padded.to(DEVICE, non_blocking=True)

    batch_size = x_ref_padded.size(0)
    total_batch_loss = 0.0
    steps_updated = 0
    gate_mean_acc = 0.0
    gate_entropy_acc = 0.0
    gate_obs = 0
    ended_by_cap = False

    # 分项损失累计（步平均）
    comps_acc = {
        'gate_bce': 0.0,
        'token_ce': 0.0,
        'token_ce_eval': 0.0,
        'gate_entropy_reg': 0.0,
        'budget_reg': 0.0,
        'think_nce': 0.0,
        'target_step_count': 0.0,  # 本 batch 内含有效标签（非PAD）的步数计数
    }

    h_prev = torch.zeros(batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM, device=DEVICE)

    original_lrs = [param_group['lr'] for param_group in optimizer.param_groups]

    # 新增：温度退火参数 (从config导入)
    initial_temperature = config.INITIAL_TEMPERATURE
    temperature_decay = config.TEMPERATURE_DECAY

    # 自适应思考：解析最小/最大思考步（-1 表示不限制）
    min_t = getattr(config, 'MIN_THINKING_STEPS', -1)
    max_t = getattr(config, 'MAX_THINKING_STEPS', -1)
    min_t = None if min_t is None or min_t < 0 else int(min_t)
    max_t = None if max_t is None or max_t < 0 else int(max_t)

    steps_available = len(steps_data)
    # 计算应运行的内部步数：至少满足 min_t，至多不超过 max_t（若配置）
    steps_to_run = steps_available
    if min_t is not None:
        steps_to_run = max(steps_to_run, min_t)
    if max_t is not None:
        steps_to_run = min(steps_to_run, max_t)
    # 若启用扩展，则当数据步不足时补空思考直至 MAX
    if getattr(config, 'EXTEND_THINKING_TO_MAX', False) and (max_t is not None):
        steps_to_run = max(steps_to_run, max_t)
    # 安全上限（硬性强制输出步）：不改变循环上界，仅在命中该步时强制输出
    safety_max_t = getattr(config, 'SAFETY_MAX_THINKING_STEPS', None)
    try:
        safety_max_t = int(safety_max_t) if (safety_max_t is not None and int(safety_max_t) > 0) else None
    except Exception:
        safety_max_t = None
    # 不再自动裁剪或提示 SAFETY_MAX_THINKING_STEPS，严格按配置使用
    # 归一化分母采用计划运行的步数；若允许延长至安全步，则以安全步作为 cap
    cap = steps_to_run
    if getattr(config, 'EXTEND_THINKING_TO_SAFETY', False) and (safety_max_t is not None):
        cap = max(cap, safety_max_t)

    # 评估用“样本级目标”（例如最终答案）：取该样本在本batch内最后一个非空 target 作为全局标签，用于在每个思考步评估 token_ce（不参与反传）
    final_target_padded = None
    try:
        for s in range(steps_available - 1, -1, -1):
            _tgt = steps_data[s][1]
            # 仅当 target 存在且包含至少一个非 PAD token 时，才作为最终目标
            if _tgt is not None and _tgt.numel() > 0:
                # 若为 2D，检查是否存在非PAD元素
                if _tgt.dim() == 2:
                    if (_tgt != config.PAD_token).any():
                        final_target_padded = _tgt.to(DEVICE, non_blocking=True)
                        break
                else:
                    final_target_padded = _tgt.to(DEVICE, non_blocking=True)
                    break
    except Exception:
        final_target_padded = None

    # 梯度累积设置（在样本的“内部思考步”维度累积）
    acc_steps = max(1, int(getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1) or 1))
    optimizer.zero_grad()
    # 详细追踪：若开启详细打印收集，则缓存本 batch 第一个样本的详细步信息
    collect_detail = bool(getattr(config, 'COLLECT_DETAIL_FOR_DEBUG', True))
    detail_trace = [] if collect_detail else None

    # 循环上界：若启用延长至安全步，则延长到 safety
    loop_steps = cap if (getattr(config, 'EXTEND_THINKING_TO_SAFETY', False) and (safety_max_t is not None)) else steps_to_run
    for t in range(loop_steps):
        is_safety_step = (safety_max_t is not None and (t + 1) == safety_max_t)
        # 若超过数据提供的步数，则补“思考步”（无输入、无监督）
        if t < steps_available:
            # 忽略数据集中的 gate 标签
            x_t_padded, target_padded, _ = steps_data[t]
        else:
            x_t_padded, target_padded = None, None
        # 若命中安全步且无 target，则强制作为“发声步”：使用最终目标作为监督
        if is_safety_step and target_padded is None and final_target_padded is not None:
            target_padded = final_target_padded
        x_t_padded = x_t_padded.to(DEVICE, non_blocking=True) if x_t_padded is not None else None
        if target_padded is not None:
            target_padded = target_padded.to(DEVICE, non_blocking=True)

        # 1. 在每个时间步内部动态调整学习率
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = original_lrs[i] * (config.INNER_STEP_LR_DECAY ** t)

        # 新增：计算当前步骤的温度
        current_temperature = max(initial_temperature * (temperature_decay ** t), config.MIN_TEMPERATURE)

        # 累积中不要在每个时间步 zero_grad（外部已 zero），只在 step 时机再清零
        # optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            # 构造控制向量（时机意识）：t_norm, remain_norm, min_done, target_speak_ratio
            # 缓升：将分母扩大 extra，使 MIN_THINKING_STEPS 处的 t_norm < 1，减缓时间偏置增长
            extra = float(getattr(config, 'GATE_TIME_RAMP_EXTRA_STEPS', 0) or 0)
            denom = max(cap + extra, 1)
            t_norm = torch.full((batch_size, 1), (t + 1) / denom, device=DEVICE)
            remain_norm = torch.full((batch_size, 1), max(denom - (t + 1), 0) / denom, device=DEVICE)
            min_done = torch.full((batch_size, 1), 1.0 if (min_t is not None and (t + 1) >= min_t) or (min_t is None) else 0.0, device=DEVICE)
            budget = torch.full((batch_size, 1), float(getattr(config, 'TARGET_SPEAK_RATIO', 0.0) or 0.0), device=DEVICE)
            control = torch.cat([t_norm, remain_norm, min_done, budget], dim=1)

            # 反馈向量（上一时刻的摘要）：[threshold, gap, gate_p, top1, min_done, t_norm]
            feedback_vec = None
            if getattr(config, 'USE_DYNAMIC_FEEDBACK', False):
                try:
                    th = float(getattr(config, 'GATE_THRESHOLD', 0.5) or 0.5)
                    gate_p = torch.sigmoid(gate_pred.detach()) if 'gate_pred' in locals() else torch.zeros(batch_size, 1, device=DEVICE)
                    gap = gate_p - th
                    # top1 置信度（静态头输出）
                    if 'output_logits' in locals() and output_logits is not None:
                        if output_logits.dim() == 3:
                            logits_for_conf = output_logits[:, -1, :]
                        else:
                            logits_for_conf = output_logits
                        probs = torch.softmax(logits_for_conf.detach(), dim=-1)
                        top1 = probs.max(dim=-1).values.mean().view(1, 1).expand(batch_size, 1)
                    else:
                        top1 = torch.zeros(batch_size, 1, device=DEVICE)
                    feedback_vec = torch.cat([
                        torch.full((batch_size,1), th, device=DEVICE),
                        gap, gate_p,
                        top1,
                        min_done, t_norm
                    ], dim=1)
                except Exception:
                    feedback_vec = None

            h_next, gate_pred, output_logits = model(
                x_t_padded, x_ref_padded, h_prev, temperature=current_temperature, control=control, feedback_vec=feedback_vec
            )

            # 统计门控均值/熵（gate_pred 为 logits，这里按概率统计）
            # 数值保护 + 温度平滑（日志统计用，不影响反传）
            gate_logits = torch.nan_to_num(gate_pred, nan=0.0, posinf=10.0, neginf=-10.0)
            gp_temp = float(getattr(config, 'GATE_PROB_LOG_TEMPERATURE', 1.0) or 1.0)
            if gp_temp != 1.0 and gp_temp > 0:
                gate_logits = gate_logits / gp_temp
            gate_prob = torch.sigmoid(gate_logits)
            gate_mean_acc += float(gate_prob.mean().item())
            p = torch.clamp(gate_prob, 1e-6, 1 - 1e-6)
            gate_entropy_acc += float(-(p * p.log() + (1 - p) * (1 - p).log()).mean().item())
            gate_obs += 1

            # 监控：温度累计（用于批平均）
            try:
                comps_acc['temperature'] = comps_acc.get('temperature', 0.0) + float(current_temperature)
            except Exception:
                pass

            # THINK_LOSS warmup：前 N 步内线性升温
            warmup_steps = int(getattr(config, 'THINK_WARMUP_STEPS', 0) or 0)
            tlw_cfg = float(getattr(config, 'THINK_LOSS_WEIGHT', 0.0) or 0.0)
            if warmup_steps > 0 and tlw_cfg > 0:
                tlw_eff = tlw_cfg * min(1.0, (global_total_steps + steps_updated + 1) / warmup_steps)
            else:
                tlw_eff = tlw_cfg

            # ========== 基于门控与安全步的“发声决策”与监督选择 ==========
            th_now = float(getattr(config, 'GATE_THRESHOLD', 0.5) or 0.5)
            dyn_eps = float(getattr(config, 'GATE_DYNAMIC_THRESHOLD_EPS', 0.0) or 0.0)
            if dyn_eps != 0 and cap > 0:
                th_now = max(0.0, min(1.0, th_now - dyn_eps * ((t + 1) / cap)))
            gate_prob_mean_now = float(torch.sigmoid(gate_pred).mean().item()) if gate_pred is not None else 0.0
            min_done_now = (min_t is None) or ((t + 1) >= min_t)
            decided_to_speak = (is_safety_step or (min_done_now and (gate_prob_mean_now >= th_now)))
            # 若到达 MAX 仍未触发，则在 MAX 强制发声（符合“在 MIN~MAX 区间内发声”的要求）
            if (not decided_to_speak) and (max_t is not None) and ((t + 1) == max_t):
                decided_to_speak = True

            # 若无 target 且决定发声，使用最终目标监督；并将 gate_target 置为 1
            y_for_loss = target_padded
            if y_for_loss is None and decided_to_speak and final_target_padded is not None:
                y_for_loss = final_target_padded
            step_loss, comps = compute_loss(
                gate_pred, None, output_logits, y_for_loss,
                gate_entropy_weight=getattr(config, 'GATE_ENTROPY_WEIGHT', 0.0),
                target_speak_ratio=getattr(config, 'TARGET_SPEAK_RATIO', None),
                h_prev=h_prev, h_next=h_next,
                think_loss_weight=tlw_eff,
                info_nce_tau=getattr(config, 'THINK_INFO_TAU', 0.1),
                return_components=True,
            )
            # 若损失为 NaN/Inf，记录上下文并回退为0，避免污染统计
            if torch.isnan(step_loss) or torch.isinf(step_loss):
                try:
                    dbg = {
                        't': t+1,
                        'has_target': target_padded is not None,
                        'x_t_shape': tuple(x_t_padded.shape) if x_t_padded is not None else None,
                        'logits_shape': tuple(output_logits.shape) if output_logits is not None else None,
                        'gate_mean': float(torch.sigmoid(gate_pred).mean().item()) if gate_pred is not None else None,
                    }
                    print(f"[WARNING] NaN/Inf step_loss detected, context: {dbg}")
                except Exception:
                    pass
                step_loss = torch.zeros((), device=DEVICE)
            # 统计：该步是否存在非PAD标签
            try:
                if target_padded is not None and target_padded.dim() == 2:
                    nonpad_mask = (target_padded != config.PAD_token)
                    has_valid = float(nonpad_mask.any(dim=1).float().mean().item())
                    comps_acc['target_step_count'] += has_valid
            except Exception:
                pass
            # 累计分项
            try:
                for k in list(comps_acc.keys()):
                    comps_acc[k] += comps.get(k, 0.0)
            except Exception:
                pass

            # 详细收集（仅记录第一个样本，避免日志爆炸）
            try:
                if collect_detail:
                    sample_idx = 0
                    # 当前步的输入/标签形状、门控/温度、损失分项
                    rec = {
                        't': t+1,
                        'x_t_shape': tuple(x_t_padded.shape) if x_t_padded is not None else None,
                        'target_shape': tuple(target_padded.shape) if target_padded is not None else None,
                        'gate_prob_mean': float(torch.sigmoid(gate_pred).mean().item()) if gate_pred is not None else None,
                        'temperature': float(current_temperature),
                        'loss': float(step_loss.detach().item()) if hasattr(step_loss, 'item') else float(step_loss),
                        'comps': comps,
                    }
                    # 记录该步的 x_t token ids（样本0）供打印解码
                    try:
                        if x_t_padded is not None and x_t_padded.size(0) > sample_idx and x_t_padded.dim() == 2:
                            ids0 = x_t_padded[sample_idx]
                            # 截断到前 32 个非PAD token
                            nonpad = ids0[ids0 != config.PAD_token]
                            if nonpad.numel() > 0:
                                rec['x_t_ids'] = [int(i) for i in nonpad[:32].detach().cpu().tolist()]
                    except Exception:
                        pass
                    # 提取该样本的 token 预测与目标（末位时间步）
                    try:
                        if output_logits is not None:
                            if output_logits.dim() == 3:
                                logits_for_tok = output_logits[sample_idx, -1, :]
                            else:
                                logits_for_tok = output_logits[sample_idx]
                            pred_id = int(torch.argmax(logits_for_tok).item())
                            rec['pred_token'] = pred_id
                            # top-5 预测
                            try:
                                probs_last = torch.softmax(logits_for_tok, dim=-1)
                                k = min(5, probs_last.size(0))
                                topk_vals, topk_idx = torch.topk(probs_last, k=k, dim=-1)
                                rec['topk'] = list(zip([int(i) for i in topk_idx.detach().cpu().tolist()], [float(v) for v in topk_vals.detach().cpu().tolist()]))
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # 记录该步是否“发声”：由门控/安全步或数据标签触发
                    rec['is_speak_step'] = bool(decided_to_speak)
                    detail_trace.append(rec)
            except Exception:
                pass

            # --- 调试：token_ce 相关诊断打印（按需开启） ---
            try:
                if getattr(config, 'DEBUG_TOKEN_CE', False):
                    dbg_every = int(getattr(config, 'DEBUG_TOKEN_CE_EVERY_N', 1000) or 1000)
                    dbg_step = global_total_steps + steps_updated + 1
                    if dbg_every > 0 and (dbg_step % dbg_every == 0):
                        lines = [
                            f"[debug/token_ce] step={dbg_step} t={t+1} use_seq_ce={getattr(config, 'USE_SEQUENCE_CE', False)}"
                        ]
                        if target_padded is None:
                            lines.append(" target=None (thinking step)")
                            if final_target_padded is not None:
                                nonpad_final = int((final_target_padded != config.PAD_token).sum().item())
                                lines.append(f" final_target_shape={tuple(final_target_padded.shape)} final_nonpad={nonpad_final}")
                        else:
                            nonpad = int((target_padded != config.PAD_token).sum().item())
                            lines.append(f" target_shape={tuple(target_padded.shape)} nonpad={nonpad}")
                        if output_logits is None:
                            lines.append(" logits=None")
                        else:
                            shp = tuple(output_logits.shape)
                            lines.append(f" logits_shape={shp} dim={output_logits.dim()}")
                            if output_logits.dim() == 3 and target_padded is not None and target_padded.dim() == 2:
                                L = output_logits.size(1); L_t = target_padded.size(1); L_eff = min(L, L_t)
                                lines.append(f" L={L} L_t={L_t} L_eff={L_eff}")
                        try:
                            lines.append(f" comps.token_ce={comps.get('token_ce', None)}")
                        except Exception:
                            pass
                        print("\n".join(lines))
            except Exception:
                pass
            # --- 调试结束 ---

            # 思考期 token_ce 评估（不参与反传）：针对没有 target_padded 的步，用最终目标评估静态头输出质量
            try:
                if target_padded is None and final_target_padded is not None and output_logits is not None:
                    from .utils import compute_loss as _cl
                    with torch.no_grad():
                        _, eval_comps = _cl(
                            gate_pred, None, output_logits, final_target_padded,
                            gate_entropy_weight=0.0, target_speak_ratio=None,
                            return_components=True,
                        )
                        comps_acc['token_ce_eval'] = comps_acc.get('token_ce_eval', 0.0) + eval_comps.get('token_ce', 0.0)
            except Exception:
                pass

            # 思考步蒸馏（弱反馈）：将思考步的分布拉向基于“最终目标序列”的软 teacher 分布（不直接用 gold 做硬CE）
            try:
                distill_w = float(getattr(config, 'THOUGHT_DISTILL_WEIGHT', 0.0) or 0.0)
                if distill_w > 0 and output_logits is not None and final_target_padded is not None and target_padded is None:
                    import torch.nn.functional as F
                    # 构造 teacher：用序列解码器在最终目标序列上跑一次 TF，初态用当前 h_next（表示“思考到此时”为止的内部状态）
                    teacher_p = None
                    try:
                        if getattr(model, 'use_sequence_ce', False) and hasattr(model, 'seq_decoder') and hasattr(model, 'seq_out'):
                            with torch.no_grad():
                                dec_inp = model.embed(final_target_padded)
                                dec_h0 = h_next.unsqueeze(0)
                                dec_out, _ = model.seq_decoder(dec_inp, dec_h0)  # (B,L,H)
                                seq_logits = model.seq_out(dec_out)              # (B,L,V)
                                # 使用“非PAD且非EOS”的时间步做掩码平均，避免末位偏向 EOS
                                tau = float(getattr(config, 'THOUGHT_DISTILL_TAU', 1.0) or 1.0)
                                probs = F.softmax(seq_logits / max(tau, 1e-6), dim=-1)  # (B,L,V)
                                valid_mask = (final_target_padded != getattr(config, 'PAD_token', 0)) & (final_target_padded != getattr(config, 'EOS_token', 2))
                                if valid_mask.any():
                                    weights = valid_mask.float()  # (B,L)
                                    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
                                    teacher_p = (probs * weights.unsqueeze(-1)).sum(dim=1).detach()  # (B,V)
                                else:
                                    # 兜底：取中间位置
                                    mid = max(0, probs.size(1) // 2)
                                    teacher_p = probs[:, mid, :].detach()
                    except Exception:
                        teacher_p = None
                    if teacher_p is not None:
                        tau = float(getattr(config, 'THOUGHT_DISTILL_TAU', 1.0) or 1.0)
                        student_logits = output_logits[:, -1, :] if output_logits.dim() == 3 else output_logits
                        if student_logits.size(-1) == teacher_p.size(-1):
                            student_logp = F.log_softmax(student_logits / max(tau, 1e-6), dim=-1)
                            kl = F.kl_div(student_logp, teacher_p, reduction='batchmean')
                            scheme = str(getattr(config, 'THOUGHT_DISTILL_SCHEME', 't_norm') or 't_norm').lower()
                            if scheme == 't_norm':
                                w_scalar = float(((t + 1) / max(cap, 1)))
                            elif scheme == 'gate_prob':
                                w_scalar = float(torch.sigmoid(gate_pred.detach()).mean().item())
                            else:
                                w_scalar = 1.0
                            step_loss = step_loss + distill_w * w_scalar * kl
            except Exception:
                pass
            # 思考步弱监督 token_ce（小权重+warmup+可选加权方案）
            try:
                # 辅助 CE 触发检查（仅调试）
                try:
                    if getattr(config, 'DEBUG_TOKEN_CE', False):
                        w_cfg_dbg = float(getattr(config, 'THINK_STEP_CE_WEIGHT', 0.0) or 0.0)
                        cond = {
                            'w_cfg>0': (w_cfg_dbg > 0),
                            'is_thinking_step': (target_padded is None),
                            'has_final_target': (final_target_padded is not None and (final_target_padded != config.PAD_token).any().item() if final_target_padded is not None else False),
                            'has_logits': (output_logits is not None),
                        }
                        print(f"[debug/aux_check] step={global_total_steps+steps_updated+1} t={t+1} {cond}")
                except Exception:
                    pass

                w_cfg = float(getattr(config, 'THINK_STEP_CE_WEIGHT', 0.0) or 0.0)
                if w_cfg > 0 and final_target_padded is not None and output_logits is not None and target_padded is None:
                    # warmup 到当前步的有效权重
                    w_steps = int(getattr(config, 'THINK_STEP_CE_WARMUP_STEPS', 0) or 0)
                    if w_steps > 0:
                        w_eff = w_cfg * min(1.0, (global_total_steps + steps_updated + 1) / w_steps)
                    else:
                        w_eff = w_cfg
                    # 加权方案
                    scheme = str(getattr(config, 'THINK_STEP_CE_SCHEME', 't_norm') or 't_norm').lower()
                    if scheme == 't_norm':
                        weight_scalar = float(((t + 1) / max(cap, 1)))
                    elif scheme == 'gate_prob':
                        weight_scalar = float(torch.sigmoid(gate_pred.detach()).mean().item())
                    else:
                        weight_scalar = 1.0
                    # 仅取 CE 分项并加到总损失
                    from .utils import compute_loss as _cl
                    aux_loss, aux_comps = _cl(
                        gate_pred=None, gate_target=None,
                        output_logits=output_logits, target_padded=final_target_padded,
                        gate_entropy_weight=0.0, target_speak_ratio=None,
                        return_components=True,
                    )
                    token_ce_aux = aux_comps.get('token_ce', 0.0) if isinstance(aux_comps, dict) else 0.0
                    add_val = w_eff * weight_scalar * token_ce_aux
                    step_loss = step_loss + (w_eff * weight_scalar * (aux_loss if aux_loss is not None else 0.0))
                    comps_acc['token_ce'] = comps_acc.get('token_ce', 0.0) + add_val
                    # 调试输出：辅助 token_ce 是否被触发及其数值
                    if getattr(config, 'DEBUG_TOKEN_CE', False):
                        try:
                            nonpad_final = int((final_target_padded != config.PAD_token).sum().item()) if final_target_padded is not None else -1
                            print(f"[debug/token_aux] step={global_total_steps+steps_updated+1} t={t+1} w_eff={w_eff:.4f} scheme={scheme} w_scalar={weight_scalar:.4f} ce_aux={float(token_ce_aux):.6f} add={float(add_val):.6f} final_nonpad={nonpad_final}")
                        except Exception:
                            pass
            except Exception:
                pass


            # 监控：动态组核心RNN与静态头的内部统计（若可用）
            try:
                if hasattr(model.dynamic_group.core_rnn, 'last_selection_stats') and model.dynamic_group.core_rnn.last_selection_stats:
                    sel = model.dynamic_group.core_rnn.last_selection_stats
                    comps_acc['cell_avg_row_max'] = comps_acc.get('cell_avg_row_max', 0.0) + sel.get('avg_row_max_prob', 0.0)
                    comps_acc['cell_avg_row_entropy'] = comps_acc.get('cell_avg_row_entropy', 0.0) + sel.get('avg_row_entropy', 0.0)
                if hasattr(model.static_head, 'last_sampling_stats') and model.static_head.last_sampling_stats:
                    samp = model.static_head.last_sampling_stats
                    comps_acc['sampler_avg_topk_weight'] = comps_acc.get('sampler_avg_topk_weight', 0.0) + (samp.get('avg_topk_weight') or 0.0)
                    comps_acc['sampler_avg_topk_max'] = comps_acc.get('sampler_avg_topk_max', 0.0) + (samp.get('avg_topk_max') or 0.0)
                    comps_acc['sampler_coverage_ratio'] = comps_acc.get('sampler_coverage_ratio', 0.0) + (samp.get('coverage_ratio') or 0.0)
            except Exception:
                pass

            # 思考过程追踪（可选）：记录静态头输出的Top-K预测（仅取第一个样本，避免日志过大）
            try:
                trace_every = int(getattr(config, 'THINK_TRACE_EVERY_N_STEPS', 0) or 0)
                trace_topk = int(getattr(config, 'THINK_TRACE_TOPK', 0) or 0)
                if trace_every > 0 and trace_topk > 0:
                    should_trace = ((global_total_steps + steps_updated + 1) % trace_every == 0)
                    if should_trace and output_logits is not None:
                        if output_logits.dim() == 3:
                            logits_for_trace = output_logits[:, -1, :]
                        else:
                            logits_for_trace = output_logits
                        probs = torch.softmax(logits_for_trace, dim=-1)
                        probs_0 = probs[0]
                        k = min(trace_topk, probs_0.size(0))
                        topk_vals, topk_idx = torch.topk(probs_0, k=k, dim=-1)
                        # 写入文件
                        os.makedirs('logs', exist_ok=True)
                        with open(os.path.join('logs', 'thinking_trace.txt'), 'a', encoding='utf-8') as f:
                            gp0 = float(torch.sigmoid(gate_pred[0]).item())
                            f.write(f"step={global_total_steps+steps_updated+1}, t={t+1}, gate_p={gp0:.4f}, top{ k }: ")
                            f.write(', '.join([f"{int(i)}:{float(v):.4f}" for i, v in zip(topk_idx.tolist(), topk_vals.tolist())]))
                            f.write("\n")
            except Exception:
                pass

        # 梯度累积：对损失做平均再反传
        # 数值保护：若 step_loss 为 NaN/Inf，跳过该步反传，避免污染梯度
        if torch.isnan(step_loss) or torch.isinf(step_loss):
            print(f"[WARNING] step_loss is NaN/Inf at inner step t={t+1}, skip backward")
        else:
            scaled_loss = step_loss / acc_steps
            scaler.scale(scaled_loss).backward()

        # 满足累积步数或到步尾时再执行优化器更新
        if ((t + 1) % acc_steps == 0):
            try:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            except Exception as _e:
                print(f"[WARNING] optimizer.step/update failed (possibly due to NaN grads): {_e}; performing gradient reset")
            optimizer.zero_grad()

        total_batch_loss += step_loss.item()
        h_prev = h_next.detach()
        steps_updated += 1

        # 可选：达到最小步后，基于门控阈值提前停止多步思考；若命中安全步（并已强制输出）则直接终止
        try:
            if is_safety_step:
                break
            # 到达 MAX 后也结束该样本（无论门控情况），以确保在 MIN~MAX 区间内发声一次
            if (max_t is not None) and ((t + 1) == max_t):
                break
            if getattr(config, 'USE_GATED_MULTISTEP', False) and (min_t is None or (t + 1) >= min_t):
                th = float(getattr(config, 'GATE_THRESHOLD', 0.5) or 0.5)
                # 早停阈值加动态偏移：越接近MAX越容易触发
                dyn_eps = float(getattr(config, 'GATE_DYNAMIC_THRESHOLD_EPS', 0.1) or 0.0)
                if dyn_eps != 0 and cap > 0:
                    t_frac = (t + 1) / cap
                    th = max(0.0, min(1.0, th - dyn_eps * t_frac))
                gate_prob_mean = float(torch.sigmoid(gate_pred).mean().item()) if gate_pred is not None else 0.0
                if gate_prob_mean >= th:
                    # 若允许延长到安全步并尚未达到安全步，则不早停
                    if not (getattr(config, 'EXTEND_THINKING_TO_SAFETY', False) and (safety_max_t is not None) and ((t + 1) < safety_max_t)):
                        break
        except Exception:
            pass

        # 基于门控的多步思考早停（需达到最小思考步才生效）
    # 循环结束后，如仍有未更新的累积梯度则执行一次优化器更新
    if (steps_updated % acc_steps) != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = original_lrs[i]

        # 伪代码说明：
        # for t in steps:
        #   ... 前向得到 gate_pred
        #   if USE_GATED_MULTISTEP and gate_pred >= GATE_THRESHOLD:
        #       break  # 提前终止内部步，模拟“已决定输出”

    # 计算门控统计
    gate_mean = (gate_mean_acc / max(1, gate_obs)) if gate_obs > 0 else 0.0
    gate_entropy = (gate_entropy_acc / max(1, gate_obs)) if gate_obs > 0 else 0.0

    # 分项平均（按步）
    comps_avg = {k: (v / max(1, steps_updated)) for k, v in comps_acc.items()}

    return total_batch_loss, steps_updated, gate_mean, gate_entropy, int(ended_by_cap), comps_avg, detail_trace

# 剪枝/再生长：按配置周期触发（最小实现，模块级函数）
def maybe_prune_regrow(model, total_steps):
    try:
        if not getattr(config, 'PRUNE_ENABLE', False):
            return
        if total_steps < getattr(config, 'PRUNE_START_STEPS', 0):
            return
        if total_steps % max(1, getattr(config, 'PRUNE_EVERY_STEPS', 1000)) != 0:
            return
        cell = getattr(getattr(model, 'dynamic_group', None), 'core_rnn', None)
        if cell is None or not hasattr(cell, 'hh_mask'):
            return
        # 记录剪枝前非零数
        with torch.no_grad():
            before_nz = int(cell.hh_mask.sum().item())
        # 剪枝
        cell.prune_by_magnitude(
            sparsity_step=getattr(config, 'PRUNE_SPARSE_STEP', 0.05),
            min_keep=getattr(config, 'PRUNE_MIN_KEEP', 4),
        )
        # 再生长（可选）
        if getattr(config, 'REGROW_ENABLE', False):
            cell.regrow_by_hebb(
                per_row=getattr(config, 'REGROW_PER_ROW', 1),
                init_std=getattr(config, 'REGROW_INIT_STD', 1e-3),
            )
        # 记录剪枝后非零数
        with torch.no_grad():
            after_nz = int(cell.hh_mask.sum().item())
        logging.info(f"[sparsify] step={total_steps} nnz: {before_nz}->{after_nz}")
    except Exception as e:
        logging.warning(f"剪枝/再生长触发失败: {e}")

# ========= Live 内存流式：直接从 jsonl 生成内存 chunk =========
from typing import Iterable

def stream_live_jsonl(file_path: str, vocab, chunk_size: int = 1000, start_line: int = 0) -> Iterable[tuple[str, list, int]]:
    """逐行读取 jsonl，在线处理为 (x_ref, steps_data) 并按 chunk_size 产出内存块。
    - start_line: 断点续读的起始行号（0-based），用于从上次位置继续。
    返回: (chunk_name, data_list, end_line)，end_line 为本次产出后应记录的续读起点。
    仅在 ENABLE_LIVE_STREAM_TRAIN 且 LIVE_STREAM_MODE=="memory" 时使用。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到 jsonl 文件: {file_path}")
    from src.data_processing.prepare_binary_data import process_dialogue_to_tensors
    buf = []
    idx = 0
    end_line = start_line
    with open(file_path, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f):
            if ln < max(0, int(start_line)):
                continue
            try:
                dialogue = json.loads(line)
                item = process_dialogue_to_tensors(dialogue, vocab)
                if item:
                    buf.append(item)
                end_line = ln + 1  # 下次从该行开始
                if len(buf) >= max(1, int(chunk_size)):
                    yield f"live_chunk_{idx}", buf, end_line
                    buf = []
                    idx += 1
            except Exception:
                # 跳过异常行
                continue
    if buf:
        yield f"live_chunk_{idx}", buf, end_line

# ========= Live 内存队列生产者 =========
class _LiveProducer(threading.Thread):
    def __init__(self, file_path: str, vocab, queue: Queue, chunk_size: int = 1000, start_line: int = 0):
        super().__init__(daemon=True)
        self.file_path = file_path
        self.vocab = vocab
        self.queue = queue
        self.chunk_size = max(1, int(chunk_size))
        self.start_line = max(0, int(start_line))
        self._stop_evt = threading.Event()
        self.end_line = self.start_line

    def stop(self):
        self._stop_evt.set()

    def run(self):
        try:
            from src.data_processing.prepare_binary_data import process_dialogue_to_tensors
            buf = []
            idx = 0
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for ln, line in enumerate(f):
                    if self._stop_evt.is_set():
                        break
                    if ln < self.start_line:
                        continue
                    try:
                        dialogue = json.loads(line)
                        item = process_dialogue_to_tensors(dialogue, self.vocab)
                        if item:
                            buf.append(item)
                        self.end_line = ln + 1  # 下次从该行开始
                        if len(buf) >= self.chunk_size:
                            self.queue.put((f"live_chunk_{idx}", buf, self.end_line))
                            buf = []
                            idx += 1
                    except Exception:
                        continue
            if buf and not self._stop_evt.is_set():
                self.queue.put((f"live_chunk_{idx}", buf, self.end_line))
        finally:
            # 发送哨兵，通知消费端结束
            try:
                self.queue.put(None)
            except Exception:
                pass

# ========= Chunk 流式加载工具 =========
from typing import List

def _sorted_chunk_paths(directory: str) -> List[str]:
    paths = glob.glob(os.path.join(directory, 'chunk_*.pt'))
    paths = sorted(paths, key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
    return paths

class _ChunkListDataset(Dataset):
    """将内存中的 chunk 列表封装为 Dataset 以配合 DataLoader 使用"""
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

def _load_chunk(path: str):
    return torch.load(path, weights_only=True)

def stream_chunks(directory: str, prefetch: bool = True):  # prefetch workers从config读取
    """逐块流式加载，当前块训练时预取下一块"""
    paths = _sorted_chunk_paths(directory)
    if not paths:
        raise FileNotFoundError(f"在 '{directory}' 中未找到数据块文件 (chunk_*.pt)。")
    future = None
    executor = ThreadPoolExecutor(max_workers=getattr(config, 'STREAM_PREFETCH_WORKERS', 1)) if prefetch else None
    for i, path in enumerate(paths):
        if executor:
            if future is None:
                future = executor.submit(_load_chunk, path)
            data = future.result()
            future = executor.submit(_load_chunk, paths[i+1]) if i+1 < len(paths) else None
        else:
            data = _load_chunk(path)
        yield os.path.basename(path), data
        del data
        gc.collect()
    if executor:
        executor.shutdown(wait=True)



def validate_model(model, epoch=None, optimizer=None, csv_writer=None):
    """验证模型：支持流式按 chunk 验证以降低内存占用。"""
    model.eval()
    val_data_path = os.path.join(config.LCCC_PROCESSED_PATH, "valid")
    if not os.path.exists(val_data_path):
        print("警告：找不到二进制验证数据目录 valid/，跳过验证。")
        return float('inf')

    total_val_loss = 0.0
    total_steps = 0

    # 抽样大小配置
    sample_size = int(getattr(config, 'VALIDATE_SAMPLE_SIZE', 0) or 0)
    shuffle_when_subsample = bool(getattr(config, 'VALIDATE_SHUFFLE_WHEN_SUBSAMPLE', True))

    with torch.inference_mode():
        # 验证阶段使用硬采样
        model.dynamic_group.core_rnn.use_hard_sampling = True

        if getattr(config, 'USE_STREAMING_VALIDATE', True):
            # 按训练同样方式流式加载验证集
            try:
                seen_samples = 0
                for chunk_name, chunk_data in stream_chunks(val_data_path, prefetch=getattr(config, 'STREAM_PREFETCH', True)):
                    try:
                        # 若启用抽样，则在 chunk 级别进行截断抽样
                        if sample_size > 0:
                            if shuffle_when_subsample:
                                import random as _rnd
                                _rnd.shuffle(chunk_data)
                            remaining = max(0, sample_size - seen_samples)
                            if remaining <= 0:
                                break
                            chunk_data = chunk_data[:remaining]

                        val_dataset = _ChunkListDataset(chunk_data)
                        # 注意：pin_memory/num_workers 可通过配置控制
                        val_loader = DataLoader(
                            val_dataset,
                            batch_size=getattr(config, 'VAL_BATCH_SIZE', config.BATCH_SIZE),
                            collate_fn=binary_collate_fn,
                            num_workers=getattr(config, 'VALIDATE_NUM_WORKERS', getattr(config, 'STREAM_DATALOADER_NUM_WORKERS', 0)),
                            shuffle=False,
                            pin_memory=bool(getattr(config, 'VALIDATE_PIN_MEMORY', False)),
                        )
                        pbar = tqdm(val_loader, desc=f"正在验证 | {chunk_name}")
                        for x_ref_padded, steps_data in pbar:
                            x_ref_padded = x_ref_padded.to(DEVICE, non_blocking=bool(getattr(config, 'VALIDATE_PIN_MEMORY', False)))
                            # 预编码 x_ref 并复用到每个内部步
                            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                                x_ref_emb = model.embed(x_ref_padded)
                                x_ref_mask = (x_ref_padded != config.PAD_token)
                                x_ref_encoded, _ = model.dynamic_group.encoder_rnn(x_ref_emb)
                            h_prev = torch.zeros(x_ref_padded.size(0), config.DYNAMIC_GROUP_HIDDEN_DIM, device=DEVICE)

                            for x_t, y_t, _ in steps_data:
                                x_t = x_t.to(DEVICE, non_blocking=bool(getattr(config, 'VALIDATE_PIN_MEMORY', False))) if x_t is not None else None
                                y_t = y_t.to(DEVICE, non_blocking=bool(getattr(config, 'VALIDATE_PIN_MEMORY', False))) if y_t is not None else None

                                with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                                    h_next, gate_pred, output_logits = model(
                                        x_t, x_ref_padded, h_prev, temperature=0.1,
                                        x_ref_encoded=x_ref_encoded, x_ref_mask=x_ref_mask
                                    )

                                step_loss, comps = compute_loss(
                                    gate_pred, None, output_logits, y_t,
                                    gate_entropy_weight=getattr(config, 'GATE_ENTROPY_WEIGHT', 0.0),
                                    target_speak_ratio=getattr(config, 'TARGET_SPEAK_RATIO', None),
                                    return_components=True,
                                )

                                total_val_loss += step_loss.item()
                                # 记录到 CSV（验证分项）：标注 split=val
                                if csv_writer is not None:
                                    epoch_val = (epoch + 1) if epoch is not None else ''
                                    lr_val = optimizer.param_groups[0]['lr'] if optimizer is not None else ''
                                    csv_writer.writerow([epoch_val, total_steps, step_loss.item(), lr_val, '', f"{comps.get('token_ce', 0.0):.6f}", f"{comps.get('gate_bce', 0.0):.6f}", f"{comps.get('think_nce', 0.0):.6f}", 'val'])
                                h_prev = h_next
                                total_steps += 1
                        seen_samples += len(val_dataset)
                    finally:
                        # 显式释放当前 chunk 引用（stream_chunks 内也会释放其内部数据）
                        del chunk_data
                        gc.collect()
                    if sample_size > 0 and seen_samples >= sample_size:
                        break
            except FileNotFoundError as e:
                print(f"警告：验证数据块加载失败: {e}")
                return float('inf')
        else:
            # 回退到整目录 DataLoader（非流式）
            try:
                val_dataset = BinaryDialogueDataset(val_data_path)
            except FileNotFoundError as e:
                print(f"警告：初始化验证数据集失败: {e}")
                return float('inf')

            # 若启用抽样：先取前 sample_size 条（可选打乱），再构造 DataLoader
            if sample_size > 0:
                indices = list(range(len(val_dataset)))
                if shuffle_when_subsample:
                    import random as _rnd
                    _rnd.shuffle(indices)
                indices = indices[:sample_size]
                subset = [val_dataset[i] for i in indices]
                val_dataset = _ChunkListDataset(subset)

            val_loader = DataLoader(
                val_dataset,
                batch_size=getattr(config, 'VAL_BATCH_SIZE', config.BATCH_SIZE),
                collate_fn=binary_collate_fn,
                num_workers=getattr(config, 'VALIDATE_NUM_WORKERS', min(os.cpu_count(), 2)),
                pin_memory=bool(getattr(config, 'VALIDATE_PIN_MEMORY', False)),
                persistent_workers=True,
                prefetch_factor=2,
                shuffle=False,
            )
            pbar = tqdm(val_loader, desc="正在验证")
            for x_ref_padded, steps_data in pbar:
                x_ref_padded = x_ref_padded.to(DEVICE, non_blocking=bool(getattr(config, 'VALIDATE_PIN_MEMORY', False)))
                with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    x_ref_emb = model.embed(x_ref_padded)
                    x_ref_mask = (x_ref_padded != config.PAD_token)
                    x_ref_encoded, _ = model.dynamic_group.encoder_rnn(x_ref_emb)
                h_prev = torch.zeros(x_ref_padded.size(0), config.DYNAMIC_GROUP_HIDDEN_DIM, device=DEVICE)

                for x_t, y_t, _ in steps_data:
                    x_t = x_t.to(DEVICE, non_blocking=bool(getattr(config, 'VALIDATE_PIN_MEMORY', False))) if x_t is not None else None
                    y_t = y_t.to(DEVICE, non_blocking=bool(getattr(config, 'VALIDATE_PIN_MEMORY', False))) if y_t is not None else None

                    with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                        h_next, gate_pred, output_logits = model(
                            x_t, x_ref_padded, h_prev, temperature=0.1,
                            x_ref_encoded=x_ref_encoded, x_ref_mask=x_ref_mask
                        )

                    step_loss = compute_loss(
                        gate_pred, None, output_logits, y_t,
                        gate_entropy_weight=getattr(config, 'GATE_ENTROPY_WEIGHT', 0.0),
                        target_speak_ratio=getattr(config, 'TARGET_SPEAK_RATIO', None),
                    )

                    total_val_loss += step_loss.item()
                    h_prev = h_next
                    total_steps += 1

    # 恢复软采样
    model.dynamic_group.core_rnn.use_hard_sampling = False
    avg_val_loss = total_val_loss / total_steps if total_steps > 0 else 0
    model.train()
    return avg_val_loss


def test_model(model):
    """测试集评估：支持流式按 chunk 测试，只计算损失（与验证一致）。"""
    model.eval()
    test_data_path = os.path.join(config.LCCC_PROCESSED_PATH, "test")
    if not os.path.exists(test_data_path):
        print("警告：找不到二进制测试数据目录 test/，跳过测试。")
        return float('inf')

    total_test_loss = 0.0
    total_steps = 0

    with torch.inference_mode():
        model.dynamic_group.core_rnn.use_hard_sampling = True

        if getattr(config, 'USE_STREAMING_TEST', True):
            try:
                for chunk_name, chunk_data in stream_chunks(test_data_path, prefetch=getattr(config, 'STREAM_PREFETCH', True)):
                    try:
                        test_dataset = _ChunkListDataset(chunk_data)
                        test_loader = DataLoader(
                            test_dataset,
                            batch_size=getattr(config, 'TEST_BATCH_SIZE', getattr(config, 'VAL_BATCH_SIZE', config.BATCH_SIZE)),
                            collate_fn=binary_collate_fn,
                            num_workers=getattr(config, 'TEST_NUM_WORKERS', getattr(config, 'STREAM_DATALOADER_NUM_WORKERS', 0)),
                            shuffle=False,
                            pin_memory=bool(getattr(config, 'TEST_PIN_MEMORY', False)),
                        )
                        pbar = tqdm(test_loader, desc=f"正在测试 | {chunk_name}")
                        for x_ref_padded, steps_data in pbar:
                            x_ref_padded = x_ref_padded.to(DEVICE, non_blocking=bool(getattr(config, 'TEST_PIN_MEMORY', False)))
                            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                                x_ref_emb = model.embed(x_ref_padded)
                                x_ref_mask = (x_ref_padded != config.PAD_token)
                                x_ref_encoded, _ = model.dynamic_group.encoder_rnn(x_ref_emb)
                            h_prev = torch.zeros(x_ref_padded.size(0), config.DYNAMIC_GROUP_HIDDEN_DIM, device=DEVICE)

                            for x_t, y_t, _ in steps_data:
                                x_t = x_t.to(DEVICE, non_blocking=bool(getattr(config, 'TEST_PIN_MEMORY', False))) if x_t is not None else None
                                y_t = y_t.to(DEVICE, non_blocking=bool(getattr(config, 'TEST_PIN_MEMORY', False))) if y_t is not None else None

                                with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                                    h_next, gate_pred, output_logits = model(
                                        x_t, x_ref_padded, h_prev, temperature=0.1,
                                        x_ref_encoded=x_ref_encoded, x_ref_mask=x_ref_mask
                                    )

                                step_loss, comps = compute_loss(
                                    gate_pred, None, output_logits, y_t,
                                    gate_entropy_weight=getattr(config, 'GATE_ENTROPY_WEIGHT', 0.0),
                                    target_speak_ratio=getattr(config, 'TARGET_SPEAK_RATIO', None),
                                    return_components=True,
                                )

                                total_test_loss += step_loss.item()
                                h_prev = h_next
                                total_steps += 1  # 可选：这里也可以累计 comps 以打印更细测试分项
                    finally:
                        del chunk_data
                        gc.collect()
            except FileNotFoundError as e:
                print(f"警告：测试数据块加载失败: {e}")
                return float('inf')
        else:
            try:
                test_dataset = BinaryDialogueDataset(test_data_path)
            except FileNotFoundError as e:
                print(f"警告：初始化测试数据集失败: {e}")
                return float('inf')

            test_loader = DataLoader(
                test_dataset,
                batch_size=getattr(config, 'TEST_BATCH_SIZE', getattr(config, 'VAL_BATCH_SIZE', config.BATCH_SIZE)),
                collate_fn=binary_collate_fn,
                num_workers=getattr(config, 'TEST_NUM_WORKERS', min(os.cpu_count(), 2)),
                pin_memory=bool(getattr(config, 'TEST_PIN_MEMORY', False)),
                persistent_workers=True,
                prefetch_factor=2,
                shuffle=False,
            )
            pbar = tqdm(test_loader, desc="正在测试")
            for x_ref_padded, steps_data in pbar:
                x_ref_padded = x_ref_padded.to(DEVICE, non_blocking=bool(getattr(config, 'TEST_PIN_MEMORY', False)))
                with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    x_ref_emb = model.embed(x_ref_padded)
                    x_ref_mask = (x_ref_padded != config.PAD_token)
                    x_ref_encoded, _ = model.dynamic_group.encoder_rnn(x_ref_emb)
                h_prev = torch.zeros(x_ref_padded.size(0), config.DYNAMIC_GROUP_HIDDEN_DIM, device=DEVICE)

                for x_t, y_t, _ in steps_data:
                    x_t = x_t.to(DEVICE, non_blocking=bool(getattr(config, 'TEST_PIN_MEMORY', False))) if x_t is not None else None
                    y_t = y_t.to(DEVICE, non_blocking=bool(getattr(config, 'TEST_PIN_MEMORY', False))) if y_t is not None else None

                    with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                        h_next, gate_pred, output_logits = model(
                            x_t, x_ref_padded, h_prev, temperature=0.1,
                            x_ref_encoded=x_ref_encoded, x_ref_mask=x_ref_mask
                        )

                    step_loss = compute_loss(
                        gate_pred, None, output_logits, y_t,
                        gate_entropy_weight=getattr(config, 'GATE_ENTROPY_WEIGHT', 0.0),
                        target_speak_ratio=getattr(config, 'TARGET_SPEAK_RATIO', None),
                    )

                    total_test_loss += step_loss.item()
                    h_prev = h_next
                    total_steps += 1

    model.dynamic_group.core_rnn.use_hard_sampling = False
    avg_test_loss = total_test_loss / total_steps if total_steps > 0 else 0
    model.train()
    return avg_test_loss

def save_checkpoint(model, optimizer, scaler, epoch, steps, directory, is_best=False):
    if not os.path.exists(directory): os.makedirs(directory)
    filename = f"checkpoint_epoch_{epoch}_steps_{steps}.pth"
    filepath = os.path.join(directory, filename)

    checkpoints = sorted(glob.glob(os.path.join(directory, "checkpoint_*.pth")), key=os.path.getmtime)
    if len(checkpoints) >= config.MAX_CHECKPOINTS_TO_KEEP:
        for old_checkpoint in checkpoints[:len(checkpoints) - config.MAX_CHECKPOINTS_TO_KEEP + 1]:
            os.remove(old_checkpoint)

    model_state = model.state_dict()
    torch.save({
        'epoch': epoch, 'total_steps': steps,
        'model_state_dict': model_state, 'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }, filepath)

    print(f"检查点已保存至 {filepath}")
    if is_best:
        best_filepath = os.path.join(config.BEST_MODEL_DIR, "best_model.pth")
        os.makedirs(config.BEST_MODEL_DIR, exist_ok=True)
        torch.save({'model_state_dict': model_state}, best_filepath)
        print(f"*** 新的最佳模型已保存至 {best_filepath} ***")

def load_latest_checkpoint(model, optimizer, scaler, directory):
    # 替换为 utils 中的函数，但保留 optimizer 和 scaler 加载
    if not os.path.exists(directory): return 0, 0
    checkpoints = glob.glob(os.path.join(directory, "checkpoint_*.pth"))
    if not checkpoints: return 0, 0

    latest_checkpoint_path = max(checkpoints, key=os.path.getmtime)
    print(f"正在从最新的检查点加载: {latest_checkpoint_path}")
    # 显式指定 weights_only=False（我们需要优化器/Scaler等对象）
    checkpoint = torch.load(latest_checkpoint_path, map_location=DEVICE, weights_only=False)

    # 安全部分加载：仅加载与当前模型形状一致的权重，忽略形状不匹配的层
    try:
        ckpt_state = checkpoint.get('model_state_dict', {})
        model_state = model.state_dict()
        filtered_state = {}
        skipped = []
        for k, v in ckpt_state.items():
            if k in model_state and hasattr(v, 'shape') and hasattr(model_state[k], 'shape'):
                if v.shape == model_state[k].shape:
                    filtered_state[k] = v
                else:
                    skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            else:
                # 若不存在或不是张量（如缓冲区），尝试保留
                if k in model_state:
                    try:
                        filtered_state[k] = v
                    except Exception:
                        skipped.append((k, 'NA', 'NA'))
        incompatible = model.load_state_dict(filtered_state, strict=False)
        if skipped:
            print(f"提示：跳过 {len(skipped)} 个形状不匹配的层（例如 {skipped[0][0]}: {skipped[0][1]} -> {skipped[0][2]}）。")
        # 可选打印 missing/unexpected
        try:
            if incompatible.missing_keys or incompatible.unexpected_keys:
                print(f"提示：部分权重未匹配。missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}")
        except Exception:
            pass
    except Exception as e:
        print(f"警告：加载模型权重时发生异常，尝试忽略不兼容层：{e}")

    # 优化器与 Scaler
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 将优化器状态迁移到当前设备
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)
    except Exception as e:
        print(f"警告：优化器状态加载失败（可能是优化器类型变化）。将使用新优化器。详情: {e}")
    if 'scaler_state_dict' in checkpoint:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except Exception:
            pass

    start_epoch = checkpoint.get('epoch', 0)
    total_steps = checkpoint.get('total_steps', 0)

    # 打印部分加载信息（已在上方以 incompatible 对象形式显示）
    try:
        _ = incompatible  # 仅为避免未使用警告
    except Exception:
        pass

    print(f"已从 '{latest_checkpoint_path}' 加载 (轮次 {start_epoch}, 总步数 {total_steps})。")
    return start_epoch, total_steps

def train_model():
    try:
        print("开始训练流程...")
        logging.info("开始训练")

        vocab_path = os.path.join(config.LCCC_PROCESSED_PATH, "vocabulary.json")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        # 修正：JSON 会将数值键序列化为字符串，需将 index2word 的键转为 int，避免日志解码成 <UNK:*>
        try:
            if isinstance(vocab_dict.get('index2word'), dict):
                vocab_dict['index2word'] = {int(k): v for k, v in vocab_dict['index2word'].items()}
        except Exception:
            pass
        vocab = Vocabulary("lccc")
        vocab.__dict__.update(vocab_dict)
        config.VOCAB_SIZE = vocab.num_words
        print(f"词汇表加载完毕。大小: {vocab.num_words}")

        # 新增：启用 TF32 以加速（Ampere+ GPU）
        if torch.cuda.is_available() and getattr(config, 'ALLOW_TF32', True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # 使用 arch_config.CONTEXT 中的 heads/type（若存在），与新模块化上下文保持一致
        heads = None
        attn_type = None
        try:
            if _arch_cfg is not None and hasattr(_arch_cfg, 'CONTEXT'):
                heads = int(_arch_cfg.CONTEXT.get('heads', config.NUM_ATTENTION_HEADS))
                attn_type = str(_arch_cfg.CONTEXT.get('type', config.ATTENTION_TYPE))
        except Exception:
            heads = None
            attn_type = None
        if heads is None:
            heads = config.NUM_ATTENTION_HEADS
        if attn_type is None:
            attn_type = config.ATTENTION_TYPE

        model = HGD_MemNet(
            vocab_size=config.VOCAB_SIZE, embed_dim=config.EMBEDDING_DIM,
            dynamic_hidden_dim=config.DYNAMIC_GROUP_HIDDEN_DIM,
            static_hidden_dim=config.STATIC_HEAD_HIDDEN_DIM,
            num_attention_heads=heads
        ).to(DEVICE)

        # 打印模型架构信息
        if heads == 0:
            print("模型架构: 纯HGD-MemNet (无注意力机制)")
        elif heads == 1:
            print(f"模型架构: HGD-MemNet + 单头注意力 ({attn_type})")
        else:
            print(f"模型架构: HGD-MemNet + {heads}头注意力 ({attn_type})")

        # 新增：训练时设置软采样
        model.dynamic_group.core_rnn.use_hard_sampling = False

        print("注意：由于动态学习率调整，模型JIT脚本化已被禁用。")

        # 优化器（优化2）：AdamW 默认 + Adam 兼容 + no_decay 参数分组
        decay_params, no_decay_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_bias = n.endswith('.bias')
            is_norm = ('norm' in n.lower()) or ('layernorm' in n.lower()) or ('.bn' in n.lower())
            is_embed = ('embed' in n.lower()) or ('embedding' in n.lower())
            if (p.ndim >= 2) and (not is_bias) and (not is_norm) and (not is_embed):
                decay_params.append(p)
            else:
                no_decay_params.append(p)
        param_groups = [
            {'params': decay_params, 'weight_decay': getattr(config, 'WEIGHT_DECAY', 1e-2)},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        if getattr(config, 'OPTIMIZER', 'adam').lower() == 'adamw':
            optimizer = torch.optim.AdamW(param_groups, lr=config.LEARNING_RATE)
        else:
            optimizer = torch.optim.Adam(param_groups, lr=config.LEARNING_RATE)
        scaler = GradScaler(enabled=torch.cuda.is_available())

        start_epoch, total_steps = load_latest_checkpoint(model, optimizer, scaler, config.CHECKPOINT_DIR)

        train_data_dir = os.path.join(config.LCCC_PROCESSED_PATH, "train")

        best_val_loss = float('inf')

        # 学习率调度器（优化2）：step/plateau 按配置选择
        if getattr(config, 'LR_SCHEDULER', 'none').lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=getattr(config, 'LR_DECAY_RATE', 0.95))
        elif getattr(config, 'LR_SCHEDULER', 'none').lower() == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=getattr(config, 'LR_PLATEAU_FACTOR', 0.5),
                patience=getattr(config, 'LR_PLATEAU_PATIENCE', 5),
                min_lr=getattr(config, 'LR_PLATEAU_MIN_LR', 1e-6),
                verbose=True,
            )
        else:
            scheduler = None

        # 统一随机种子（如设置）
        from datetime import datetime
        if getattr(config, 'SEED', None) is not None:
            set_seed(config.SEED, getattr(config, 'DETERMINISTIC', False))
        else:
            # 若未指定，仍可记录一次性随机种子，便于复现
            try:
                seed_now = int(datetime.now().timestamp())
                set_seed(seed_now, False)
                print(f"本次运行随机种子: {seed_now}")
            except Exception:
                pass

        # TensorBoard/CSV 记录器初始化（可选）
        writer = None
        if getattr(config, 'USE_TENSORBOARD', False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(log_dir=getattr(config, 'TENSORBOARD_LOG_DIR', './runs'))
            except Exception as e:
                print(f"TensorBoard 初始化失败: {e}")
                writer = None
        csv_f = None
        csv_writer = None
        if getattr(config, 'USE_CSV_LOGGER', False):
            os.makedirs(os.path.dirname(getattr(config, 'CSV_LOG_PATH', './logs/train_metrics.csv')), exist_ok=True)
            import csv
            csv_f = open(getattr(config, 'CSV_LOG_PATH', './logs/train_metrics.csv'), 'a', newline='', encoding='utf-8')
            csv_writer = csv.writer(csv_f)
            # 写入表头（若为空）
            if csv_f.tell() == 0:
                csv_writer.writerow(['epoch', 'total_steps', 'avg_step_loss', 'lr', 'gate_mean', 'token_ce', 'gate_bce', 'think_nce', 'split'])

        # 详细日志全局计数与下一个阈值（避免跨轮次重置导致连发）
        try:
            train_model._files_counter
        except Exception:
            train_model._files_counter = 0
        try:
            n_detail = int(getattr(config, 'PRINT_DETAIL_EVERY_N_STEPS', 0) or 0)
        except Exception:
            n_detail = 0
        if n_detail > 0:
            # 如果尚未设置下一阈值，则设为严格大于当前计数的最近倍数
            if not hasattr(train_model, '_detail_next_at') or train_model._detail_next_at is None:
                k = (train_model._files_counter // n_detail) + 1
                train_model._detail_next_at = k * n_detail
        else:
            train_model._detail_next_at = None

        # 统一的详细日志输出器：支持落盘与控制台摘要
        def _emit_detail_report(lines_summary: list[str], files_counter_val: int, total_steps_val: int, avg_loss_val: float, chunk_name_val: str | None = None):
            to_console = bool(getattr(config, 'DETAIL_LOG_TO_CONSOLE', True))
            to_file = bool(getattr(config, 'DETAIL_LOG_TO_FILE', True))
            try:
                if to_file:
                    os.makedirs('logs', exist_ok=True)
                    with open(os.path.join('logs', 'detail_report.txt'), 'a', encoding='utf-8') as f:
                        f.write("\n".join(lines_summary))
                        f.write("\n")
                if to_console:
                    print("\n".join(lines_summary))
                else:
                    # 控制台只打一行摘要，避免刷屏
                    if chunk_name_val is None:
                        print(f"[detail] saved files={files_counter_val} steps={total_steps_val} avg={avg_loss_val:.4f}")
                    else:
                        print(f"[detail] chunk={chunk_name_val} saved files={files_counter_val} steps={total_steps_val} avg={avg_loss_val:.4f}")
            except Exception:
                pass

        for epoch in range(start_epoch, config.NUM_EPOCHS):
            if getattr(config, 'ENABLE_LIVE_STREAM_TRAIN', False) and getattr(config, 'LIVE_STREAM_MODE', 'memory') == 'memory':
                # ========= 在线内存流式训练：逐行处理 jsonl 并按内存 chunk 训练 =========
                print("启用在线内存流式训练（内存队列 / 按行处理 jsonl）")
                # 读取进度
                progress_path = os.path.join(getattr(config, 'CHECKPOINT_DIR', './checkpoints'), 'train_progress.json')
                start_line = 0
                try:
                    if os.path.exists(progress_path):
                        with open(progress_path, 'r', encoding='utf-8') as pf:
                            prog = json.load(pf)
                            start_line = int(prog.get('live_start_line', 0) or 0)
                except Exception:
                    start_line = 0
                print(f"从 jsonl 第 {start_line} 行开始流式训练…")

                for chunk_name, chunk_data, end_line in stream_live_jsonl(
                    getattr(config, 'LCCC_TRAIN_FILE', ''), vocab,
                    chunk_size=getattr(config, 'LIVE_STREAM_CHUNK_SIZE', 1000),
                    start_line=start_line,
                ):
                    try:
                        train_dataset = _ChunkListDataset(chunk_data)
                        train_loader = DataLoader(
                            train_dataset,
                            batch_size=config.BATCH_SIZE,
                            collate_fn=binary_collate_fn,
                            num_workers=getattr(config, 'STREAM_DATALOADER_NUM_WORKERS', 0),
                            shuffle=True,
                            pin_memory=torch.cuda.is_available(),
                        )
                        pbar = tqdm(train_loader, desc=f"轮次 {epoch+1}/{config.NUM_EPOCHS} | {chunk_name}")
                        for x_ref_padded, steps_data in pbar:
                            batch_loss, steps_in_batch, gate_mean, gate_entropy, cap_hit, comps_avg, detail_trace = train_batch_stepwise(
                                x_ref_padded, steps_data, model, optimizer, scaler, global_total_steps=total_steps
                            )
                            total_steps += steps_in_batch
                            avg_loss = batch_loss / max(1, steps_in_batch)
                            pbar.set_postfix({
                                "平均步损失": f"{avg_loss:.4f}",
                                "token_ce": f"{comps_avg.get('token_ce', 0.0):.4f}",
                                "token_ce_eval": f"{comps_avg.get('token_ce_eval', 0.0):.4f}",
                                "gate_bce": f"{comps_avg.get('gate_bce', 0.0):.4f}",
                                "think_nce": f"{comps_avg.get('think_nce', 0.0):.4f}",
                                "有标签步占比": f"{comps_avg.get('target_step_count', 0.0):.2f}",
                                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                                "τ(温度)": f"{comps_avg.get('temperature', 0.0):.3f}",
                                "cell.maxP": f"{comps_avg.get('cell_avg_row_max', 0.0):.3f}",
                                "cell.ent": f"{comps_avg.get('cell_avg_row_entropy', 0.0):.3f}",
                                "sam.cover": f"{comps_avg.get('sampler_coverage_ratio', 0.0):.3f}",
                                "总更新步数": total_steps,
                                "门控均值": f"{gate_mean:.3f}",
                                "cap触发": cap_hit,
                            })
                            # 可选：按文件粒度的详细打印（每第 PRINT_DETAIL_EVERY_N_STEPS 个“文件”后）
                            try:
                                # 严格阈值：仅当累计样本数 >= 下一阈值时打印，然后推进下一阈值
                                if train_model._detail_next_at is not None:
                                    try:
                                        train_model._files_counter += int(x_ref_padded.size(0))
                                    except Exception:
                                        pass
                                    if train_model._files_counter >= train_model._detail_next_at:
                                        # 汇总成多行文本（落盘或摘要）
                                        lines = []
                                        try:
                                            toks = _decode_ids_to_tokens(x_ref_padded[0], vocab, max_len=64)
                                        except Exception:
                                            toks = []
                                        try:
                                            L0 = int((x_ref_padded[0] != config.PAD_token).sum().item())
                                            emb_shape = [L0, getattr(config, 'EMBEDDING_DIM', None)]
                                        except Exception:
                                            emb_shape = None
                                        lines.append("===== [DETAIL REPORT] =====")
                                        lines.append(f"chunk={chunk_name} batch_size={x_ref_padded.size(0) if hasattr(x_ref_padded, 'size') else '?'} total_files={train_model._files_counter} total_steps={total_steps} avg_loss={avg_loss:.6f}")
                                        lines.append(f"dialogue_tokens[0]={' '.join(toks)}")
                                        lines.append(f"embed_shape_for_sample0={emb_shape}")
                                        lines.append(f"lr={optimizer.param_groups[0]['lr']:.6f} acc_steps={int(getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1) or 1)} temp0={getattr(config, 'INITIAL_TEMPERATURE', 1.0)} decay={getattr(config, 'TEMPERATURE_DECAY', 0.95)}")
                                        for rec in (detail_trace or [])[:min(64, len(detail_trace or []))]:
                                            xt = rec.get('x_t_ids')
                                            xt_dec = ''
                                            try:
                                                if xt:
                                                    xt_dec = ' '.join([vocab.index2word.get(i, f"<UNK:{i}>") for i in xt])
                                            except Exception:
                                                pass
                                            pred_tok = ''
                                            try:
                                                pid = rec.get('pred_token')
                                                if pid is not None:
                                                    pred_tok = vocab.index2word.get(pid, f"<UNK:{pid}>")
                                            except Exception:
                                                pass
                                            topk = rec.get('topk') or []
                                            topk_fmt = ''
                                            try:
                                                if topk:
                                                    topk_fmt = ', '.join([f"{vocab.index2word.get(i, f'<UNK:{i}>')}:{p:.6f}" for i, p in topk])
                                            except Exception:
                                                pass
                                            lines.append(f"t={rec.get('t')} speak={rec.get('is_speak_step')} temp={rec.get('temperature'):.3f} gate_p={rec.get('gate_prob_mean'):.4f} loss={rec.get('loss'):.6f} pred='{pred_tok}' top5=[{topk_fmt}] x_t='{xt_dec}' comps={rec.get('comps')}")
                                        lines.append("===== [END DETAIL] =====")
                                        _emit_detail_report(lines, train_model._files_counter, total_steps, avg_loss, chunk_name)
                                        # 推进下一阈值
                                        train_model._detail_next_at += int(getattr(config, 'PRINT_DETAIL_EVERY_N_STEPS', 0) or 0)
                            except Exception:
                                pass

                            if writer is not None:
                                writer.add_scalar('train/avg_step_loss', avg_loss, total_steps)
                                writer.add_scalar('train/token_ce', comps_avg.get('token_ce', 0.0), total_steps)
                                writer.add_scalar('train/gate_bce', comps_avg.get('gate_bce', 0.0), total_steps)
                                writer.add_scalar('train/think_nce', comps_avg.get('think_nce', 0.0), total_steps)
                                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], total_steps)
                                writer.add_scalar('train/gate_mean', gate_mean, total_steps)
                                writer.add_scalar('train/gate_entropy', gate_entropy, total_steps)
                                writer.add_scalar('train/cap_hit', cap_hit, total_steps)
                                if 'temperature' in comps_avg:
                                    writer.add_scalar('train/temperature', comps_avg.get('temperature', 0.0), total_steps)
                                if 'cell_avg_row_max' in comps_avg:
                                    writer.add_scalar('train/cell_avg_row_max', comps_avg.get('cell_avg_row_max', 0.0), total_steps)
                                if 'cell_avg_row_entropy' in comps_avg:
                                    writer.add_scalar('train/cell_avg_row_entropy', comps_avg.get('cell_avg_row_entropy', 0.0), total_steps)
                                if 'sampler_avg_topk_weight' in comps_avg:
                                    writer.add_scalar('train/sampler_avg_topk_weight', comps_avg.get('sampler_avg_topk_weight', 0.0), total_steps)
                                if 'sampler_avg_topk_max' in comps_avg:
                                    writer.add_scalar('train/sampler_avg_topk_max', comps_avg.get('sampler_avg_topk_max', 0.0), total_steps)
                                if 'sampler_coverage_ratio' in comps_avg:
                                    writer.add_scalar('train/sampler_coverage_ratio', comps_avg.get('sampler_coverage_ratio', 0.0), total_steps)
                            if csv_writer is not None:
                                csv_writer.writerow([epoch+1, total_steps, avg_loss, optimizer.param_groups[0]['lr'], f"{gate_mean:.4f}", f"{comps_avg.get('token_ce', 0.0):.6f}", f"{comps_avg.get('gate_bce', 0.0):.6f}", f"{comps_avg.get('think_nce', 0.0):.6f}", 'train'])
                            maybe_prune_regrow(model, total_steps)
                            if total_steps > 0 and total_steps % config.SAVE_CHECKPOINT_EVERY_N_BATCHES == 0:
                                save_checkpoint(model, optimizer, scaler, epoch, total_steps, config.CHECKPOINT_DIR)
                                try:
                                    print("检查点保存完成，继续训练…")
                                except Exception:
                                    pass
                            if total_steps > 0 and total_steps % config.VALIDATE_EVERY_N_STEPS == 0:
                                val_loss = validate_model(model, epoch=epoch, optimizer=optimizer, csv_writer=csv_writer)
                                print(f"\n--- 验证 (总步数: {total_steps}) | 当前损失: {val_loss:.4f} | 最佳损失: {best_val_loss:.4f} ---")
                                # 调度器（plateau）按验证损失 step

                    finally:
                        # 记录进度，用于断点续训
                        try:
                            prog = {
                                'epoch': epoch+1,
                                'total_steps': total_steps,
                                'last_chunk': chunk_name,
                                'live_start_line': int(end_line),
                            }
                            os.makedirs(getattr(config, 'CHECKPOINT_DIR', './checkpoints'), exist_ok=True)
                            with open(progress_path, 'w', encoding='utf-8') as pf:
                                json.dump(prog, pf, ensure_ascii=False)
                        except Exception:
                            pass

            elif getattr(config, 'USE_STREAMING_TRAIN', True):
                # ========= 离线流式训练：逐块加载与训练 =========
                print("启用流式训练：按 chunk 逐块加载与训练（后台预取）")
                for chunk_name, chunk_data in stream_chunks(train_data_dir, prefetch=getattr(config, 'STREAM_PREFETCH', True)):
                    try:
                        train_dataset = _ChunkListDataset(chunk_data)
                        train_loader = DataLoader(
                            train_dataset,
                            batch_size=config.BATCH_SIZE,
                            collate_fn=binary_collate_fn,
                            num_workers=getattr(config, 'STREAM_DATALOADER_NUM_WORKERS', 0),
                            shuffle=True,
                            pin_memory=torch.cuda.is_available(),
                        )
                        pbar = tqdm(train_loader, desc=f"轮次 {epoch+1}/{config.NUM_EPOCHS} | {chunk_name}")
                        for x_ref_padded, steps_data in pbar:
                            batch_loss, steps_in_batch, gate_mean, gate_entropy, cap_hit, comps_avg, detail_trace = train_batch_stepwise(
                                x_ref_padded, steps_data, model, optimizer, scaler, global_total_steps=total_steps
                            )
                            total_steps += steps_in_batch
                            avg_loss = batch_loss / max(1, steps_in_batch)
                            pbar.set_postfix({
                                "平均步损失": f"{avg_loss:.4f}",
                                "token_ce": f"{comps_avg.get('token_ce', 0.0):.4f}",
                                "token_ce_eval": f"{comps_avg.get('token_ce_eval', 0.0):.4f}",
                                "gate_bce": f"{comps_avg.get('gate_bce', 0.0):.4f}",
                                "think_nce": f"{comps_avg.get('think_nce', 0.0):.4f}",
                                "有标签步占比": f"{comps_avg.get('target_step_count', 0.0):.2f}",
                                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                                "τ(温度)": f"{comps_avg.get('temperature', 0.0):.3f}",
                                "cell.maxP": f"{comps_avg.get('cell_avg_row_max', 0.0):.3f}",
                                "cell.ent": f"{comps_avg.get('cell_avg_row_entropy', 0.0):.3f}",
                                "sam.cover": f"{comps_avg.get('sampler_coverage_ratio', 0.0):.3f}",
                                "总更新步数": total_steps,
                                "门控均值": f"{gate_mean:.3f}",
                                "cap触发": cap_hit,
                            })
                            # 可选：按文件粒度的详细打印（每第 PRINT_DETAIL_EVERY_N_STEPS 个“文件”后）
                            try:
                                if train_model._detail_next_at is not None:
                                    try:
                                        train_model._files_counter += int(x_ref_padded.size(0))
                                    except Exception:
                                        pass
                                    if train_model._files_counter >= train_model._detail_next_at:
                                        try:
                                            toks = _decode_ids_to_tokens(x_ref_padded[0], vocab, max_len=64)
                                        except Exception:
                                            toks = []
                                        try:
                                            L0 = int((x_ref_padded[0] != config.PAD_token).sum().item())
                                            emb_shape = [L0, getattr(config, 'EMBEDDING_DIM', None)]
                                        except Exception:
                                            emb_shape = None
                                        print("\n===== [DETAIL REPORT] =====")
                                        print(f"chunk={chunk_name} batch_size={x_ref_padded.size(0) if hasattr(x_ref_padded, 'size') else '?'} total_files={files_counter} total_steps={total_steps} avg_loss={avg_loss:.6f}")
                                        print(f"dialogue_tokens[0]={' '.join(toks)}")
                                        print(f"embed_shape_for_sample0={emb_shape}")
                                        print(f"lr={optimizer.param_groups[0]['lr']:.6f} acc_steps={int(getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1) or 1)} temp0={getattr(config, 'INITIAL_TEMPERATURE', 1.0)} decay={getattr(config, 'TEMPERATURE_DECAY', 0.95)}")
                                        for rec in (detail_trace or [])[:min(64, len(detail_trace or []))]:
                                            xt = rec.get('x_t_ids')
                                            xt_dec = ''
                                            try:
                                                if xt:
                                                    xt_dec = ' '.join([vocab.index2word.get(i, f"<UNK:{i}>") for i in xt])
                                            except Exception:
                                                pass
                                            pred_tok = ''
                                            try:
                                                pid = rec.get('pred_token')
                                                if pid is not None:
                                                    pred_tok = vocab.index2word.get(pid, f"<UNK:{pid}>")
                                            except Exception:
                                                pass
                                            topk = rec.get('topk') or []
                                            topk_fmt = ''
                                            try:
                                                if topk:
                                                    topk_fmt = ', '.join([f"{vocab.index2word.get(i, f'<UNK:{i}>')}:{p:.6f}" for i, p in topk])
                                            except Exception:
                                                pass
                                            print(f"t={rec.get('t')} speak={rec.get('is_speak_step')} temp={rec.get('temperature'):.3f} gate_p={rec.get('gate_prob_mean'):.4f} loss={rec.get('loss'):.6f} pred='{pred_tok}' top5=[{topk_fmt}] x_t='{xt_dec}' comps={rec.get('comps')}")
                                        print("===== [END DETAIL] =====\n")
                                        train_model._detail_next_at += int(getattr(config, 'PRINT_DETAIL_EVERY_N_STEPS', 0) or 0)
                            except Exception:
                                pass

                            if writer is not None:
                                writer.add_scalar('train/avg_step_loss', avg_loss, total_steps)
                                writer.add_scalar('train/token_ce', comps_avg.get('token_ce', 0.0), total_steps)
                                writer.add_scalar('train/gate_bce', comps_avg.get('gate_bce', 0.0), total_steps)
                                writer.add_scalar('train/think_nce', comps_avg.get('think_nce', 0.0), total_steps)
                                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], total_steps)
                                writer.add_scalar('train/gate_mean', gate_mean, total_steps)
                                writer.add_scalar('train/gate_entropy', gate_entropy, total_steps)
                                writer.add_scalar('train/cap_hit', cap_hit, total_steps)
                                # 新监控
                                if 'temperature' in comps_avg:
                                    writer.add_scalar('train/temperature', comps_avg.get('temperature', 0.0), total_steps)
                                if 'cell_avg_row_max' in comps_avg:
                                    writer.add_scalar('train/cell_avg_row_max', comps_avg.get('cell_avg_row_max', 0.0), total_steps)
                                if 'cell_avg_row_entropy' in comps_avg:
                                    writer.add_scalar('train/cell_avg_row_entropy', comps_avg.get('cell_avg_row_entropy', 0.0), total_steps)
                                if 'sampler_avg_topk_weight' in comps_avg:
                                    writer.add_scalar('train/sampler_avg_topk_weight', comps_avg.get('sampler_avg_topk_weight', 0.0), total_steps)
                                if 'sampler_avg_topk_max' in comps_avg:
                                    writer.add_scalar('train/sampler_avg_topk_max', comps_avg.get('sampler_avg_topk_max', 0.0), total_steps)
                                if 'sampler_coverage_ratio' in comps_avg:
                                    writer.add_scalar('train/sampler_coverage_ratio', comps_avg.get('sampler_coverage_ratio', 0.0), total_steps)
                            if csv_writer is not None:
                                csv_writer.writerow([epoch+1, total_steps, avg_loss, optimizer.param_groups[0]['lr'], f"{gate_mean:.4f}", f"{comps_avg.get('token_ce', 0.0):.6f}", f"{comps_avg.get('gate_bce', 0.0):.6f}", f"{comps_avg.get('think_nce', 0.0):.6f}", 'train'])
                            maybe_prune_regrow(model, total_steps)
                            if total_steps > 0 and total_steps % config.SAVE_CHECKPOINT_EVERY_N_BATCHES == 0:
                                save_checkpoint(model, optimizer, scaler, epoch, total_steps, config.CHECKPOINT_DIR)
                            if total_steps > 0 and total_steps % config.VALIDATE_EVERY_N_STEPS == 0:
                                val_loss = validate_model(model, epoch=epoch, optimizer=optimizer, csv_writer=csv_writer)
                                print(f"\n--- 验证 (总步数: {total_steps}) | 当前损失: {val_loss:.4f} | 最佳损失: {best_val_loss:.4f} ---")
                                # 调度器（plateau）按验证损失 step
                                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    scheduler.step(val_loss)
                                if val_loss < best_val_loss:
                                    best_val_loss = val_loss
                                    print("发现新的最佳模型！正在保存...")
                                    logging.info(f"新最佳验证损失: {val_loss}")
                                    save_checkpoint(model, optimizer, scaler, epoch, total_steps, config.CHECKPOINT_DIR, is_best=True)
                    finally:
                        del chunk_data
                        gc.collect()
            else:
                # ========= 传统整目录 DataLoader =========
                train_dataset = BinaryDialogueDataset(train_data_dir)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=config.BATCH_SIZE,
                    collate_fn=binary_collate_fn,
                    num_workers=min(os.cpu_count(), 2),
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=True,
                    prefetch_factor=2,
                    shuffle=True,
                )
                pbar = tqdm(train_loader, desc=f"轮次 {epoch+1}/{config.NUM_EPOCHS}")
                for x_ref_padded, steps_data in pbar:
                    batch_loss, steps_in_batch, gate_mean, gate_entropy, cap_hit, comps_avg, detail_trace = train_batch_stepwise(
                        x_ref_padded, steps_data, model, optimizer, scaler
                    )
                    total_steps += steps_in_batch
                    avg_loss = batch_loss / steps_in_batch if steps_in_batch > 0 else 0
                    pbar.set_postfix({
                        "平均步损失": f"{avg_loss:.4f}",
                        "token_ce": f"{comps_avg.get('token_ce', 0.0):.4f}",
                        "gate_bce": f"{comps_avg.get('gate_bce', 0.0):.4f}",
                        "think_nce": f"{comps_avg.get('think_nce', 0.0):.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                        "总更新步数": total_steps,
                        "门控均值": f"{gate_mean:.3f}",
                        "cap触发": cap_hit,
                    })
                    if writer is not None:
                        writer.add_scalar('train/avg_step_loss', avg_loss, total_steps)
                        writer.add_scalar('train/token_ce', comps_avg.get('token_ce', 0.0), total_steps)
                        writer.add_scalar('train/gate_bce', comps_avg.get('gate_bce', 0.0), total_steps)
                        writer.add_scalar('train/think_nce', comps_avg.get('think_nce', 0.0), total_steps)
                        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], total_steps)
                        writer.add_scalar('train/gate_mean', gate_mean, total_steps)
                        writer.add_scalar('train/gate_entropy', gate_entropy, total_steps)
                        writer.add_scalar('train/cap_hit', cap_hit, total_steps)
                    if csv_writer is not None:
                        csv_writer.writerow([epoch+1, total_steps, avg_loss, optimizer.param_groups[0]['lr'], f"{gate_mean:.4f}", f"{comps_avg.get('token_ce', 0.0):.6f}", f"{comps_avg.get('gate_bce', 0.0):.6f}", f"{comps_avg.get('think_nce', 0.0):.6f}", 'train'])
                    maybe_prune_regrow(model, total_steps)
                    # 可选：按文件粒度的详细打印（每第 PRINT_DETAIL_EVERY_N_STEPS 个“文件”后）
                    try:
                        n = int(getattr(config, 'PRINT_DETAIL_EVERY_N_STEPS', 0) or 0)
                        if n > 0:
                            try:
                                files_counter = getattr(train_model, '_files_counter', 0)
                            except Exception:
                                files_counter = 0
                            try:
                                files_counter += int(x_ref_padded.size(0))
                            except Exception:
                                files_counter += 0
                            setattr(train_model, '_files_counter', files_counter)
                            if files_counter % n == 0:
                                try:
                                    toks = _decode_ids_to_tokens(x_ref_padded[0], vocab, max_len=64)
                                except Exception:
                                    toks = []
                                try:
                                    L0 = int((x_ref_padded[0] != config.PAD_token).sum().item())
                                    emb_shape = [L0, getattr(config, 'EMBEDDING_DIM', None)]
                                except Exception:
                                    emb_shape = None
                                print("\n===== [DETAIL REPORT] =====")
                                print(f"total_files={files_counter} total_steps={total_steps} avg_loss={avg_loss:.6f}")
                                print(f"dialogue_tokens[0]={' '.join(toks)}")
                                print(f"embed_shape_for_sample0={emb_shape}")
                                print(f"lr={optimizer.param_groups[0]['lr']:.6f} acc_steps={int(getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1) or 1)} temp0={getattr(config, 'INITIAL_TEMPERATURE', 1.0)} decay={getattr(config, 'TEMPERATURE_DECAY', 0.95)}")
                                for rec in (detail_trace or [])[:min(64, len(detail_trace or []))]:
                                    xt = rec.get('x_t_ids')
                                    xt_dec = ''
                                    try:
                                        if xt:
                                            xt_dec = ' '.join([vocab.index2word.get(i, f"<UNK:{i}>") for i in xt])
                                    except Exception:
                                        pass
                                    pred_tok = ''
                                    try:
                                        pid = rec.get('pred_token')
                                        if pid is not None:
                                            pred_tok = vocab.index2word.get(pid, f"<UNK:{pid}>")
                                    except Exception:
                                        pass
                                    topk = rec.get('topk') or []
                                    topk_fmt = ''
                                    try:
                                        if topk:
                                            topk_fmt = ', '.join([f"{vocab.index2word.get(i, f'<UNK:{i}>')}:{p:.6f}" for i, p in topk])
                                    except Exception:
                                        pass
                                    print(f"t={rec.get('t')} speak={rec.get('is_speak_step')} temp={rec.get('temperature'):.3f} gate_p={rec.get('gate_prob_mean'):.4f} loss={rec.get('loss'):.6f} pred='{pred_tok}' top5=[{topk_fmt}] x_t='{xt_dec}' comps={rec.get('comps')}")
                                print("===== [END DETAIL] =====\n")
                    except Exception:
                        pass
                    if total_steps > 0 and total_steps % config.SAVE_CHECKPOINT_EVERY_N_BATCHES == 0:
                        save_checkpoint(model, optimizer, scaler, epoch, total_steps, config.CHECKPOINT_DIR)
                        try:
                            print("检查点保存完成，继续训练…")
                        except Exception:
                            pass
                    if total_steps > 0 and total_steps % config.VALIDATE_EVERY_N_STEPS == 0:
                        val_loss = validate_model(model, epoch=epoch, optimizer=optimizer, csv_writer=csv_writer)
                        print(f"\n--- 验证 (总步数: {total_steps}) | 当前损失: {val_loss:.4f} | 最佳损失: {best_val_loss:.4f} ---")
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            print("发现新的最佳模型！正在保存...")
                            logging.info(f"新最佳验证损失: {val_loss}")
                            save_checkpoint(model, optimizer, scaler, epoch, total_steps, config.CHECKPOINT_DIR, is_best=True)

            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler.step()  # 仅 StepLR 在每轮末 step；ReduceLROnPlateau 已在验证时 step(val_loss)
            # 轮次汇总日志
            try:
                print(f"\n[汇总] 轮次 {epoch+1}/{config.NUM_EPOCHS} 完成 | 最佳验证损失: {best_val_loss:.4f} | 总更新步: {total_steps} | 当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
                logging.info(f"epoch={epoch+1} total_steps={total_steps} best_val_loss={best_val_loss:.6f} lr={optimizer.param_groups[0]['lr']:.6f}")
            except Exception:
                pass

            # 资源回收（每轮结束）
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc as _gc
                _gc.collect()
            except Exception:
                pass

        print("\n训练全部完成！")
        logging.info("训练完成")
        # 关闭记录器
        try:
            if writer is not None:
                writer.close()
            if csv_f is not None:
                csv_f.close()
        except Exception:
            pass
    except Exception as e:
        logging.error(f"训练出错: {e}")
        raise

if __name__ == "__main__":
    # 若启用在线内存流式训练，直接进入训练主流程（不依赖预处理的 chunk 目录）
    if getattr(config, 'ENABLE_LIVE_STREAM_TRAIN', False) and getattr(config, 'LIVE_STREAM_MODE', 'memory') == 'memory':
        train_model()
    else:
        train_dir = os.path.join(config.LCCC_PROCESSED_PATH, "train")
        if not os.path.isdir(train_dir):
            print(f"错误: 未找到处理后的二进制数据目录 '{train_dir}'。")
            print("请先运行 'python -m src.data_processing.prepare_binary_data' 来生成数据，或启用 ENABLE_LIVE_STREAM_TRAIN。")
        else:
            train_model()
