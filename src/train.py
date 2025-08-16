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
import logging  # 新增：日志记录
import gc
from concurrent.futures import ThreadPoolExecutor

# 设置日志
logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 可选：确定性设置（由 config 决定）
if getattr(config, 'DETERMINISTIC', False):
    cudnn.benchmark = False
    cudnn.deterministic = True

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
    # 在不改动数据管线的前提下，cap 不能超过提供的 steps；若 max_t 为 None 则使用 steps_available
    cap = min(steps_available, max_t) if max_t is not None else steps_available

    # 评估用“样本级目标”（例如最终答案）：取该样本在本batch内最后一个非空 target 作为全局标签，用于在每个思考步评估 token_ce（不参与反传）
    final_target_padded = None
    try:
        for s in range(steps_available - 1, -1, -1):
            _tgt = steps_data[s][1]
            if _tgt is not None:
                final_target_padded = _tgt.to(DEVICE, non_blocking=True)
                break
    except Exception:
        final_target_padded = None

    for t in range(steps_available):
        x_t_padded, target_padded, gate_target = steps_data[t]
        x_t_padded = x_t_padded.to(DEVICE, non_blocking=True) if x_t_padded is not None else None
        gate_target = gate_target.to(DEVICE, non_blocking=True)
        if target_padded is not None:
            target_padded = target_padded.to(DEVICE, non_blocking=True)

        # 1. 在每个时间步内部动态调整学习率
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = original_lrs[i] * (config.INNER_STEP_LR_DECAY ** t)

        # 新增：计算当前步骤的温度
        current_temperature = max(initial_temperature * (temperature_decay ** t), config.MIN_TEMPERATURE)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            # 构造控制向量（时机意识）：t_norm, remain_norm, min_done, target_speak_ratio
            t_norm = torch.full((batch_size, 1), (t + 1) / max(cap, 1), device=DEVICE)
            remain_norm = torch.full((batch_size, 1), max(cap - (t + 1), 0) / max(cap, 1), device=DEVICE)
            min_done = torch.full((batch_size, 1), 1.0 if (min_t is not None and (t + 1) >= min_t) or (min_t is None) else 0.0, device=DEVICE)
            budget = torch.full((batch_size, 1), float(getattr(config, 'TARGET_SPEAK_RATIO', 0.0) or 0.0), device=DEVICE)
            control = torch.cat([t_norm, remain_norm, min_done, budget], dim=1)

            h_next, gate_pred, output_logits = model(x_t_padded, x_ref_padded, h_prev, temperature=current_temperature, control=control)

            # 统计门控均值/熵（gate_pred 为 logits，这里按概率统计）
            gate_prob = torch.sigmoid(gate_pred)
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

            step_loss, comps = compute_loss(
                gate_pred, gate_target, output_logits, target_padded,
                gate_entropy_weight=getattr(config, 'GATE_ENTROPY_WEIGHT', 0.0),
                target_speak_ratio=getattr(config, 'TARGET_SPEAK_RATIO', None),
                h_prev=h_prev, h_next=h_next,
                think_loss_weight=tlw_eff,
                info_nce_tau=getattr(config, 'THINK_INFO_TAU', 0.1),
                return_components=True,
            )
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

        scaler.scale(step_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_batch_loss += step_loss.item()
        h_prev = h_next.detach()
        steps_updated += 1

        # 基于门控的多步思考早停（需达到最小思考步才生效）
        if getattr(config, 'USE_GATED_MULTISTEP', False):
            reached_min = (min_t is None) or ((t + 1) >= min_t)
            if reached_min and (gate_pred.mean().item() >= config.GATE_THRESHOLD):
                break

        # 达到最大思考步上限（cap）则强制停止本样本的内部思考
        if (t + 1) >= cap:
            ended_by_cap = True
            break

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

    return total_batch_loss, steps_updated, gate_mean, gate_entropy, int(ended_by_cap), comps_avg

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

    with torch.no_grad():
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
                        val_loader = DataLoader(
                            val_dataset,
                            batch_size=config.BATCH_SIZE,
                            collate_fn=binary_collate_fn,
                            num_workers=getattr(config, 'STREAM_DATALOADER_NUM_WORKERS', 0),
                            shuffle=False,
                        )
                        pbar = tqdm(val_loader, desc=f"正在验证 | {chunk_name}")
                        for x_ref_padded, steps_data in pbar:
                            x_ref_padded = x_ref_padded.to(DEVICE, non_blocking=True)
                            h_prev = torch.zeros(x_ref_padded.size(0), config.DYNAMIC_GROUP_HIDDEN_DIM, device=DEVICE)

                            for x_t, y_t, g_t in steps_data:
                                x_t = x_t.to(DEVICE, non_blocking=True) if x_t is not None else None
                                y_t = y_t.to(DEVICE, non_blocking=True) if y_t is not None else None
                                g_t = g_t.to(DEVICE, non_blocking=True)

                                h_next, gate_pred, output_logits = model(x_t, x_ref_padded, h_prev, temperature=0.1)

                                step_loss, comps = compute_loss(
                                    gate_pred, g_t, output_logits, y_t,
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
                batch_size=config.BATCH_SIZE,
                collate_fn=binary_collate_fn,
                num_workers=min(os.cpu_count(), 2),
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True,
                prefetch_factor=2,
                shuffle=False,
            )
            pbar = tqdm(val_loader, desc="正在验证")
            for x_ref_padded, steps_data in pbar:
                x_ref_padded = x_ref_padded.to(DEVICE, non_blocking=True)
                h_prev = torch.zeros(x_ref_padded.size(0), config.DYNAMIC_GROUP_HIDDEN_DIM, device=DEVICE)

                for x_t, y_t, g_t in steps_data:
                    x_t = x_t.to(DEVICE, non_blocking=True) if x_t is not None else None
                    y_t = y_t.to(DEVICE, non_blocking=True) if y_t is not None else None
                    g_t = g_t.to(DEVICE, non_blocking=True)

                    h_next, gate_pred, output_logits = model(x_t, x_ref_padded, h_prev, temperature=0.1)

                    step_loss = compute_loss(
                        gate_pred, g_t, output_logits, y_t,
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

    with torch.no_grad():
        model.dynamic_group.core_rnn.use_hard_sampling = True

        if getattr(config, 'USE_STREAMING_TEST', True):
            try:
                for chunk_name, chunk_data in stream_chunks(test_data_path, prefetch=getattr(config, 'STREAM_PREFETCH', True)):
                    try:
                        test_dataset = _ChunkListDataset(chunk_data)
                        test_loader = DataLoader(
                            test_dataset,
                            batch_size=config.BATCH_SIZE,
                            collate_fn=binary_collate_fn,
                            num_workers=getattr(config, 'STREAM_DATALOADER_NUM_WORKERS', 0),
                            shuffle=False,
                        )
                        pbar = tqdm(test_loader, desc=f"正在测试 | {chunk_name}")
                        for x_ref_padded, steps_data in pbar:
                            x_ref_padded = x_ref_padded.to(DEVICE, non_blocking=True)
                            h_prev = torch.zeros(x_ref_padded.size(0), config.DYNAMIC_GROUP_HIDDEN_DIM, device=DEVICE)

                            for x_t, y_t, g_t in steps_data:
                                x_t = x_t.to(DEVICE, non_blocking=True) if x_t is not None else None
                                y_t = y_t.to(DEVICE, non_blocking=True) if y_t is not None else None
                                g_t = g_t.to(DEVICE, non_blocking=True)

                                h_next, gate_pred, output_logits = model(x_t, x_ref_padded, h_prev, temperature=0.1)

                                step_loss, comps = compute_loss(
                                    gate_pred, g_t, output_logits, y_t,
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
                batch_size=config.BATCH_SIZE,
                collate_fn=binary_collate_fn,
                num_workers=min(os.cpu_count(), 2),
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True,
                prefetch_factor=2,
                shuffle=False,
            )
            pbar = tqdm(test_loader, desc="正在测试")
            for x_ref_padded, steps_data in pbar:
                x_ref_padded = x_ref_padded.to(DEVICE, non_blocking=True)
                h_prev = torch.zeros(x_ref_padded.size(0), config.DYNAMIC_GROUP_HIDDEN_DIM, device=DEVICE)

                for x_t, y_t, g_t in steps_data:
                    x_t = x_t.to(DEVICE, non_blocking=True) if x_t is not None else None
                    y_t = y_t.to(DEVICE, non_blocking=True) if y_t is not None else None
                    g_t = g_t.to(DEVICE, non_blocking=True)

                    h_next, gate_pred, output_logits = model(x_t, x_ref_padded, h_prev, temperature=0.1)

                    step_loss = compute_loss(
                        gate_pred, g_t, output_logits, y_t,
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
    # 显式指定 weights_only=False（我们需要优化器/Scaler等对象）；未来可改为安全加载仅权重
    checkpoint = torch.load(latest_checkpoint_path, map_location=DEVICE, weights_only=False)

    # 允许部分加载（序列CE等新模块在旧权重中缺失）
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
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

    # 打印部分加载信息
    try:
        if missing or unexpected:
            print(f"提示：部分权重未匹配。missing={len(missing)} unexpected={len(unexpected)}")
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
        vocab = Vocabulary("lccc")
        vocab.__dict__.update(vocab_dict)
        config.VOCAB_SIZE = vocab.num_words
        print(f"词汇表加载完毕。大小: {vocab.num_words}")

        # 新增：启用 TF32 以加速（Ampere+ GPU）
        if torch.cuda.is_available() and getattr(config, 'ALLOW_TF32', True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model = HGD_MemNet(
            vocab_size=config.VOCAB_SIZE, embed_dim=config.EMBEDDING_DIM,
            dynamic_hidden_dim=config.DYNAMIC_GROUP_HIDDEN_DIM,
            static_hidden_dim=config.STATIC_HEAD_HIDDEN_DIM,
            num_attention_heads=config.NUM_ATTENTION_HEADS
        ).to(DEVICE)

        # 打印模型架构信息
        if config.NUM_ATTENTION_HEADS == 0:
            print("模型架构: 纯HGD-MemNet (无注意力机制)")
        elif config.NUM_ATTENTION_HEADS == 1:
            print(f"模型架构: HGD-MemNet + 单头注意力 ({config.ATTENTION_TYPE})")
        else:
            print(f"模型架构: HGD-MemNet + {config.NUM_ATTENTION_HEADS}头注意力 ({config.ATTENTION_TYPE})")

        # 新增：训练时设置软采样
        model.dynamic_group.core_rnn.use_hard_sampling = False

        print("注意：由于动态学习率调整，模型JIT脚本化已被禁用。")

        # 优化器（优化2）：AdamW 默认 + Adam 兼容
        if getattr(config, 'OPTIMIZER', 'adam').lower() == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=getattr(config, 'WEIGHT_DECAY', 1e-2))
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=getattr(config, 'WEIGHT_DECAY', 1e-5))
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

        for epoch in range(start_epoch, config.NUM_EPOCHS):
            if getattr(config, 'USE_STREAMING_TRAIN', True):
                # ========= 流式训练：逐块加载与训练 =========
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
                        )
                        pbar = tqdm(train_loader, desc=f"轮次 {epoch+1}/{config.NUM_EPOCHS} | {chunk_name}")
                        for x_ref_padded, steps_data in pbar:
                            batch_loss, steps_in_batch, gate_mean, gate_entropy, cap_hit, comps_avg = train_batch_stepwise(
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
                            # 可选：周期性详细打印（避免被 tqdm 覆盖）
                            try:
                                n = int(getattr(config, 'PRINT_DETAIL_EVERY_N_STEPS', 0) or 0)
                                if n > 0 and (total_steps % n == 0):
                                    print(
                                        f"[detail] step={total_steps} avg={avg_loss:.4f} "
                                        f"gate_bce={comps_avg.get('gate_bce', 0.0):.4f} "
                                        f"token_ce={comps_avg.get('token_ce', 0.0):.4f} "
                                        f"token_ce_eval={comps_avg.get('token_ce_eval', 0.0):.4f} "
                                        f"think_nce={comps_avg.get('think_nce', 0.0):.4f} "
                                        f"target_ratio={comps_avg.get('target_step_count', 0.0):.2f} "
                                        f"tau={comps_avg.get('temperature', 0.0):.3f}"
                                    )
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
                    batch_loss, steps_in_batch, gate_mean, gate_entropy, cap_hit, comps_avg = train_batch_stepwise(
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
                    if total_steps > 0 and total_steps % config.SAVE_CHECKPOINT_EVERY_N_BATCHES == 0:
                        save_checkpoint(model, optimizer, scaler, epoch, total_steps, config.CHECKPOINT_DIR)
                    if total_steps > 0 and total_steps % config.VALIDATE_EVERY_N_STEPS == 0:
                        val_loss = validate_model(model, epoch=epoch, optimizer=optimizer, csv_writer=csv_writer)
                        print(f"\n--- 验证 (总步数: {total_steps}) | 当前损失: {val_loss:.4f} | 最佳损失: {best_val_loss:.4f} ---")
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            print("发现新的最佳模型！正在保存...")
                            logging.info(f"新最佳验证损失: {val_loss}")
                            save_checkpoint(model, optimizer, scaler, epoch, total_steps, config.CHECKPOINT_DIR, is_best=True)

            scheduler.step()  # 每轮更新学习率
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
    train_dir = os.path.join(config.LCCC_PROCESSED_PATH, "train")
    if not os.path.isdir(train_dir):
        print(f"错误: 未找到处理后的二进制数据目录 '{train_dir}'。")
        print("请先运行 'python -m src.data_processing.prepare_binary_data' 来生成数据。")
    else:
        train_model()
