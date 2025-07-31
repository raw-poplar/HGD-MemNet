# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import glob
from tqdm import tqdm
from torch.cuda.amp import GradScaler

# 新增：性能优化
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # 启用 cudnn 基准测试以加速

import config
from .model import HGD_MemNet
from .dataset import Vocabulary, BinaryDialogueDataset, binary_collate_fn
from .utils import load_model_from_checkpoint, compute_loss  # 新增：导入工具函数
import logging  # 新增：日志记录

# 设置日志
logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_batch_stepwise(x_ref_padded, steps_data, model, optimizer, scaler):
    """
    按照描述进行训练：在每个“思考步骤”中都进行一次损失计算和反向传播，
    并动态调整学习率。
    """
    model.train()
    x_ref_padded = x_ref_padded.to(DEVICE)
    
    batch_size = x_ref_padded.size(0)
    total_batch_loss = 0.0
    steps_updated = 0
    
    h_prev = torch.zeros(batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM, device=DEVICE)
    model.reset_virtual_weights()  # [新增] 重置虚拟权重，确保每个批次都有一个干净的短期记忆环境
    
    original_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    
    # 新增：温度退火参数 (从config导入)
    initial_temperature = config.INITIAL_TEMPERATURE
    temperature_decay = config.TEMPERATURE_DECAY

    for t in range(len(steps_data)):
        x_t_padded, target_padded, gate_target = steps_data[t]
        x_t_padded = x_t_padded.to(DEVICE) if x_t_padded is not None else None
        gate_target = gate_target.to(DEVICE)
        if target_padded is not None:
            target_padded = target_padded.to(DEVICE)

        # 1. 在每个时间步内部动态调整学习率
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = original_lrs[i] * (config.INNER_STEP_LR_DECAY ** t)

        # 新增：计算当前步骤的温度
        current_temperature = max(initial_temperature * (temperature_decay ** t), config.MIN_TEMPERATURE)

        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            h_next, gate_pred, output_logits = model(x_t_padded, x_ref_padded, h_prev, temperature=current_temperature)  # 更新：传入温度
            
            # 使用 utils 中的 compute_loss
            step_loss = compute_loss(gate_pred, gate_target, output_logits, target_padded)
            
        # 2. 立即进行反向传播和参数更新
        scaler.scale(step_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_batch_loss += step_loss.item()
        h_prev = h_next.detach()
        steps_updated += 1
        
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = original_lrs[i]
        
    return total_batch_loss, steps_updated

def validate_model(model):
    """验证模型，只计算损失。"""
    model.eval()
    val_data_path = os.path.join(config.LCCC_PROCESSED_PATH, "valid")
    if not os.path.exists(val_data_path):
        print("警告：找不到二进制验证数据目录 valid/，跳过验证。")
        return float('inf')

    try:
        val_dataset = BinaryDialogueDataset(val_data_path)
    except FileNotFoundError as e:
        print(f"警告：初始化验证数据集失败: {e}")
        return float('inf')

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        collate_fn=binary_collate_fn,
        num_workers=min(os.cpu_count(), 2),  # 同样限制为2
        shuffle=False
    )

    total_val_loss = 0
    total_steps = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="正在验证")
        for x_ref_padded, steps_data in pbar:
            x_ref_padded = x_ref_padded.to(DEVICE)
            h_prev = torch.zeros(x_ref_padded.size(0), config.DYNAMIC_GROUP_HIDDEN_DIM, device=DEVICE)
            model.reset_virtual_weights() # [新增] 同样在验证时重置，保证评估的公平性
            
            for x_t, y_t, g_t in steps_data:
                x_t = x_t.to(DEVICE) if x_t is not None else None
                y_t = y_t.to(DEVICE) if y_t is not None else None
                g_t = g_t.to(DEVICE)

                # 新增：在验证时使用硬采样和低温
                model.dynamic_group.core_rnn.use_hard_sampling = True  # 切换到硬采样
                h_next, gate_pred, output_logits = model(x_t, x_ref_padded, h_prev, temperature=0.1)  # 低温
                model.dynamic_group.core_rnn.use_hard_sampling = False  # 恢复
                
                # 使用 utils 中的 compute_loss
                step_loss = compute_loss(gate_pred, g_t, output_logits, y_t)
                
                total_val_loss += step_loss.item()
                h_prev = h_next
                total_steps += 1

    avg_val_loss = total_val_loss / total_steps if total_steps > 0 else 0
    model.train()
    return avg_val_loss

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
    checkpoint = torch.load(latest_checkpoint_path, map_location=DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scaler_state_dict' in checkpoint: scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    total_steps = checkpoint.get('total_steps', 0)
    
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
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        model = HGD_MemNet(
            vocab_size=config.VOCAB_SIZE, embed_dim=config.EMBEDDING_DIM,
            dynamic_hidden_dim=config.DYNAMIC_GROUP_HIDDEN_DIM,
            static_hidden_dim=config.STATIC_HEAD_HIDDEN_DIM
        ).to(DEVICE)
        
        # 新增：训练时设置软采样
        model.dynamic_group.core_rnn.use_hard_sampling = False

        print("注意：由于动态学习率调整，模型JIT脚本化已被禁用。")

        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)  # 新增：L2正则
        scaler = GradScaler(enabled=torch.cuda.is_available())

        start_epoch, total_steps = load_latest_checkpoint(model, optimizer, scaler, config.CHECKPOINT_DIR)

        train_data_dir = os.path.join(config.LCCC_PROCESSED_PATH, "train")
        
        best_val_loss = float('inf')

        # 新增：全局学习率调度器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)  # 每轮衰减

        for epoch in range(start_epoch, config.NUM_EPOCHS):
            train_dataset = BinaryDialogueDataset(train_data_dir)
            # 注意：不再需要除以梯度累积步数
            train_loader = DataLoader(
                train_dataset, batch_size=config.BATCH_SIZE,
                collate_fn=binary_collate_fn, num_workers=min(os.cpu_count(), 2),  # 限制为2以降低内存使用
                pin_memory=True, shuffle=True
            )

            pbar = tqdm(train_loader, desc=f"轮次 {epoch+1}/{config.NUM_EPOCHS}")
            
            for x_ref_padded, steps_data in pbar:
                batch_loss, steps_in_batch = train_batch_stepwise(x_ref_padded, steps_data, model, optimizer, scaler)
                
                total_steps += steps_in_batch
                avg_loss = batch_loss / steps_in_batch if steps_in_batch > 0 else 0
                pbar.set_postfix({"平均步损失": f"{avg_loss:.4f}", "总更新步数": total_steps})

                if total_steps > 0 and total_steps % config.SAVE_CHECKPOINT_EVERY_N_BATCHES == 0:
                    save_checkpoint(model, optimizer, scaler, epoch, total_steps, config.CHECKPOINT_DIR)
                
                if total_steps > 0 and total_steps % config.VALIDATE_EVERY_N_STEPS == 0:
                    val_loss = validate_model(model)
                    print(f"\n--- 验证 (总步数: {total_steps}) | 当前损失: {val_loss:.4f} | 最佳损失: {best_val_loss:.4f} ---")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print("发现新的最佳模型！正在保存...")
                        logging.info(f"新最佳验证损失: {val_loss}")
                        save_checkpoint(model, optimizer, scaler, epoch, total_steps, config.CHECKPOINT_DIR, is_best=True)

            scheduler.step()  # 每轮更新学习率

            print(f"\n轮次 {epoch+1}/{config.NUM_EPOCHS} 完成.\n")
        print("\n训练全部完成！")
        logging.info("训练完成")
    except Exception as e:
        logging.error(f"训练出错: {e}")
        raise

if __name__ == "__main__":
    train_dir = os.path.join(config.LCCC_PROCESSED_PATH, "train")
    if not os.path.isdir(train_dir):
        print(f"错误: 未找到处理后的二进制数据目录 '{train_dir}'。")
        print("请先运行 'python -m src.prepare_binary_data' 来生成数据。")
    else:
        train_model() 
