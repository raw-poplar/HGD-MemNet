# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import glob
import re
from torch.amp import GradScaler
from tqdm import tqdm

import config
from .model import HGD_MemNet
from .dataset import Vocabulary, BinaryDialogueDataset, binary_collate_fn

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.jit.script
def train_batch_stepwise_jitted(x_ref_padded, steps_data, model: HGD_MemNet):
    batch_size = x_ref_padded.size(0)
    total_batch_loss = torch.tensor(0.0, device=x_ref_padded.device)
    h_prev = torch.zeros(batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM, device=x_ref_padded.device)
    
    for t in range(len(steps_data)):
        x_t_padded, target_padded, gate_target = steps_data[t]
        
        with torch.cuda.amp.autocast(enabled=True):
            h_next, gate_pred, output_logits = model(x_t_padded, x_ref_padded, h_prev.detach())
            
            gate_pred_logits = torch.log(gate_pred / (1.0 - gate_pred) + 1e-9)
            step_loss = nn.BCEWithLogitsLoss()(gate_pred_logits, gate_target)
            
            if target_padded is not None:
                output_loss = torch.tensor(0.0, device=x_ref_padded.device)
                seq_len = target_padded.size(1)
                for i in range(seq_len):
                    token_loss = nn.CrossEntropyLoss(ignore_index=config.PAD_token)(
                        output_logits, target_padded[:, i]
                    )
                    output_loss += token_loss
                if seq_len > 0:
                    step_loss += output_loss / float(seq_len)
        
        total_batch_loss += step_loss
        h_prev = h_next
        
    return total_batch_loss

def train_batch_stepwise(x_ref_padded, steps_data, model):
    x_ref_padded = x_ref_padded.to(DEVICE)
    steps_data_device = []
    for x_t, y_t, g_t in steps_data:
        steps_data_device.append((
            x_t.to(DEVICE),
            y_t.to(DEVICE) if y_t is not None else None,
            g_t.to(DEVICE)
        ))
    return train_batch_stepwise_jitted(x_ref_padded, steps_data_device, model)

def validate_model(model):
    model.eval()
    # 验证数据现在也是一个目录
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
        batch_size=config.BATCH_SIZE // config.GRADIENT_ACCUMULATION_STEPS,
        collate_fn=binary_collate_fn,
        num_workers=min(os.cpu_count(), 4),
        shuffle=False
    )

    total_val_loss = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="正在验证")
        for x_ref_padded, steps_data in pbar:
            batch_loss = train_batch_stepwise(x_ref_padded, steps_data, model)
            total_val_loss += batch_loss.item()
            pbar.set_postfix({"验证损失": f"{batch_loss.item():.4f}"})
    
    avg_val_loss = total_val_loss / len(val_loader) if val_loader else 0
    model.train()
    return avg_val_loss

def save_checkpoint(model, optimizer, scaler, epoch, steps, batches_in_epoch, directory, is_best=False):
    if not os.path.exists(directory): os.makedirs(directory)
    filename = f"checkpoint_epoch_{epoch}_steps_{steps}.pth"
    filepath = os.path.join(directory, filename)
    
    checkpoints = sorted(glob.glob(os.path.join(directory, "checkpoint_*.pth")), key=os.path.getmtime)
    if len(checkpoints) >= config.MAX_CHECKPOINTS_TO_KEEP:
        for old_checkpoint in checkpoints[:len(checkpoints) - config.MAX_CHECKPOINTS_TO_KEEP + 1]:
            os.remove(old_checkpoint)

    model_state = model.state_dict()
    torch.save({
        'epoch': epoch, 'total_steps': steps, 'batches_in_epoch': batches_in_epoch,
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
    if not os.path.exists(directory): return 0, 0, 0
    checkpoints = glob.glob(os.path.join(directory, "checkpoint_*.pth"))
    if not checkpoints: return 0, 0, 0

    latest_checkpoint_path = max(checkpoints, key=os.path.getmtime)
    print(f"正在从最新的检查点加载: {latest_checkpoint_path}")
    checkpoint = torch.load(latest_checkpoint_path, map_location=DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scaler_state_dict' in checkpoint: scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    total_steps = checkpoint.get('total_steps', 0)
    batches_in_epoch = checkpoint.get('batches_in_epoch', 0)
    
    print(f"已从 '{latest_checkpoint_path}' 加载 (轮次 {start_epoch}, 总步数 {total_steps})。")
    return start_epoch, total_steps, batches_in_epoch

def train_model():
    print("开始训练流程...")
    vocab_path = os.path.join(config.LCCC_PROCESSED_PATH, "vocabulary.json")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)
    vocab = Vocabulary("lccc")
    vocab.__dict__.update(vocab_dict)
    config.VOCAB_SIZE = vocab.num_words
    print(f"词汇表加载完毕。大小: {vocab.num_words}")

    model = HGD_MemNet(
        vocab_size=config.VOCAB_SIZE, embed_dim=config.EMBEDDING_DIM,
        context_dim=config.CONTEXT_VECTOR_DIM, dynamic_hidden_dim=config.DYNAMIC_GROUP_HIDDEN_DIM,
        static_hidden_dim=config.STATIC_HEAD_HIDDEN_DIM
    ).to(DEVICE)
    
    try:
        model = torch.jit.script(model)
        print("模型已成功JIT脚本化，以提升性能。")
    except Exception as e:
        print(f"警告：模型JIT脚本化失败: {e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    start_epoch, total_steps, batches_to_skip = load_latest_checkpoint(model, optimizer, scaler, config.CHECKPOINT_DIR)

    # 训练数据现在是一个目录
    train_data_dir = os.path.join(config.LCCC_PROCESSED_PATH, "train")
    try:
        # 创建一个临时数据集仅用于获取总长度
        train_len_dataset = BinaryDialogueDataset(train_data_dir)
        steps_per_epoch = len(train_len_dataset) // (config.BATCH_SIZE // config.GRADIENT_ACCUMULATION_STEPS)
        del train_len_dataset
    except FileNotFoundError:
        steps_per_epoch = 0

    optimizer.zero_grad()
    best_val_loss = float('inf')

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        # 每次 epoch 都重新创建数据集对象，以支持 shuffle
        train_dataset = BinaryDialogueDataset(train_data_dir)
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE // config.GRADIENT_ACCUMULATION_STEPS,
            collate_fn=binary_collate_fn, num_workers=min(os.cpu_count(), 4),
            pin_memory=True, shuffle=True
        )

        pbar = tqdm(train_loader, desc=f"轮次 {epoch+1}/{config.NUM_EPOCHS}", total=steps_per_epoch)
        
        for batch_idx, (x_ref_padded, steps_data) in enumerate(pbar):
            if epoch == start_epoch and batch_idx < batches_to_skip:
                continue

            batch_loss = train_batch_stepwise(x_ref_padded, steps_data, model)
            scaler.scale(batch_loss / config.GRADIENT_ACCUMULATION_STEPS).backward()
            
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                total_steps += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                pbar.set_postfix({"训练损失": f"{batch_loss.item():.4f}", "总步数": total_steps})

                if total_steps > 0 and total_steps % config.SAVE_CHECKPOINT_EVERY_N_BATCHES == 0:
                    save_checkpoint(model, optimizer, scaler, epoch, total_steps, batch_idx + 1, config.CHECKPOINT_DIR)
                
                if total_steps > 0 and total_steps % config.VALIDATE_EVERY_N_STEPS == 0:
                    val_loss = validate_model(model)
                    print(f"\n--- 验证 (步数: {total_steps}) | 当前损失: {val_loss:.4f} | 最佳损失: {best_val_loss:.4f} ---")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print("发现新的最佳模型！正在保存...")
                        save_checkpoint(model, optimizer, scaler, epoch, total_steps, batch_idx + 1, config.CHECKPOINT_DIR, is_best=True)

        print(f"\n轮次 {epoch+1}/{config.NUM_EPOCHS} 完成.\n")
    print("\n训练全部完成！")

if __name__ == "__main__":
    # 检查训练数据目录是否存在
    train_dir = os.path.join(config.LCCC_PROCESSED_PATH, "train")
    if not os.path.isdir(train_dir):
        print(f"错误: 未找到处理后的二进制数据目录 '{train_dir}'。")
        print("请先运行 'python -m src.prepare_binary_data' 来生成数据。")
    else:
        train_model() 