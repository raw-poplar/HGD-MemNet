# -*- coding: utf-8 -*-
import torch
from torch import nn
from typing import List, Tuple
import json
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate

import config
from .model import HGD_MemNet
from .dataset import Vocabulary, BinaryDialogueDataset, binary_collate_fn
from .train import DEVICE # 从训练脚本中导入设备设置
from functools import partial

import os
import torch
from torch.utils.data import DataLoader
import config
from .dataset import BinaryDialogueDataset, binary_collate_fn
from .utils import load_model_from_checkpoint, compute_loss
import nltk
from nltk.translate.bleu_score import sentence_bleu
from math import exp

nltk.download('punkt')

# --- JIT 编译优化 ---
# 同样将贪心解码的核心逻辑进行JIT编译，以提升评估速度
@torch.jit.script
def greedy_decode_jitted(model, x_ref_padded, steps_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], vocab_size : int, gate_threshold : float, eos_token : int):
    # batch_size = x_ref_padded.size(0)
    # h_prev = torch.zeros(batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM, device=x_ref_padded.device)
    #
    # # JIT需要显式的类型
    # decoded_indices = torch.full((batch_size, config.MAX_CONVO_LENGTH), config.PAD_token, dtype=torch.long, device=x_ref_padded.device)
    # finished = torch.zeros(batch_size, dtype=torch.bool, device=x_ref_padded.device)
    # 
    # for t in range(len(steps_data)):
    #     if torch.all(finished):
    #         break
    #         
    #     x_t_padded, _, _ = steps_data[t]
    #     
    #     h_next, gate_pred, output_logits = model(x_t_padded, x_ref_padded, h_prev)
    #     
    #     # JIT兼容的门控检查
    #     for i in range(batch_size):
    #         if not finished[i] and gate_pred[i] > gate_threshold:
    #             # 如果门控开启，生成输出
    #             topi = torch.argmax(output_logits[i], dim=-1)
    #             
    #             # JIT不支持动态列表追加，我们填充张量
    #             # 此处简化为只取第一个token，实际贪心解码需要一个循环
    #             # 为了保持评估逻辑一致性，我们暂时不在JIT中做完整解码
    #             # 注意：这里的逻辑是为了JIT兼容，但丢失了多步解码能力
    #             # 完整的贪心搜索在JIT中实现比较复杂，这里暂时只做一步生成
    #             if topi.item() != eos_token:
    #                 decoded_indices[i, 0] = topi.item()
    #             finished[i] = True
    #
    #     h_prev = h_next
    #
    # return decoded_indices
    pass


def greedy_decode(model, x_ref_padded, steps_data, vocab):
    """
    使用贪心策略为一批序列生成回答
    此版本是包装器，调用JIT版本或原始实现以获得最佳性能
    """
    model.eval()
    
    with torch.no_grad():
        # 1. 移动数据到设备
        x_ref_padded = x_ref_padded.to(DEVICE)
        steps_data_device = []
        for x_t, y_t, g_t in steps_data:
            steps_data_device.append((
                x_t.to(DEVICE),
                y_t.to(DEVICE) if y_t is not None else None,
                g_t.to(DEVICE)
            ))

        # --- 实际解码逻辑 ---
        # 完整的贪心解码循环在JIT中不易实现，我们保留原始的Python循环逻辑
        # JIT的优势主要体现在没有复杂控制流的数值计算上
        batch_size = x_ref_padded.size(0)
        h_prev = torch.zeros(batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM, device=DEVICE)
        
        decoded_sentences = [""] * batch_size
        finished = [False] * batch_size
        
        for t, (x_t_padded, _, _) in enumerate(steps_data_device):
            if all(finished):
                break
            
            h_next, gate_pred, output_logits = model(x_t_padded, x_ref_padded, h_prev)
            
            for i in range(batch_size):
                if not finished[i] and gate_pred[i].item() > config.GATE_THRESHOLD:
                    topv, topi = output_logits[i].topk(1)
                    idx = topi.item()
                    if idx != config.EOS_token:
                        word = vocab.index2word.get(str(idx), "<UNK>")
                        decoded_sentences[i] = word # 简化为单步解码
                    finished[i] = True

            h_prev = h_next

        return decoded_sentences


def evaluate_model(args):
    """主评估函数"""
    print("--- 开始评估 ---")
    
    # 1. 加载词汇表
    print("正在加载词汇表...")
    vocab_path = os.path.join(args.data_dir, "../", "vocabulary.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"在 {vocab_path} 未找到词汇表文件。请确保它与您的数据文件在同一目录中。")
        
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)
    vocab = Vocabulary(vocab_dict['name'])
    vocab.__dict__.update(vocab_dict)
    print(f"词汇表加载完毕。大小: {vocab.num_words}")
    
    config.VOCAB_SIZE = vocab.num_words

    # 2. 初始化模型
    print("正在初始化模型...")
    model = HGD_MemNet(
        vocab_size=config.VOCAB_SIZE,
        embed_dim=config.EMBEDDING_DIM,
        context_dim=config.CONTEXT_VECTOR_DIM,
        dynamic_hidden_dim=config.DYNAMIC_GROUP_HIDDEN_DIM,
        static_hidden_dim=config.STATIC_HEAD_HIDDEN_DIM
    ).to(DEVICE)

    # 尝试脚本化以获得性能提升
    try:
        model = torch.jit.script(model)
        print("模型已成功JIT脚本化，以提升评估性能。")
    except Exception as e:
        print(f"警告：模型JIT脚本化失败: {e}")

    # 3. 加载检查点
    print(f"正在从 {args.checkpoint} 加载检查点...")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"在 {args.checkpoint} 未找到检查点文件。")
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    # 兼容不同类型的检查点
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 4. 准备数据加载器 (使用分块二进制)
    print(f"正在从二进制数据目录 {args.data_dir} 加载数据...")
    try:
        dataset = BinaryDialogueDataset(args.data_dir)
    except FileNotFoundError as e:
        print(f"错误：初始化评估数据集失败: {e}")
        return

    # 在评估时也使用多进程加载
    num_workers = min(os.cpu_count(), 4)
    print(f"使用 {num_workers} 个工作进程加载评估数据。")
    
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=binary_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    # 5. 生成并收集结果
    print("正在生成预测...")
    predictions = []
    references = []
    for x_ref_padded, steps_data in tqdm(data_loader, desc="评估中"):
        # 假设每个剧本的最后一个非空target_text是参考答案
        # 我们需要从 steps_data 中提取参考答案的原始文本
        batch_refs = []
        for single_convo_steps in zip(*steps_data):
            # 这部分逻辑变得复杂，因为数据是批量处理的
            # 为了简化，我们暂时跳过从二进制数据中提取reference
            pass
        
        # 使用解码函数生成预测
        preds = greedy_decode(model, x_ref_padded, steps_data, vocab)
        predictions.extend(preds)
    
    print("警告：由于数据加载方式的改变，评估指标计算中的'参考答案'部分暂未实现。")
    print(f"已生成 {len(predictions)} 条预测。")

    # 6. 计算指标 (暂时跳过，因为references为空)
    # print("\n正在计算评估指标...")
    
    # Hugging Face Evaluate
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    
    # bleu_results = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])
    # rouge_results = rouge_metric.compute(predictions=predictions, references=[[r] for r in references])
    
    # BERTScore
    print(f"正在使用语言='{args.lang}'计算BERTScore...")
    bertscore_metric = evaluate.load("bertscore")
    # bertscore_results = bertscore_metric.compute(predictions=predictions, references=references, lang=args.lang)

    print("\n--- 评估结果 ---")
    # print(f"BLEU-1: {bleu_results['bleu']:.4f}")
    # print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
    # print(f"BERTScore-F1: {sum(bertscore_results['f1']) / len(bertscore_results['f1']):.4f}")
    print("--------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 HGD-MemNet 模型。")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点文件的路径 (例如, model_final.pth)。")
    parser.add_argument("--data_dir", type=str, required=True, help="包含处理后的测试数据块的目录 (例如, ./data/lccc_processed/test)。")
    parser.add_argument("--batch_size", type=int, default=16, help="用于评估的批处理大小。")
    parser.add_argument("--lang", type=str, default="zh", help="BERTScore模型的语言 (例如, 'en', 'zh')。")
    
    args = parser.parse_args()
    if not os.path.isdir(args.data_dir):
        print(f"错误：提供的数据路径 '{args.data_dir}' 不是一个目录。")
        print("请提供包含 chunk_*.pt 文件的有效目录。")
    else:
        evaluate_model(args) 