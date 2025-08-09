# -*- coding: utf-8 -*-
import torch
import json
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
# import evaluate  # 可选：如需BLEU/ROUGE/BERTScore请安装并解开注释

import config
from .model import HGD_MemNet
from .dataset import Vocabulary, BinaryDialogueDataset, binary_collate_fn
from .train import DEVICE # 从训练脚本中导入设备设置
# from functools import partial

import os
# import torch
from torch.utils.data import DataLoader
import config
from .dataset import BinaryDialogueDataset, binary_collate_fn
# from .utils import load_model_from_checkpoint, compute_loss  # 可选：如需加载检查点或计算损失再启用
# import nltk
# from nltk.translate.bleu_score import sentence_bleu
# from math import exp

# nltk.download('punkt')

# --- JIT 编译优化 ---
# 同样将贪心解码的核心逻辑进行JIT编译，以提升评估速度
@torch.jit.script
def greedy_decode_jitted(model : HGD_MemNet, x_ref_padded, steps_data, _vocab_size: int, gate_threshold: float, eos_token: int):
    batch_size = x_ref_padded.size(0)
    h_prev = torch.zeros(batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM, device=x_ref_padded.device)

    # JIT需要显式的类型
    decoded_indices = torch.full((batch_size, config.MAX_CONVO_LENGTH), config.PAD_token, dtype=torch.long, device=x_ref_padded.device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=x_ref_padded.device)

    for t in range(len(steps_data)):
        if torch.all(finished):
            break

        x_t_padded, _, _ = steps_data[t]

        h_next, gate_pred, output_logits = model(x_t_padded, x_ref_padded, h_prev)

        # JIT兼容的门控检查
        for i in range(batch_size):
            if not finished[i] and gate_pred[i] > gate_threshold:
                # 如果门控开启，生成输出
                topi = torch.argmax(output_logits[i], dim=-1)

                # JIT不支持动态列表追加，我们填充张量
                # 此处简化为只取第一个token，实际贪心解码需要一个循环
                # 为了保持评估逻辑一致性，我们暂时不在JIT中做完整解码
                # 注意：这里的逻辑是为了JIT兼容，但丢失了多步解码能力
                # 完整的贪心搜索在JIT中实现比较复杂，这里暂时只做一步生成
                if topi.item() != eos_token:
                    decoded_indices[i, 0] = topi.item()
                finished[i] = True

        h_prev = h_next

    return decoded_indices


def greedy_decode(model, x_ref_padded, steps_data, vocab, min_think: int | None = None, max_think: int | None = None, temperature: float = 0.1):
    """
    使用带有自适应思考步长与门控的简化贪心策略生成回答。
    - min_think: 最小思考步（None 表示不限制）
    - max_think: 最大思考步（None 表示由数据 steps 决定）
    - temperature: 推理温度（传给模型动态组）
    """
    model.eval()

    with torch.no_grad():
        # 1. 移动数据到设备
        x_ref_padded = x_ref_padded.to(DEVICE)
        steps_data_device = []
        for x_t, y_t, g_t in steps_data:
            x_t_dev = x_t.to(DEVICE) if x_t is not None else None
            steps_data_device.append((
                x_t_dev,
                y_t.to(DEVICE) if y_t is not None else None,
                g_t.to(DEVICE)
            ))

        # --- 实际解码逻辑 ---
        batch_size = x_ref_padded.size(0)
        h_prev = torch.zeros(batch_size, config.DYNAMIC_GROUP_HIDDEN_DIM, device=DEVICE)

        decoded_sentences = [""] * batch_size
        finished = [False] * batch_size

        steps_available = len(steps_data_device)
        cap = min(steps_available, max_think) if (max_think is not None) else steps_available

        for t in range(cap):
            if all(finished):
                break
            x_t_padded, _, _ = steps_data_device[t]

            # 构造控制向量：t_norm, remain_norm, min_done, budget
            t_norm = torch.full((batch_size, 1), (t + 1) / max(cap, 1), device=DEVICE)
            remain_norm = torch.full((batch_size, 1), max(cap - (t + 1), 0) / max(cap, 1), device=DEVICE)
            min_done_flag = 1.0 if (min_think is None) or ((t + 1) >= min_think) else 0.0
            min_done = torch.full((batch_size, 1), min_done_flag, device=DEVICE)
            budget = torch.full((batch_size, 1), float(getattr(config, 'TARGET_SPEAK_RATIO', 0.0) or 0.0), device=DEVICE)
            control = torch.cat([t_norm, remain_norm, min_done, budget], dim=1)

            h_next, gate_pred, output_logits = model(x_t_padded, x_ref_padded, h_prev, temperature=temperature, control=control)

            for i in range(batch_size):
                allow_speak = (min_think is None) or ((t + 1) >= min_think)
                force_speak = (t + 1) >= cap
                if not finished[i] and (force_speak or (allow_speak and gate_pred[i].item() > config.GATE_THRESHOLD)):
                    topi = output_logits[i].argmax(dim=-1)
                    idx = topi.item()
                    if idx != config.EOS_token:
                        # 这里的词表映射逻辑简化
                        word = vocab.index2word.get(idx, "<UNK>") if hasattr(vocab, 'index2word') else str(idx)
                        decoded_sentences[i] = word
                    finished[i] = True

            h_prev = h_next

        return decoded_sentences


def evaluate_model(args):
    """主评估函数"""
    print("--- 开始评估 ---")

    # 1. 加载词汇表
    print("正在加载词汇表...")
    vocab_path = os.path.join(os.path.dirname(args.data_file), "vocabulary.json")
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
    # references = []  # 指标计算暂时关闭
    for x_ref_padded, steps_data in tqdm(data_loader, desc="评估中"):
        # 假设每个剧本的最后一个非空target_text是参考答案
        # 我们需要从 steps_data 中提取参考答案的原始文本
        batch_refs = []
        for single_convo_steps in zip(*steps_data):
            # 这部分逻辑变得复杂，因为数据是批量处理的
            # 为了简化，我们暂时跳过从二进制数据中提取reference
            pass

        # 使用解码函数生成预测
        # 解析 min/max 参数
        min_t = None if args.min_think is None or args.min_think < 0 else int(args.min_think)
        max_t = None if args.max_think is None or args.max_think < 0 else int(args.max_think)
        preds = greedy_decode(model, x_ref_padded, steps_data, vocab, min_think=min_t, max_think=max_t, temperature=0.1)
        predictions.extend(preds)

    print("警告：由于数据加载方式的改变，评估指标计算中的'参考答案'部分暂未实现。")
    print(f"已生成 {len(predictions)} 条预测。")

    # 6. 计算指标 (暂时跳过，因为references为空)
    # print("\n正在计算评估指标...")

    # 如需BLEU/ROUGE/BERTScore，请安装 evaluate/nltk/bertscore 并在上方解开注释后补充计算逻辑。
    print("\n--- 评估完成（已生成预测；指标计算被跳过） ---")
    print("提示：安装并启用 evaluate/nltk/bertscore 后可恢复指标计算。")
    print("--------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 HGD-MemNet 模型。")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点文件的路径 (例如, model_final.pth)。")
    parser.add_argument("--data_dir", type=str, required=True, help="包含处理后的测试数据块的目录 (例如, ./data/lccc_processed/test)。")
    parser.add_argument("--batch_size", type=int, default=16, help="用于评估的批处理大小。")
    parser.add_argument("--lang", type=str, default="zh", help="BERTScore模型的语言 (例如, 'en', 'zh')。")
    parser.add_argument("--min-think", type=int, default=-1, help="最小思考步数（-1 不限制）")
    parser.add_argument("--max-think", type=int, default=-1, help="最大思考步数（-1 不限制）")


    args = parser.parse_args()
    if not os.path.isdir(args.data_dir):
        print(f"错误：提供的数据路径 '{args.data_dir}' 不是一个目录。")
        print("请提供包含 chunk_*.pt 文件的有效目录。")
    else:
        evaluate_model(args)