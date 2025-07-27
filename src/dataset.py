# -*- coding: utf-8 -*-

from torch.utils.data import IterableDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import json
import os
import config # 导入全局配置
from tqdm import tqdm
import glob # 导入 glob 模块用于查找文件
import random
import psutil  # 新增：内存监控


class Vocabulary:
    """词汇表类，用于跟踪单词到索引的映射"""
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        # 使用来自 config 的统一定义
        self.index2word = {
            config.PAD_token: "PAD",
            config.SOS_token: "SOS",
            config.EOS_token: "EOS",
            config.UNK_token: "UNK"
        }
        self.num_words = 4  # PAD, SOS, EOS, UNK

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = [k for k, v in self.word2count.items() if v >= min_count]
        print(f'保留 {len(keep_words)} / {len(self.word2index)} = {len(keep_words) / len(self.word2index):.4f}')
        # 重置词典
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            config.PAD_token: "PAD",
            config.SOS_token: "SOS",
            config.EOS_token: "EOS",
            config.UNK_token: "UNK"
        }
        self.num_words = 4
        for word in keep_words:
            self.addWord(word)

class DialogueDataset(IterableDataset):
    """
    一个可迭代的数据集，用于从 .jsonl 文件流式传输对话。
    （保留以备旧工作流使用）
    """
    def __init__(self, file_path, skip_lines=0):
        super(DialogueDataset, self).__init__()
        self.file_path = file_path
        self.file_handle = open(self.file_path, 'r', encoding='utf-8')
        if skip_lines > 0:
            print(f"从 {os.path.basename(self.file_path)} 跳过 {skip_lines} 行...")
            for _ in tqdm(range(skip_lines), desc="正在跳过已处理的数据"):
                self.file_handle.readline()

    def __iter__(self):
        return map(json.loads, self.file_handle)

    def __del__(self):
        if hasattr(self, 'file_handle') and self.file_handle:
            self.file_handle.close()


# --- 升级: 用于处理分块二进制 .pt 文件的智能 Dataset ---
class BinaryDialogueDataset(Dataset):
    """
    一个高效的数据集，它能处理一个包含多个 .pt 数据块的目录。
    它会按需加载数据块，从而将内存占用降到最低。
    """
    def __init__(self, directory_path):
        self.directory_path = directory_path
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"二进制数据目录未找到: {directory_path}。请先运行 'prepare_binary_data.py'。")
        
        # 查找并排序所有数据块文件
        self.chunk_files = sorted(
            glob.glob(os.path.join(directory_path, "chunk_*.pt")),
            key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
        )

        if not self.chunk_files:
            raise FileNotFoundError(f"在 '{directory_path}' 中未找到数据块文件 (chunk_*.pt)。")

        # 计算每个块的样本数和总样本数
        self.chunk_lengths = [len(torch.load(f)) for f in self.chunk_files]
        self.cumulative_lengths = [0] + list(torch.cumsum(torch.tensor(self.chunk_lengths), dim=0))
        self.total_length = self.cumulative_lengths[-1].item()

        # 用于缓存当前加载的块，以避免重复IO
        self.current_chunk_index = -1
        self.current_chunk_data = None
        
        print(f"已初始化分块数据集，在 {len(self.chunk_files)} 个块中找到 {self.total_length} 个样本。")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_length:
            raise IndexError("索引超出范围")

        # 1. 找到索引 `idx` 属于哪个数据块
        chunk_index = -1
        for i in range(len(self.cumulative_lengths) - 1):
            if self.cumulative_lengths[i] <= idx < self.cumulative_lengths[i+1]:
                chunk_index = i
                break
        
        # 2. 如果需要的块不是当前缓存的块，则加载新块
        if chunk_index != self.current_chunk_index:
            self.current_chunk_data = torch.load(self.chunk_files[chunk_index])
            self.current_chunk_index = chunk_index
        
        # 3. 计算在块内的局部索引
        local_idx = idx - self.cumulative_lengths[chunk_index]
        
        return self.current_chunk_data[local_idx]


def binary_collate_fn(batch):
    """
    一个高效的 Collate 函数，用于处理来自 BinaryDialogueDataset 的、已经是张量的数据。
    """
    x_ref_tensors, steps_data_lists = zip(*batch)

    x_ref_padded = pad_sequence(x_ref_tensors, batch_first=True, padding_value=config.PAD_token)
    
    max_num_steps = max(len(steps) for steps in steps_data_lists) if steps_data_lists else 0
    batched_steps_data = []

    for t in range(max_num_steps):
        x_t_batch, target_batch, gate_target_batch = [], [], []
        has_target_in_step = False

        for steps_list in steps_data_lists:
            if t < len(steps_list):
                x_t, target, gate_target = steps_list[t]
                x_t_batch.append(x_t)
                target_batch.append(target)
                gate_target_batch.append(gate_target)
                if target.numel() > 0:
                    has_target_in_step = True
            else:
                x_t_batch.append(torch.tensor([], dtype=torch.long))
                target_batch.append(torch.tensor([], dtype=torch.long))
                gate_target_batch.append(torch.tensor([0.0], dtype=torch.float))

        x_t_padded = pad_sequence(x_t_batch, batch_first=True, padding_value=config.PAD_token)
        
        target_padded = None
        if has_target_in_step:
            padded_targets = pad_sequence(target_batch, batch_first=True, padding_value=config.PAD_token)
            target_padded = padded_targets

        gate_target_tensor = torch.cat(gate_target_batch)
        batched_steps_data.append((x_t_padded, target_padded, gate_target_tensor))

    return x_ref_padded, batched_steps_data 