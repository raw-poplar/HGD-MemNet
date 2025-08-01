# -*- coding: utf-8 -*-
import os
import sys
import json
from tqdm import tqdm
import ijson
import logging

# 将项目根目录添加到系统路径中，以便导入config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config
from src.dataset import Vocabulary

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_vocab_from_lccc(input_files, output_path, vocab_size):
    """
    从 LCCC 数据集的多个 .jsonl 文件构建词汇表。

    Args:
        input_files (list of str): .jsonl 文件路径的列表。
        output_path (str): 保存 vocabulary.json 的目录。
        vocab_size (int): 词汇表的最大大小。
    """
    logging.info("开始构建词汇表...")
    
    word_counts = {}

    for input_file in input_files:
        if not os.path.exists(input_file):
            logging.warning(f"找不到输入文件: {input_file}，已跳过。")
            continue
        
        logging.info(f"正在扫描文件: {input_file}")
        
        # 使用 ijson 流式解析大型jsonl文件
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                # 使用tqdm估算总行数以显示进度条
                try:
                    total_lines = sum(1 for line in open(input_file, 'r', encoding='utf-8'))
                    f.seek(0) # 重置文件指针
                    pbar = tqdm(f, total=total_lines, desc=f"处理 {os.path.basename(input_file)}", unit="行")
                except Exception:
                    pbar = tqdm(f, desc=f"处理 {os.path.basename(input_file)}", unit="行")

                for line in pbar:
                    try:
                        dialogue = json.loads(line)
                        if isinstance(dialogue, list):
                            # LCCC-base format is a list of strings (sentences)
                            for sentence in dialogue:
                                if isinstance(sentence, str):
                                    # Split by character, which is suitable for Chinese
                                    words = list(sentence.replace(" ", ""))
                                    for word in words:
                                        word_counts[word] = word_counts.get(word, 0) + 1
                    except (json.JSONDecodeError, TypeError):
                        logging.warning(f"跳过无效的JSON行: {line.strip()}")
                        continue
        except FileNotFoundError:
            # 这个检查在循环开始时已经做了，但为了安全起见保留
            logging.error(f"错误: 找不到文件 {input_file}")
            continue

    logging.info(f"所有文件扫描完毕。找到了 {len(word_counts)} 个独立词元。")

    # 按频率排序并选择前N个词
    vocab = Vocabulary("LCCC_vocab")
    sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    top_words = sorted_words[:vocab_size - 4] # 为4个特殊词元留出空间

    for word, _ in top_words:
        vocab.addWord(word)
    
    logging.info(f"词汇表构建完成，总大小为 {vocab.num_words} (包括特殊词元)。")

    # 保存词汇表
    os.makedirs(output_path, exist_ok=True)
    vocab_save_path = os.path.join(output_path, "vocabulary.json")
    with open(vocab_save_path, 'w', encoding='utf-8') as f:
        json.dump(vocab.__dict__, f, ensure_ascii=False, indent=4)
        
    logging.info(f"词汇表已成功保存到: {vocab_save_path}")


if __name__ == "__main__":
    # 从config.py中获取所有数据集文件的路径
    dataset_files = [
        config.LCCC_TRAIN_FILE,
        config.LCCC_VALID_FILE,
        config.LCCC_TEST_FILE
    ]
    processed_data_path = config.LCCC_PROCESSED_PATH
    vocabulary_size = config.VOCAB_SIZE

    print("--- 开始从所有LCCC数据集文件生成词汇表 ---")
    build_vocab_from_lccc(dataset_files, processed_data_path, vocabulary_size)
    print("--- 词汇表生成完毕！ ---")
