#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
词汇表构建工具

从train, valid, test三个jsonl文件中提取指定数量的最常用词汇，
生成词汇表文件用于模型训练。

功能特性:
1. 支持自定义数据路径
2. 支持指定词汇表大小
3. 支持中文分词
4. 支持词频统计和过滤
5. 生成JSON格式的词汇表文件
"""

import os
import json
import sys
from collections import Counter
from tqdm import tqdm
import argparse
# jieba 为可选依赖，不强制安装；不可用时会自动回退到简易分词
try:
    import jieba  # type: ignore
    JIEBA_AVAILABLE = True
except Exception:
    jieba = None
    JIEBA_AVAILABLE = False
import re
import unicodedata

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import config
from src.dataset import Vocabulary


class VocabularyBuilder:
    """词汇表构建器"""
    
    def __init__(self, data_path=None, vocab_size=None, min_freq=None):
        """
        初始化词汇表构建器
        
        Args:
            data_path: 数据文件路径，默认使用config中的路径
            vocab_size: 词汇表大小，默认30000
            min_freq: 最小词频，默认2
        """
        self.data_path = data_path or config.LCCC_PROCESSED_PATH
        self.vocab_size = vocab_size or getattr(config, 'VOCAB_BUILD_SIZE', 30000)
        self.min_freq = min_freq or getattr(config, 'VOCAB_MIN_FREQ', 2)
        self.word_counter = Counter()

        # 文本预处理：模式与过滤器
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        self.emoji_pattern = re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF\U00002190-\U00002BFF]")
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
        self.email_pattern = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
        self.num_pattern = re.compile(r'\d{5,}')  # 5位以上长数字合并为占位符

        print(f"词汇表构建器初始化完成")
        print(f"   数据路径: {self.data_path}")
        print(f"   目标词汇表大小: {self.vocab_size}")
        print(f"   最小词频: {self.min_freq}")
    
    def preprocess_text(self, text):
        """
        预处理文本：
        - NFKC 归一化与去空白
        - 去除 URL、邮箱、emoji
        - 小写化英文
        - 使用 jieba 分词（不可用时退化为简单切分）
        - 过滤：仅保留中文词、英文词；数字仅保留<=4位，超过归一化为 <NUM>
        """
        if not text or not isinstance(text, str):
            return []

        # 基础清理与归一化
        text = unicodedata.normalize('NFKC', text).strip()
        if not text:
            return []

        # 去除 URL/邮箱/emoji
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.emoji_pattern.sub('', text)

        # 英文小写化
        text = text.lower()

        # 分词（jieba 不可用时回退）
        try:
            words = jieba.lcut(text)
        except Exception:
            # 退化：按空白切分 + 保留中文连续块
            parts = re.split(r"\s+", text)
            words = []
            for p in parts:
                words.extend(re.findall(r"[\u4e00-\u9fff]+|[A-Za-z]+|\d+", p))

        # 过滤与清理
        cleaned_words = []
        for word in words:
            word = word.strip()
            if not word:
                continue
            if self.chinese_pattern.search(word) or word.isalpha():
                cleaned_words.append(word)
            elif word.isdigit():
                if len(word) <= 4:
                    cleaned_words.append(word)
                else:
                    cleaned_words.append('<NUM>')

        return cleaned_words
    
    def process_jsonl_file(self, file_path):
        """
        处理单个jsonl文件，统计词频
        
        Args:
            file_path: jsonl文件路径
        """
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return
        
        print(f"处理文件: {os.path.basename(file_path)}")
        
        # 单遍扫描处理文件内容（避免预先统计总行数的二次 IO）
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc=f"处理{os.path.basename(file_path)}", unit="lines")):
                try:
                    dialogue = json.loads(line.strip())

                    # 处理对话中的每个句子（兼容 text / input_text / target_text 三种字段）
                    for turn in dialogue:
                        texts = []
                        if isinstance(turn, dict):
                            if 'text' in turn and turn['text']:
                                texts.append(turn['text'])
                            if 'input_text' in turn and turn['input_text']:
                                texts.append(turn['input_text'])
                            if 'target_text' in turn and turn['target_text']:
                                texts.append(turn['target_text'])
                        elif isinstance(turn, str):
                            texts.append(turn)

                        # 预处理并统计词频
                        for t in texts:
                            words = self.preprocess_text(t)
                            self.word_counter.update(words)

                except json.JSONDecodeError as e:
                    print(f"第{line_num+1}行JSON解析错误: {e}")
                    continue
                except Exception as e:
                    print(f"第{line_num+1}行处理错误: {e}")
                    continue
    
    def build_vocabulary(self):
        """
        构建词汇表
        
        Returns:
            Vocabulary对象
        """
        print(f"\n开始构建词汇表...")
        
        # 处理所有数据文件
        data_files = ['train.jsonl', 'valid.jsonl', 'test.jsonl']
        
        for data_file in data_files:
            file_path = os.path.join(self.data_path, data_file)
            self.process_jsonl_file(file_path)
        
        print(f"\n词频统计完成:")
        print(f"   总词汇数: {len(self.word_counter)}")
        print(f"   总词频: {sum(self.word_counter.values())}")
        
        # 过滤低频词
        filtered_words = {word: count for word, count in self.word_counter.items() 
                         if count >= self.min_freq}
        
        print(f"   过滤后词汇数: {len(filtered_words)}")
        
        # 选择最常用的词汇（基于最小词频过滤后的计数）
        filtered_counter = Counter(filtered_words)
        most_common_words = filtered_counter.most_common(self.vocab_size)

        print(f"   选择词汇数: {len(most_common_words)}")

        # 创建词汇表对象
        vocab = Vocabulary("lccc")

        # 添加词汇到词汇表
        for word, _ in tqdm(most_common_words, desc="构建词汇表"):
            vocab.addWord(word)

        print(f"\n词汇表构建完成:")
        print(f"   词汇表大小: {vocab.num_words}")
        print(f"   包含特殊词元: PAD, SOS, EOS, UNK")
        
        return vocab
    
    def save_vocabulary(self, vocab, output_path=None):
        """
        保存词汇表到文件
        
        Args:
            vocab: Vocabulary对象
            output_path: 输出文件路径，默认为数据路径下的vocabulary.json
        """
        if output_path is None:
            output_path = os.path.join(self.data_path, "vocabulary.json")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存词汇表
        vocab_dict = {
            'name': vocab.name,
            'word2index': vocab.word2index,
            'word2count': vocab.word2count,
            'index2word': vocab.index2word,
            'num_words': vocab.num_words,
            'trimmed': vocab.trimmed
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
        
        print(f"词汇表已保存到: {output_path}")
        
        # 保存词频统计信息
        stats_path = output_path.replace('.json', '_stats.json')
        stats = {
            'vocab_size': vocab.num_words,
            'total_words_processed': sum(self.word_counter.values()),
            'unique_words_found': len(self.word_counter),
            'min_frequency': self.min_freq,
            'top_10_words': self.word_counter.most_common(10)
        }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"统计信息已保存到: {stats_path}")
        
        return output_path
    
    def print_vocabulary_stats(self, vocab):
        """
        打印词汇表统计信息
        
        Args:
            vocab: Vocabulary对象
        """
        print(f"\n词汇表统计信息:")
        print(f"   词汇表名称: {vocab.name}")
        print(f"   总词汇数: {vocab.num_words}")
        print(f"   是否修剪: {vocab.trimmed}")
        
        print(f"\n最常用的10个词:")
        top_words = self.word_counter.most_common(10)
        for i, (word, count) in enumerate(top_words, 1):
            print(f"   {i:2d}. {word:<10} (出现 {count:,} 次)")
        
        print(f"\n词频分布:")
        freq_ranges = [
            (1, 1), (2, 5), (6, 10), (11, 50), (51, 100), 
            (101, 500), (501, 1000), (1001, float('inf'))
        ]
        
        for min_freq, max_freq in freq_ranges:
            if max_freq == float('inf'):
                count = sum(1 for freq in self.word_counter.values() if freq >= min_freq)
                print(f"   {min_freq}+ 次: {count:,} 个词")
            else:
                count = sum(1 for freq in self.word_counter.values() if min_freq <= freq <= max_freq)
                print(f"   {min_freq}-{max_freq} 次: {count:,} 个词")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="构建词汇表工具")
    parser.add_argument("--data_path", type=str, 
                       default="F:/modelTrain/data/lccc_processed",
                       help="数据文件路径 (默认: F:/modelTrain/data/lccc_processed)")
    parser.add_argument("--vocab_size", type=int, default=getattr(config, 'VOCAB_BUILD_SIZE', 30000),
                       help=f"词汇表大小 (默认: {getattr(config, 'VOCAB_BUILD_SIZE', 30000)})")
    parser.add_argument("--min_freq", type=int, default=getattr(config, 'VOCAB_MIN_FREQ', 2),
                       help=f"最小词频 (默认: {getattr(config, 'VOCAB_MIN_FREQ', 2)})")
    parser.add_argument("--output", type=str, 
                       help="输出文件路径 (默认: 数据路径下的vocabulary.json)")
    
    args = parser.parse_args()
    
    print("词汇表构建工具")
    print("=" * 50)
    
    try:
        # 创建词汇表构建器
        builder = VocabularyBuilder(
            data_path=args.data_path,
            vocab_size=args.vocab_size,
            min_freq=args.min_freq
        )
        
        # 构建词汇表
        vocab = builder.build_vocabulary()
        
        # 打印统计信息
        builder.print_vocabulary_stats(vocab)
        
        # 保存词汇表
        output_path = builder.save_vocabulary(vocab, args.output)
        
        print(f"\n词汇表构建完成!")
        print(f"   输出文件: {output_path}")
        
    except Exception as e:
        print(f"构建过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
