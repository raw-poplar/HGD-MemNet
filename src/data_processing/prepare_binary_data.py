import torch
import json
import os
from tqdm import tqdm
import glob  # chunking support
from multiprocessing import Pool
from itertools import islice  # batch slicing
import logging  # 日志记录
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.dataset import Vocabulary


# 统一移除输出中的emoji，避免控制台/日志环境兼容问题
import re, builtins as _builtins
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF\U00002190-\U00002BFF]")
_print = _builtins.print

def print(*args, **kwargs):
    def _strip(s):
        return _EMOJI_RE.sub('', s) if isinstance(s, str) else s
    return _print(*[_strip(a) for a in args], **kwargs)

# --- 辅助函数 ---
def preprocess_text_for_indexing(text):
    """
    与词表构建一致的文本预处理（简化版）
    返回词汇列表，用于索引映射
    """
    if not text or not isinstance(text, str):
        return []

    import unicodedata
    import re

    # 基础清理与归一化
    text = unicodedata.normalize('NFKC', text).strip()
    if not text:
        return []

    # 去除 URL/邮箱/emoji（与词表构建保持一致）
    url_pattern = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
    email_pattern = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
    emoji_pattern = re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF\U00002190-\U00002BFF]")

    text = url_pattern.sub(' ', text)
    text = email_pattern.sub(' ', text)
    text = emoji_pattern.sub('', text)

    # 英文小写化
    text = text.lower()

    # 分词（优先使用 jieba，不可用时回退）
    try:
        import jieba
        words = jieba.lcut(text)
    except Exception:
        # 退化：按空白切分 + 保留中文连续块
        parts = re.split(r"\s+", text)
        words = []
        for p in parts:
            words.extend(re.findall(r"[\u4e00-\u9fff]+|[A-Za-z]+|\d+", p))

    # 过滤与清理（与词表构建保持一致）
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    cleaned_words = []
    for word in words:
        word = word.strip()
        if not word:
            continue
        if chinese_pattern.search(word) or word.isalpha():
            cleaned_words.append(word)
        elif word.isdigit():
            if len(word) <= 4:
                cleaned_words.append(word)
            else:
                cleaned_words.append('<NUM>')

    return cleaned_words

def indexesFromSentence(vocab, sentence):
    """将句子转换为索引列表（使用与词表一致的分词）"""
    if sentence is None:
        return []

    # 使用与词表构建一致的预处理
    words = preprocess_text_for_indexing(sentence)

    UNK_idx = vocab.word2index.get("<UNK>", config.UNK_token)
    return [vocab.word2index.get(word, UNK_idx) for word in words] + [config.EOS_token]

# 全局变量，用于多进程
_global_vocab = None

def init_worker(vocab_state):
    """初始化工作进程"""
    global _global_vocab
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.dataset import Vocabulary
    _global_vocab = Vocabulary("worker")
    _global_vocab.__dict__.update(vocab_state)

def process_batch_optimized(batch_lines):
    """优化的批处理函数 - 返回可序列化的数据"""
    global _global_vocab
    results = []
    for line in batch_lines:
        try:
            dialogue = json.loads(line)
            tensor_data = process_dialogue_to_tensors(dialogue, _global_vocab)
            if tensor_data:
                # 转换为可序列化的格式
                x_ref, steps_data = tensor_data
                serializable_data = (
                    x_ref.tolist(),  # 转换为列表
                    [(x_t.tolist() if x_t is not None else None,
                      target.tolist(),
                      gate.tolist()) for x_t, target, gate in steps_data]
                )
                results.append(serializable_data)
        except Exception:
            # 静默跳过错误的行
            pass
    return results

def deserialize_tensor_data(serializable_data):
    """将序列化数据转换回张量"""
    x_ref_list, steps_list = serializable_data
    x_ref = torch.tensor(x_ref_list, dtype=torch.long)
    steps_data = []
    for x_t_list, target_list, gate_list in steps_list:
        x_t = torch.tensor(x_t_list, dtype=torch.long) if x_t_list is not None else None
        target = torch.tensor(target_list, dtype=torch.long)
        gate = torch.tensor(gate_list, dtype=torch.float)
        steps_data.append((x_t, target, gate))
    return (x_ref, steps_data)

def process_dialogue_to_tensors(dialogue_list, vocab):
    """
    将对话列表转换为张量格式

    Args:
        dialogue_list: 对话列表，每个元素是 {"text": "..."}
        vocab: 词汇表对象

    Returns:
        tuple: (x_ref_tensor, steps_data) 或 None
    """
    if len(dialogue_list) < 2:
        return None

    try:
        # 参考句子（第一句话）
        x_ref_text = dialogue_list[0].get("text", "")
        x_ref_tensor = torch.tensor(indexesFromSentence(vocab, x_ref_text), dtype=torch.long)

        steps_data = []

        # 处理每一轮对话
        for i in range(1, len(dialogue_list)):
            target_text = dialogue_list[i].get("text", "")

            # 思考步骤 (THINKING_STEPS 次)
            for _ in range(config.THINKING_STEPS):
                thinking_step_x_t = None  # 修改：思考步骤中x_t为None
                thinking_step_target = torch.tensor([], dtype=torch.long)
                thinking_step_gate = torch.tensor([0.0], dtype=torch.float)
                steps_data.append((thinking_step_x_t, thinking_step_target, thinking_step_gate))

            # 输出步骤
            x_t_text = dialogue_list[i-1].get("text", "")
            output_step_x_t = torch.tensor(indexesFromSentence(vocab, x_t_text), dtype=torch.long)
            output_step_target = torch.tensor(indexesFromSentence(vocab, target_text), dtype=torch.long)
            output_step_gate = torch.tensor([1.0], dtype=torch.float)
            steps_data.append((output_step_x_t, output_step_target, output_step_gate))

        return (x_ref_tensor, steps_data)

    except Exception:
        return None


def merge_chunks(chunk_dir, data_type, delete_chunks_after_merge=True):
    """合并所有 chunk 文件成一个单一的 .pt 文件"""
    merged_data = []
    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.pt")), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for chunk_file in chunk_files:
        chunk_data = torch.load(chunk_file, weights_only=True)
        merged_data.extend(chunk_data)

    merged_file = os.path.join(chunk_dir, f"{data_type}.pt")
    torch.save(merged_data, merged_file)
    print(f"已合并所有块到 {merged_file}")

    if delete_chunks_after_merge:
        for chunk_file in chunk_files:
            os.remove(chunk_file)
        print(f"已删除单个 chunk 文件")


def process_line(line, vocab):
    """Worker 函数：处理单行 jsonl"""
    try:
        dialogue = json.loads(line)
        return process_dialogue_to_tensors(dialogue, vocab)
    except:
        return None

def count_lines_in_file(file_path):
    """快速计算文件行数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except:
        return None

def convert_to_binary(data_type, vocab, input_dir, output_dir, num_workers=1):
    """
    将 JSONL 文件转换为二进制格式，支持多进程和断点续传

    Args:
        data_type: 数据类型 ("train", "valid", "test")
        vocab: 词汇表对象
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        num_workers: 工作进程数
    """

    # 配置参数
    CHUNK_SIZE = 10000  # 每个 chunk 包含的对话数
    SAVE_PARTIAL_EVERY = 5000  # 每处理多少个对话保存一次 partial

    input_file = os.path.join(input_dir, f"{data_type}.jsonl")
    chunk_dir = os.path.join(output_dir, data_type)

    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return

    os.makedirs(chunk_dir, exist_ok=True)

    # 断点续传：检查是否有 resume 文件
    resume_file = os.path.join(chunk_dir, f"resume_{data_type}.json")
    chunk_index = 0
    processed_lines = 0
    has_partial = False

    if os.path.exists(resume_file):
        try:
            with open(resume_file, 'r') as rf:
                resume_data = json.load(rf)
                chunk_index = resume_data.get('chunk_index', 0)
                processed_lines = resume_data.get('processed_lines', 0)
                has_partial = resume_data.get('has_partial', False)
                print(f"从断点续传：块 {chunk_index}，已处理 {processed_lines} 行")
        except:
            print("resume 文件损坏，从头开始")

    # 如果有 partial 文件，尝试加载
    processed_dialogues = []
    if has_partial:
        partial_file = os.path.join(chunk_dir, f"partial_{chunk_index}.pt")
        try:
            processed_dialogues = torch.load(partial_file, weights_only=True)
            print(f"加载了 partial 文件，当前 processed_dialogues 大小: {len(processed_dialogues)}")
        except Exception as e:
            print(f"警告：加载 partial 文件失败 ({e})，\n从块 {chunk_index} 和 {processed_lines} 行续传。当前 processed_dialogues 大小: {len(processed_dialogues)}")
            processed_dialogues = []

    print(f"开始分块处理 {input_file}...")

    # 计算总行数（用于进度条）
    total_lines = count_lines_in_file(input_file)
    if total_lines:
        print(f"文件总行数: {total_lines}")

    # 准备词汇表状态用于多进程
    vocab_state = vocab.__dict__.copy()

    # 动态调整批次大小
    batch_size = max(100, 1000 // max(num_workers, 1))

    print(f"使用 {num_workers} 个进程处理 {data_type}")

    with open(input_file, 'r', encoding='utf-8') as f:
        # 跳过已处理的行
        for _ in range(processed_lines):
            next(f, None)

        # 创建进度条
        pbar = tqdm(total=total_lines - processed_lines if total_lines else None,
                   desc=f"转换 {data_type}",
                   initial=0)

        if num_workers > 1:
            # 多进程处理
            with Pool(num_workers, initializer=init_worker, initargs=(vocab_state,)) as pool:
                while True:
                    # 读取批次
                    batch_lines = list(islice(f, batch_size))
                    if not batch_lines:
                        break

                    # 提交批处理任务
                    future = pool.apply_async(process_batch_optimized, (batch_lines,))

                    try:
                        # 获取结果（设置超时）
                        batch_results = future.get(timeout=60)

                        # 反序列化并添加到结果
                        for serializable_data in batch_results:
                            tensor_data = deserialize_tensor_data(serializable_data)
                            processed_dialogues.append(tensor_data)

                    except Exception as e:
                        print(f"批处理失败: {e}")
                        # 回退到单进程处理这个批次
                        for line in batch_lines:
                            try:
                                dialogue = json.loads(line)
                                tensor_data = process_dialogue_to_tensors(dialogue, vocab)
                                if tensor_data:
                                    processed_dialogues.append(tensor_data)
                            except:
                                pass

                    processed_lines += len(batch_lines)
                    pbar.update(len(batch_lines))

                    # 检查是否需要保存 chunk 或 partial
                    if len(processed_dialogues) >= CHUNK_SIZE:
                        # 保存完整的 chunk
                        chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_index}.pt")
                        torch.save(processed_dialogues, chunk_file)
                        pbar.write(f"已保存块 {chunk_index} 到 {chunk_file} (批次 {len(batch_lines)})")

                        processed_dialogues = []
                        chunk_index += 1

                        # 更新 resume 状态
                        with open(resume_file, 'w') as rf:
                            json.dump({'chunk_index': chunk_index, 'processed_lines': processed_lines, 'has_partial': False}, rf)

                    elif len(processed_dialogues) > 0 and len(processed_dialogues) % SAVE_PARTIAL_EVERY == 0:
                        # 保存 partial 文件
                        partial_file = os.path.join(chunk_dir, f"partial_{chunk_index}.pt")
                        torch.save(processed_dialogues, partial_file)
                        pbar.write(f"已保存 partial 块 {chunk_index} 到 {partial_file} (大小: {len(processed_dialogues)})")

                        # 更新 resume 状态
                        with open(resume_file, 'w') as rf:
                            json.dump({'chunk_index': chunk_index, 'processed_lines': processed_lines, 'has_partial': True}, rf)
        else:
            # 单进程处理
            for line in f:
                try:
                    dialogue = json.loads(line)
                    tensor_data = process_dialogue_to_tensors(dialogue, vocab)
                    if tensor_data:
                        processed_dialogues.append(tensor_data)
                except:
                    pass

                processed_lines += 1
                pbar.update(1)

                # 检查是否需要保存 chunk 或 partial
                if len(processed_dialogues) >= CHUNK_SIZE:
                    # 保存完整的 chunk
                    chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_index}.pt")
                    torch.save(processed_dialogues, chunk_file)
                    pbar.write(f"已保存块 {chunk_index} 到 {chunk_file}")

                    processed_dialogues = []
                    chunk_index += 1

                    # 更新 resume 状态
                    with open(resume_file, 'w') as rf:
                        json.dump({'chunk_index': chunk_index, 'processed_lines': processed_lines, 'has_partial': False}, rf)

                elif len(processed_dialogues) > 0 and len(processed_dialogues) % SAVE_PARTIAL_EVERY == 0:
                    # 保存 partial 文件
                    partial_file = os.path.join(chunk_dir, f"partial_{chunk_index}.pt")
                    torch.save(processed_dialogues, partial_file)
                    pbar.write(f"已保存 partial 块 {chunk_index} 到 {partial_file} (大小: {len(processed_dialogues)})")

                    # 更新 resume 状态
                    with open(resume_file, 'w') as rf:
                        json.dump({'chunk_index': chunk_index, 'processed_lines': processed_lines, 'has_partial': True}, rf)

        pbar.close()

    # 保存最后的 chunk（如果有剩余数据）
    if processed_dialogues:
        chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_index}.pt")
        torch.save(processed_dialogues, chunk_file)
        pbar.write(f"已保存最后一个块 {chunk_index} 到 {chunk_file}")

    # 保存最终 resume 状态（即使完成，也保存以记录），但设置 has_partial=False
    with open(resume_file, 'w') as rf:
        json.dump({'chunk_index': chunk_index + 1 if processed_dialogues else chunk_index, 'processed_lines': processed_lines, 'has_partial': False}, rf)

    # 新增：如果处理完成（文件结束），合并所有 chunk
    if total_lines is not None and processed_lines >= total_lines:
        # 在合并前，如果有 partial 文件，保存为 chunk
        partial_pattern = os.path.join(chunk_dir, "partial_*.pt")
        for partial_file in glob.glob(partial_pattern):
            partial_index = int(partial_file.split('_')[-1].split('.')[0])
            target_chunk = os.path.join(chunk_dir, f"chunk_{partial_index}.pt")
            os.rename(partial_file, target_chunk)
            print(f"已将 partial {partial_file} 转换为 chunk {target_chunk}")
        merge_chunks(chunk_dir, data_type)
        # 可选：删除 resume 文件
        if os.path.exists(resume_file):
            os.remove(resume_file)

    # 处理完成后，可以选择删除 resume 文件，但保留以防万一
    print(f"{data_type} 数据集分块处理完成，保存在 '{chunk_dir}' 目录中。")
    logging.info(f"{data_type} 处理完成。")


def main():
    """主函数"""
    import argparse

    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="将LCCC数据集转换为二进制格式")
    parser.add_argument("--num_workers", type=int, default=1, help="工作进程数")
    args = parser.parse_args()

    try:
        print("--- 开始将LCCC数据集转换为二进制格式 ---")

        # 1. 加载词汇表
        vocab_path = os.path.join(config.LCCC_PROCESSED_PATH, "vocabulary.json")
        print(f"正在从 {vocab_path} 加载词汇表...")

        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)

        vocab = Vocabulary("lccc")
        vocab.__dict__.update(vocab_dict)
        print(f"词汇表加载完毕。大小: {vocab.num_words}")

        # 3. 转换所有数据集类型
        for data_type in ["train", "valid", "test"]:
            print(f"\n=== 开始处理 {data_type} 数据集 ===")
            try:
                convert_to_binary(
                    data_type=data_type,
                    vocab=vocab,
                    input_dir=config.LCCC_PROCESSED_PATH,
                    output_dir=config.LCCC_PROCESSED_PATH, # 输出到处理后的数据目录
                    num_workers=args.num_workers
                )
                print(f"✅ {data_type} 数据集处理完成")
            except Exception as e:
                print(f"❌ {data_type} 数据集处理失败: {e}")
                logging.error(f"{data_type} 处理失败: {e}")
                # 继续处理下一个数据集
                continue

        print("--- 所有数据文件分块转换完成！ ---")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        logging.error(f"处理失败: {e}")

if __name__ == "__main__":
    main()
