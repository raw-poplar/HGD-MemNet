
import torch
import json
import os
from tqdm import tqdm
import ijson
from functools import partial
import psutil
import glob # Added for chunking
import json  # 新增，确保已导入
from multiprocessing import Pool, cpu_count  # 新增 for 多进程
from itertools import islice # Added for islice
import logging  # 新增：日志记录
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config
from src.dataset import Vocabulary

# --- 辅助函数 ---
def indexesFromSentence(vocab, sentence):
    """将句子转换为索引列表"""
    if sentence is None:
        return []
    UNK_idx = vocab.word2index.get("<UNK>", config.UNK_token)
    return [vocab.word2index.get(word, UNK_idx) for word in sentence.split(' ')] + [config.EOS_token]

def process_dialogue_to_tensors(dialogue_list, vocab):
    """
    将单个对话（一个字典列表）处理成张量格式。
    这是修正后的版本，以处理 .jsonl 文件中的实际数据格式。
    """
    # 校验输入是否为非空列表
    if not isinstance(dialogue_list, list) or not dialogue_list:
        return None # 如果数据无效，则跳过

    # 1. 将第一句话作为参照输入 x_ref
    # 从字典中获取文本
    x_ref_text = dialogue_list[0].get("text", "") 
    x_ref_tensor = torch.tensor(indexesFromSentence(vocab, x_ref_text), dtype=torch.long)

    # 2. 构建“思考-输出”步骤
    steps_data = []
    # 从第二句话开始，两两配对，模拟问答
    for i in range(1, len(dialogue_list)):
        # 当前的输入 x_t 是前一句话的文本
        x_t_text = dialogue_list[i-1].get("text", "")
        # 目标 target 是当前这句话的文本
        target_text = dialogue_list[i].get("text", "")

        # 模拟模型的 "思考" 步骤
        # 在这个简化的设定中，我们为每个真实回答前都添加N个“思考”步骤
        # 在这些步骤中，模型没有输出，只有门控目标为0
        for _ in range(config.THINKING_STEPS):
            thinking_step_x_t = torch.tensor(indexesFromSentence(vocab, x_t_text), dtype=torch.long)
            # 思考步骤没有目标输出
            thinking_step_target = torch.tensor([], dtype=torch.long) 
            # 门控目标为0，表示不输出
            thinking_step_gate = torch.tensor([0.0], dtype=torch.float)
            steps_data.append((thinking_step_x_t, thinking_step_target, thinking_step_gate))

        # 添加真实的 "输出" 步骤
        output_step_x_t = torch.tensor(indexesFromSentence(vocab, x_t_text), dtype=torch.long)
        output_step_target = torch.tensor(indexesFromSentence(vocab, target_text), dtype=torch.long)
        # 门控目标为1，表示需要输出
        output_step_gate = torch.tensor([1.0], dtype=torch.float)
        steps_data.append((output_step_x_t, output_step_target, output_step_gate))

    # 只有当步骤列表不为空时，才认为这是一个有效的样本
    if not steps_data:
        return None
        
    return (x_ref_tensor, steps_data)


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
    except json.JSONDecodeError:
        return None


def convert_to_binary(data_type, vocab, input_dir, output_dir, num_workers=1):
    """
    读取 .jsonl 文件，处理所有对话，并将它们分块保存为 .pt 文件。
    支持多进程加速。
    """
    input_file = os.path.join(input_dir, f"{data_type}.jsonl")
    
    # 为数据块创建一个子目录
    chunk_dir = os.path.join(output_dir, data_type)
    os.makedirs(chunk_dir, exist_ok=True)
    
    # 不再删除旧的块文件，支持续传
    # 为 resume，检查现有的块并加载状态
    resume_file = os.path.join(chunk_dir, "resume.json")
    SAVE_PARTIAL_EVERY = 1000  # 每处理多少个对话保存一次 partial
    if os.path.exists(resume_file):
        with open(resume_file, 'r') as rf:
            resume_state = json.load(rf)
        chunk_index = resume_state['chunk_index']
        processed_lines = resume_state['processed_lines']
        has_partial = resume_state.get('has_partial', False)
        if has_partial:
            partial_file = os.path.join(chunk_dir, f"partial_{chunk_index}.pt")
            try:
                if os.path.exists(partial_file):
                    processed_dialogues = torch.load(partial_file, weights_only=True)
                    print(f"加载 partial 文件 {partial_file}，当前 processed_dialogues 大小: {len(processed_dialogues)}")
                else:
                    processed_dialogues = []
                    print("警告：has_partial=True 但 partial 文件不存在，从空开始")
            except Exception as e:
                print(f"警告：加载 partial 文件失败 ({e})，重置为从空开始")
                processed_dialogues = []
        else:
            processed_dialogues = []
        print(f"从块 {chunk_index} 和 {processed_lines} 行续传。当前 processed_dialogues 大小: {len(processed_dialogues)}")
    else:
        # 无 resume，从头开始
        processed_dialogues = []
        chunk_index = 0
        processed_lines = 0
        has_partial = False

    if not os.path.exists(input_file):
        print(f"警告：找不到输入文件 {input_file}，跳过该文件。")
        return

    print(f"开始分块处理 {input_file}...")

    total_lines = None
    try:
        # 优化：仅当文件小于1GB时才计算总行数，以避免在处理大文件时卡住
        if os.path.exists(input_file) and os.path.getsize(input_file) < 1 * 1024 * 1024 * 1024:
            with open(input_file, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
    except Exception as e:
        print(f"无法计算总行数，将不显示总进度：{e}")
        total_lines = None

    # 新增：限制 num_workers 不超过可用 CPU
    num_workers = min(num_workers, cpu_count())
    print(f"使用 {num_workers} 个进程处理 {data_type}")

    with open(input_file, 'r', encoding='utf-8') as f:
        # 跳过已处理的行
        for _ in range(processed_lines):
            next(f)
        
        pbar = tqdm(total=total_lines - processed_lines if total_lines else None, desc=f"转换 {data_type}", unit=" 对话", initial=processed_lines, mininterval=0.01)
        
        if num_workers > 1:
            # 多进程模式：分批读取并处理
            BATCH_SIZE = 1000  # 每批处理的行数
            pool = Pool(num_workers)
            while True:
                batch_lines = list(islice(f, BATCH_SIZE))
                if not batch_lines:
                    break
                # 并行处理批次
                results = pool.starmap(process_line, [(line, vocab) for line in batch_lines])
                for result in results:
                    processed_lines += 1
                    if result:
                        processed_dialogues.append(result)
                    pbar.update(1)
                    # chunking 和 partial 保存逻辑 (同单线程)
                    if len(processed_dialogues) >= config.CHUNK_SIZE:
                        chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_index}.pt")
                        torch.save(processed_dialogues, chunk_file)
                        pbar.write(f"已保存块 {chunk_index} 到 {chunk_file}")
                        chunk_index += 1
                        processed_dialogues = []
                        # 更新 resume，无 partial
                        with open(resume_file, 'w') as rf:
                            json.dump({'chunk_index': chunk_index, 'processed_lines': processed_lines, 'has_partial': False}, rf)
                    elif len(processed_dialogues) % SAVE_PARTIAL_EVERY == 0 and len(processed_dialogues) > 0:
                        partial_file = os.path.join(chunk_dir, f"partial_{chunk_index}.pt")
                        torch.save(processed_dialogues, partial_file)
                        pbar.write(f"已保存 partial 块 {chunk_index} 到 {partial_file} (大小: {len(processed_dialogues)})")
                        # 更新 resume，有 partial
                        with open(resume_file, 'w') as rf:
                            json.dump({'chunk_index': chunk_index, 'processed_lines': processed_lines, 'has_partial': True}, rf)
                # 批次后更新 resume
                with open(resume_file, 'w') as rf:
                    json.dump({'chunk_index': chunk_index, 'processed_lines': processed_lines, 'has_partial': len(processed_dialogues) > 0}, rf)
            pool.close()
            pool.join()
        else:
            # 单线程模式 (原有循环)
            for line in f:
                processed_lines += 1
                try:
                    dialogue = json.loads(line)
                    tensor_data = process_dialogue_to_tensors(dialogue, vocab)
                    if tensor_data:
                        processed_dialogues.append(tensor_data)
                    
                    pbar.update(1)
                    # 当达到块大小时，保存并重置
                    if len(processed_dialogues) >= config.CHUNK_SIZE:
                        chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_index}.pt")
                        torch.save(processed_dialogues, chunk_file)
                        pbar.write(f"已保存块 {chunk_index} 到 {chunk_file}")
                        chunk_index += 1
                        processed_dialogues = []
                        # 更新 resume，无 partial
                        with open(resume_file, 'w') as rf:
                            json.dump({'chunk_index': chunk_index, 'processed_lines': processed_lines, 'has_partial': False}, rf)
                    elif len(processed_dialogues) % SAVE_PARTIAL_EVERY == 0 and len(processed_dialogues) > 0:
                        partial_file = os.path.join(chunk_dir, f"partial_{chunk_index}.pt")
                        torch.save(processed_dialogues, partial_file)
                        pbar.write(f"已保存 partial 块 {chunk_index} 到 {partial_file} (大小: {len(processed_dialogues)})")
                        # 更新 resume，有 partial
                        with open(resume_file, 'w') as rf:
                            json.dump({'chunk_index': chunk_index, 'processed_lines': processed_lines, 'has_partial': True}, rf)

                except json.JSONDecodeError:
                    continue # 跳过无效行
                
                mem = psutil.virtual_memory()
                pbar.set_postfix({"内存占用": f"{mem.percent}%", "块": chunk_index})

    # 保存最后一个不足一个块大小的剩余数据
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


if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes')
        args = parser.parse_args()
        
        print("--- 开始将LCCC数据集转换为二进制格式 ---")
        logging.info("开始数据转换流程")

        # 1. 确保目标目录存在
        output_path = config.LCCC_PROCESSED_PATH
        os.makedirs(output_path, exist_ok=True)
        
        # 2. 加载词汇表
        vocab_path = os.path.join(config.LCCC_PROCESSED_PATH, "vocabulary.json")
        print(f"正在从 {vocab_path} 加载词汇表...")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"在 {vocab_path} 未找到词汇表文件。请先运行 preprocess_lccc.py。")
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        vocab = Vocabulary("lccc")
        vocab.__dict__.update(vocab_dict)
        print(f"词汇表加载完毕。大小: {vocab.num_words}")

        # 3. 转换所有数据集类型
        for data_type in ["train", "valid", "test"]:
            convert_to_binary(
                data_type=data_type,
                vocab=vocab,
                input_dir=config.LCCC_PROCESSED_PATH,
                output_dir=config.LCCC_PROCESSED_PATH, # 输出到处理后的数据目录
                num_workers=args.num_workers
            )

        print("\n--- 所有数据文件分块转换完成！ ---")
        logging.info("数据转换完成")
    except Exception as e:
        logging.error(f"主流程出错: {e}")
        print(f"错误: {e}") 