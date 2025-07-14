
import torch
import json
import os
from tqdm import tqdm
import ijson
from functools import partial
import psutil
import glob # Added for chunking

# 导入项目特定的模块
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


def convert_to_binary(data_type, vocab, input_dir, output_dir):
    """
    读取 .jsonl 文件，处理所有对话，并将它们分块保存为 .pt 文件。
    """
    input_file = os.path.join(input_dir, f"{data_type}.jsonl")
    
    # 为数据块创建一个子目录
    chunk_dir = os.path.join(output_dir, data_type)
    os.makedirs(chunk_dir, exist_ok=True)
    
    # 先删除旧的块文件，以防重新运行时出错
    for old_chunk in glob.glob(os.path.join(chunk_dir, "*.pt")):
        os.remove(old_chunk)

    if not os.path.exists(input_file):
        print(f"警告：找不到输入文件 {input_file}，跳过该文件。")
        return

    print(f"开始分块处理 {input_file}...")

    processed_dialogues = []
    chunk_index = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
    except Exception:
        total_lines = None

    with open(input_file, 'r', encoding='utf-8') as f:
        pbar = tqdm(f, total=total_lines, desc=f"转换 {data_type}", unit=" 对话")
        for line in pbar:
            try:
                dialogue = json.loads(line)
                tensor_data = process_dialogue_to_tensors(dialogue, vocab)
                if tensor_data:
                    processed_dialogues.append(tensor_data)
                
                # 当达到块大小时，保存并重置
                if len(processed_dialogues) >= config.CHUNK_SIZE:
                    chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_index}.pt")
                    torch.save(processed_dialogues, chunk_file)
                    pbar.write(f"已保存块 {chunk_index} 到 {chunk_file}")
                    processed_dialogues = []
                    chunk_index += 1

            except json.JSONDecodeError:
                continue # 跳过无效行
            
            mem = psutil.virtual_memory()
            pbar.set_postfix({"内存占用": f"{mem.percent}%", "块": chunk_index})

    # 保存最后一个不足一个块大小的剩余数据
    if processed_dialogues:
        chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_index}.pt")
        torch.save(processed_dialogues, chunk_file)
        pbar.write(f"已保存最后一个块 {chunk_index} 到 {chunk_file}")

    print(f"{data_type} 数据集分块处理完成，保存在 '{chunk_dir}' 目录中。")


if __name__ == "__main__":
    print("--- 开始将LCCC数据集转换为二进制格式 ---")

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
            output_dir=config.LCCC_PROCESSED_PATH # 输出到处理后的数据目录
        )

    print("\n--- 所有数据文件分块转换完成！ ---")
    print(f"二进制数据块已保存到 '{config.LCCC_PROCESSED_PATH}' 下的相应子目录中。")
    print("现在您可以运行训练脚本了。") 