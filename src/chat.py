import sys
import torch
import os
import json
import config
from .model import HGD_MemNet
from .utils import load_model_from_checkpoint
from .dataset import Vocabulary

def load_vocabulary(path=None):
    """加载词汇表"""
    if path is None:
        path = os.path.join(config.LCCC_PROCESSED_PATH, 'vocabulary.json')
    with open(path, 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)
    vocab = Vocabulary('lccc')
    vocab.__dict__.update(vocab_dict)
    return vocab

def load_model_for_chat(device=None):
    """为聊天加载模型"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_from_checkpoint(config.BEST_MODEL_DIR, device)
    if model:
        model.eval()
    return model

def generate_response(model, vocab, user_input, history, h_prev):
    """生成回应的核心逻辑"""
    # 伪代码: 需要一个真实的 sentence -> index 的转换
    def indexesFromSentence(voc, sentence):
        return [voc.word2index.get(word, config.UNK_token) for word in sentence.split(' ')]

    x_t = torch.tensor(indexesFromSentence(vocab, user_input)).unsqueeze(0).to(next(model.parameters()).device)
    
    # 简化历史记录处理
    history_indices = []
    if history:
        history_indices = indexesFromSentence(vocab, " ".join(history))
    x_ref = torch.tensor(history_indices).unsqueeze(0).to(model.device)
    
    with torch.no_grad():
        h_next, gate, output_logits = model(x_t, x_ref, h_prev)

    # 简化: 取 argmax 生成回应
    topi = torch.argmax(output_logits, dim=-1)
    response_ids = topi.squeeze().tolist()
    
    # 确保 response_ids 是一个列表
    if not isinstance(response_ids, list):
        response_ids = [response_ids]

    response = ' '.join([vocab.index2word.get(str(idx), '<UNK>') for idx in response_ids])
    
    new_history = history + [user_input, response]
    return response, new_history, h_next

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("正在加载模型...")
    model = load_model_for_chat(device)
    if not model:
        print("模型加载失败，退出。")
        return
    print("模型加载成功。")

    print("正在加载词汇表...")
    vocab = load_vocabulary()
    print("词汇表加载成功。")

    history = []
    h_prev = torch.zeros(1, config.DYNAMIC_GROUP_HIDDEN_DIM, device=device)

    print('\n欢迎使用 HGD-MemNet 聊天！输入 "exit" 退出。')
    while True:
        try:
            user_input = input('您: ')
            if user_input.lower() == 'exit':
                print("再见!")
                break
            
            response, history, h_prev = generate_response(model, vocab, user_input, history, h_prev)
            print(f'模型: {response}')

        except KeyboardInterrupt:
            print("\n再见!")
            break

if __name__ == '__main__':
    main()
