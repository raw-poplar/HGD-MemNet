import sys
import torch
import config
from .model import HGD_MemNet
from .utils import load_model_from_checkpoint

from src.dataset import Vocabulary  # 假设需要词汇表来转换索引到词

# 加载词汇表（简化，假设 vocabulary.json 存在）
vocab_path = os.path.join(config.LCCC_PROCESSED_PATH, 'vocabulary.json')
with open(vocab_path, 'r') as f:
    vocab_dict = json.load(f)
vocab = Vocabulary('lccc')
vocab.__dict__.update(vocab_dict)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_from_checkpoint(config.BEST_MODEL_DIR, device)
    model.eval()

    history = []  # 对话历史
    h_prev = torch.zeros(1, config.DYNAMIC_GROUP_HIDDEN_DIM, device=device)

    print('欢迎使用 HGD-MemNet 聊天！输入 "exit" 退出。')
    while True:
        user_input = input('您: ')
        if user_input.lower() == 'exit':
            break

        # 简化：转换为张量（需实现实际 tokenization）
        x_t = torch.tensor(indexesFromSentence(vocab, user_input)).unsqueeze(0).to(device)  # 伪代码
        x_ref = torch.tensor(history).to(device) if history else torch.tensor([])  # 简化

        h_next, gate, output_logits = model(x_t, x_ref, h_prev)

        # 简化：取 argmax 生成回应
        response_ids = torch.argmax(output_logits, dim=1).tolist()
        response = ' '.join([vocab.index2word.get(idx, '<UNK>') for idx in response_ids])

        print(f'模型: {response}')
        history.append(user_input)
        history.append(response)
        h_prev = h_next

if __name__ == '__main__':
    main() 