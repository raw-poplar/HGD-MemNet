import streamlit as st
import torch
import config
from .model import HGD_MemNet
from .utils import load_model_from_checkpoint

st.title('HGD-MemNet 聊天演示')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model_from_checkpoint(config.BEST_MODEL_DIR, device)
model.eval()

# 初始化对话历史
if 'history' not in st.session_state:
    st.session_state.history = []

user_input = st.text_input('输入您的消息:')
if st.button('发送'):
    if user_input:
        # 简化：转换为张量并运行模型（需实现实际逻辑）
        x_t = torch.tensor([user_input])  # 伪代码，需要实际转换
        x_ref = torch.tensor(st.session_state.history)  # 伪代码
        h_prev = torch.zeros(1, config.DYNAMIC_GROUP_HIDDEN_DIM, device=device)
        _, gate, output = model(x_t, x_ref, h_prev)
        response = '模型回应: ' + str(output)  # 简化
        st.session_state.history.append(user_input)
        st.session_state.history.append(response)
        st.write(response)

# 显示历史
for msg in st.session_state.history:
    st.write(msg) 