# -*- coding: utf-8 -*-

# ------------------------------------
# 模型维度和结构相关的超参数
# ------------------------------------
# --- 内存与性能关键参数 ---
# 词汇表大小 (由预处理脚本生成，此为占位符)
VOCAB_SIZE = 5000
# 词嵌入维度 (影响模型大小和内存)
EMBEDDING_DIM = 128
# 动态神经组隐藏层维度 (影响模型大小和内存)
DYNAMIC_GROUP_HIDDEN_DIM = 256
# 静态网络隐藏层维度 (影响模型大小和内存)
STATIC_HEAD_HIDDEN_DIM = 128


# --- 其他模型参数 ---
# 上下文向量维度
CONTEXT_VECTOR_DIM = 128
# 采样比例 (模型内部使用，对内存影响较小)
FIXED_SAMPLING_RATIO = 0.3
RANDOM_SAMPLING_RATIO = 0.05


# ------------------------------------
# 特殊词元 (Special Tokens)
# ------------------------------------
PAD_token = 0  # 用于填充短句
SOS_token = 1  # 句子开头
EOS_token = 2  # 句子结尾
UNK_token = 3  # 未知词


# ------------------------------------
# 训练过程相关的超参数
# ------------------------------------
# --- 内存与性能关键参数 ---
# 逻辑批处理大小。实际GPU处理的大小会是 BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS。
BATCH_SIZE = 12
# 梯度累积步骤。用于在不增加内存消耗的情况下模拟更大的批次。
# 例如，BATCH_SIZE=32, ACCUMULATION_STEPS=4, 则GPU每次处理8个样本，累积4次后更新一次权重。
GRADIENT_ACCUMULATION_STEPS = 4


# --- 其他训练参数 ---
# 迭代优化训练的“思考步数”
THINKING_STEPS = 5
# 学习率
LEARNING_RATE = 0.001
# 学习率衰减策略 (简单起见，我们先设为固定值，后续可改为衰减)
LR_DECAY_RATE = 0.95
# 训练轮次
NUM_EPOCHS = 20


# ------------------------------------
# 验证与模型保存
# ------------------------------------
# 每N个更新步数后进行一次验证
VALIDATE_EVERY_N_STEPS = 200

# 保存最佳模型的目录
BEST_MODEL_DIR = "./best_model"


# ------------------------------------
# 门控和输出逻辑相关的超参数
# ------------------------------------
# 门控网络的输出阈值
GATE_THRESHOLD = 0.8


# ------------------------------------
# 检查点 (Checkpoint) 相关设置
# ------------------------------------
# 保存检查点的目录
CHECKPOINT_DIR = "./checkpoints"

# 每处理多少个批次(batch)的数据后就保存一次检查点
SAVE_CHECKPOINT_EVERY_N_BATCHES = 300

# 最多保留多少个最新的检查点文件
MAX_CHECKPOINTS_TO_KEEP = 5


# ------------------------------------
# 数据集和文件路径
# ------------------------------------

# --- 康奈尔电影对话数据集 (原始) ---
RAW_DATASET_PATH_CORNELL = "./data/cornell movie-dialogs corpus"
MOVIE_LINES_FILE = f"{RAW_DATASET_PATH_CORNELL}/movie_lines.txt"
MOVIE_CONVERSATIONS_FILE = f"{RAW_DATASET_PATH_CORNELL}/movie_conversations.txt"
PROCESSED_DATA_PATH_CORNELL = "./data/cornell_processed/processed_dialogues.jsonl"


# --- LCCC 中文对话数据集 ---
# 原始数据路径
LCCC_RAW_PATH = "./src/data/LCCC"
LCCC_TRAIN_FILE = f"{LCCC_RAW_PATH}/LCCC-base_train.json"
LCCC_VALID_FILE = f"{LCCC_RAW_PATH}/LCCC-base_valid.json"
LCCC_TEST_FILE = f"{LCCC_RAW_PATH}/LCCC-base_test.json"

# 处理后的数据保存路径
LCCC_PROCESSED_PATH = "./data/lccc_processed"


# 最小和最大对话长度（按句子数）
MIN_CONVO_LENGTH = 2
MAX_CONVO_LENGTH = 25 


# ------------------------------------
# 二进制数据处理
# ------------------------------------
# 在内存中一次处理的对话数量，用于生成二进制数据块
CHUNK_SIZE = 100000 