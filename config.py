# -*- coding: utf-8 -*-

import os

# ==============================================================================
# 配置文件使用说明（建议先读）
# ------------------------------------------------------------------------------
# 1) 本文件集中定义了模型结构、训练流程、数据路径与日志等配置。
#    - 可直接修改常量；或在训练/脚本中通过 `config.XXX = ...` 动态覆盖。
#    - 少数参数也支持环境变量覆盖（如 DATASET_PATH）。
#
# 2) 常见“如何选”简表：
#    - 内存吃紧：降低 EMBEDDING_DIM / DYNAMIC_GROUP_HIDDEN_DIM；减少 NUM_ATTENTION_HEADS；
#                 降低 BATCH_SIZE、提高 GRADIENT_ACCUMULATION_STEPS；关闭 USE_CONTEXTUAL_SAMPLER。
#    - 想要更强表达：适当增大 EMBEDDING_DIM/HIDDEN_DIM；开启多头注意力(>=2)；开启 USE_CONTEXTUAL_SAMPLER。
#    - 训练不稳定：
#        • 降低 LEARNING_RATE；
#        • 提高 GATE_THRESHOLD（更保守的发声）；
#        • 降低 INITIAL_TEMPERATURE 或提高 MIN_TEMPERATURE（减少采样噪声）；
#        • 关闭 USE_SOFT_TOPK_TRAINING（改为硬采样）以降低梯度噪声（可能影响收敛速度）。
#
# 3) 重要耦合关系：
#    - 多头注意力：DYNAMIC_GROUP_HIDDEN_DIM % NUM_ATTENTION_HEADS == 0 必须为真。
#    - 批大小与累积步：等效批次 ≈ BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS。
#    - 退火温度：训练循环按步内 t 使用 current_temperature = max(INITIAL_TEMPERATURE * TEMPERATURE_DECAY**t, MIN_TEMPERATURE)。
#    - 门控与步数：USE_GATED_MULTISTEP 与 MIN/MAX_THINKING_STEPS 一起决定何时“说话”。
#
# 4) 数据与mask：
#    - PAD_token 会用于构建注意力 padding mask，确保注意力不“看见”填充位。
#    - collate 函数会用 PAD_token 进行 pad，模型内部自动使用该 mask。
#
# 5) 剪枝/再生长（动态稀疏）：
#    - PRUNE_ENABLE / REGROW_ENABLE 控制是否周期性对核心RNN的连接进行稀疏化与再生长；
#    - 剪枝/再生长的统计（hebb/usage）在训练态通过 EMA 积累，暖身后再触发更稳健。
#
# 6) 路径与数据准备：
#    - DATASET_PATH 可通过环境变量覆盖（建议使用绝对路径或外接磁盘路径）；
#    - 预处理脚本会生成分块化的二进制数据（chunk_*.pt），训练脚本按需加载，降低内存占用。
#
# 7) 建议流程：
#    - 先运行数据预处理：python -m src.data_processing.prepare_binary_data --num_workers=4
#    - 再训练：python -m src.train
# ==============================================================================


# ------------------------------------
# 模型维度和结构相关的超参数
# ------------------------------------
# --- 内存与性能关键参数 ---
# VOCAB_SIZE: 词汇表大小。
#   - 实际值由词汇表构建脚本生成（vocabulary.json），此处为默认占位/上限。
#   - 影响 Embedding 与输出层（分类头）的参数量与显存占用。
VOCAB_SIZE = 5000

# EMBEDDING_DIM: 词嵌入维度。
#   - 越大表示能力越强，但显存与计算开销成正比；建议在 64–256 之间试验。
#   - 对应 nn.Embedding(vocab_size, embed_dim)。
EMBEDDING_DIM = 96

# DYNAMIC_GROUP_HIDDEN_DIM: 动态神经组隐藏维度（核心状态 h 的大小）。
#   - 影响动态组（GRU 编码器 + 核心RNN）与注意力（若启用）的维度；
#   - 若使用多头注意力，需保证 hidden_dim % NUM_ATTENTION_HEADS == 0。
DYNAMIC_GROUP_HIDDEN_DIM = 192

# STATIC_HEAD_HIDDEN_DIM: 静态决策头的中间层维度。
#   - 仅影响门控网络与输出网络的中间层宽度，对总参数量影响较 EMBEDDING/隐藏层小。
STATIC_HEAD_HIDDEN_DIM = 96


# --- 其他模型参数 ---
# CONTEXT_VECTOR_DIM: 预留的上下文向量维度（当前实现未直接使用）。
#   - 如未来引入独立上下文投影/融合，可使用该维度进行约束。
CONTEXT_VECTOR_DIM = 128

# 采样比例（用于静态头从动态组隐藏状态中“固定 + 随机”采样的占比）。
#   - FIXED_SAMPLING_RATIO: 固定采样比例（从前 N 个神经元）；
#   - RANDOM_SAMPLING_RATIO: 随机采样比例（从余下神经元池中按概率选取）。
#   - 二者之和不应超过 1.0；实际采样数会取整。
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
# BATCH_SIZE: 逻辑批大小（非等效）。
#   - 实际每次反向传播等效批次 ≈ BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS；
#   - 小显存推荐减小 BATCH_SIZE 并增大 GRADIENT_ACCUMULATION_STEPS。
BATCH_SIZE = 4

# GRADIENT_ACCUMULATION_STEPS: 梯度累积步数。
#   - 例如 BATCH_SIZE=4, ACCUMULATION=12，则等效批次≈48；
#   - 注意与学习率的相互作用（可适当线性放缩LR）。
GRADIENT_ACCUMULATION_STEPS = 12

# LOW_MEMORY_MODE: 低内存模式。
#   - True 时自动下调 BATCH_SIZE 并上调累积步，保持等效批次不变。
LOW_MEMORY_MODE = True
if LOW_MEMORY_MODE:
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 24  # 2*24≈48 等效批次大小


# --- 其他训练参数 ---
# THINKING_STEPS: 为每个对话轮设计的“内部思考步数”。
#   - 数据预处理阶段会为每个目标轮生成若干“思考步”（gate=0）和一个“输出步”（gate=1）；
#   - 训练时可搭配 USE_GATED_MULTISTEP 早停，模拟“思考到位即输出”。
THINKING_STEPS = 5

# 说明：若使用自适应思考（MIN/MAX_THINKING_STEPS），THINKING_STEPS 作为数据侧“思考步的最大规划参考”，
# 实际训练中是否提前/延后由门控与 MIN/MAX 决定。


# 新增：自适应思考步数（-1 表示不限制；训练/推理时可运行时覆盖）
MIN_THINKING_STEPS = -1
MAX_THINKING_STEPS = -1
# 安全兜底，防止极端情况下长时间不发声
SAFETY_MAX_THINKING_STEPS = 64

# LEARNING_RATE: 初始学习率；LR_DECAY_RATE: 轮级别的学习率衰减（train.py中通过StepLR示例）。
LEARNING_RATE = 0.001
LR_DECAY_RATE = 0.95

# NUM_EPOCHS: 总训练轮数；INNER_STEP_LR_DECAY: 同一个batch内部，随思考步 t 衰减学习率的因子。
NUM_EPOCHS = 20
INNER_STEP_LR_DECAY = 0.95

# 温度退火参数（控制 ReservoirRNNCell 的探索→收敛）。
#   - INITIAL_TEMPERATURE: 初始温度，越高越发散，建议 1.0–2.0；
#   - TEMPERATURE_DECAY: 每步退火率（<1.0），建议 0.9–0.99；
#   - MIN_TEMPERATURE: 最小温度下限，防止完全确定性导致停滞。
INITIAL_TEMPERATURE = 1.5
TEMPERATURE_DECAY = 0.95
MIN_TEMPERATURE = 0.1

# ------------------------------------
# 注意力机制相关参数
# ------------------------------------
# NUM_ATTENTION_HEADS: 注意力头数量（0/1/多）。
# 关于注意力头数量（NUM_ATTENTION_HEADS）：
#   - 0: 无注意力（使用历史编码的平均池化作为上下文），速度最快、最省显存；
#   - 1: 单头 Bahdanau 注意力；
#   - >=2: 多头注意力（Transformer 风格），注意 DYNAMIC_GROUP_HIDDEN_DIM % NUM_ATTENTION_HEADS == 0。
# ATTENTION_DROPOUT: 注意力权重上的 dropout（防过拟合）；
# ATTENTION_HEAD_DIM: 单头维度；默认为 None，表示使用 hidden_dim // num_heads；
# USE_ATTENTION_BIAS: 注意力线性映射是否带偏置；
# ATTENTION_TEMPERATURE: 注意力温度（softmax 锐度控制）。
# ATTENTION_TYPE: 注意力类型（"bahdanau"/"dot_product"/"multi_head"）。
NUM_ATTENTION_HEADS = 0
ATTENTION_DROPOUT = 0.1
ATTENTION_HEAD_DIM = None
USE_ATTENTION_BIAS = True
ATTENTION_TEMPERATURE = 1.0
# 训练与推理的“时机意识”控制向量（control）
#   - t_norm: 当前思考步 / cap；remain_norm: 剩余步 / cap；min_done: 是否达最小步；budget: 目标发声比例
#   - CONTROL_GATE_ALPHA: 若 >0，会用 control 的压缩信息与 gate_pred 做凸组合（无参数微调），默认关闭

ATTENTION_TYPE = "bahdanau"



# ------------------------------------
# 流式训练与数据加载
# ------------------------------------
# 是否启用按 chunk 流式训练；开启后训练会逐块加载数据，并在训练当前块时后台预取下一块
USE_STREAMING_TRAIN = True
# 预取下一块开关
STREAM_PREFETCH = True
# 预取使用的后台线程数（目前实现仅使用1，保留配置作为未来扩展）
STREAM_PREFETCH_WORKERS = 1
# 流式 DataLoader 的工作线程（Windows 建议 0 以避免多进程开销）
STREAM_DATALOADER_NUM_WORKERS = 0

# ------------------------------------
# 词表构建参数（用于 build_vocabulary.py 默认值）
# ------------------------------------
VOCAB_BUILD_SIZE = 30000
VOCAB_MIN_FREQ = 2

# ------------------------------------
# 随机性与日志设置
# ------------------------------------
# 随机种子（None 表示不固定）与确定性选项
SEED = None
DETERMINISTIC = False
# 允许 TF32（Ampere+ GPU 上建议开启以加速）
ALLOW_TF32 = True

# 日志与可视化
USE_TENSORBOARD = False
TENSORBOARD_LOG_DIR = "./runs"
USE_CSV_LOGGER = True
CSV_LOG_PATH = "./logs/train_metrics.csv"


# ------------------------------------
# 验证与模型保存
# ------------------------------------
# VALIDATE_EVERY_N_STEPS: 每 N 个“内部步”后在验证集上评估一次（频繁会减慢训练）。
VALIDATE_EVERY_N_STEPS = 200

# BEST_MODEL_DIR: 保存验证集最优模型权重的目录。
BEST_MODEL_DIR = "./best_model"


# ------------------------------------
# 门控与输出逻辑
# ------------------------------------
# GATE_THRESHOLD: 门控网络判定“输出”的阈值（0~1）。
#   - 语义：当 gate_pred >= GATE_THRESHOLD 且达到 MIN_THINKING_STEPS 时，模型触发“说话”；否则继续“思考”。
#   - 越高越保守：更倾向多思考几步再说；越低越激进：更快开始说话。
#   - 与 MIN/MAX_THINKING_STEPS 配合：达到 MAX_THINKING_STEPS 即使 gate 低也会强制说话（cap 行为）。
#   - 典型范围：0.5–0.9。建议从 0.7–0.85 区间微调，观测 gate_mean、gate_entropy 与 cap 触发率。
GATE_THRESHOLD = 0.8

# USE_SOFT_TOPK_TRAINING: 训练时静态头是否使用“近似可微 Top‑k”来替换硬采样，利于学习采样权重。
USE_SOFT_TOPK_TRAINING = True
# 是否使用基于上下文的随机采样评分器（增强版采样器）
USE_CONTEXTUAL_SAMPLER = True


# 门控正则与思考损失（可选）
# 目标发声比例（用于预算正则，不是硬约束）
TARGET_SPEAK_RATIO = 0.2
# 门控熵正则权重（鼓励适度不确定性，防止塌缩）
GATE_ENTROPY_WEIGHT = 1e-3
# 思考信息量损失（InfoNCE 等）的权重（0 表示关闭）
THINK_LOSS_WEIGHT = 0.0
# InfoNCE 温度
THINK_INFO_TAU = 0.1
# 通过控制向量对门控进行无参数微调的强度（0 关闭；建议 <=0.2）
CONTROL_GATE_ALPHA = 0.0


# USE_GATED_MULTISTEP: 是否启用“门控多步思考”早停（训练循环中当 gate>=阈值时提前跳出该样本的内部步）。
# ------------------------------------
# 剪枝与再生长（动态稀疏）设置
# ------------------------------------
PRUNE_ENABLE = False           # 开关：是否启用训练中剪枝
USAGE_EMA_BETA = 0.9          # 使用频度的EMA系数（与 HEBB_EMA_BETA 可一致）

PRUNE_START_STEPS = 2000       # 从第多少个总更新步开始剪枝（暖身）
PRUNE_EVERY_STEPS = 1000       # 每多少个总更新步进行一次剪枝
PRUNE_SPARSE_STEP = 0.05       # 每次新增剪枝比例（相对于当前激活连接的比例）
PRUNE_MIN_KEEP = 4             # 每行（每个输出神经元）至少保留的连接数

REGROW_ENABLE = False          # 开关：是否在剪枝后再生长
REGROW_PER_ROW = 1             # 每行再生长的连接数
REGROW_INIT_STD = 1e-3         # 新生连接权重初始化标准差

HEBB_EMA_BETA = 0.9            # 赫布分数的EMA系数（越大越平滑）

USE_GATED_MULTISTEP = False


# ------------------------------------
# 检查点 (Checkpoint) 相关设置
# ------------------------------------
# CHECKPOINT_DIR: 训练过程中的通用检查点目录（包含最近若干轮/步权重）。
CHECKPOINT_DIR = "./checkpoints"

# SAVE_CHECKPOINT_EVERY_N_BATCHES: 每处理多少个 batch（内部步聚合后的单位）保存一次检查点。
SAVE_CHECKPOINT_EVERY_N_BATCHES = 300

# MAX_CHECKPOINTS_TO_KEEP: 最多保留的检查点个数（滚动删除旧的）。
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
# 可通过环境变量 DATASET_PATH 覆盖默认根路径；建议使用相对路径或外部磁盘路径。
# 示例：在 PowerShell 中可设：$env:DATASET_PATH="D:/datasets"
# 原始 LCCC 文件命名可根据你的实际下载情况调整。

dataset_path = os.environ.get('DATASET_PATH', 'F:/modelTrain')
LCCC_RAW_PATH = os.path.join(dataset_path, 'LCCC')
LCCC_TRAIN_FILE = os.path.join(LCCC_RAW_PATH, 'LCCC-base_train.json')
LCCC_VALID_FILE = os.path.join(LCCC_RAW_PATH, 'LCCC-base_valid.json')
LCCC_TEST_FILE = os.path.join(LCCC_RAW_PATH, 'LCCC-base_test.json')

# 处理后的数据保存路径（预处理脚本输出目录）。
LCCC_PROCESSED_PATH = os.path.join(dataset_path, 'data', 'lccc_processed')


# 最小与最大对话长度（按句子计数），用于数据清洗或过滤。
MIN_CONVO_LENGTH = 2
MAX_CONVO_LENGTH = 25


# ------------------------------------
# 二进制数据处理
# ------------------------------------
# 在内存中一次处理的对话数量，用于生成二进制数据块 - 减小以适应内存
CHUNK_SIZE = 20000