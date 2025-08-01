# -*- coding: utf-8 -*-
"""
测试配置文件
提供测试专用的配置参数，避免影响主配置
"""

import os
import sys
import tempfile

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 测试专用的小规模配置
TEST_VOCAB_SIZE = 1000
TEST_EMBEDDING_DIM = 32
TEST_DYNAMIC_GROUP_HIDDEN_DIM = 64
TEST_STATIC_HEAD_HIDDEN_DIM = 32
TEST_BATCH_SIZE = 2
TEST_SEQ_LEN = 10
TEST_THINKING_STEPS = 3

# 特殊词元
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

# 采样比例
FIXED_SAMPLING_RATIO = 0.3
RANDOM_SAMPLING_RATIO = 0.05

# 温度参数
INITIAL_TEMPERATURE = 1.5
TEMPERATURE_DECAY = 0.95
MIN_TEMPERATURE = 0.1

# 门控阈值
GATE_THRESHOLD = 0.5

# 临时目录用于测试
TEST_DATA_DIR = tempfile.mkdtemp()
TEST_CHECKPOINT_DIR = os.path.join(TEST_DATA_DIR, "checkpoints")
TEST_MODEL_DIR = os.path.join(TEST_DATA_DIR, "models")

# 创建测试目录
os.makedirs(TEST_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TEST_MODEL_DIR, exist_ok=True)

def cleanup_test_dirs():
    """清理测试目录"""
    import shutil
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
