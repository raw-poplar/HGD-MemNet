# 数据处理模块

这个模块包含了将LCCC数据集转换为二进制格式的所有工具和脚本。

## 📁 文件结构

```
src/data_processing/
├── __init__.py                    # 模块初始化文件
├── README.md                      # 本文件
├── prepare_binary_data.py         # 主要的数据转换脚本
├── merge_tools.py                 # 数据合并工具集合
├── data_utils.py                  # 数据处理工具函数
├── debug_tools.py                 # 调试和检查工具
└── legacy_merge_scripts.py        # 遗留合并脚本（备用）
```

## 🚀 主要功能

### 1. 数据转换 (`prepare_binary_data.py`)

将原始JSONL格式的对话数据转换为PyTorch张量格式。

**特性**:
- ✅ 多进程并行处理
- ✅ 断点续传支持
- ✅ 内存优化
- ✅ 分块保存

**使用方法**:
```bash
# 基本使用
python -m src.data_processing.prepare_binary_data

# 指定工作进程数
python -m src.data_processing.prepare_binary_data --num_workers=4
```

### 2. 数据合并 (`merge_tools.py`)

将分块的chunk文件合并为最终的训练/验证/测试文件。

**合并方法**:
- `simple`: 简单一次性合并
- `optimized`: 流式处理，内存友好
- `large`: 专门处理大文件
- `ultra`: 超级优化版，分段保存，支持断点续传

**使用方法**:
```bash
# 使用优化方法合并所有数据集
python -m src.data_processing.merge_tools --method=optimized --dataset=all

# 合并特定数据集
python -m src.data_processing.merge_tools --method=large --dataset=train

# 使用超级优化方法（推荐用于大文件）
python -m src.data_processing.merge_tools --method=ultra --dataset=train --workers=2 --batch-size=20

# 自定义参数的超级优化合并
python -m src.data_processing.merge_tools --method=ultra --dataset=train --workers=1 --batch-size=10 --save-interval=30

# 合并后验证和清理
python -m src.data_processing.merge_tools --verify --cleanup
```

### 3. 数据工具 (`data_utils.py`)

提供各种数据处理相关的工具函数。

**功能**:
- 数据完整性检查
- 处理时间估算
- 磁盘空间管理
- 文件清理

**使用方法**:
```bash
# 检查所有数据
python -m src.data_processing.data_utils --check --estimate --space

# 清理partial文件
python -m src.data_processing.data_utils --cleanup

# 检查特定数据集
python -m src.data_processing.data_utils --check --dataset=train
```

### 4. 调试工具 (`debug_tools.py`)

提供调试和检查功能。

**功能**:
- chunk文件加载测试
- 处理状态分析
- 数据兼容性检查
- 性能基准测试

**使用方法**:
```bash
# 运行所有调试测试
python -m src.data_processing.debug_tools

# 测试chunk文件加载
python -m src.data_processing.debug_tools --test-loading --dataset=valid

# 性能基准测试
python -m src.data_processing.debug_tools --benchmark --dataset=train
```

## 📋 完整处理流程

### 步骤1: 数据转换
```bash
# 将JSONL文件转换为chunk文件
python -m src.data_processing.prepare_binary_data --num_workers=4
```

### 步骤2: 检查状态
```bash
# 检查转换结果
python -m src.data_processing.data_utils --check
```

### 步骤3: 合并数据
```bash
# 合并chunk文件为最终数据集（推荐用于大文件）
python -m src.data_processing.merge_tools --method=ultra --dataset=all --workers=2

# 或使用传统优化方法
python -m src.data_processing.merge_tools --method=optimized --dataset=all
```

### 步骤4: 验证结果
```bash
# 验证合并结果
python -m src.data_processing.merge_tools --verify
```

### 步骤5: 清理文件
```bash
# 清理临时文件
python -m src.data_processing.data_utils --cleanup
python -m src.data_processing.merge_tools --cleanup
```

## � 新增超级优化方法 (ultra)

### 特性
- ✅ **分段保存**: 避免内存溢出，支持超大文件
- ✅ **真正的断点续传**: 从任意中断点继续
- ✅ **内存友好**: 最小内存占用
- ✅ **高效I/O**: 减少文件读写次数
- ✅ **多线程支持**: 可配置工作线程数
- ✅ **自动清理**: 完成后自动清理临时文件

### 使用场景
- 处理超大数据集（>100GB）
- 内存受限的环境
- 需要长时间运行的合并任务
- 网络不稳定可能中断的环境

### 参数说明
- `--workers`: 工作线程数 (默认: 2)
- `--batch-size`: 批处理大小 (默认: 20)
- `--save-interval`: 保存间隔，每处理多少个chunk保存一次 (默认: 50)

## �🔧 配置说明

### 环境变量
- `DATASET_PATH`: 数据集根目录路径
- `LCCC_PROCESSED_PATH`: 处理后数据的存储路径

### 重要参数
- `CHUNK_SIZE`: 每个chunk包含的对话数 (默认: 10000)
- `SAVE_PARTIAL_EVERY`: 保存partial文件的频率 (默认: 5000)
- `THINKING_STEPS`: 思考步骤数 (默认: 3)

## 📊 数据格式

### 输入格式 (JSONL)
```json
[
    {"text": "你好"},
    {"text": "你好，有什么可以帮助你的吗？"},
    {"text": "我想了解一下产品信息"}
]
```

### 输出格式 (PyTorch张量)
```python
(x_ref_tensor, steps_data)
# x_ref_tensor: 参考句子的张量
# steps_data: [(x_t, target, gate), ...] 步骤数据列表
```

## 🚨 故障排除

### 常见问题

1. **内存不足**
   - 减少 `num_workers` 参数
   - 使用 `optimized` 合并方法
   - 增加系统虚拟内存

2. **磁盘空间不足**
   - 清理partial文件
   - 分步合并（先小数据集，后大数据集）
   - 删除不必要的chunk文件

3. **处理速度慢**
   - 增加 `num_workers` 参数
   - 使用SSD存储
   - 关闭其他占用资源的程序

4. **文件损坏**
   - 使用调试工具检查文件完整性
   - 重新处理损坏的chunk
   - 检查磁盘健康状态

### 调试命令
```bash
# 检查chunk文件完整性
python -m src.data_processing.debug_tools --test-loading

# 分析处理状态
python -m src.data_processing.debug_tools --analyze-status

# 测试数据兼容性
python -m src.data_processing.debug_tools --test-compatibility
```

## 📈 性能优化

### 方法选择建议
- **小文件 (<10GB)**: 使用 `simple` 方法
- **中等文件 (10-100GB)**: 使用 `optimized` 方法
- **大文件 (>100GB)**: 使用 `ultra` 方法
- **内存受限环境**: 优先使用 `ultra` 方法

### 参数调优建议
- **workers**:
  - 内存充足: 2-4个线程
  - 内存受限: 1个线程
  - SSD存储: 可适当增加到4-8个
- **batch-size**:
  - 小内存: 10-20
  - 大内存: 20-50
  - SSD存储: 可增加到50-100
- **save-interval** (ultra方法):
  - 快速存储: 50-100
  - 慢速存储: 20-50

### 多进程优化
- 使用2-8个工作进程
- 根据CPU核心数和内存大小调整
- 监控系统资源使用情况

### 内存优化
- 使用流式处理
- 及时清理不需要的数据
- 分批处理大文件

### I/O优化
- 使用SSD存储
- 避免频繁的小文件操作
- 批量读写数据

## 🔗 相关文件

- `config.py`: 全局配置文件
- `src/dataset.py`: 数据集和词汇表定义
- `src/tests/`: 相关测试文件

## 📝 更新日志

### v1.0.0
- 初始版本
- 基本的数据转换和合并功能
- 多进程支持
- 断点续传功能

### 未来计划
- [ ] 增加数据压缩功能
- [ ] 支持增量更新
- [ ] 添加数据统计分析
- [ ] 优化大文件处理性能
