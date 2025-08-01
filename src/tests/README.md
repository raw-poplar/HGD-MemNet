# HGD-MemNet 测试套件

这个目录包含了HGD-MemNet项目的完整测试套件，涵盖单元测试、集成测试和性能测试。

## 测试结构

```
src/tests/
├── README.md                    # 本文件
├── test_config.py              # 测试配置
├── run_tests.py                # 测试运行脚本
├── test_model_components.py    # 模型组件单元测试
├── test_model_integration.py   # 模型集成测试
├── test_training.py            # 训练流程测试
├── test_data_processing.py     # 数据处理测试
└── test_performance.py         # 性能测试
```

## 测试类型

### 1. 单元测试 (Unit Tests)
- **test_model_components.py**: 测试各个模型组件
  - Attention机制
  - ReservoirRNNCell
  - DynamicGroup
  - StaticHead
- **test_data_processing.py**: 测试数据处理功能
  - Vocabulary类
  - 数据加载和整理
  - 数据验证

### 2. 集成测试 (Integration Tests)
- **test_model_integration.py**: 测试完整模型
  - 端到端前向传播
  - 模型保存和加载
  - 不同模式下的行为
- **test_training.py**: 测试训练流程
  - 损失计算
  - 梯度流动
  - 训练稳定性

### 3. 性能测试 (Performance Tests)
- **test_performance.py**: 测试性能和效率
  - 前向传播速度
  - 内存使用
  - 批次和序列长度扩展性
  - GPU vs CPU性能对比

## 运行测试

### 快速开始

```bash
# 运行所有测试
python src/tests/run_tests.py

# 或者直接使用pytest
cd src/tests
python -m pytest -v
```

### 运行特定类型的测试

```bash
# 单元测试
python src/tests/run_tests.py --type unit

# 集成测试
python src/tests/run_tests.py --type integration

# 性能测试
python src/tests/run_tests.py --type performance

# 覆盖率测试
python src/tests/run_tests.py --type coverage
```

### 检查依赖

```bash
python src/tests/run_tests.py --check-deps
```

## 测试配置

测试使用较小的配置参数以确保快速执行：

- `TEST_VOCAB_SIZE = 1000`
- `TEST_EMBEDDING_DIM = 32`
- `TEST_DYNAMIC_GROUP_HIDDEN_DIM = 64`
- `TEST_STATIC_HEAD_HIDDEN_DIM = 32`
- `TEST_BATCH_SIZE = 2`
- `TEST_SEQ_LEN = 10`

这些配置在`test_config.py`中定义，可以根据需要调整。

## 测试覆盖范围

### 模型组件测试
- ✅ 前向传播正确性
- ✅ 输出形状验证
- ✅ 梯度流动检查
- ✅ 参数初始化
- ✅ 训练/评估模式切换

### 功能测试
- ✅ 温度参数效果
- ✅ 注意力权重归一化
- ✅ 门控机制
- ✅ 随机采样稳定性
- ✅ 序列处理

### 性能测试
- ✅ 前向传播速度
- ✅ 内存使用效率
- ✅ 批次大小扩展性
- ✅ 序列长度扩展性
- ✅ GPU加速效果

### 稳定性测试
- ✅ 梯度稳定性
- ✅ 数值稳定性
- ✅ 训练收敛性
- ✅ 内存泄漏检查

## 依赖要求

```
pytest>=6.0.0
torch>=1.9.0
numpy>=1.19.0
psutil>=5.8.0
coverage>=5.0.0  # 可选，用于覆盖率测试
```

## 持续集成

这些测试设计为可以在CI/CD环境中运行：

```yaml
# GitHub Actions 示例
- name: Run tests
  run: |
    pip install pytest torch numpy psutil
    python src/tests/run_tests.py --type all
```

## 故障排除

### 常见问题

1. **CUDA相关测试失败**
   - GPU测试会在没有CUDA的环境中自动跳过
   - 如果有CUDA但测试失败，检查GPU内存是否足够

2. **内存不足**
   - 调整`test_config.py`中的批次大小和模型维度
   - 确保系统有足够的可用内存

3. **导入错误**
   - 确保项目根目录在Python路径中
   - 检查所有依赖是否正确安装

4. **性能测试超时**
   - 性能测试的阈值可能需要根据硬件调整
   - 在较慢的机器上可以放宽时间限制

### 调试技巧

```bash
# 详细输出
python src/tests/run_tests.py --verbose

# 运行单个测试文件
python -m pytest src/tests/test_model_components.py -v

# 运行特定测试函数
python -m pytest src/tests/test_model_components.py::TestAttention::test_attention_forward -v

# 显示print输出
python -m pytest src/tests/test_performance.py -s
```

## 贡献指南

添加新测试时请遵循以下原则：

1. **命名规范**: 测试函数以`test_`开头
2. **文档**: 每个测试类和方法都应有清晰的文档字符串
3. **独立性**: 测试应该相互独立，不依赖执行顺序
4. **清理**: 使用`setup_method`和`teardown_method`进行资源管理
5. **断言**: 使用有意义的断言消息
6. **覆盖**: 新功能应该有相应的测试覆盖

## 性能基准

在标准硬件上的预期性能（仅供参考）：

- **前向传播**: < 10ms (CPU, 测试配置)
- **内存使用**: < 50MB 增长
- **GPU加速**: 2-5x 提升（取决于批次大小）

实际性能会根据硬件配置有所不同。
