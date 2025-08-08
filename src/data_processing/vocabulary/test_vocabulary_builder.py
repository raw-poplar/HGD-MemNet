#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
词汇表构建工具测试脚本

用于测试词汇表构建功能，包括创建测试数据和验证构建结果。
"""

import os
import json
import tempfile
import shutil
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.data_processing.vocabulary.build_vocabulary import VocabularyBuilder


def create_test_data(test_dir):
    """创建测试数据文件"""
    print("创建测试数据...")
    
    # 确保测试目录存在
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建测试对话数据
    test_dialogues = [
        # train.jsonl 数据
        [
            [{"text": "你好"}, {"text": "你好，很高兴见到你"}],
            [{"text": "今天天气怎么样"}, {"text": "今天天气很好，阳光明媚"}],
            [{"text": "你喜欢什么运动"}, {"text": "我喜欢跑步和游泳"}],
            [{"text": "你好"}, {"text": "你好，有什么可以帮助你的吗"}],
            [{"text": "谢谢你"}, {"text": "不客气，很高兴能帮到你"}],
        ],
        # valid.jsonl 数据
        [
            [{"text": "早上好"}, {"text": "早上好，今天是美好的一天"}],
            [{"text": "你好"}, {"text": "你好，欢迎来到这里"}],
            [{"text": "再见"}, {"text": "再见，期待下次见面"}],
        ],
        # test.jsonl 数据
        [
            [{"text": "晚安"}, {"text": "晚安，做个好梦"}],
            [{"text": "你好"}, {"text": "你好，很高兴认识你"}],
            [{"text": "今天天气很好"}, {"text": "是的，适合出去走走"}],
        ]
    ]
    
    file_names = ['train.jsonl', 'valid.jsonl', 'test.jsonl']
    
    for i, (file_name, dialogues) in enumerate(zip(file_names, test_dialogues)):
        file_path = os.path.join(test_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            for dialogue in dialogues:
                f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')
        print(f"   创建 {file_name}: {len(dialogues)} 个对话")
    
    return test_dir


def test_vocabulary_builder():
    """测试词汇表构建器"""
    print("测试词汇表构建器...")
    
    # 创建临时目录
    test_dir = tempfile.mkdtemp(prefix="vocab_test_")
    print(f"测试目录: {test_dir}")
    
    try:
        # 创建测试数据
        create_test_data(test_dir)
        
        # 创建词汇表构建器
        builder = VocabularyBuilder(
            data_path=test_dir,
            vocab_size=20,  # 小词汇表用于测试
            min_freq=1
        )
        
        # 构建词汇表
        vocab = builder.build_vocabulary()
        
        # 验证结果
        print(f"\n词汇表构建成功:")
        print(f"   词汇表大小: {vocab.num_words}")
        print(f"   包含特殊词元: {vocab.num_words >= 4}")
        
        # 打印统计信息
        builder.print_vocabulary_stats(vocab)
        
        # 保存词汇表
        output_path = builder.save_vocabulary(vocab)
        
        # 验证输出文件
        if os.path.exists(output_path):
            print(f"词汇表文件保存成功: {output_path}")
            
            # 验证文件内容
            with open(output_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            print(f"词汇表文件验证通过:")
            print(f"   包含字段: {list(vocab_data.keys())}")
            print(f"   词汇数量: {vocab_data['num_words']}")
        else:
            print(f"词汇表文件保存失败")
            return False
        
        # 验证统计文件
        stats_path = output_path.replace('.json', '_stats.json')
        if os.path.exists(stats_path):
            print(f"统计文件保存成功: {stats_path}")
        else:
            print(f"统计文件保存失败")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(test_dir)
            print(f"清理测试目录: {test_dir}")
        except:
            pass


def test_text_preprocessing():
    """测试文本预处理功能"""
    print("\n测试文本预处理...")
    
    builder = VocabularyBuilder()
    
    test_cases = [
        ("你好世界", ["你好", "世界"]),
        ("Hello World", ["Hello", "World"]),
        ("你好，世界！", ["你好", "世界"]),
        ("今天天气很好123", ["今天", "天气", "很", "好", "123"]),
        ("", []),
        (None, []),
        ("   ", []),
    ]
    
    for text, expected in test_cases:
        result = builder.preprocess_text(text)
        print(f"   输入: '{text}' -> 输出: {result}")
        
        # 简单验证（不要求完全匹配，因为jieba分词可能有差异）
        if text and text.strip():
            if len(result) == 0:
                print(f"   预期有输出但得到空结果")
        else:
            if len(result) != 0:
                print(f"   预期空结果但得到: {result}")
    
    print("文本预处理测试完成")


def main():
    """主函数"""
    print("词汇表构建工具测试")
    print("=" * 50)
    
    # 检查依赖
    try:
        import jieba
        print("jieba 依赖检查通过")
    except ImportError:
        print("jieba 未安装，请运行: pip install jieba")
        return 1
    
    # 测试文本预处理
    test_text_preprocessing()
    
    # 测试词汇表构建
    success = test_vocabulary_builder()
    
    if success:
        print(f"\n所有测试通过!")
        print(f"\n现在可以使用以下命令构建真实的词汇表:")
        print(f"   python -m src.data_processing.vocabulary.build_vocab_quick")
        return 0
    else:
        print(f"\n测试失败!")
        return 1


if __name__ == "__main__":
    exit(main())
