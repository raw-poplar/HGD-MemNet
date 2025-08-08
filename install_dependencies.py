#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目依赖安装脚本

安装HGD-MemNet项目所需的所有依赖包。
"""

import subprocess
import sys
import os


def install_package(package_name):
    """安装Python包"""
    try:
        print(f"安装 {package_name}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_name
        ], capture_output=True, text=True, check=True)
        print(f"{package_name} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{package_name} 安装失败:")
        print(f"   错误信息: {e.stderr}")
        return False


def check_package(package_name):
    """检查包是否已安装"""
    try:
        __import__(package_name)
        print(f"{package_name} 已安装")
        return True
    except ImportError:
        print(f"{package_name} 未安装")
        return False


def install_from_requirements():
    """从requirements.txt安装依赖"""
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"未找到 {requirements_file} 文件")
        return False
    
    try:
        print(f"从 {requirements_file} 安装依赖...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ], capture_output=True, text=True, check=True)
        print("requirements.txt 中的依赖安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print("从 requirements.txt 安装失败:")
        print(f"   错误信息: {e.stderr}")
        return False


def main():
    """主函数"""
    print("HGD-MemNet 项目依赖安装")
    print("=" * 40)
    
    # 首先尝试从requirements.txt安装
    print("步骤1: 安装基础依赖")
    if install_from_requirements():
        print("基础依赖安装完成")
    else:
        print("基础依赖安装失败，尝试手动安装关键包...")
        
        # 手动安装关键包
        key_packages = ["torch", "tqdm", "jieba"]
        for package in key_packages:
            install_package(package)
    
    print("\n步骤2: 检查关键依赖")
    
    # 检查关键依赖
    critical_packages = {
        "torch": "PyTorch深度学习框架",
        "tqdm": "进度条显示",
        "jieba": "中文分词（词汇表构建必需）",
        "json": "JSON处理（内置模块）"
    }
    
    all_installed = True
    for package, description in critical_packages.items():
        print(f"检查 {package} ({description})...")
        if not check_package(package):
            all_installed = False
    
    print("\n安装结果:")
    if all_installed:
        print("所有关键依赖都已安装!")
        print("\n下一步:")
        print("1. 准备数据文件 (train.jsonl, valid.jsonl, test.jsonl)")
        print("2. 构建词汇表: python -m src.data_processing.vocabulary.build_vocab_quick")
        print("3. 开始训练: python -m src.train")
        return 0
    else:
        print("部分依赖安装失败，请手动安装缺失的包")
        print("手动安装命令:")
        for package in critical_packages:
            if package != "json":  # json是内置模块
                print(f"   pip install {package}")
        return 1


if __name__ == "__main__":
    exit(main())
