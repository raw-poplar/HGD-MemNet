#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速检查partial文件
"""

import os
import glob

def check_partial_files():
    """检查partial文件"""
    print("🔍 检查partial文件...")
    
    # 搜索所有可能的partial文件
    patterns = [
        "F:/modelTrain/data/lccc_processed/*/partial_*.pt",
        "F:/modelTrain/data/lccc_processed/partial_*.pt",
        "./data/lccc_processed/*/partial_*.pt",
        "./*/partial_*.pt",
        "./**/partial_*.pt"
    ]
    
    found_files = []
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        found_files.extend(files)
    
    # 去重
    found_files = list(set(found_files))
    
    if not found_files:
        print("✅ 未找到任何partial文件")
        return
    
    print(f"📦 找到 {len(found_files)} 个partial文件:")
    
    for file_path in found_files:
        print(f"\n📄 {file_path}")
        
        # 检查文件大小
        try:
            size = os.path.getsize(file_path)
            print(f"   大小: {size / (1024*1024):.2f} MB")
            
            if size == 0:
                print(f"   ⚠️  文件为空")
            elif size < 1024:
                print(f"   ⚠️  文件过小，可能损坏")
            
        except Exception as e:
            print(f"   ❌ 无法读取文件信息: {e}")
        
        # 尝试用torch加载
        try:
            import torch
            data = torch.load(file_path, weights_only=True)
            print(f"   ✅ 可以正常加载，包含 {len(data)} 个对话")
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            print(f"   💡 建议删除此文件")
            
            # 询问是否删除
            response = input(f"   是否删除损坏的文件? (y/n): ")
            if response.lower() == 'y':
                try:
                    # 备份
                    backup_path = file_path + ".backup"
                    os.rename(file_path, backup_path)
                    print(f"   ✅ 已重命名为: {backup_path}")
                except Exception as e:
                    print(f"   ❌ 重命名失败: {e}")

if __name__ == "__main__":
    check_partial_files()
