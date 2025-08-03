#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理损坏的partial文件
"""

import os
import sys
import torch
import json
sys.path.append('.')
import config

def cleanup_partial_files():
    """清理损坏的partial文件"""
    print("🧹 清理损坏的partial文件...")
    
    data_types = ["train", "valid", "test"]
    cleaned_files = []
    
    for data_type in data_types:
        chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
        if not os.path.exists(chunk_dir):
            continue
            
        print(f"\n📁 检查 {data_type} 目录: {chunk_dir}")
        
        # 查找所有partial文件
        partial_files = []
        for file in os.listdir(chunk_dir):
            if file.startswith("partial_") and file.endswith(".pt"):
                partial_files.append(file)
        
        if not partial_files:
            print(f"   ✅ 无partial文件")
            continue
            
        print(f"   📦 找到 {len(partial_files)} 个partial文件")
        
        for partial_file in partial_files:
            partial_path = os.path.join(chunk_dir, partial_file)
            print(f"   🔍 检查: {partial_file}")
            
            try:
                # 尝试加载文件
                data = torch.load(partial_path, weights_only=True)
                print(f"      ✅ 文件正常，包含 {len(data)} 个对话")
            except Exception as e:
                print(f"      ❌ 文件损坏: {e}")
                print(f"      🗑️  删除损坏文件: {partial_file}")
                
                # 备份文件名
                backup_path = partial_path + ".backup"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                
                # 重命名为备份
                os.rename(partial_path, backup_path)
                cleaned_files.append(partial_path)
                
                print(f"      💾 已备份为: {partial_file}.backup")
    
    return cleaned_files

def check_resume_files():
    """检查resume文件状态"""
    print(f"\n📋 检查resume文件状态...")
    
    data_types = ["train", "valid", "test"]
    
    for data_type in data_types:
        chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
        resume_file = os.path.join(chunk_dir, f"resume_{data_type}.json")
        
        if os.path.exists(resume_file):
            try:
                with open(resume_file, 'r') as f:
                    resume_data = json.load(f)
                print(f"   📄 {data_type}: {resume_data}")
                
                # 检查是否有partial标记但partial文件不存在
                if resume_data.get('has_partial', False):
                    chunk_index = resume_data.get('chunk_index', 0)
                    partial_file = os.path.join(chunk_dir, f"partial_{chunk_index}.pt")
                    if not os.path.exists(partial_file):
                        print(f"      ⚠️  标记有partial但文件不存在，更新resume文件")
                        resume_data['has_partial'] = False
                        with open(resume_file, 'w') as f:
                            json.dump(resume_data, f)
                        print(f"      ✅ 已更新resume文件")
                        
            except Exception as e:
                print(f"   ❌ {data_type} resume文件损坏: {e}")
        else:
            print(f"   📄 {data_type}: 无resume文件")

def check_chunk_integrity():
    """检查chunk文件完整性"""
    print(f"\n🔍 检查chunk文件完整性...")
    
    data_types = ["train", "valid", "test"]
    
    for data_type in data_types:
        chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
        if not os.path.exists(chunk_dir):
            continue
            
        chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".pt")]
        if not chunk_files:
            print(f"   📦 {data_type}: 无chunk文件")
            continue
            
        # 获取chunk编号
        chunk_numbers = []
        for f in chunk_files:
            try:
                num = int(f.split('_')[1].split('.')[0])
                chunk_numbers.append(num)
            except:
                pass
        
        if chunk_numbers:
            chunk_numbers.sort()
            print(f"   📦 {data_type}: {len(chunk_files)} 个chunk文件")
            print(f"      编号范围: {min(chunk_numbers)} - {max(chunk_numbers)}")
            
            # 检查连续性
            expected = list(range(min(chunk_numbers), max(chunk_numbers) + 1))
            missing = set(expected) - set(chunk_numbers)
            if missing:
                print(f"      ⚠️  缺失编号: {sorted(missing)}")
            else:
                print(f"      ✅ 编号连续")
            
            # 检查最后几个文件的完整性
            last_chunks = sorted(chunk_numbers)[-3:]  # 检查最后3个
            for chunk_num in last_chunks:
                chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_num}.pt")
                try:
                    data = torch.load(chunk_file, weights_only=True)
                    print(f"      ✅ chunk_{chunk_num}.pt: {len(data)} 个对话")
                except Exception as e:
                    print(f"      ❌ chunk_{chunk_num}.pt 损坏: {e}")

def main():
    print("🧹 Partial文件清理工具")
    print("=" * 50)
    
    # 1. 清理损坏的partial文件
    cleaned_files = cleanup_partial_files()
    
    # 2. 检查resume文件
    check_resume_files()
    
    # 3. 检查chunk文件完整性
    check_chunk_integrity()
    
    # 4. 总结
    print(f"\n📊 清理总结:")
    if cleaned_files:
        print(f"   🗑️  清理了 {len(cleaned_files)} 个损坏的partial文件")
        for file in cleaned_files:
            print(f"      - {file}")
        print(f"   💾 原文件已备份为 .backup")
    else:
        print(f"   ✅ 未发现损坏的partial文件")
    
    print(f"\n💡 建议:")
    print(f"1. 现在可以安全地继续运行数据转换")
    print(f"2. 系统会从正确的位置继续处理")
    print(f"3. 如果问题持续，可能是磁盘空间或内存不足")
    
    print(f"\n🚀 继续处理命令:")
    print(f"python -m src.prepare_binary_data --num_workers=2")

if __name__ == "__main__":
    main()
