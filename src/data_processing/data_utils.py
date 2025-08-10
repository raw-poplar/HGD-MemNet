#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理工具函数

包含各种数据处理相关的工具函数：
1. 数据完整性检查
2. 处理时间估算
3. 磁盘空间管理
4. 文件操作工具
"""

import os
import json
import torch
import glob
import shutil
import sys
from typing import Dict

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config


# 统一移除输出中的emoji，避免控制台/日志环境兼容问题
import re, builtins as _builtins
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF\U00002190-\U00002BFF]")
_print = _builtins.print

def print(*args, **kwargs):
    def _strip(s):
        return _EMOJI_RE.sub('', s) if isinstance(s, str) else s
    return _print(*[_strip(a) for a in args], **kwargs)

def check_data_integrity(data_type: str = None) -> Dict:
    """检查数据完整性"""
    print("🔍 检查数据完整性...")

    if data_type:
        data_types = [data_type]
    else:
        data_types = ["train", "valid", "test"]

    results = {}

    for dtype in data_types:
        print(f"\n📁 检查 {dtype}:")

        result = {
            "original_file": None,
            "chunk_files": [],
            "merged_file": None,
            "partial_files": [],
            "resume_file": None
        }

        # 检查原始文件
        original_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{dtype}.jsonl")
        if os.path.exists(original_file):
            size_mb = os.path.getsize(original_file) / (1024*1024)
            result["original_file"] = {
                "exists": True,
                "size_mb": size_mb,
                "path": original_file
            }
            print(f"   📄 原始文件: {size_mb:.1f} MB")
        else:
            result["original_file"] = {"exists": False}
            print(f"   📄 原始文件: 不存在")

        # 检查chunk目录
        chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, dtype)
        if os.path.exists(chunk_dir):
            # chunk文件
            chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".pt")]
            chunk_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

            for chunk_file in chunk_files:
                chunk_path = os.path.join(chunk_dir, chunk_file)
                size_mb = os.path.getsize(chunk_path) / (1024*1024)
                chunk_num = int(chunk_file.split('_')[1].split('.')[0])
                result["chunk_files"].append({
                    "name": chunk_file,
                    "number": chunk_num,
                    "size_mb": size_mb,
                    "path": chunk_path
                })

            print(f"   📦 chunk文件: {len(chunk_files)} 个")

            # partial文件
            partial_files = [f for f in os.listdir(chunk_dir) if f.startswith("partial_") and f.endswith(".pt")]
            for partial_file in partial_files:
                partial_path = os.path.join(chunk_dir, partial_file)
                size_mb = os.path.getsize(partial_path) / (1024*1024)
                result["partial_files"].append({
                    "name": partial_file,
                    "size_mb": size_mb,
                    "path": partial_path
                })

            if partial_files:
                print(f"   🔄 partial文件: {len(partial_files)} 个")

            # resume文件
            resume_file = os.path.join(chunk_dir, f"resume_{dtype}.json")
            if os.path.exists(resume_file):
                try:
                    with open(resume_file, 'r') as f:
                        resume_data = json.load(f)
                    result["resume_file"] = resume_data
                    print(f"   📋 resume文件: {resume_data}")
                except:
                    result["resume_file"] = {"error": "damaged"}
                    print(f"   📋 resume文件: 损坏")

        # 检查合并文件
        merged_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{dtype}.pt")
        if os.path.exists(merged_file):
            size_gb = os.path.getsize(merged_file) / (1024**3)
            try:
                data = torch.load(merged_file, weights_only=True)
                dialogue_count = len(data)
                del data
                result["merged_file"] = {
                    "exists": True,
                    "size_gb": size_gb,
                    "dialogue_count": dialogue_count,
                    "path": merged_file
                }
                print(f"   📄 合并文件: {size_gb:.2f} GB, {dialogue_count:,} 个对话")
            except:
                result["merged_file"] = {
                    "exists": True,
                    "size_gb": size_gb,
                    "error": "cannot_load"
                }
                print(f"   📄 合并文件: {size_gb:.2f} GB (无法加载)")
        else:
            result["merged_file"] = {"exists": False}
            print(f"   📄 合并文件: 不存在")

        results[dtype] = result

    return results

def estimate_processing_time(data_type: str = None) -> Dict:
    """估算处理时间"""
    print("⏱️  估算处理时间...")

    if data_type:
        data_types = [data_type]
    else:
        data_types = ["train", "valid", "test"]

    estimates = {}

    for dtype in data_types:
        print(f"\n📊 估算 {dtype}:")

        # 检查原始文件
        original_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{dtype}.jsonl")
        if not os.path.exists(original_file):
            print(f"   ❌ 原始文件不存在")
            continue

        # 文件大小
        file_size_mb = os.path.getsize(original_file) / (1024*1024)
        print(f"   📄 文件大小: {file_size_mb:.1f} MB")

        # 估算行数
        with open(original_file, 'r', encoding='utf-8') as f:
            sample_lines = []
            for i, line in enumerate(f):
                if i >= 100:
                    break
                sample_lines.append(len(line.encode('utf-8')))

        if sample_lines:
            avg_line_size = sum(sample_lines) / len(sample_lines)
            estimated_lines = int(file_size_mb * 1024 * 1024 / avg_line_size)
            print(f"   📊 估算行数: {estimated_lines:,}")

            # 处理时间估算
            speeds = {
                "单线程": 100,
                "2线程": 180,
                "4线程": 300,
                "8线程": 400
            }

            time_estimates = {}
            for mode, speed in speeds.items():
                time_seconds = estimated_lines / speed
                time_minutes = time_seconds / 60
                time_hours = time_minutes / 60

                if time_hours > 1:
                    time_str = f"{time_hours:.1f} 小时"
                elif time_minutes > 1:
                    time_str = f"{time_minutes:.1f} 分钟"
                else:
                    time_str = f"{time_seconds:.1f} 秒"

                time_estimates[mode] = {
                    "seconds": time_seconds,
                    "display": time_str
                }
                print(f"   {mode}: {time_str}")

            estimates[dtype] = {
                "file_size_mb": file_size_mb,
                "estimated_lines": estimated_lines,
                "time_estimates": time_estimates
            }

    return estimates

def get_disk_usage() -> Dict:
    """获取磁盘使用情况"""
    try:
        total, used, free = shutil.disk_usage(config.LCCC_PROCESSED_PATH)
        return {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "free_gb": free / (1024**3),
            "usage_percent": (used / total) * 100
        }
    except Exception as e:
        return {"error": str(e)}

def estimate_space_requirements() -> Dict:
    """估算空间需求"""
    print("💾 估算空间需求...")

    space_info = get_disk_usage()
    if "error" in space_info:
        print(f"❌ 无法获取磁盘信息: {space_info['error']}")
        return space_info

    print(f"💿 当前磁盘状态:")
    print(f"   总空间: {space_info['total_gb']:.1f} GB")
    print(f"   已使用: {space_info['used_gb']:.1f} GB")
    print(f"   可用空间: {space_info['free_gb']:.1f} GB")
    print(f"   使用率: {space_info['usage_percent']:.1f}%")

    # 估算各类文件的空间占用
    file_sizes = {}

    # chunk文件
    chunk_size = 0
    for data_type in ["train", "valid", "test"]:
        chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
        if os.path.exists(chunk_dir):
            for file in os.listdir(chunk_dir):
                if file.startswith("chunk_") and file.endswith(".pt"):
                    file_path = os.path.join(chunk_dir, file)
                    chunk_size += os.path.getsize(file_path)

    file_sizes["chunks_gb"] = chunk_size / (1024**3)

    # 原始文件
    original_size = 0
    for data_type in ["train", "valid", "test"]:
        original_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{data_type}.jsonl")
        if os.path.exists(original_file):
            original_size += os.path.getsize(original_file)

    file_sizes["original_gb"] = original_size / (1024**3)

    # 合并文件
    merged_size = 0
    for data_type in ["train", "valid", "test"]:
        merged_file = os.path.join(config.LCCC_PROCESSED_PATH, f"{data_type}.pt")
        if os.path.exists(merged_file):
            merged_size += os.path.getsize(merged_file)

    file_sizes["merged_gb"] = merged_size / (1024**3)

    print(f"\n📊 文件空间占用:")
    print(f"   原始文件: {file_sizes['original_gb']:.1f} GB")
    print(f"   chunk文件: {file_sizes['chunks_gb']:.1f} GB")
    print(f"   合并文件: {file_sizes['merged_gb']:.1f} GB")

    # 估算合并所需空间
    estimated_merged_size = file_sizes["chunks_gb"] * 0.8  # 估算合并后大小
    required_space = estimated_merged_size * 1.2  # 20%缓冲

    print(f"\n💡 空间建议:")
    print(f"   估算合并后大小: {estimated_merged_size:.1f} GB")
    print(f"   建议可用空间: {required_space:.1f} GB")

    if space_info["free_gb"] >= required_space:
        print(f"   ✅ 空间充足")
    else:
        print(f"   ⚠️  空间可能不足")
        print(f"   需要释放: {required_space - space_info['free_gb']:.1f} GB")

    space_info.update(file_sizes)
    space_info["estimated_merged_gb"] = estimated_merged_size
    space_info["required_space_gb"] = required_space

    return space_info

def cleanup_partial_files(data_type: str = None, confirm: bool = True) -> int:
    """清理partial文件"""
    if data_type:
        data_types = [data_type]
    else:
        data_types = ["train", "valid", "test"]

    total_cleaned = 0

    for dtype in data_types:
        chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, dtype)
        if not os.path.exists(chunk_dir):
            continue

        partial_files = [f for f in os.listdir(chunk_dir) if f.startswith("partial_") and f.endswith(".pt")]

        if not partial_files:
            continue

        print(f"📁 {dtype}: 找到 {len(partial_files)} 个partial文件")

        if confirm:
            response = input(f"是否删除 {dtype} 的partial文件? (y/n): ")
            if response.lower() != 'y':
                continue

        # 删除partial文件
        for partial_file in partial_files:
            partial_path = os.path.join(chunk_dir, partial_file)
            try:
                os.remove(partial_path)
                total_cleaned += 1
            except Exception as e:
                print(f"❌ 删除 {partial_file} 失败: {e}")

    if total_cleaned > 0:
        print(f"✅ 总共清理了 {total_cleaned} 个partial文件")

    return total_cleaned

def count_dialogues_in_chunks(data_type: str) -> int:
    """统计chunk文件中的对话数量"""
    chunk_dir = os.path.join(config.LCCC_PROCESSED_PATH, data_type)
    if not os.path.exists(chunk_dir):
        return 0

    chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".pt")]

    if not chunk_files:
        return 0

    # 采样统计
    sample_size = min(5, len(chunk_files))
    sample_files = chunk_files[:sample_size]

    total_in_samples = 0
    for chunk_file in sample_files:
        chunk_path = os.path.join(chunk_dir, chunk_file)
        try:
            data = torch.load(chunk_path, weights_only=True)
            total_in_samples += len(data)
            del data
        except:
            pass

    if sample_size > 0:
        avg_per_chunk = total_in_samples / sample_size
        estimated_total = int(avg_per_chunk * len(chunk_files))
        return estimated_total

    return 0

def main():
    """主函数 - 数据工具的命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="数据处理工具")
    parser.add_argument("--check", action="store_true", help="检查数据完整性")
    parser.add_argument("--estimate", action="store_true", help="估算处理时间")
    parser.add_argument("--space", action="store_true", help="检查磁盘空间")
    parser.add_argument("--cleanup", action="store_true", help="清理partial文件")
    parser.add_argument("--dataset", choices=["train", "valid", "test"], help="指定数据集")

    args = parser.parse_args()

    if not any([args.check, args.estimate, args.space, args.cleanup]):
        # 默认执行所有检查
        args.check = args.estimate = args.space = True

    print("🔧 数据处理工具")
    print("=" * 40)

    if args.check:
        check_data_integrity(args.dataset)

    if args.estimate:
        estimate_processing_time(args.dataset)

    if args.space:
        estimate_space_requirements()

    if args.cleanup:
        cleanup_partial_files(args.dataset)

if __name__ == "__main__":
    main()
