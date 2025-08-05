#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理主入口脚本

提供统一的命令行接口来执行所有数据处理任务。
"""

import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="HGD-MemNet 数据处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整的数据处理流程
  python -m src.data_processing.main --full-pipeline --workers=4
  
  # 只进行数据转换
  python -m src.data_processing.main --convert --workers=4
  
  # 只进行数据合并
  python -m src.data_processing.main --merge --method=optimized
  
  # 检查数据状态
  python -m src.data_processing.main --check
  
  # 调试和测试
  python -m src.data_processing.main --debug
        """
    )
    
    # 主要操作
    parser.add_argument("--full-pipeline", action="store_true",
                       help="执行完整的数据处理流程")
    parser.add_argument("--convert", action="store_true",
                       help="执行数据转换 (JSONL -> chunks)")
    parser.add_argument("--merge", action="store_true",
                       help="执行数据合并 (chunks -> final files)")
    parser.add_argument("--check", action="store_true",
                       help="检查数据状态和完整性")
    parser.add_argument("--debug", action="store_true",
                       help="运行调试和测试工具")
    parser.add_argument("--cleanup", action="store_true",
                       help="清理临时文件")
    
    # 转换参数
    parser.add_argument("--workers", type=int, default=2,
                       help="工作进程数 (默认: 2)")
    
    # 合并参数
    parser.add_argument("--merge-method", choices=["simple", "optimized", "large"],
                       default="optimized", help="合并方法 (默认: optimized)")
    parser.add_argument("--dataset", choices=["train", "valid", "test", "all"],
                       default="all", help="要处理的数据集 (默认: all)")
    
    # 其他选项
    parser.add_argument("--verify", action="store_true",
                       help="验证处理结果")
    parser.add_argument("--force", action="store_true",
                       help="强制覆盖现有文件")
    parser.add_argument("--verbose", action="store_true",
                       help="显示详细输出")
    
    args = parser.parse_args()
    
    # 如果没有指定任何操作，显示帮助
    if not any([args.full_pipeline, args.convert, args.merge, args.check, args.debug, args.cleanup]):
        parser.print_help()
        return
    
    print("🚀 HGD-MemNet 数据处理工具")
    print("=" * 50)
    
    try:
        if args.full_pipeline:
            run_full_pipeline(args)
        else:
            if args.convert:
                run_convert(args)
            
            if args.merge:
                run_merge(args)
            
            if args.check:
                run_check(args)
            
            if args.debug:
                run_debug(args)
            
            if args.cleanup:
                run_cleanup(args)
        
        print("\n✅ 所有操作完成!")
        
    except KeyboardInterrupt:
        print("\n⏸️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 操作失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

def run_full_pipeline(args):
    """运行完整的数据处理流程"""
    print("🔄 执行完整数据处理流程...")
    
    # 1. 检查当前状态
    print("\n📊 步骤1: 检查当前状态")
    run_check(args)
    
    # 2. 数据转换
    print("\n🔄 步骤2: 数据转换")
    run_convert(args)
    
    # 3. 数据合并
    print("\n📦 步骤3: 数据合并")
    run_merge(args)
    
    # 4. 验证结果
    if args.verify:
        print("\n🔍 步骤4: 验证结果")
        run_verify(args)
    
    # 5. 清理临时文件
    if args.cleanup:
        print("\n🧹 步骤5: 清理临时文件")
        run_cleanup(args)

def run_convert(args):
    """运行数据转换"""
    print(f"🔄 数据转换 (workers={args.workers})...")
    
    from .prepare_binary_data import main as convert_main
    
    # 临时修改sys.argv来传递参数
    original_argv = sys.argv.copy()
    sys.argv = ["prepare_binary_data.py", f"--num_workers={args.workers}"]
    
    try:
        convert_main()
    finally:
        sys.argv = original_argv

def run_merge(args):
    """运行数据合并"""
    print(f"📦 数据合并 (method={args.merge_method}, dataset={args.dataset})...")
    
    from .merge_tools import main as merge_main
    
    # 临时修改sys.argv来传递参数
    original_argv = sys.argv.copy()
    merge_args = [
        "merge_tools.py",
        f"--method={args.merge_method}",
        f"--dataset={args.dataset}",
        f"--workers={args.workers}"
    ]
    
    if args.verify:
        merge_args.append("--verify")
    if args.cleanup:
        merge_args.append("--cleanup")
    
    sys.argv = merge_args
    
    try:
        merge_main()
    finally:
        sys.argv = original_argv

def run_check(args):
    """运行数据检查"""
    print("🔍 数据状态检查...")
    
    from .data_utils import main as utils_main
    
    # 临时修改sys.argv来传递参数
    original_argv = sys.argv.copy()
    check_args = ["data_utils.py", "--check", "--estimate", "--space"]
    
    if args.dataset != "all":
        check_args.extend(["--dataset", args.dataset])
    
    sys.argv = check_args
    
    try:
        utils_main()
    finally:
        sys.argv = original_argv

def run_debug(args):
    """运行调试工具"""
    print("🔧 调试工具...")
    
    from .debug_tools import main as debug_main
    
    # 临时修改sys.argv来传递参数
    original_argv = sys.argv.copy()
    debug_args = ["debug_tools.py"]
    
    if args.dataset != "all":
        debug_args.extend(["--dataset", args.dataset])
    
    sys.argv = debug_args
    
    try:
        debug_main()
    finally:
        sys.argv = original_argv

def run_cleanup(args):
    """运行清理操作"""
    print("🧹 清理临时文件...")
    
    from .data_utils import cleanup_partial_files
    
    # 清理partial文件
    dataset = None if args.dataset == "all" else args.dataset
    cleaned_count = cleanup_partial_files(dataset, confirm=not args.force)
    
    if cleaned_count > 0:
        print(f"✅ 清理了 {cleaned_count} 个文件")
    else:
        print("✅ 无需清理")

def run_verify(args):
    """运行验证操作"""
    print("🔍 验证处理结果...")
    
    from .merge_tools import verify_merged_files
    verify_merged_files()

def show_status():
    """显示当前处理状态"""
    print("📊 当前处理状态:")
    
    try:
        from .debug_tools import analyze_processing_status
        analyze_processing_status()
    except Exception as e:
        print(f"❌ 无法获取状态: {e}")

def interactive_mode():
    """交互模式"""
    print("🎮 交互模式")
    print("=" * 30)
    
    while True:
        print("\n可用操作:")
        print("1. 检查数据状态")
        print("2. 数据转换")
        print("3. 数据合并")
        print("4. 调试工具")
        print("5. 清理文件")
        print("6. 完整流程")
        print("0. 退出")
        
        try:
            choice = input("\n请选择操作 (0-6): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                show_status()
            elif choice == "2":
                workers = input("工作进程数 (默认2): ").strip() or "2"
                args = type('Args', (), {
                    'workers': int(workers),
                    'dataset': 'all',
                    'verbose': True
                })()
                run_convert(args)
            elif choice == "3":
                method = input("合并方法 (simple/optimized/large, 默认optimized): ").strip() or "optimized"
                dataset = input("数据集 (train/valid/test/all, 默认all): ").strip() or "all"
                args = type('Args', (), {
                    'merge_method': method,
                    'dataset': dataset,
                    'workers': 2,
                    'verify': True,
                    'cleanup': False
                })()
                run_merge(args)
            elif choice == "4":
                args = type('Args', (), {'dataset': 'all'})()
                run_debug(args)
            elif choice == "5":
                args = type('Args', (), {'dataset': 'all', 'force': False})()
                run_cleanup(args)
            elif choice == "6":
                workers = input("工作进程数 (默认2): ").strip() or "2"
                args = type('Args', (), {
                    'workers': int(workers),
                    'merge_method': 'optimized',
                    'dataset': 'all',
                    'verify': True,
                    'cleanup': True,
                    'verbose': True
                })()
                run_full_pipeline(args)
            else:
                print("❌ 无效选择")
                
        except KeyboardInterrupt:
            print("\n⏸️  操作中断")
        except Exception as e:
            print(f"❌ 操作失败: {e}")

if __name__ == "__main__":
    # 如果没有命令行参数，启动交互模式
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        main()
