#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试运行脚本
提供便捷的测试运行接口
"""

import sys
import os
import argparse
import pytest
import subprocess
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_unit_tests():
    """运行单元测试"""
    print("=" * 60)
    print("运行单元测试...")
    print("=" * 60)
    
    test_files = [
        "test_model_components.py",
        "test_neuron_architectures.py",
        "test_data_processing.py"
    ]
    
    for test_file in test_files:
        print(f"\n运行 {test_file}...")
        result = pytest.main([
            os.path.join(os.path.dirname(__file__), test_file),
            "-v",
            "--tb=short"
        ])
        if result != 0:
            print(f"{test_file} 测试失败")
            return False
        else:
            print(f"{test_file} 测试通过")
    
    return True

def run_integration_tests():
    """运行集成测试"""
    print("=" * 60)
    print("运行集成测试...")
    print("=" * 60)
    
    test_files = [
        "test_model_integration.py",
        "test_training.py"
    ]
    
    for test_file in test_files:
        print(f"\n运行 {test_file}...")
        result = pytest.main([
            os.path.join(os.path.dirname(__file__), test_file),
            "-v",
            "--tb=short"
        ])
        if result != 0:
            print(f"{test_file} 测试失败")
            return False
        else:
            print(f"{test_file} 测试通过")
    
    return True

def run_performance_tests():
    """运行性能测试"""
    print("=" * 60)
    print("运行性能测试...")
    print("=" * 60)

    test_files = [
        "test_performance.py",
        "test_quick_performance.py",
        "test_multiprocessing_performance.py"
    ]

    for test_file in test_files:
        print(f"\n运行 {test_file}...")
        result = pytest.main([
            os.path.join(os.path.dirname(__file__), test_file),
            "-v",
            "--tb=short",
            "-s"  # 显示print输出
        ])
        if result != 0:
            print(f"{test_file} 测试失败")
            return False
        else:
            print(f"{test_file} 测试通过")

    return True

def run_utility_tests():
    """运行实用测试（数据兼容性、设置检查等）"""
    print("=" * 60)
    print("运行实用测试...")
    print("=" * 60)

    utility_scripts = [
        "test_current_setup.py",
        "test_data_compatibility.py",
        "test_existing_chunks.py",
        "test_basic_processing.py",
        "test_find_chunks.py"
    ]

    for script in utility_scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)
        if os.path.exists(script_path):
            print(f"\n运行 {script}...")
            try:
                result = subprocess.run([
                    sys.executable, script_path
                ], capture_output=True, text=True, timeout=120)

                if result.returncode == 0:
                    print(f"{script} 执行成功")
                    if result.stdout:
                        print("输出:")
                        print(result.stdout[-500:])  # 显示最后500字符
                else:
                    print(f"{script} 执行失败")
                    if result.stderr:
                        print("错误:")
                        print(result.stderr[-500:])
                    return False
            except subprocess.TimeoutExpired:
                print(f"{script} 执行超时")
                return False
            except Exception as e:
                print(f"{script} 执行出错: {e}")
                return False
        else:
            print(f"{script} 不存在，跳过")

    return True

def run_all_tests():
    """运行所有测试"""
    print("开始运行完整测试套件...")
    
    success = True
    
    # 运行单元测试
    if not run_unit_tests():
        success = False
    
    # 运行集成测试
    if not run_integration_tests():
        success = False
    
    # 运行性能测试
    if not run_performance_tests():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("所有测试通过！")
    else:
        print("部分测试失败，请检查上述输出")
    print("=" * 60)
    
    return success

def run_coverage_report():
    """运行测试覆盖率报告"""
    print("=" * 60)
    print("生成测试覆盖率报告...")
    print("=" * 60)
    
    try:
        # 安装coverage如果没有
        subprocess.run([sys.executable, "-m", "pip", "install", "coverage"], 
                      capture_output=True, check=False)
        
        # 运行覆盖率测试
        test_dir = os.path.dirname(__file__)
        src_dir = os.path.dirname(test_dir)
        
        cmd = [
            sys.executable, "-m", "coverage", "run",
            "--source", src_dir,
            "-m", "pytest",
            test_dir,
            "-v"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("覆盖率测试完成")
            
            # 生成报告
            subprocess.run([sys.executable, "-m", "coverage", "report"])
            
            # 生成HTML报告
            html_result = subprocess.run([
                sys.executable, "-m", "coverage", "html",
                "--directory", os.path.join(project_root, "htmlcov")
            ], capture_output=True)
            
            if html_result.returncode == 0:
                print(f"HTML覆盖率报告已生成: {project_root}/htmlcov/index.html")
        else:
            print("覆盖率测试失败")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"覆盖率测试出错: {e}")
        return False
    
    return True

def check_dependencies():
    """检查测试依赖"""
    print("检查测试依赖...")
    
    required_packages = [
        "pytest",
        "torch",
        "numpy",
        "psutil"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package}")
        except ImportError:
            print(f"{package} (缺失)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n请安装缺失的包: pip install {' '.join(missing_packages)}")
        return False
    
    print("所有依赖已满足")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HGD-MemNet 测试运行器")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "performance", "utility", "all", "coverage"],
        default="all",
        help="要运行的测试类型"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="检查测试依赖"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    # 检查依赖
    if args.check_deps:
        if not check_dependencies():
            sys.exit(1)
        return
    
    # 设置详细输出
    if args.verbose:
        os.environ["PYTEST_VERBOSE"] = "1"
    
    # 运行指定类型的测试
    success = True
    
    if args.type == "unit":
        success = run_unit_tests()
    elif args.type == "integration":
        success = run_integration_tests()
    elif args.type == "performance":
        success = run_performance_tests()
    elif args.type == "utility":
        success = run_utility_tests()
    elif args.type == "coverage":
        success = run_coverage_report()
    elif args.type == "all":
        success = run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
