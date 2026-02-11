#!/usr/bin/env python3
"""
测试seed-range参数的不同使用方式
"""
import sys
sys.path.append('.')

from exper import parser

def test_seed_range():
    """测试不同的seed-range参数使用方式"""
    
    print("=== 测试seed-range参数 ===\n")
    
    # 测试1: 不使用seed-range参数
    print("1. 测试不使用seed-range参数:")
    try:
        test_args = ["--label", "test"]
        args = parser.get_parser().parse_args(test_args)
        args = vars(args)
        print(f"   seed_range: {args['seed_range']}")
        print(f"   seed: {args['seed']}")
        print("   ✓ 成功")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    print()
    
    # 测试2: 正确使用seed-range参数
    print("2. 测试正确使用seed-range参数 (--seed-range 1 3):")
    try:
        test_args = ["--label", "test", "--seed-range", "1", "3"]
        args = parser.get_parser().parse_args(test_args)
        args = vars(args)
        print(f"   seed_range: {args['seed_range']}")
        if args["seed_range"] is not None:
            seed_list = list(range(args["seed_range"][0], args["seed_range"][1] + 1))
            print(f"   生成的seed列表: {seed_list}")
        print("   ✓ 成功")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    print()
    
    # 测试3: 错误使用seed-range参数（只有一个参数）
    print("3. 测试错误使用seed-range参数 (--seed-range 1):")
    try:
        test_args = ["--label", "test", "--seed-range", "1"]
        args = parser.get_parser().parse_args(test_args)
        args = vars(args)
        print(f"   seed_range: {args['seed_range']}")
        print("   ✓ 成功")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    print()
    
    # 测试4: 错误使用seed-range参数（三个参数）
    print("4. 测试错误使用seed-range参数 (--seed-range 1 2 3):")
    try:
        test_args = ["--label", "test", "--seed-range", "1", "2", "3"]
        args = parser.get_parser().parse_args(test_args)
        args = vars(args)
        print(f"   seed_range: {args['seed_range']}")
        print("   ✓ 成功")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    print()
    
    # 测试5: 同时使用seed和seed-range
    print("5. 测试同时使用seed和seed-range:")
    try:
        test_args = ["--label", "test", "--seed", "10", "--seed-range", "1", "3"]
        args = parser.get_parser().parse_args(test_args)
        args = vars(args)
        print(f"   原始seed: {args['seed']}")
        print(f"   seed_range: {args['seed_range']}")
        if args["seed_range"] is not None:
            args["seed"] = list(range(args["seed_range"][0], args["seed_range"][1] + 1))
            print(f"   最终seed: {args['seed']}")
        print("   ✓ 成功")
    except Exception as e:
        print(f"   ✗ 失败: {e}")

if __name__ == "__main__":
    test_seed_range() 