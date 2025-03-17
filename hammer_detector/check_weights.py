#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查HAMMER所需的所有预训练模型和权重文件
"""

import os
import sys
import argparse
import json

def check_file(file_path, required=True):
    """检查文件是否存在"""
    exists = os.path.exists(file_path)
    return {
        'path': file_path,
        'exists': exists,
        'required': required,
        'status': '✅' if exists else ('❌' if required else '⚠️')
    }

def check_weights():
    """检查所有必要的权重和预训练文件"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义所有需要检查的文件
    files_to_check = [
        # HAMMER模型权重
        {
            'path': os.path.join(current_dir, 'weights', 'checkpoint_best.pth'),
            'description': 'HAMMER模型权重文件',
            'required': True
        },
        # DeiT预训练权重
        {
            'path': os.path.join(current_dir, 'pretrained', 'deit_base_patch16_224.pth'),
            'description': 'DeiT预训练权重文件',
            'required': True
        },
        # BERT模型文件
        {
            'path': os.path.join(current_dir, 'pretrained', 'bert', 'config.json'),
            'description': 'BERT配置文件',
            'required': True
        },
        {
            'path': os.path.join(current_dir, 'pretrained', 'bert', 'pytorch_model.bin'),
            'description': 'BERT模型权重文件',
            'required': True
        },
        {
            'path': os.path.join(current_dir, 'pretrained', 'bert', 'vocab.txt'),
            'description': 'BERT词汇表文件',
            'required': True
        },
        # Tokenizer文件
        {
            'path': os.path.join(current_dir, 'tokenizer', 'vocab.txt'),
            'description': 'BERT分词器词汇表文件',
            'required': True
        },
        {
            'path': os.path.join(current_dir, 'tokenizer', 'special_tokens_map.json'),
            'description': 'BERT分词器特殊标记映射文件',
            'required': False
        },
        {
            'path': os.path.join(current_dir, 'tokenizer', 'tokenizer_config.json'),
            'description': 'BERT分词器配置文件',
            'required': False
        }
    ]
    
    # 检查每个文件
    results = []
    for file_info in files_to_check:
        result = check_file(file_info['path'], file_info['required'])
        result['description'] = file_info['description']
        results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='检查HAMMER所需的所有预训练模型和权重文件')
    parser.add_argument('--json', action='store_true', help='以JSON格式输出结果')
    args = parser.parse_args()
    
    # 检查文件
    results = check_weights()
    
    if args.json:
        # 输出JSON格式
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        # 输出表格形式
        print("\n=== HAMMER预训练模型和权重文件检查 ===\n")
        print(f"{'状态':<4} {'文件描述':<30} {'文件路径':<70}")
        print("-" * 104)
        
        missing_required = []
        missing_optional = []
        
        for result in results:
            print(f"{result['status']:<4} {result['description']:<30} {result['path']:<70}")
            
            if not result['exists'] and result['required']:
                missing_required.append(result)
            elif not result['exists'] and not result['required']:
                missing_optional.append(result)
        
        print("\n=== 检查结果摘要 ===\n")
        print(f"总计检查: {len(results)} 个文件")
        print(f"存在: {sum(1 for r in results if r['exists'])} 个文件")
        print(f"缺失(必需): {len(missing_required)} 个文件")
        print(f"缺失(可选): {len(missing_optional)} 个文件")
        
        if missing_required:
            print("\n⚠️ 警告: 以下必需文件缺失:")
            for result in missing_required:
                print(f"  - {result['description']}: {result['path']}")
            print("\n请参考README.md中的离线使用指南下载这些文件。")
            return 1
        else:
            print("\n✅ 所有必需文件都已存在，HAMMER可以离线使用。")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 