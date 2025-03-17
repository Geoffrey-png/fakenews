#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
设置HAMMER预训练模型目录结构
"""

import os
import sys
import shutil
import argparse

def create_directory_structure(base_dir):
    """创建预训练模型目录结构"""
    # 创建预训练模型根目录
    pretrained_dir = os.path.join(base_dir, 'pretrained')
    os.makedirs(pretrained_dir, exist_ok=True)
    
    # 创建预训练模型子目录
    bert_dir = os.path.join(pretrained_dir, 'bert')
    os.makedirs(bert_dir, exist_ok=True)
    
    # 创建DeiT预训练权重目录
    os.makedirs(os.path.join(pretrained_dir), exist_ok=True)
    
    # 创建本地tokenizer目录
    tokenizer_dir = os.path.join(base_dir, 'tokenizer')
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    print("已创建以下目录结构：")
    print(f"- {pretrained_dir}")
    print(f"  |- {bert_dir}")
    print(f"  |- deit_base_patch16_224.pth (需要手动添加)")
    print(f"- {tokenizer_dir}")
    
    # 创建README文件
    readme_path = os.path.join(pretrained_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# 预训练模型目录\n\n")
        f.write("请将以下文件放置在对应目录：\n\n")
        f.write("1. BERT预训练模型文件：\n")
        f.write("   - 路径：`bert/`\n")
        f.write("   - 所需文件：config.json, vocab.txt, pytorch_model.bin等\n\n")
        f.write("2. DeiT预训练权重：\n")
        f.write("   - 文件名：`deit_base_patch16_224.pth`\n\n")
        f.write("3. BERT Tokenizer文件：\n")
        f.write("   - 路径：`../tokenizer/`\n")
        f.write("   - 所需文件：vocab.txt, special_tokens_map.json, tokenizer_config.json等\n\n")
        f.write("## 获取预训练模型\n\n")
        f.write("您可以从以下位置下载预训练模型：\n\n")
        f.write("1. BERT模型：\n")
        f.write("   - 官方地址：https://huggingface.co/bert-base-uncased\n")
        f.write("   - 您需要下载并放置在bert目录下的文件：\n")
        f.write("     - config.json\n")
        f.write("     - pytorch_model.bin\n")
        f.write("     - vocab.txt\n\n")
        f.write("2. DeiT预训练权重：\n")
        f.write("   - 官方地址：https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth\n\n")
        f.write("## 离线使用说明\n\n")
        f.write("如需在离线环境使用，请确保在有网络连接的环境中下载上述文件，然后将它们复制到相应目录。\n")
    
    print(f"已创建README文件：{readme_path}")
    
    # 创建weights目录（如果不存在）
    weights_dir = os.path.join(base_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    print(f"已创建权重目录：{weights_dir}")
    
    # 创建weights目录下的README
    weights_readme_path = os.path.join(weights_dir, 'README.md')
    with open(weights_readme_path, 'w', encoding='utf-8') as f:
        f.write("# 模型权重目录\n\n")
        f.write("请将HAMMER模型权重文件放置在此目录：\n\n")
        f.write("- 文件名：`checkpoint_best.pth`\n")
    
    print(f"已创建权重目录README文件：{weights_readme_path}")
    
    return {
        'pretrained_dir': pretrained_dir,
        'bert_dir': bert_dir,
        'tokenizer_dir': tokenizer_dir,
        'weights_dir': weights_dir
    }

def main():
    parser = argparse.ArgumentParser(description='设置HAMMER预训练模型目录结构')
    args = parser.parse_args()
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建目录结构
    dirs = create_directory_structure(current_dir)
    
    print("\n设置完成！")
    print("\n使用说明：")
    print("1. 下载预训练模型文件并放置在相应目录")
    print("2. 确保HAMMER模型权重文件放置在weights目录")
    print("3. 参考pretrained/README.md获取详细说明")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 