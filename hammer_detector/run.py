#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HAMMER 伪造检测启动脚本
提供命令行界面，便于用户快速使用HAMMER检测功能
"""

import os
import sys
import argparse
from detect_image import detect_fake, main as detect_main


def setup_path():
    """设置环境路径"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 将当前目录添加到系统路径
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        
    # 创建权重目录（如果不存在）
    weights_dir = os.path.join(current_dir, 'weights')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        print(f"创建权重目录: {weights_dir}")
        print("请将模型权重文件(checkpoint_best.pth)放置于此目录")


def main():
    """主函数"""
    # 设置路径
    setup_path()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="HAMMER伪造检测工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # detect命令
    detect_parser = subparsers.add_parser("detect", help="检测伪造内容")
    detect_parser.add_argument('--image', required=True, help='输入图像路径')
    detect_parser.add_argument('--text', default='', help='相关文本')
    detect_parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    detect_parser.add_argument('--checkpoint', default='weights/checkpoint_best.pth', help='模型权重路径')
    detect_parser.add_argument('--visualize', action='store_true', help='可视化结果')
    detect_parser.add_argument('--output', default=None, help='输出图像路径')
    
    # demo命令
    demo_parser = subparsers.add_parser("demo", help="运行演示")
    demo_parser.add_argument('--image', default='../images/1.jpg', help='输入图像路径')
    demo_parser.add_argument('--text', default='这是一张图片，可能包含伪造内容。', help='相关文本')
    
    # 解析参数
    args = parser.parse_args()
    
    # 根据子命令执行相应功能
    if args.command == "detect":
        # 执行检测
        detect_main()
    elif args.command == "demo":
        # 运行演示
        from example import main as example_main
        example_main()
    else:
        # 默认显示帮助信息
        parser.print_help()


if __name__ == "__main__":
    main() 