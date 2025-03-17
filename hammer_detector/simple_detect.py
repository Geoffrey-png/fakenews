#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单的HAMMER检测脚本
"""

import os
import sys
import argparse
from pathlib import Path
import json

# 确保当前目录在系统路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from models.detector import HAMMERDetector
except ImportError as e:
    print(f"导入HAMMERDetector失败: {str(e)}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='使用HAMMER模型检测图像篡改')
    parser.add_argument('--image', required=True, help='输入图像的路径')
    parser.add_argument('--text', default=None, help='可选的与图像相关的文本')
    parser.add_argument('--output', default=None, help='输出结果JSON文件路径')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    parser.add_argument('--english', action='store_true', help='使用英文输出')
    args = parser.parse_args()

    # 打印当前工作目录
    print(f"当前工作目录: {os.getcwd()}")
    
    # 检查图像路径
    print(f"输入图像: {args.image}")
    image_path = args.image
    
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)
    
    print(f"图像完整路径: {image_path}")
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"错误: 找不到图像文件: {image_path}")
        return 1
    
    # 设置输出路径
    output_path = args.output
    if not output_path:
        basename = os.path.basename(image_path)
        filename, _ = os.path.splitext(basename)
        output_path = os.path.join(os.getcwd(), f"{filename}_result.json")
    
    # 初始化检测器
    try:
        print("初始化检测器...")
        detector = HAMMERDetector(
            config_path=None,
            model_path=None,
            use_english=args.english
        )
        print("检测器初始化成功")
    except Exception as e:
        print(f"初始化检测器时出错: {str(e)}")
        return 1
    
    # 执行检测
    try:
        print("\n执行检测...")
        result = detector.detect(
            image_path=image_path,
            text=args.text,
            visualize=args.visualize
        )
        
        # 输出检测结果
        print("\n检测结果:")
        print(f"篡改概率: {result['forgery_score']:.4f}")
        
        if result['forgery_score'] > 0.5:
            print("结论: 图像可能被篡改")
            print("操作类型:", result['manipulation_type'])
            
            if 'bboxes' in result and result['bboxes']:
                print("篡改区域位置:")
                for i, box in enumerate(result['bboxes']):
                    x1, y1, x2, y2 = box
                    print(f"  区域 {i+1}: 左上角坐标({x1:.1f}, {y1:.1f}), 右下角坐标({x2:.1f}, {y2:.1f})")
        else:
            print("结论: 图像可能是原始的/未被篡改")
        
        # 保存到JSON文件
        detector.save_results(result, output_path)
        print(f"\n结果已保存至: {output_path}")
        
        return 0
    
    except Exception as e:
        print(f"检测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)