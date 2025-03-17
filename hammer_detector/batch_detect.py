#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HAMMER 批量检测脚本
"""

import os
import sys
import argparse
import json
import time
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description='使用HAMMER模型批量检测多张图像')
    parser.add_argument('--image_dir', required=True, type=str, help='包含图像的目录路径')
    parser.add_argument('--output', type=str, default='detection_results.json', help='输出结果的JSON文件路径')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化结果')
    parser.add_argument('--image_ext', type=str, default='.jpg,.jpeg,.png', help='要处理的图像扩展名，用逗号分隔')
    parser.add_argument('--english', action='store_true', help='使用英文标签输出结果')
    args = parser.parse_args()

    # 打印当前工作目录
    print(f"当前工作目录: {os.getcwd()}")
    
    # 检查图像目录
    print(f"输入图像目录: {args.image_dir}")
    image_dir = args.image_dir
    
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(image_dir):
        image_dir = os.path.abspath(image_dir)
    
    print(f"图像目录完整路径: {image_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(image_dir) or not os.path.isdir(image_dir):
        print(f"错误: 找不到图像目录: {image_dir}")
        return 1
    
    # 获取所有图像文件
    valid_extensions = tuple(args.image_ext.lower().split(','))
    image_files = []
    
    for root, _, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in valid_extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"错误: 在目录 {image_dir} 中未找到图像文件")
        return 1
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 配置文件和模型权重路径
    config_path = os.path.join(current_dir, "config.yaml")
    model_path = os.path.join(current_dir, "weights", "checkpoint_best.pth")
    
    # 初始化检测器
    try:
        print("初始化检测器...")
        detector = HAMMERDetector(
            config_path=config_path,
            model_path=model_path,
            use_english=args.english
        )
        print("检测器初始化成功")
    except Exception as e:
        print(f"初始化检测器时出错: {str(e)}")
        return 1
    
    # 创建输出目录（如果启用可视化）
    if args.visualize:
        output_dir = os.path.join(os.path.dirname(args.output), "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        print(f"可视化结果将保存到: {output_dir}")
    else:
        output_dir = None
    
    # 执行批量检测
    try:
        print("\n开始批量检测...")
        start_time = time.time()
        
        results = {}
        fake_count = 0
        real_count = 0
        
        for image_file in tqdm(image_files, desc="处理图像"):
            try:
                image_name = os.path.basename(image_file)
                
                # 执行检测
                result = detector.detect(image_file)
                
                # 保存可视化结果
                if args.visualize:
                    output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_result.jpg")
                    detector.save_visualization(image_file, result, output_path)
                
                # 保存结果
                results[image_name] = result
                
                # 计数统计
                if result["is_fake"]:
                    fake_count += 1
                else:
                    real_count += 1
                
            except Exception as e:
                print(f"处理图像 {image_file} 时出错: {e}")
                results[os.path.basename(image_file)] = {"error": str(e)}
        
        # 保存所有结果到JSON文件
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 打印统计信息
        end_time = time.time()
        total_time = end_time - start_time
        print("\n检测完成!")
        print(f"总图像数: {len(image_files)}")
        print(f"检测为伪造的图像数: {fake_count}")
        print(f"检测为真实的图像数: {real_count}")
        print(f"总处理时间: {total_time:.2f} 秒, 平均每张 {total_time/len(image_files):.2f} 秒")
        print(f"结果已保存至: {args.output}")
        
        return 0
    
    except Exception as e:
        print(f"批量检测过程中出错: {e}")
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