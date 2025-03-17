#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HAMMER 伪造检测主模块
提供用于检测图像和文本中伪造内容的功能接口
"""

import os
import sys
import json
import yaml
import torch
import numpy as np

# 修复Windows中文编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from transformers import BertTokenizer

# 修改导入路径
from hammer_detector.models.HAMMER import HAMMER
from hammer_detector.models.box_ops import box_cxcywh_to_xyxy
from hammer_detector.tools import preprocess_image, preprocess_text


# 篡改类型标签映射
MANIPULATION_TYPES = {
    0: "未篡改",
    1: "换脸",
    2: "面部属性修改",
    3: "文本交换",
    4: "文本属性修改", 
    5: "物体添加",
    6: "物体删除",
    7: "物体修改"
}


def load_config(config_path):
    """
    加载配置文件
    
    参数:
        config_path: 配置文件路径
        
    返回:
        config: 配置字典
    """
    # 明确指定使用UTF-8编码打开文件，避免Windows系统默认使用GBK编码
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config, checkpoint_path=None):
    """
    构建HAMMER模型
    
    参数:
        config: 配置字典
        checkpoint_path: 模型权重路径
        
    返回:
        model: HAMMER模型
    """
    # 实例化模型
    model = HAMMER(config)
    
    # 加载预训练权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"加载模型检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理可能的不匹配
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # 加载状态字典
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"加载结果: {msg}")
    else:
        print("警告: 未加载模型权重!")
        
    return model


def visualize_result(image_path, fake_region, manipulation_type, confidence, output_path=None):
    """可视化检测结果
    
    Args:
        image_path: 图像路径
        fake_region: 伪造区域，格式为 (cx, cy, w, h)，归一化坐标 [0,1]
        manipulation_type: 伪造类型
        confidence: 置信度 (百分比)
        output_path: 输出图像路径
        
    Returns:
        保存的可视化结果图像路径
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    
    # 计算边界框坐标
    cx, cy, w, h = fake_region
    cx, cy = cx * width, cy * height
    w, h = w * width, h * height
    
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2
    
    # 绘制边界框
    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
    
    # 添加标签
    label = f"{manipulation_type} ({confidence:.1f}%)"
    
    # 尝试寻找支持中文的字体
    try:
        # 尝试加载系统中常用的支持中文的字体
        font_paths = [
            os.path.join(os.environ.get('SystemRoot', 'C:\Windows'), 'Fonts', 'simsun.ttc'),  # Windows中文字体
            os.path.join(os.environ.get('SystemRoot', 'C:\Windows'), 'Fonts', 'simhei.ttf'),  # Windows中文字体
            os.path.join(os.environ.get('SystemRoot', 'C:\Windows'), 'Fonts', 'msyh.ttc'),    # 微软雅黑
            os.path.join(os.environ.get('SystemRoot', 'C:\Windows'), 'Fonts', 'arial.ttf'),   # 英文备选
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux中文字体
            "/System/Library/Fonts/PingFang.ttc"  # macOS中文字体
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, 15)
                    print(f"成功加载字体: {font_path}", flush=True)
                    break
                except Exception as e:
                    print(f"尝试加载字体 {font_path} 失败: {e}", flush=True)
                    continue
        
        # 如果没有找到合适的字体，使用默认字体
        if font is None:
            print("未找到支持中文的字体，使用默认字体。可能无法正确显示中文。", flush=True)
            font = ImageFont.load_default()
            
        # 测量文本大小以确定背景矩形的大小
        try:
            if hasattr(font, 'getbbox'):
                # 较新版本的Pillow
                bbox = font.getbbox(label)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                # 旧版本的Pillow
                text_width, text_height = font.getsize(label)
            
            # 绘制背景矩形和文本
            draw.rectangle([(x1, y1-text_height-10), (x1 + text_width + 10, y1)], fill="red")
            draw.text((x1+5, y1-text_height-5), label, fill="white", font=font)
        except Exception as e:
            # 如果测量文本大小失败，使用简化版本
            print(f"文本渲染发生错误: {e}，使用简化版本", flush=True)
            draw.rectangle([(x1, y1-20), (x1 + 150, y1)], fill="red")
            # 使用ASCII编码版本的标签，避免Unicode错误
            ascii_label = f"Fake ({confidence:.1f}%)"
            draw.text((x1+5, y1-15), ascii_label, fill="white", font=font)
    except Exception as e:
        print(f"绘制标签时发生错误: {e}", flush=True)
        # 在出错的情况下，使用最简单的英文标记，确保至少能显示边界框
        draw.rectangle([(x1, y1-20), (x1 + 100, y1)], fill="red")
        draw.text((x1+5, y1-15), "Fake", fill="white")
    
    # 设置输出路径
    if output_path is None:
        output_path = os.path.join(os.path.dirname(image_path), "detection_result.png")
        
    # 保存结果
    image.save(output_path)
    
    return output_path


def detect_fake(args):
    """
    检测图像和文本中的伪造内容
    
    参数:
        args: 包含以下字段的对象:
            - image: 图像路径
            - text: 相关文本 (可选)
            - config: 配置文件路径
            - checkpoint: 模型权重路径
            - visualize: 是否可视化结果
            - output: 输出图像路径 (可选)
            
    返回:
        result: 包含检测结果的字典
    """
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 构建模型
    model = build_model(config, args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # 图像预处理
    image = preprocess_image(args.image, config['image_res'])
    image = image.unsqueeze(0).to(device)
    
    # 文本预处理 (如果有)
    if args.text and args.text.strip():
        tokenizer = BertTokenizer.from_pretrained(config['bert_config'])
        text = preprocess_text(args.text, config['max_words'], tokenizer)
        text = {k: v.to(device) for k, v in text.items()}
    else:
        text = None
    
    # 执行推理
    with torch.no_grad():
        output = model(image, text)
        
        # 解析输出
        pred_real_fake = output['real_fake_output'].softmax(dim=-1)
        pred_bbox = output['bbox_output'].cpu().numpy()[0]  # [cx, cy, w, h]
        pred_labels = output['multi_label_output'].sigmoid()
        
        # 获取预测结果
        is_fake_prob = pred_real_fake[0, 1].item()  # 伪造概率
        is_fake = bool(is_fake_prob > 0.5)
        
        # 获取篡改类型
        if is_fake:
            # 找出最可能的篡改类型
            manipulation_idx = torch.argmax(pred_labels[0]).item() + 1  # +1因为索引0表示"未篡改"
            manipulation_type = MANIPULATION_TYPES.get(manipulation_idx, "未知")
        else:
            manipulation_type = MANIPULATION_TYPES[0]  # "未篡改"
            
    # 组织结果
    result = {
        'is_fake': is_fake,
        'confidence': is_fake_prob * 100,  # 转为百分比
        'manipulation_type': manipulation_type,
        'fake_region': pred_bbox.tolist() if is_fake else None
    }
    
    # 可视化结果
    if args.visualize and is_fake:
        output_path = visualize_result(
            args.image, 
            pred_bbox, 
            manipulation_type, 
            result['confidence'], 
            args.output
        )
        result['visualization_path'] = output_path
    
    return result


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='HAMMER伪造检测器')
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--text', default='', help='相关文本')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', default='weights/checkpoint_best.pth', help='模型权重路径')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    parser.add_argument('--output', default=None, help='输出图像路径')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.image):
        print(f"错误: 图像文件 '{args.image}' 不存在!")
        return 1
        
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 '{args.config}' 不存在!")
        return 1
        
    if not os.path.exists(args.checkpoint):
        print(f"警告: 模型权重文件 '{args.checkpoint}' 不存在!")
        
    # 执行检测
    result = detect_fake(args)
    
    # 输出结果
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 