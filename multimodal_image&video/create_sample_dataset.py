import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import shutil

def create_directory_structure():
    """创建数据集目录结构"""
    dirs = [
        'datasets/DGM4/metadata',
        'datasets/DGM4/images'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def create_sample_image(image_path, text, has_fake_region=False):
    """创建示例图片，可选择是否包含假区域"""
    # 创建一个随机背景色的图片
    img = Image.new('RGB', (256, 256), color=(random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)))
    draw = ImageDraw.Draw(img)
    
    # 添加一些随机形状作为背景
    for _ in range(3):
        x1 = random.randint(0, 200)
        y1 = random.randint(0, 200)
        x2 = x1 + random.randint(20, 50)
        y2 = y1 + random.randint(20, 50)
        color = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
        draw.rectangle([x1, y1, x2, y2], fill=color)

    # 如果需要假区域，添加一个特殊的区域
    fake_box = None
    if has_fake_region:
        x1 = random.randint(50, 150)
        y1 = random.randint(50, 150)
        w = random.randint(30, 50)
        h = random.randint(30, 50)
        x2 = x1 + w
        y2 = y1 + h
        # 使用红色标记假区域
        draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0))
        fake_box = [x1, y1, x2, y2]

    # 添加文本
    try:
        # 尝试使用系统字体
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        # 如果找不到系统字体，使用默认字体
        font = ImageFont.load_default()
    
    draw.text((10, 10), text, fill=(0, 0, 0), font=font)
    
    # 保存图片
    img.save(image_path)
    return fake_box

def create_sample_text():
    """创建示例文本"""
    templates = [
        "这是一个{}的新闻报道，关于{}的内容。",
        "最近发生了一件{}的事情，与{}有关。",
        "据报道，{}方面出现了{}的情况。",
        "专家表示，{}领域存在{}的现象。"
    ]
    
    subjects = ["科技", "经济", "教育", "医疗", "环境", "社会"]
    descriptions = ["有趣", "重要", "突发", "值得关注", "普遍", "特殊"]
    
    template = random.choice(templates)
    return template.format(random.choice(descriptions), random.choice(subjects))

def create_dataset(num_train=100, num_test=20):
    """创建完整的示例数据集"""
    # 创建目录结构
    create_directory_structure()
    
    # 创建训练集和测试集
    datasets = {
        'train': num_train,
        'test': num_test
    }
    
    for dataset_type, num_samples in datasets.items():
        data = []
        for i in range(num_samples):
            # 生成文本
            text = create_sample_text()
            
            # 决定是否包含假信息
            is_fake = random.random() < 0.5
            
            # 创建图片文件名
            image_name = f"{dataset_type}_{i+1}.jpg"
            image_path = f"datasets/DGM4/images/{image_name}"
            
            # 创建图片并获取假区域位置（如果有的话）
            fake_box = create_sample_image(image_path, text, has_fake_region=is_fake)
            
            # 创建假文本位置（随机选择1-3个位置）
            text_length = len(text)
            fake_text_pos = []
            if is_fake:
                num_fake_pos = random.randint(1, 3)
                fake_text_pos = sorted(random.sample(range(min(text_length, 50)), num_fake_pos))
            
            # 创建数据条目
            entry = {
                "image": f"DGM4/images/{image_name}",
                "text": text,
                "label": 1 if is_fake else 0,
                "fake_cls": 1 if is_fake else 0,
                "fake_image_box": fake_box if fake_box else [0, 0, 0, 0],
                "fake_text_pos": fake_text_pos
            }
            data.append(entry)
        
        # 保存JSON文件
        json_path = f"datasets/DGM4/metadata/{dataset_type}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print("开始创建示例数据集...")
    create_dataset(num_train=100, num_test=20)
    print("数据集创建完成！")
    print("训练集大小：100个样本")
    print("测试集大小：20个样本")
    print("数据位置：datasets/DGM4/") 