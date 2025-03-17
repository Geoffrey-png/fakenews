#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HAMMER 检测器接口类
提供简单易用的Python API，方便集成到其他项目中
"""

import os
import torch
import json
import yaml
import sys
from PIL import Image

# 修改为绝对导入
from models.HAMMER import HAMMER
from models.box_ops import box_cxcywh_to_xyxy
from tools.utils import preprocess_image, preprocess_text


class HAMMERDetector:
    """
    HAMMER检测器接口类
    提供简单的伪造检测API
    """
    
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
    
    # 英文标签映射（备用）
    MANIPULATION_TYPES_EN = {
        0: "No manipulation",
        1: "Face swap",
        2: "Face attribute modification",
        3: "Text swap",
        4: "Text attribute modification", 
        5: "Object addition",
        6: "Object removal",
        7: "Object modification"
    }
    
    def __init__(self, config_path, model_path, device=None, use_english=False):
        """
        初始化HAMMER检测器
        
        Args:
            config_path (str): 配置文件路径
            model_path (str): 模型权重路径
            device (str, optional): 使用的设备，可以是'cpu'或'cuda'，默认为None自动选择
            use_english (bool, optional): 是否使用英文标签，默认为False
        """
        # 设置标签语言
        self.use_english = use_english
        
        # 设置默认路径
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            config_path = os.path.join(parent_dir, 'config.yaml')
            
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            model_path = os.path.join(parent_dir, 'weights', 'checkpoint_best.pth')
        
        # 加载配置
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 构建模型
        self.model = HAMMER(self.config)
        
        # 加载预训练权重
        if model_path and os.path.exists(model_path):
            print(f"加载模型检查点: {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 处理可能的不匹配
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
                
            # 加载状态字典
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f"加载结果: {msg}")
        else:
            raise FileNotFoundError(f"未找到模型权重文件: {model_path}")
        
        # 将模型移动到设备
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 初始化tokenizer（如需要）
        self.tokenizer = None
        
    def get_tokenizer(self):
        """获取文本分词器"""
        if self.tokenizer is None:
            # 使用本地tokenizer
            from transformers import BertTokenizer
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            tokenizer_path = os.path.join(parent_dir, 'tokenizer')
            
            if os.path.exists(tokenizer_path):
                self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            else:
                raise FileNotFoundError(f"未找到tokenizer目录: {tokenizer_path}")
                
        return self.tokenizer
    
    def get_manipulation_type(self, idx):
        """获取篡改类型名称"""
        if self.use_english:
            return self.MANIPULATION_TYPES_EN.get(idx, "Unknown")
        else:
            return self.MANIPULATION_TYPES.get(idx, "未知")
    
    def detect(self, image_path, text=None, visualize=False, output_path=None):
        """
        检测图像和文本中的伪造内容
        
        参数:
            image_path: 图像路径
            text: 相关文本 (可选)
            visualize: 是否可视化结果
            output_path: 输出图像路径 (可选)
            
        返回:
            result: 包含检测结果的字典
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 图像预处理
        image = preprocess_image(image_path, self.config['image_res'])
        image = image.unsqueeze(0).to(self.device)
        
        # 文本预处理 (如果有)
        if text and text.strip():
            tokenizer = self.get_tokenizer()
            text_input = preprocess_text(text, self.config['max_words'], tokenizer)
            text_input = {k: v.to(self.device) for k, v in text_input.items()}
        else:
            text_input = None
        
        # 执行推理
        with torch.no_grad():
            output = self.model(image, text_input)
            
            # 解析输出
            pred_real_fake = output['real_fake_output'].softmax(dim=-1)
            pred_bbox = output['bbox_output'].cpu().numpy()[0]  # [cx, cy, w, h]
            pred_labels = output['multi_label_output'].sigmoid()
            
            # 获取预测结果
            forgery_score = pred_real_fake[0, 1].item()  # 伪造概率
            is_fake = bool(forgery_score > 0.5)
            
            # 获取篡改类型
            if is_fake:
                # 找出最可能的篡改类型
                manipulation_idx = torch.argmax(pred_labels[0]).item() + 1  # +1因为索引0表示"未篡改"
                manipulation_type = self.get_manipulation_type(manipulation_idx)
            else:
                manipulation_type = self.get_manipulation_type(0)  # "未篡改"
            
            # 转换边界框坐标格式，从[cx, cy, w, h]到[x1, y1, x2, y2]
            if is_fake:
                # 转换为tensor，然后使用box_cxcywh_to_xyxy函数
                box_tensor = torch.tensor(pred_bbox).unsqueeze(0)  # [1, 4]
                box_xyxy = box_cxcywh_to_xyxy(box_tensor).squeeze(0).tolist()  # [4]
            else:
                box_xyxy = None
                
        # 组织结果
        result = {
            'is_fake': is_fake,
            'forgery_score': forgery_score,  # 直接返回概率值而非百分比
            'manipulation_type': manipulation_type,
            'bboxes': [box_xyxy] if box_xyxy else []  # 返回列表格式，兼容多个边界框
        }
        
        # 可视化结果，仅当visualize=True且结果为伪造时
        if visualize and is_fake:
            try:
                from detect_image import visualize_result
                output_path = visualize_result(
                    image_path, 
                    pred_bbox, 
                    forgery_score * 100,  # 转为百分比
                    manipulation_type, 
                    output_path
                )
                result['visualization_path'] = output_path
            except Exception as e:
                print(f"可视化结果时出错: {str(e)}")
                # 可视化失败不影响返回结果
        
        return result
    
    def detect_batch(self, image_paths, texts=None, visualize=False, output_dir=None):
        """
        批量检测图像和文本中的伪造内容
        
        参数:
            image_paths: 图像路径列表
            texts: 相关文本列表 (可选)
            visualize: 是否可视化结果
            output_dir: 输出目录 (可选)
            
        返回:
            results: 包含检测结果的字典列表
        """
        results = []
        
        # 确保texts长度匹配
        if texts is None:
            texts = [None] * len(image_paths)
        elif len(texts) != len(image_paths):
            texts = texts + [None] * (len(image_paths) - len(texts))
        
        # 创建输出目录
        if visualize and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 批量处理
        for i, (image_path, text) in enumerate(zip(image_paths, texts)):
            # 设置输出路径
            if visualize and output_dir:
                basename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, f"result_{basename}")
            else:
                output_path = None
            
            # 执行检测
            try:
                result = self.detect(image_path, text, visualize, output_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"处理图像失败 {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def save_results(self, results, output_path):
        """
        保存检测结果到JSON文件
        
        参数:
            results: 检测结果
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            # 确保中文正确编码
            import json
            class ChineseJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, bytes):
                        return obj.decode('utf-8')
                    return json.JSONEncoder.default(self, obj)
                    
            # 对中文字段进行预处理，确保正确编码
            if isinstance(results, dict):
                if 'manipulation_type' in results:
                    results['manipulation_type'] = results['manipulation_type'].encode('utf-8').decode('utf-8')
            elif isinstance(results, list):
                for result in results:
                    if isinstance(result, dict) and 'manipulation_type' in result:
                        result['manipulation_type'] = result['manipulation_type'].encode('utf-8').decode('utf-8')
            
            json.dump(results, f, indent=2, ensure_ascii=False, cls=ChineseJSONEncoder)
        
        print(f"结果已保存至: {output_path}") 