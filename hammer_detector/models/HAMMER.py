#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HAMMER模型
用于多模态假新闻检测的主要模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 修复Windows中文编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import BertModel, BertConfig

from .vit import VisionTransformer


class HAMMER(nn.Module):
    """
    HAMMER (Hierarchical Attentional Multi-modal rEpResentation leaRning) 模型
    
    用于检测图像和文本中的篡改内容
    """
    
    def __init__(self, config, args=None, text_encoder=None, tokenizer=None, init_deit=True):
        """
        初始化HAMMER模型
        
        Args:
            config: 配置字典
            args: 命令行参数 (可选)
            text_encoder: 文本编码器名称 (可选)
            tokenizer: 文本分词器 (可选)
            init_deit: 是否使用预训练的DeiT权重初始化视觉编码器 (可选)
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        
        # 基于ViT的视觉编码器
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'],
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=nn.LayerNorm
        )
        
        if init_deit:
            try:
                # 获取当前文件所在目录
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # 获取项目根目录
                root_dir = os.path.dirname(current_dir)
                # 预训练权重文件路径
                deit_weights_path = os.path.join(root_dir, 'pretrained', 'deit_base_patch16_224.pth')
                
                if os.path.exists(deit_weights_path):
                    # 从本地加载预训练权重
                    print(f"正在从本地加载DeiT预训练权重: {deit_weights_path}")
                    checkpoint = torch.load(deit_weights_path, map_location="cpu")
                    state_dict = checkpoint["model"]
                    pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
                    state_dict['pos_embed'] = pos_embed_reshaped
                    
                    print("使用DeiT预训练权重初始化视觉编码器")
                    msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
                    print(f"视觉编码器初始化信息: {msg}")
                else:
                    print(f"警告: 未找到DeiT预训练权重文件: {deit_weights_path}")
                    print("视觉编码器将使用随机初始化权重")
            except Exception as e:
                print(f"警告: 无法加载预训练ViT权重: {str(e)}")
                print("视觉编码器将使用随机初始化权重")

        # 基于BERT的文本编码器
        try:
            # 获取当前文件所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 获取项目根目录
            root_dir = os.path.dirname(current_dir)
            # 本地BERT模型目录
            bert_model_dir = os.path.join(root_dir, 'pretrained', 'bert')
            
            if os.path.exists(bert_model_dir):
                print(f"正在从本地加载BERT模型: {bert_model_dir}")
                bert_config = BertConfig.from_pretrained(bert_model_dir)
                self.text_encoder = BertModel.from_pretrained(bert_model_dir, config=bert_config)
            else:
                print(f"警告: 未找到本地BERT模型目录: {bert_model_dir}")
                # 创建一个默认的BERT配置和模型
                print("使用默认BERT配置创建文本编码器")
                bert_config = BertConfig()
                self.text_encoder = BertModel(bert_config)
        except Exception as e:
            print(f"警告: 无法加载BERT模型: {str(e)}")
            # 创建一个默认的BERT配置和模型（不加载预训练权重）
            print("使用默认BERT配置创建文本编码器")
            bert_config = BertConfig()
            self.text_encoder = BertModel(bert_config)
        
        # 视觉特征投影
        self.vision_proj = nn.Linear(768, config['embed_dim'])
        self.text_proj = nn.Linear(768, config['embed_dim'])
        
        # 篡改检测头
        self.real_fake_head = nn.Linear(config['embed_dim'], 2)  # 二分类: 真/假
        self.multi_cls_head = nn.Linear(config['embed_dim'], 4)  # 多标签分类: face_swap, face_attribute, text_swap, text_attribute
        self.bbox_head = MLP(config['embed_dim'], config['embed_dim'], 4, 3)  # 边界框回归
        self.token_head = nn.Linear(768, 2)  # 文本标记二分类: 篡改/非篡改
        
        # 初始化参数
        self.real_fake_head.apply(self._init_weights)
        self.multi_cls_head.apply(self._init_weights)
        self.bbox_head.apply(self._init_weights)
        self.token_head.apply(self._init_weights)
    
    def _init_weights(self, m):
        """权重初始化函数"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, image, text_input=None, label=None, fake_image_box=None, fake_token_pos=None, is_train=False):
        """
        前向传播
        
        Args:
            image: 图像输入
            text_input: 文本输入 (可选)
            label: 标签 (可选，训练时使用)
            fake_image_box: 伪造区域边界框 (可选，训练时使用)
            fake_token_pos: 伪造文本位置 (可选，训练时使用)
            is_train: 是否处于训练模式
            
        Returns:
            dict: 包含以下键值的字典:
                - real_fake_output: 真假分类输出
                - multi_label_output: 多标签分类输出
                - bbox_output: 边界框回归输出
                - token_output: 文本标记分类输出 (如果有文本输入)
        """
        # 图像特征提取
        image_embeds = self.visual_encoder(image)  # [batch_size, 196+1, 768]
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0]), dim=-1)  # [batch_size, embed_dim]
        
        # 文本特征提取(如果有文本输入)
        if text_input is not None:
            # 处理字典形式的输入
            if isinstance(text_input, dict):
                text_output = self.text_encoder(
                    input_ids=text_input['input_ids'], 
                    attention_mask=text_input['attention_mask'],
                    return_dict=True
                )
            else:
                # 处理直接传入的编码后文本
                text_output = self.text_encoder(
                    input_ids=text_input.input_ids, 
                    attention_mask=text_input.attention_mask,
                    return_dict=True
                )
            text_embeds = text_output.last_hidden_state  # [batch_size, seq_len, 768]
            text_feat = F.normalize(self.text_proj(text_embeds[:, 0]), dim=-1)  # [batch_size, embed_dim]
            
            # 融合特征
            multimodal_feat = (image_feat + text_feat) / 2
        else:
            # 如果没有文本，只使用图像特征
            multimodal_feat = image_feat
            text_embeds = None
        
        # 预测
        logits_real_fake = self.real_fake_head(multimodal_feat)  # 真假分类
        logits_multicls = self.multi_cls_head(multimodal_feat)  # 多标签分类
        output_coord = self.bbox_head(multimodal_feat).sigmoid()  # 边界框回归
        
        # 文本标记预测(如果有文本输入)
        logits_tok = None
        if text_embeds is not None:
            logits_tok = self.token_head(text_embeds)  # [batch_size, seq_len, 2]
        
        # 返回字典形式的结果
        outputs = {
            'real_fake_output': logits_real_fake,  # [batch_size, 2]
            'multi_label_output': logits_multicls,  # [batch_size, 4]
            'bbox_output': output_coord,  # [batch_size, 4] (cx, cy, w, h)
        }
        
        if logits_tok is not None:
            outputs['token_output'] = logits_tok  # [batch_size, seq_len, 2]
        
        return outputs


class MLP(nn.Module):
    """多层感知机，用于边界框回归"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    """
    插值位置嵌入，处理不同尺寸的图像
    """
    # 不需要插值，直接返回
    if pos_embed_checkpoint.shape == visual_encoder.pos_embed.shape:
        return pos_embed_checkpoint
    
    # 位置嵌入需要插值调整
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    
    # 提取分类标记和补丁嵌入
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)
    
    if orig_size != new_size:
        print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        
        # 使用双线性插值调整大小
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        
        return new_pos_embed
    else:
        return pos_embed_checkpoint 