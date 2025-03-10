#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import jieba
from transformers import BertTokenizer, AutoTokenizer
import sys
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试导入模型类
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.train import BertFakeNewsClassifier
    logger.info("成功从src.train导入BertFakeNewsClassifier")
except ImportError:
    logger.warning("无法从src.train导入BertFakeNewsClassifier，将创建自己的模型类")
    
    # 定义自己的模型类（如果无法导入原始类）
    import torch.nn as nn
    
    class BertFakeNewsClassifier(nn.Module):
        def __init__(self, model_name='bert-base-chinese', num_classes=2):
            super(BertFakeNewsClassifier, self).__init__()
            from transformers import BertModel
            
            try:
                self.bert = BertModel.from_pretrained(model_name)
                logger.info(f"成功加载BERT模型: {model_name}")
            except Exception as e:
                logger.error(f"加载BERT模型失败: {e}")
                # 尝试本地加载
                try:
                    bert_local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'bert_local')
                    if os.path.exists(bert_local_path):
                        self.bert = BertModel.from_pretrained(bert_local_path)
                        logger.info(f"成功从本地加载BERT模型: {bert_local_path}")
                    else:
                        raise ValueError(f"找不到本地BERT模型: {bert_local_path}")
                except Exception as e:
                    logger.error(f"加载本地BERT模型失败: {e}")
                    raise
            
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        def forward(self, input_ids, attention_mask, token_type_ids):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits

class FakeNewsPredictor:
    def __init__(self, model_path):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 加载分词器
        self._load_tokenizer()
        
        # 加载模型
        self._load_model(model_path)
    
    def _load_tokenizer(self):
        """加载分词器，尝试多种可能的路径"""
        try:
            # 首先尝试从本地路径加载
            bert_local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'bert_local')
            if os.path.exists(bert_local_path):
                self.tokenizer = BertTokenizer.from_pretrained(bert_local_path, local_files_only=True)
                logger.info(f"成功从本地路径加载分词器: {bert_local_path}")
                return
        except Exception as e:
            logger.warning(f"从本地路径加载分词器失败: {e}")
        
        # 尝试在线加载
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
            logger.info("成功在线加载bert-base-chinese分词器")
            return
        except Exception as e:
            logger.warning(f"在线加载bert-base-chinese分词器失败: {e}")
        
        # 尝试从其他可能的本地路径加载
        possible_paths = [
            '../models/bert_local',
            '../models/bert-base-chinese',
            './models/bert_local',
            './models/bert-base-chinese',
            os.path.abspath('../models/bert_local'),
            os.path.abspath('./models/bert_local')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    logger.info(f"尝试从路径加载分词器: {path}")
                    self.tokenizer = BertTokenizer.from_pretrained(path, local_files_only=True)
                    logger.info(f"成功从{path}加载分词器")
                    return
                except Exception as e:
                    logger.warning(f"从{path}加载分词器失败: {e}")
        
        # 如果所有尝试都失败
        raise RuntimeError("无法加载分词器，请检查模型路径和网络连接")
    
    def _load_model(self, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        logger.info(f"加载模型: {model_path}")
        self.model = BertFakeNewsClassifier().to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"加载模型权重失败: {e}")
            raise
        
        self.model.eval()
    
    def predict(self, text, max_len=128):
        """
        预测文本类别
        
        Args:
            text: 新闻文本
            max_len: 最大序列长度
        
        Returns:
            dict: 包含预测类别和概率的字典
        """
        # 中文分词
        text = ' '.join(list(jieba.cut(text)))
        
        # 编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 移动到设备
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, token_type_ids)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        return {
            'class': int(predicted_class.cpu().numpy()[0]),
            'probabilities': probabilities.cpu().numpy()[0].tolist()
        } 