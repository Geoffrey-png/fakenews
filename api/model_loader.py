#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import jieba
from transformers import BertTokenizer, AutoTokenizer
import sys
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 预加载jieba词典，避免首次分词慢
jieba_lock = threading.Lock()
jieba_initialized = False

def initialize_jieba():
    global jieba_initialized
    with jieba_lock:
        if not jieba_initialized:
            logger.info("预加载jieba词典...")
            jieba.initialize()
            # 可以添加自定义词典
            # jieba.load_userdict("path/to/userdict.txt")
            jieba_initialized = True
            logger.info("jieba词典加载完成")

# 在后台线程中预加载
threading.Thread(target=initialize_jieba, daemon=True).start()

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
        
        # 添加预测缓存
        self.prediction_cache = {}
        self.max_cache_size = 5000  # 缓存大小上限
        logger.info("预测结果缓存已初始化，最大缓存数: 5000")
        
        # 设置批处理大小
        self.batch_size = 8  # 可以根据设备性能调整
        logger.info(f"批处理大小设置为: {self.batch_size}")
    
    def _cache_prediction(self, text, result):
        """缓存预测结果"""
        # 限制缓存大小
        if len(self.prediction_cache) >= self.max_cache_size:
            # 移除最旧的20%缓存
            remove_count = self.max_cache_size // 5
            keys_to_remove = list(self.prediction_cache.keys())[:remove_count]
            for key in keys_to_remove:
                del self.prediction_cache[key]
                
        # 添加到缓存
        self.prediction_cache[text] = result
    
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
        预测文本类别 (优化版)
        
        Args:
            text: 新闻文本
            max_len: 最大序列长度
        
        Returns:
            dict: 包含预测类别和概率的字典
        """
        # 检查缓存
        if text in self.prediction_cache:
            logger.info("使用缓存的预测结果")
            # 创建深拷贝以避免修改缓存的结果
            result = self.prediction_cache[text].copy()
            
            # 添加标签字段以兼容前端
            if 'label' not in result:
                result['label'] = '真实新闻' if result.get('class', 1) == 0 else '虚假新闻'
                
            # 确保confidence键存在
            if 'confidence' not in result and 'probabilities' in result:
                result['confidence'] = {
                    '真实新闻': result['probabilities'][0],
                    '虚假新闻': result['probabilities'][1]
                }
                
            return result
            
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
        
        # 构建结果
        predicted_class_val = int(predicted_class.cpu().numpy()[0])
        probabilities_val = probabilities.cpu().numpy()[0].tolist()
        
        result = {
            'class': predicted_class_val,
            'probabilities': probabilities_val,
            'label': '真实新闻' if predicted_class_val == 0 else '虚假新闻',
            'confidence': {
                '真实新闻': probabilities_val[0],
                '虚假新闻': probabilities_val[1]
            }
        }
        
        # 缓存结果
        self._cache_prediction(text, result)
        
        return result

    def predict_batch(self, texts, max_len=128):
        """
        批量预测多个文本
        
        Args:
            texts: 新闻文本列表
            max_len: 最大序列长度
            
        Returns:
            list: 包含每个文本预测结果的列表
        """
        # 初始化结果列表，先填充None
        results = [None] * len(texts)
        
        # 需要处理的文本索引
        to_process_indices = []
        to_process_texts = []
        
        # 先检查缓存
        for i, text in enumerate(texts):
            if text in self.prediction_cache:
                logger.info(f"文本 {i} 使用缓存结果")
                result = self.prediction_cache[text].copy()
                
                # 确保包含所有必要字段
                if 'label' not in result:
                    result['label'] = '真实新闻' if result.get('class', 1) == 0 else '虚假新闻'
                
                if 'confidence' not in result and 'probabilities' in result:
                    result['confidence'] = {
                        '真实新闻': result['probabilities'][0],
                        '虚假新闻': result['probabilities'][1]
                    }
                    
                results[i] = result
            else:
                to_process_indices.append(i)
                to_process_texts.append(text)
        
        # 如果所有文本都在缓存中找到
        if not to_process_texts:
            return results
            
        # 处理未缓存的文本
        for i in range(0, len(to_process_texts), self.batch_size):
            batch_texts = to_process_texts[i:i+self.batch_size]
            batch_indices = to_process_indices[i:i+self.batch_size]
            
            # 分词
            tokenized_texts = [' '.join(list(jieba.cut(text))) for text in batch_texts]
            
            # 编码
            encodings = [self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_len,
                return_token_type_ids=True,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for text in tokenized_texts]
            
            # 准备批处理数据
            input_ids = torch.cat([enc['input_ids'] for enc in encodings], dim=0).to(self.device)
            attention_mask = torch.cat([enc['attention_mask'] for enc in encodings], dim=0).to(self.device)
            token_type_ids = torch.cat([enc['token_type_ids'] for enc in encodings], dim=0).to(self.device)
            
            # 批量预测
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
            
            # 处理批量结果
            for j, (idx, pred_class, probs) in enumerate(
                zip(batch_indices, predicted_classes.cpu().numpy(), probabilities.cpu().numpy())
            ):
                pred_class_val = int(pred_class)
                probs_val = probs.tolist()
                
                result = {
                    'class': pred_class_val,
                    'probabilities': probs_val,
                    'label': '真实新闻' if pred_class_val == 0 else '虚假新闻',
                    'confidence': {
                        '真实新闻': probs_val[0],
                        '虚假新闻': probs_val[1]
                    }
                }
                
                # 更新结果列表
                results[idx] = result
                
                # 缓存结果
                self._cache_prediction(batch_texts[j], result)
                
        return results 