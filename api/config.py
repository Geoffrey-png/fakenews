#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# 基本路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# 确保目录存在
os.makedirs(LOGS_DIR, exist_ok=True)

# 模型配置
MODEL_PATH = os.path.join(MODELS_DIR, 'fake_news_bert_model_final.pth')
BERT_LOCAL_PATH = os.path.join(MODELS_DIR, 'bert_local')

# 服务配置
API_HOST = os.environ.get('API_HOST', '0.0.0.0')
API_PORT = int(os.environ.get('API_PORT', 5000))
DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

# 跨域配置
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')

# 模型预测配置
MAX_SEQUENCE_LENGTH = int(os.environ.get('MAX_SEQUENCE_LENGTH', 128))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8)) 