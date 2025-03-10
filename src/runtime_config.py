import os
import torch

# 从环境变量读取配置
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 32))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', 2e-5))
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY', 0.01))
EPOCHS = int(os.environ.get('EPOCHS', 3))
MAX_LEN = int(os.environ.get('MAX_LEN', 64))
DROPOUT_RATE = float(os.environ.get('DROPOUT_RATE', 0.3))
USE_AMP = os.environ.get('USE_AMP', 'true').lower() == 'true'
SCHEDULER_TYPE = os.environ.get('SCHEDULER_TYPE', 'linear')
BALANCE_DATASET = os.environ.get('BALANCE_DATASET', 'true').lower() == 'true'
BALANCE_METHOD = os.environ.get('BALANCE_METHOD', 'downsample')
EARLY_STOPPING = os.environ.get('EARLY_STOPPING', 'false').lower() == 'true'
PATIENCE = int(os.environ.get('PATIENCE', 3))
DELTA = float(os.environ.get('DELTA', 0.001))
CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 4))
SEED = int(os.environ.get('SEED', 42))

# 从基础配置导入其他设置
from config import *
