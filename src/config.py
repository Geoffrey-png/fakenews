import os
import torch

# 路径配置
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
LOG_DIR = os.path.join(ROOT_DIR, 'runs')
REPORT_DIR = os.path.join(ROOT_DIR, 'report')

# 数据集配置
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
VAL_FILE = os.path.join(DATA_DIR, 'train.csv')  # 使用同一个文件进行训练和验证
TEST_FILE = os.path.join(DATA_DIR, 'test.xls')

# 模型配置
MODEL_NAME = 'bert-base-chinese'
MODEL_PATH = os.path.join(MODEL_DIR, 'bert_local')  # 本地BERT模型路径
PRETRAINED_MODEL_PATH = os.path.join(MODEL_DIR, 'fake_news_bert_model.pth')  # 预训练好的分类模型路径
BALANCED_MODEL_PATH = os.path.join(MODEL_DIR, 'balanced_fake_news_model.pth')  # 平衡数据集训练的模型路径
NUM_CLASSES = 2
DROPOUT_RATE = 0.3

# 分词器配置
MAX_LEN = 64  # 最大序列长度
LOCAL_FILES_ONLY = True  # 是否只使用本地文件

# 训练配置
BATCH_SIZE = 32  # 如果有GPU，使用较大的batch size
CPU_BATCH_SIZE = 8  # CPU训练时的batch size
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 200
NUM_WARMUP_STEPS = 0
SCHEDULER_TYPE = 'linear'  # 学习率调度器类型：linear, cosine, constant

# 平衡数据集配置
BALANCE_DATASET = True  # 是否平衡数据集
BALANCE_METHOD = 'downsample'  # 平衡方法：'downsample'下采样、'upsample'上采样、'weight'权重采样

# 训练选项
USE_AMP = True  # 是否使用混合精度训练(Automatic Mixed Precision)
SAVE_EVERY_EPOCH = True  # 是否每个epoch都保存模型
EARLY_STOPPING = False  # 是否使用早停
PATIENCE = 3  # 早停的耐心值
DELTA = 0.001  # 早停的阈值

# 评估配置
EVAL_BATCH_SIZE = 64

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_VISIBLE_DEVICES = '0,1,2'  # 使用的GPU ID，多个用逗号分隔
CUDNN_BENCHMARK = True
NUM_WORKERS = 4  # 数据加载的工作线程数

# 随机种子
SEED = 42

# 日志配置
LOG_INTERVAL = 10  # 每多少个batch记录一次日志

def print_config():
    """打印配置信息"""
    config_dict = {k: v for k, v in globals().items() if k.isupper()}
    print("\n=== 训练配置 ===")
    for key, value in config_dict.items():
        print(f"{key}: {value}")
    print("================")

if __name__ == "__main__":
    # 打印配置
    print_config() 