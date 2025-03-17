import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

from data_processor import FakeNewsDataset
from train import BertFakeNewsClassifier, train_model

# 忽略警告信息
warnings.filterwarnings('ignore')

def prepare_balanced_dataloaders(train_path, val_path, tokenizer, batch_size=16):
    """
    准备平衡的训练和验证数据加载器
    """
    print(f"准备平衡数据加载器...")
    
    # 加载数据
    print(f"加载数据集: {train_path}")
    try:
        if not os.path.exists(train_path):
            # 尝试修正路径
            base_name = os.path.basename(train_path)
            corrected_path = os.path.join('..', 'data', base_name)
            if os.path.exists(corrected_path):
                print(f"使用修正后的路径: {corrected_path}")
                train_path = corrected_path
            else:
                raise FileNotFoundError(f"找不到数据文件: {train_path}")
                
        df = pd.read_csv(train_path)
        print(f"原始数据集形状: {df.shape}")
        
        # 统计标签分布
        label_counts = df['label'].value_counts()
        print("原始标签分布:")
        print(label_counts)
        
        # 过滤掉'尚无定论'的样本，只保留'谣言'和'事实'
        df = df[df['label'].isin(['谣言', '事实'])]
        print(f"过滤后数据集形状: {df.shape}")
        
        # 重新统计标签分布
        label_counts = df['label'].value_counts()
        print("过滤后标签分布:")
        print(label_counts)
        
        # 分离谣言和事实样本
        fake_news = df[df['label'] == '谣言']
        real_news = df[df['label'] == '事实']
        
        # 下采样谣言样本到真实新闻的数量，以平衡类别
        num_real = len(real_news)
        fake_news_balanced = fake_news.sample(n=num_real, random_state=42)
        
        # 合并平衡后的数据集
        balanced_df = pd.concat([fake_news_balanced, real_news])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # 随机打乱数据
        
        print(f"平衡后数据集形状: {balanced_df.shape}")
        print("平衡后标签分布:")
        print(balanced_df['label'].value_counts())
        
        # 划分训练集和验证集（80%/20%）
        train_df, val_df = train_test_split(balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['label'])
        
        print(f"训练集大小: {len(train_df)}")
        print(f"验证集大小: {len(val_df)}")
        print("训练集标签分布:")
        print(train_df['label'].value_counts())
        print("验证集标签分布:")
        print(val_df['label'].value_counts())
        
        # 创建数据集
        print("创建训练集...")
        train_dataset = FakeNewsDataset(train_df, tokenizer)
        print("创建验证集...")
        val_dataset = FakeNewsDataset(val_df, tokenizer)
        
        # 创建数据加载器
        print(f"使用batch_size={batch_size}创建数据加载器")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,  # 使用随机打乱而不是采样器
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        raise e

def main():
    # 设置CUDA相关配置
    torch.backends.cudnn.benchmark = True
    
    # 检查是否有GPU可用
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
    else:
        device = torch.device('cpu')
        print("GPU不可用，使用CPU训练")
    
    # 加载分词器
    try:
        print("尝试从本地加载bert-base-chinese分词器...")
        tokenizer = BertTokenizer.from_pretrained('../models/bert_local', local_files_only=True)
        print("成功从本地加载分词器")
    except Exception as e:
        print(f"从本地加载分词器失败: {e}")
        try:
            print("尝试在线加载bert-base-chinese分词器...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        except Exception as e:
            print(f"无法加载分词器: {e}")
            raise RuntimeError("无法加载分词器，无法继续训练")
    
    # 准备数据加载器
    batch_size = 32 if torch.cuda.is_available() else 8
    print(f"使用batch_size: {batch_size}")
    
    # 使用平衡的数据加载器
    train_loader, val_loader = prepare_balanced_dataloaders(
        '../data/train.csv', 
        '../data/train.csv',
        tokenizer,
        batch_size
    )
    
    # 初始化模型
    model = BertFakeNewsClassifier().to(device)
    
    # 使用适当的损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # 学习率调度器
    epochs = 3
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    try:
        # 训练模型
        train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, device, epochs=epochs)
        print("训练完成！")
        
        # 保存平衡模型
        torch.save(model.state_dict(), '../models/balanced_fake_news_model.pth')
        print("平衡模型已保存至 ../models/balanced_fake_news_model.pth")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        # 尝试保存中断的模型
        try:
            torch.save(model.state_dict(), '../models/balanced_model_interrupted.pth')
            print("已保存中断的模型")
        except:
            print("无法保存模型")

if __name__ == '__main__':
    main() 