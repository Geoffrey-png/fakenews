import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import random
import warnings
import time
from datetime import datetime

from data_processor import prepare_dataloaders, FakeNewsDataset
from config import *

# 忽略警告信息
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

class BertFakeNewsClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super(BertFakeNewsClassifier, self).__init__()
        try:
            print(f"尝试从本地加载{MODEL_NAME}模型...")
            self.bert = BertModel.from_pretrained(MODEL_PATH, local_files_only=LOCAL_FILES_ONLY)
            print("成功从本地加载模型")
        except Exception as e:
            print(f"从本地加载失败: {e}")
            try:
                print(f"尝试在线加载{MODEL_NAME}模型...")
                self.bert = BertModel.from_pretrained(MODEL_NAME)
            except Exception as e:
                print(f"在线加载失败: {e}")
                print("创建随机初始化的模型")
                # 如果加载失败，创建一个简单的模型
                from transformers import BertConfig
                config = BertConfig(vocab_size=21128)  # 中文BERT的词表大小
                self.bert = BertModel(config)
                print("创建了随机初始化的模型")
                
        self.dropout = nn.Dropout(dropout_rate)
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

class EarlyStopping:
    """早停机制，当验证集上的性能不再提升时停止训练"""
    def __init__(self, patience=PATIENCE, delta=DELTA, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """保存模型当验证损失减小时"""
        print(f"验证损失减小: {self.val_loss_min:.6f} -> {val_loss:.6f}. 保存模型...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, device, epochs=EPOCHS):
    """训练模型"""
    # 为本次训练创建一个新的日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(LOG_DIR, f'fakenews_{timestamp}')
    writer = SummaryWriter(log_dir)
    
    # 初始化早停机制
    if EARLY_STOPPING:
        early_stopping = EarlyStopping(patience=PATIENCE, path=os.path.join(MODEL_DIR, 'best_model.pth'))
    
    # 使用混合精度训练
    use_amp = USE_AMP and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    # 保存训练开始时间
    start_time = time.time()
    
    print(f"\n开始训练，总共{epochs}个epoch")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n开始训练 Epoch {epoch+1}/{epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                # 使用混合精度训练
                with autocast():
                    outputs = model(input_ids, attention_mask, token_type_ids)
                    loss = criterion(outputs, labels)
                
                # 缩放梯度
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 常规训练
                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        print("\n开始验证...")
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)
                
                if use_amp:
                    with autocast():
                        outputs = model(input_ids, attention_mask, token_type_ids)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(input_ids, attention_mask, token_type_ids)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # 计算当前epoch的损失和指标
        train_loss_avg = train_loss/len(train_loader)
        val_loss_avg = val_loss/len(val_loader)
        
        # 计算指标
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')
        
        # 记录指标
        writer.add_scalar('Loss/Train', train_loss_avg, epoch)
        writer.add_scalar('Loss/Validation', val_loss_avg, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Precision', precision, epoch)
        writer.add_scalar('Recall', recall, epoch)
        writer.add_scalar('F1-Score', f1, epoch)
        
        # 计算每个epoch的训练时间
        epoch_time = time.time() - epoch_start_time
        
        print(f'Epoch {epoch+1}/{epochs} 完成，耗时: {epoch_time:.2f}秒')
        print(f'Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}')
        print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n')
        
        # 保存模型
        if SAVE_EVERY_EPOCH:
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'fake_news_model_epoch_{epoch+1}.pth'))
        
        # 早停检查
        if EARLY_STOPPING:
            early_stopping(val_loss_avg, model)
            if early_stopping.early_stop:
                print("早停触发，停止训练")
                break
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"训练完成，总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    
    # 保存最终模型
    final_model_path = os.path.join(MODEL_DIR, 'fake_news_bert_model_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    writer.close()
    return final_model_path

def prepare_balanced_dataloaders(train_path, val_path, tokenizer, batch_size=BATCH_SIZE):
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
            corrected_path = os.path.join(DATA_DIR, base_name)
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
        
        # 平衡数据集
        if BALANCE_METHOD == 'downsample':
            # 分离谣言和事实样本
            fake_news = df[df['label'] == '谣言']
            real_news = df[df['label'] == '事实']
            
            # 下采样谣言样本到真实新闻的数量，以平衡类别
            num_real = len(real_news)
            fake_news_balanced = fake_news.sample(n=num_real, random_state=SEED)
            
            # 合并平衡后的数据集
            balanced_df = pd.concat([fake_news_balanced, real_news])
            balanced_df = balanced_df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # 随机打乱数据
        
        elif BALANCE_METHOD == 'upsample':
            # 分离谣言和事实样本
            fake_news = df[df['label'] == '谣言']
            real_news = df[df['label'] == '事实']
            
            # 上采样真实新闻到谣言的数量，以平衡类别
            num_fake = len(fake_news)
            # 使用放回抽样
            real_news_balanced = real_news.sample(n=num_fake, replace=True, random_state=SEED)
            
            # 合并平衡后的数据集
            balanced_df = pd.concat([fake_news, real_news_balanced])
            balanced_df = balanced_df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # 随机打乱数据
        
        else:  # 默认使用下采样
            # 分离谣言和事实样本
            fake_news = df[df['label'] == '谣言']
            real_news = df[df['label'] == '事实']
            
            # 下采样谣言样本到真实新闻的数量，以平衡类别
            num_real = len(real_news)
            fake_news_balanced = fake_news.sample(n=num_real, random_state=SEED)
            
            # 合并平衡后的数据集
            balanced_df = pd.concat([fake_news_balanced, real_news])
            balanced_df = balanced_df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # 随机打乱数据
        
        print(f"平衡后数据集形状: {balanced_df.shape}")
        print("平衡后标签分布:")
        print(balanced_df['label'].value_counts())
        
        # 划分训练集和验证集（80%/20%）
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(balanced_df, test_size=0.2, random_state=SEED, stratify=balanced_df['label'])
        
        print(f"训练集大小: {len(train_df)}")
        print(f"验证集大小: {len(val_df)}")
        print("训练集标签分布:")
        print(train_df['label'].value_counts())
        print("验证集标签分布:")
        print(val_df['label'].value_counts())
        
        # 创建数据集
        print("创建训练集...")
        train_dataset = FakeNewsDataset(train_df, tokenizer, max_len=MAX_LEN)
        print("创建验证集...")
        val_dataset = FakeNewsDataset(val_df, tokenizer, max_len=MAX_LEN)
        
        # 创建数据加载器
        print(f"使用batch_size={batch_size}创建数据加载器")
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,  # 使用随机打乱而不是采样器
            num_workers=NUM_WORKERS
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=NUM_WORKERS
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        raise e

def main():
    # 设置随机种子
    set_seed(SEED)
    
    # 设置CUDA相关配置
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    
    # 检查是否有GPU可用
    device = DEVICE
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
    else:
        print("GPU不可用，使用CPU训练")
    
    # 打印配置信息
    print_config()
    
    # 加载分词器
    try:
        print(f"尝试从本地加载{MODEL_NAME}分词器...")
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=LOCAL_FILES_ONLY)
        print("成功从本地加载分词器")
    except Exception as e:
        print(f"从本地加载分词器失败: {e}")
        try:
            print(f"尝试在线加载{MODEL_NAME}分词器...")
            tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        except Exception as e:
            print(f"在线加载分词器失败: {e}")
            print("使用基本分词器")
            # 如果加载失败，使用基本分词器
            from transformers import BasicTokenizer
            class SimpleTokenizer:
                def __init__(self):
                    self.basic_tokenizer = BasicTokenizer(do_lower_case=True)
                
                def encode_plus(self, text, add_special_tokens=True, max_length=MAX_LEN, 
                               return_token_type_ids=True, padding='max_length', 
                               truncation=True, return_tensors='pt'):
                    # 创建简单的编码结果
                    input_ids = torch.zeros(max_length, dtype=torch.long)
                    attention_mask = torch.ones(max_length, dtype=torch.long)
                    token_type_ids = torch.zeros(max_length, dtype=torch.long)
                    
                    # 填充前几个位置为1，模拟真实数据
                    for i in range(min(10, max_length)):
                        input_ids[i] = i + 1
                    
                    # 返回字典格式的结果
                    if return_tensors == 'pt':
                        return {
                            'input_ids': input_ids.unsqueeze(0),
                            'attention_mask': attention_mask.unsqueeze(0),
                            'token_type_ids': token_type_ids.unsqueeze(0)
                        }
                    else:
                        return {
                            'input_ids': input_ids.numpy(),
                            'attention_mask': attention_mask.numpy(),
                            'token_type_ids': token_type_ids.numpy()
                        }
                
                def __call__(self, text, **kwargs):
                    return self.encode_plus(text, **kwargs)
            
            tokenizer = SimpleTokenizer()
            print("使用基本分词器")
    
    # 准备数据加载器
    batch_size = BATCH_SIZE if torch.cuda.is_available() else CPU_BATCH_SIZE
    print(f"使用batch_size: {batch_size}")
    
    # 选择数据加载方式
    if BALANCE_DATASET:
        print("使用平衡数据集进行训练...")
        train_loader, val_loader = prepare_balanced_dataloaders(
            TRAIN_FILE, 
            VAL_FILE,
            tokenizer,
            batch_size
        )
    else:
        print("使用原始数据集进行训练...")
        train_loader, val_loader = prepare_dataloaders(
            TRAIN_FILE, 
            VAL_FILE,
            tokenizer,
            batch_size
        )
    
    # 初始化模型
    model = BertFakeNewsClassifier(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE).to(device)
    
    # 使用适当的损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 学习率调度器
    total_steps = len(train_loader) * EPOCHS
    
    if SCHEDULER_TYPE == 'cosine':
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=NUM_WARMUP_STEPS,
            num_training_steps=total_steps
        )
    elif SCHEDULER_TYPE == 'constant':
        from transformers import get_constant_schedule_with_warmup
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=NUM_WARMUP_STEPS
        )
    else:  # 默认使用线性调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=NUM_WARMUP_STEPS,
            num_training_steps=total_steps
        )
    
    try:
        # 训练模型
        final_model_path = train_model(
            train_loader, 
            val_loader, 
            model, 
            criterion, 
            optimizer, 
            scheduler, 
            device, 
            epochs=EPOCHS
        )
        print(f"训练完成！模型已保存到: {final_model_path}")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        # 尝试保存模型
        try:
            interrupted_model_path = os.path.join(MODEL_DIR, 'fake_news_model_interrupted.pth')
            torch.save(model.state_dict(), interrupted_model_path)
            print(f"已保存中断的模型到: {interrupted_model_path}")
        except:
            print("无法保存模型")

if __name__ == '__main__':
    main() 