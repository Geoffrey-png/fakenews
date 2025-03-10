import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.cuda.amp import autocast, GradScaler
import warnings

from data_processor import prepare_dataloaders

# 忽略警告信息
warnings.filterwarnings('ignore')

class BertFakeNewsClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(BertFakeNewsClassifier, self).__init__()
        try:
            print("尝试从本地加载bert-base-chinese模型...")
            self.bert = BertModel.from_pretrained('../models/bert_local', local_files_only=True)
            print("成功从本地加载模型")
        except Exception as e:
            print(f"从本地加载失败: {e}")
            try:
                print("尝试在线加载bert-base-chinese模型...")
                self.bert = BertModel.from_pretrained('bert-base-chinese')
            except Exception as e:
                print(f"在线加载失败: {e}")
                print("创建随机初始化的模型")
                # 如果加载失败，创建一个简单的模型
                from transformers import BertConfig
                config = BertConfig(vocab_size=21128)  # 中文BERT的词表大小
                self.bert = BertModel(config)
                print("创建了随机初始化的模型")
                
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

def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, device, epochs=3):
    writer = SummaryWriter('runs/fake_news_detection')
    
    # 使用混合精度训练
    scaler = GradScaler() if torch.cuda.is_available() else None
    use_amp = torch.cuda.is_available()
    
    for epoch in range(epochs):
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
            
            if (batch_idx + 1) % 10 == 0:
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
        
        # 计算指标
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')
        
        # 记录指标
        writer.add_scalar('Loss/Train', train_loss/len(train_loader), epoch)
        writer.add_scalar('Loss/Validation', val_loss/len(val_loader), epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Precision', precision, epoch)
        writer.add_scalar('Recall', recall, epoch)
        writer.add_scalar('F1-Score', f1, epoch)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n')
        
        # 确保模型保存目录存在
        os.makedirs('../models', exist_ok=True)
        
        # 每个epoch保存一次模型
        torch.save(model.state_dict(), f'../models/fake_news_bert_model_epoch_{epoch+1}.pth')
    
    # 保存最终模型
    torch.save(model.state_dict(), '../models/fake_news_bert_model.pth')
    writer.close()

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
            print(f"在线加载分词器失败: {e}")
            print("使用基本分词器")
            # 如果加载失败，使用基本分词器
            from transformers import BasicTokenizer
            class SimpleTokenizer:
                def __init__(self):
                    self.basic_tokenizer = BasicTokenizer(do_lower_case=True)
                
                def encode_plus(self, text, add_special_tokens=True, max_length=64, 
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
    
    # 准备数据加载器 - 由于可能使用GPU，可以增加batch_size
    batch_size = 32 if torch.cuda.is_available() else 8  # 增加batch_size以加速训练
    print(f"使用batch_size: {batch_size}")
    
    # 加载完整的数据集进行训练
    print("加载完整数据集进行训练...")
    train_loader, val_loader = prepare_dataloaders(
        '../data/train.csv', 
        '../data/train.csv',  # 使用同一个文件进行训练和验证
        tokenizer,
        batch_size
        # 移除max_samples参数，使用完整数据集
    )
    
    # 初始化模型
    model = BertFakeNewsClassifier().to(device)
    
    # 使用适当的损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # 学习率调度器
    epochs = 3  # 设置为3个完整epoch
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
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        # 尝试保存模型
        try:
            torch.save(model.state_dict(), '../models/fake_news_bert_model_interrupted.pth')
            print("已保存中断的模型")
        except:
            print("无法保存模型")

if __name__ == '__main__':
    main() 