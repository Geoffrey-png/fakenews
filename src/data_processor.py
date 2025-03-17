import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import jieba
from sklearn.model_selection import train_test_split
import os
import warnings

try:
    from config import MAX_LEN, NUM_WORKERS, SEED
except ImportError:
    # 默认配置
    MAX_LEN = 64
    NUM_WORKERS = 0
    SEED = 42

# 忽略警告信息
warnings.filterwarnings('ignore')

class FakeNewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=MAX_LEN):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 打印数据集信息
        print(f"数据集大小: {len(self.data)}")
        print(f"数据集列名: {list(self.data.columns)}")
        print(f"标签分布: {self.data['label'].value_counts()}")
        print(f"最大序列长度: {self.max_len}")
        
        # 清理数据
        self._clean_data()
        
    def _clean_data(self):
        """清理数据集中的问题"""
        # 填充缺失的content
        if 'content' in self.data.columns:
            null_content = self.data['content'].isnull()
            if null_content.any():
                print(f"发现{null_content.sum()}条缺失内容的记录，使用标题替代")
                for idx in self.data[null_content].index:
                    if pd.notna(self.data.loc[idx, 'title']):
                        self.data.loc[idx, 'content'] = self.data.loc[idx, 'title']
                    else:
                        self.data.loc[idx, 'content'] = "无内容"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if 'content' not in self.data.columns:
            raise ValueError(f"数据集缺少'content'列，现有列: {list(self.data.columns)}")
        
        # 获取文本，确保有内容
        if pd.isna(self.data.iloc[idx]['content']):
            text = "无内容"
        else:
            text = str(self.data.iloc[idx]['content'])
        
        # 如果文本太长，截断
        if len(text) > 200:  
            text = text[:200]
            
        if 'label' not in self.data.columns:
            raise ValueError(f"数据集缺少'label'列，现有列: {list(self.data.columns)}")
            
        # 处理标签
        label_text = self.data.iloc[idx]['label']
        if label_text == '谣言' or label_text == '虚假' or label_text == 1 or label_text == '1':
            label = 1
        elif label_text == '事实' or label_text == '真实' or label_text == 0 or label_text == '0':
            label = 0
        else:
            # 处理其他可能的标签
            print(f"遇到未知标签: {label_text}，默认为0")
            label = 0
        
        # 中文分词
        try:
            text = ' '.join(list(jieba.cut(text)))
        except Exception as e:
            print(f"分词失败: {e}, 文本: {text[:20]}...")
            text = ' '.join(list(text))
        
        # 编码
        try:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=True,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"编码失败: {e}, 文本: {text[:20]}...")
            # 返回一个空白编码
            return {
                'input_ids': torch.zeros(self.max_len, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_len, dtype=torch.long),
                'token_type_ids': torch.zeros(self.max_len, dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.long)
            }

def load_data(file_path, max_samples=None):
    """
    加载数据集
    """
    print(f"加载数据集: {file_path}")
    try:
        if not os.path.exists(file_path):
            # 尝试修正路径
            base_name = os.path.basename(file_path)
            corrected_path = os.path.join('..', 'data', base_name)
            if os.path.exists(corrected_path):
                print(f"使用修正后的路径: {corrected_path}")
                file_path = corrected_path
            else:
                print(f"修正后的路径也不存在: {corrected_path}")
                raise FileNotFoundError(f"找不到数据文件: {file_path}")
                
        df = pd.read_csv(file_path)
        print(f"原始数据集形状: {df.shape}")
        
        # 如果指定了最大样本数，则限制数据集大小
        if max_samples and max_samples < len(df):
            print(f"限制数据集大小为 {max_samples} 条记录")
            df = df.sample(max_samples, random_state=SEED)
            
        print(f"使用的数据集形状: {df.shape}")
        
        # 检查是否有空值
        null_count = df.isnull().sum()
        print("空值统计:")
        print(null_count)
        
        return df
    except Exception as e:
        print(f"加载数据时出错: {e}")
        # 创建一个简单的测试数据集
        print("创建简单的测试数据集")
        test_data = {
            'content': [
                '这是一条真实的新闻，内容经过严格核实。',
                '震惊！某知名明星竟然做出这种惊人之事！',
                '这是一条新闻',
                '这是一条假新闻'
            ],
            'label': ['真实', '谣言', '真实', '谣言']
        }
        return pd.DataFrame(test_data)

def prepare_dataloaders(train_path, val_path, tokenizer, batch_size=16, max_samples=None):
    """
    准备训练和验证数据加载器
    """
    print(f"准备数据加载器...")
    train_df = load_data(train_path, max_samples)
    
    # 检查数据的格式
    print(f"训练数据前5行:")
    print(train_df.head())
    
    # 随机划分训练集和验证集
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=SEED)
    
    print("创建训练集...")
    train_dataset = FakeNewsDataset(train_df, tokenizer)
    print("创建验证集...")
    val_dataset = FakeNewsDataset(val_df, tokenizer)
    
    # 创建数据加载器，添加错误处理
    print(f"使用batch_size={batch_size}创建数据加载器")
    try:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=NUM_WORKERS
        )
    except Exception as e:
        print(f"创建数据加载器失败: {e}，尝试减小batch_size")
        batch_size = max(1, batch_size // 2)
        print(f"使用减小后的batch_size={batch_size}")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )
    
    return train_loader, val_loader 