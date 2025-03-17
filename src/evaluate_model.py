import os
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import jieba
from train import BertFakeNewsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.font_manager as fm

# 忽略警告信息
warnings.filterwarnings('ignore')

class FakeNewsPredictor:
    def __init__(self, model_path='../models/fake_news_bert_model_final.pth'):
        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载分词器
        try:
            self.tokenizer = BertTokenizer.from_pretrained('../models/bert_local', local_files_only=True)
            print("成功加载本地分词器")
        except Exception as e:
            print(f"加载本地分词器失败: {e}")
            try:
                # 尝试多个可能的本地路径
                local_paths = [
                    './models/bert_local',
                    '../models/bert-base-chinese',
                    './models/bert-base-chinese',
                    os.path.abspath('../models/bert_local'),
                    os.path.abspath('./models/bert_local')
                ]
                
                for path in local_paths:
                    if os.path.exists(path):
                        try:
                            print(f"尝试从路径加载分词器: {path}")
                            self.tokenizer = BertTokenizer.from_pretrained(path, local_files_only=True)
                            print(f"从 {path} 成功加载分词器")
                            break
                        except Exception as e2:
                            print(f"从 {path} 加载失败: {e2}")
                else:  # 如果所有本地路径都失败
                    print("尝试在线下载分词器...")
                    self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                    print("在线下载分词器成功")
            except Exception as e3:
                print(f"所有尝试都失败: {e3}")
                raise RuntimeError("无法加载分词器")
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = BertFakeNewsClassifier().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("模型加载完成")
    
    def predict(self, text, max_len=128):
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

def load_test_data(file_path='../data/test.xls'):
    """加载测试数据集"""
    print(f"加载测试数据: {file_path}")
    
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到测试文件: {file_path}")
        
        # 根据文件扩展名决定读取方式
        if file_path.endswith('.xls') or file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        print(f"测试数据集形状: {df.shape}")
        print(f"测试数据集列名: {list(df.columns)}")
        
        # 检查数据
        print("\n数据集前5行:")
        print(df.head())
        
        # 对于特殊格式的数据，进行重命名和格式转换
        if len(df.columns) == 3 and '谣言' in df.columns:
            print("\n检测到特殊格式的测试数据，进行转换...")
            # 重命名列
            new_columns = ['id', 'label', 'content']
            df.columns = new_columns
            print("列名已重命名为:", new_columns)
        
        # 检查标签分布
        if 'label' in df.columns:
            print("\n标签分布:")
            print(df['label'].value_counts())
        
        return df
    
    except Exception as e:
        print(f"加载测试数据时出错: {e}")
        raise e

def plot_confusion_matrix(y_true, y_pred, labels=None, title='混淆矩阵'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    # 尝试设置多种可能的中文字体
    if labels and len(labels) == 2:
        # 将中文标签转换为英文
        eng_labels = ['True News', 'Fake News']
        eng_title = 'Confusion Matrix'
    else:
        eng_labels = labels
        eng_title = 'Confusion Matrix'
    
    # 绘制混淆矩阵
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=eng_labels, yticklabels=eng_labels, annot_kws={"size": 14})
    
    # 添加百分比标签（只使用数字和英文符号）
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.7, f'{cm_norm[i, j]:.2%}', 
                    ha='center', va='center', color='black' if cm_norm[i, j] < 0.7 else 'white', fontsize=12)
    
    # 使用英文标签
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(eng_title, fontsize=16)
    
    # 在图表中添加额外的信息（全英文）
    textstr = f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n"
    textstr += f"True News Recognition: {sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)/sum(1 for t in y_true if t == 0):.4f}\n"
    textstr += f"Fake News Recognition: {sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)/sum(1 for t in y_true if t == 1):.4f}"
    
    # 放置文本框
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.figtext(0.15, 0.05, textstr, fontsize=12, bbox=props)
    
    # 调整图表边距
    plt.tight_layout()
    
    # 保存高质量图片
    plt.savefig('../confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存至 '../confusion_matrix.png'")
    
    # 保存一份到报告目录
    os.makedirs('../report', exist_ok=True)
    plt.savefig('../report/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"混淆矩阵也已保存至 '../report/confusion_matrix.png'")

def evaluate_model(model_path, test_file_path):
    """评估模型在测试集上的表现"""
    # 加载预测器
    predictor = FakeNewsPredictor(model_path)
    
    # 加载测试数据
    test_df = load_test_data(test_file_path)
    
    # 准备输入和输出
    texts = []
    true_labels = []
    
    # 确定文本和标签列
    text_column = None
    label_column = None
    
    # 尝试查找文本列和标签列
    for col in test_df.columns:
        if col.lower() in ['content', 'text', 'title', '内容', '标题']:
            text_column = col
            break
    
    for col in test_df.columns:
        if col.lower() in ['label', 'class', 'target', '标签', '谣言']:
            label_column = col
            break
    
    if not text_column and len(test_df.columns) >= 3:
        print("未找到标准文本列名，使用第三列作为文本内容")
        text_column = test_df.columns[2]
    
    if not label_column and '谣言' in test_df.columns:
        print("使用'谣言'列作为标签")
        label_column = '谣言'
    
    if not text_column:
        raise ValueError("找不到文本列，请检查测试数据格式")
    
    print(f"使用 '{text_column}' 作为文本列，'{label_column}' 作为标签列")
    
    # 收集文本和标签
    original_ids = []
    for idx, row in test_df.iterrows():
        # 获取文本内容
        text = row[text_column] if pd.notna(row[text_column]) else ""
        if not text and 'title' in test_df.columns and pd.notna(row['title']):
            text = row['title']  # 如果内容为空，使用标题
        
        if not text:
            continue  # 跳过没有文本的行
            
        # 保存原始ID
        if 'id' in test_df.columns:
            original_ids.append(row['id'])
        else:
            original_ids.append(idx)
        
        texts.append(text)
        
        # 获取标签（如果有）
        if label_column and pd.notna(row[label_column]):
            label_text = row[label_column]
            # 将标签文本转换为数值
            if label_text in ['真实', '事实', '0', 0]:
                true_labels.append(0)
            elif label_text in ['虚假', '谣言', '1', 1]:
                true_labels.append(1)
            else:
                print(f"未知标签: {label_text}，跳过")
                texts.pop()  # 移除对应的文本
                original_ids.pop()  # 移除对应的ID
    
    # 打印信息
    print(f"评估数据数量: {len(texts)}")
    
    # 进行预测
    predictions = []
    probabilities = []
    
    print("开始预测...")
    for i, text in enumerate(texts):
        if i % 50 == 0:
            print(f"已处理 {i}/{len(texts)} 条数据")
        
        result = predictor.predict(text)
        predictions.append(result['class'])
        probabilities.append(result['probabilities'])
    
    # 创建结果DataFrame用于导出
    results_df = pd.DataFrame({
        'ID': original_ids,
        'Content': texts,
        'True_Label': true_labels,
        'Predicted_Label': predictions,
        'True_News_Probability': [prob[0] for prob in probabilities],
        'Fake_News_Probability': [prob[1] for prob in probabilities]
    })
    
    # 添加中文标签和正确性标记
    results_df['True_Label_Text'] = results_df['True_Label'].map({0: '真实新闻', 1: '虚假新闻'})
    results_df['Predicted_Label_Text'] = results_df['Predicted_Label'].map({0: '真实新闻', 1: '虚假新闻'})
    results_df['Correct'] = results_df['True_Label'] == results_df['Predicted_Label']
    
    # 保存结果到CSV
    os.makedirs('../results', exist_ok=True)
    csv_path = '../results/prediction_results.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # 使用utf-8-sig以支持Excel中文显示
    print(f"\n预测结果已保存至: {csv_path}")
    
    # 尝试保存为Excel格式
    try:
        excel_path = '../results/prediction_results.xlsx'
        results_df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"预测结果也已保存为Excel格式: {excel_path}")
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")
    
    # 如果有真实标签，计算评估指标
    if true_labels:
        print("\n===== 评估结果 =====")
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
        
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(true_labels, predictions, target_names=['真实新闻', '虚假新闻']))
        
        # 绘制混淆矩阵
        plot_confusion_matrix(true_labels, predictions, labels=['真实新闻', '虚假新闻'])
    
    # 返回预测结果
    return {
        'texts': texts,
        'true_labels': true_labels,
        'predictions': predictions,
        'probabilities': probabilities,
        'results_df': results_df
    }

def main():
    print("===== 模型评估 =====")
    
    # 标准模型评估
    print("\n评估标准模型...")
    standard_results = evaluate_model('../models/fake_news_bert_model_final.pth', '../data/test.xls')
    
    # 由于没有平衡模型，我们跳过这部分评估
    # print("\n评估平衡模型...")
    # balanced_results = evaluate_model('../models/balanced_fake_news_model.pth', '../data/test.xls')
    
    # 比较结果
    if standard_results['true_labels']:
        print("\n===== 模型性能 =====")
        
        # 计算准确率
        standard_acc = accuracy_score(standard_results['true_labels'], standard_results['predictions'])
        
        print(f"模型准确率: {standard_acc:.4f}")
        
        # 计算类别识别准确率
        true_correct = sum(1 for t, p in zip(standard_results['true_labels'], standard_results['predictions']) 
                         if t == 0 and p == 0)
        true_total = sum(1 for t in standard_results['true_labels'] if t == 0)
        
        fake_correct = sum(1 for t, p in zip(standard_results['true_labels'], standard_results['predictions']) 
                         if t == 1 and p == 1)
        fake_total = sum(1 for t in standard_results['true_labels'] if t == 1)
        
        if true_total > 0:
            print(f"真实新闻识别准确率: {true_correct/true_total:.4f} ({true_correct}/{true_total})")
        if fake_total > 0:
            print(f"虚假新闻识别准确率: {fake_correct/fake_total:.4f} ({fake_correct}/{fake_total})")

if __name__ == "__main__":
    main() 