import os
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import jieba
from train import BertFakeNewsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter
import argparse
import time

# 忽略警告信息
warnings.filterwarnings('ignore')

# 设置统一的图表风格
plt.style.use('seaborn-v0_8-whitegrid')

class FakeNewsPredictor:
    def __init__(self, model_path='./models/fake_news_bert_model_final.pth'):
        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载分词器
        try:
            self.tokenizer = BertTokenizer.from_pretrained('./models/bert_local', local_files_only=True)
            print("成功加载本地分词器")
        except Exception as e:
            print(f"加载本地分词器失败: {e}")
            try:
                # 尝试多个可能的本地路径
                local_paths = [
                    './models/bert_local',
                    './models/bert-base-chinese',
                    os.path.abspath('./models/bert_local'),
                    os.path.abspath('./models/bert-base-chinese')
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
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("模型加载完成")
        except Exception as e:
            print(f"尝试从本地加载bert-base-chinese模型...")
            try:
                model_dict = torch.load(model_path, map_location=self.device)
                # 处理可能的键不匹配问题
                if isinstance(model_dict, dict) and 'model' in model_dict:
                    self.model.load_state_dict(model_dict['model'])
                else:
                    self.model.load_state_dict(model_dict)
                print("模型加载成功")
            except Exception as e2:
                print(f"从本地加载失败: {e2}")
                print("尝试在线加载bert-base-chinese模型...")
                # 如果还失败，可以考虑其他备选方案
                raise RuntimeError(f"模型加载失败: {e}")
        
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
    
    def predict_batch(self, texts, max_len=128):
        """批量预测多个文本
        
        Args:
            texts: 文本列表
            max_len: 最大序列长度
            
        Returns:
            fake_scores: 虚假新闻概率分数列表
        """
        fake_scores = []
        
        for text in texts:
            try:
                # 使用单个预测函数
                result = self.predict(text, max_len)
                # 获取虚假新闻的概率（假设索引1对应虚假新闻）
                fake_score = result['probabilities'][1]
                fake_scores.append(fake_score)
            except Exception as e:
                print(f"预测文本时出错: {e}")
                # 如果预测失败，添加0.5（不确定）
                fake_scores.append(0.5)
        
        return fake_scores

def load_test_data(file_path='./data/test.xls'):
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

def plot_confusion_matrix(y_true, y_pred, labels=['真实', '虚假']):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    # 设置中文字体支持
    try:
        font_path = fm.findfont(fm.FontProperties(family='SimHei'))
        if font_path:
            plt.rcParams['font.family'] = 'SimHei'
        else:
            print("警告: 找不到SimHei字体，中文可能无法正确显示")
    except Exception as e:
        print(f"设置字体时出错: {e}")
    
    # 使用seaborn绘制混淆矩阵，增大字体大小
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 16},  # 增大混淆矩阵中数字的字体
                cbar_kws={'label': '样本数量'})
    
    # 增大刻度标签字体
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # 添加标题和轴标签，增大字号
    plt.title('混淆矩阵', fontsize=20)
    plt.ylabel('真实标签', fontsize=16)
    plt.xlabel('预测标签', fontsize=16)
    
    # 显示图形
    plt.tight_layout()
    plt.show()
    
    # 保存一份到报告目录
    os.makedirs('./report', exist_ok=True)
    plt.savefig('./report/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"混淆矩阵也已保存至 './report/confusion_matrix.png'")

def plot_roc_curve(y_true, y_score, title='ROC Curve'):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    
    # 添加性能信息
    tpr_at_specific_fpr = np.interp(0.1, fpr, tpr)  # TPR at 10% FPR
    plt.text(0.6, 0.3, f'TPR at 10% FPR: {tpr_at_specific_fpr:.4f}', 
             bbox=dict(facecolor='white', alpha=0.8), fontsize=12)
    
    # 保存结果
    os.makedirs('./report', exist_ok=True)
    plt.tight_layout()
    plt.savefig('./report/roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"ROC曲线已保存至 './report/roc_curve.png'")
    plt.close()

def plot_precision_recall_curve(y_true, y_score, title='Precision-Recall Curve'):
    """绘制精确率-召回率曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    
    plt.figure(figsize=(10, 8))
    plt.step(recall, precision, color='b', alpha=0.8, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.text(0.6, 0.7, f'AP = {average_precision:.4f}', 
             bbox=dict(facecolor='white', alpha=0.8), fontsize=12)
    
    # 标记F1最优点
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    max_f1_idx = np.argmax(f1_scores)
    best_precision = precision[max_f1_idx]
    best_recall = recall[max_f1_idx]
    best_f1 = f1_scores[max_f1_idx]
    
    plt.plot(best_recall, best_precision, 'ro', markersize=8, label=f'Best F1: {best_f1:.4f}')
    plt.axvline(x=best_recall, linestyle='--', color='r', alpha=0.3)
    plt.axhline(y=best_precision, linestyle='--', color='r', alpha=0.3)
    plt.legend(loc="lower left", fontsize=12)
    
    # 保存结果
    os.makedirs('./report', exist_ok=True)
    plt.tight_layout()
    plt.savefig('./report/pr_curve.png', dpi=300, bbox_inches='tight')
    print(f"PR曲线已保存至 './report/pr_curve.png'")
    plt.close()

def plot_score_distribution(true_labels, fake_scores, title='Score Distribution'):
    """绘制分数分布图"""
    plt.figure(figsize=(12, 8))
    
    # 转换为numpy数组
    y_true = np.array(true_labels)
    scores = np.array(fake_scores)
    
    # 分离真实和虚假新闻的分数
    true_news_scores = scores[y_true == 0]
    fake_news_scores = scores[y_true == 1]
    
    # 设置直方图的参数
    bins = np.linspace(0, 1, 21)  # 20个均匀分布的bin从0到1
    alpha = 0.7
    
    # 绘制分布
    plt.hist(true_news_scores, bins=bins, alpha=alpha, color='green', label='True News')
    plt.hist(fake_news_scores, bins=bins, alpha=alpha, color='red', label='Fake News')
    
    plt.xlim([0.0, 1.0])
    plt.xlabel('Fake News Score', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Distribution of Prediction Scores', fontsize=16)
    plt.legend(fontsize=12)
    
    # 使用百分比格式显示X轴
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    # 添加统计信息
    true_news_avg = np.mean(true_news_scores)
    fake_news_avg = np.mean(fake_news_scores)
    
    info_text = (f"True News Avg: {true_news_avg:.2%}\n"
                 f"Fake News Avg: {fake_news_avg:.2%}\n"
                 f"Separation: {abs(fake_news_avg - true_news_avg):.2%}")
    
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
    
    # 保存结果
    os.makedirs('./report', exist_ok=True)
    plt.tight_layout()
    plt.savefig('./report/score_distribution.png', dpi=300, bbox_inches='tight')
    print(f"分数分布图已保存至 './report/score_distribution.png'")
    plt.close()

def generate_threshold_analysis(y_true, y_score):
    """生成阈值分析图"""
    # 计算不同阈值下的性能指标
    thresholds = np.linspace(0, 1, 101)  # 0到1，步长0.01
    metrics = []
    
    for threshold in thresholds:
        y_pred = (np.array(y_score) >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # 计算指标，处理分母为0的情况
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # 转换为DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # 绘制阈值vs指标
    plt.figure(figsize=(12, 8))
    
    plt.plot(metrics_df['threshold'], metrics_df['accuracy'], label='Accuracy', lw=2)
    plt.plot(metrics_df['threshold'], metrics_df['precision'], label='Precision', lw=2)
    plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall', lw=2)
    plt.plot(metrics_df['threshold'], metrics_df['f1'], label='F1', lw=2)
    
    # 找出F1最大的阈值
    best_f1_idx = metrics_df['f1'].idxmax()
    best_threshold = metrics_df.loc[best_f1_idx, 'threshold']
    best_f1 = metrics_df.loc[best_f1_idx, 'f1']
    
    plt.axvline(x=best_threshold, linestyle='--', color='r', alpha=0.5)
    plt.text(best_threshold + 0.02, 0.5, f'Best Threshold: {best_threshold:.2f}\nBest F1: {best_f1:.4f}', 
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Performance Metrics vs. Threshold', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 使用百分比格式显示X轴
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    # 保存结果
    os.makedirs('./report', exist_ok=True)
    plt.tight_layout()
    plt.savefig('./report/threshold_analysis.png', dpi=300, bbox_inches='tight')
    print(f"阈值分析图已保存至 './report/threshold_analysis.png'")
    
    # 保存最佳阈值数据
    best_metrics = metrics_df.loc[best_f1_idx].to_dict()
    print(f"最佳阈值: {best_threshold:.4f}, 对应指标: 精确率={best_metrics['precision']:.4f}, 召回率={best_metrics['recall']:.4f}, F1={best_metrics['f1']:.4f}")
    
    # 返回最佳阈值数据以便后续使用
    return best_threshold, best_metrics

def evaluate_model(model_path, test_data_path, config=None, output_path='./report',
                 threshold=0.5, save_results=True, verbose=True, batch_size=16):
    """评估假新闻检测模型的性能
    
    参数:
        model_path: 模型路径
        test_data_path: 测试数据路径
        config: 配置
        output_path: 输出报告路径
        threshold: 分类阈值
        save_results: 是否保存结果
        verbose: 是否打印详细信息
        batch_size: 批处理大小
    """
    print(f"正在评估模型: {model_path}")
    print(f"使用测试数据: {test_data_path}")
    
    # 确保输出目录存在
    if save_results:
        os.makedirs(output_path, exist_ok=True)
        print(f"结果将保存到: {output_path}")
    
    # 初始化预测器
    predictor = FakeNewsPredictor(model_path)
    
    # 加载测试数据
    test_data = load_test_data(test_data_path)
    if test_data is None:
        print("无法加载测试数据，评估中止")
        return None
    
    # 提取标签和文本
    # 将中文标签转换为数字标签
    label_mapping = {'事实': 0, '谣言': 1}
    
    # 检查标签类型并转换
    if 'label' in test_data.columns:
        if test_data['label'].dtype == 'object':
            print(f"标签值示例: {test_data['label'].value_counts().index.tolist()}")
            print("将中文标签转换为数字标签")
            # 转换标签
            if isinstance(test_data['label'].iloc[0], str):
                test_data['numeric_label'] = test_data['label'].map(lambda x: label_mapping.get(x, -1))
                print(f"标签映射: {label_mapping}")
                print(f"转换后的标签分布: {test_data['numeric_label'].value_counts()}")
                labels = test_data['numeric_label'].values
            else:
                labels = test_data['label'].values
        else:
            labels = test_data['label'].values
    else:
        print("错误: 找不到label列")
        return None
    
    # 选择文本列
    text_column = 'content' if 'content' in test_data.columns else 'text'
    if text_column not in test_data.columns:
        print(f"错误: 找不到文本列 ({text_column})")
        return None
        
    texts = test_data[text_column].values
    
    print(f"使用文本列: {text_column}")
    
    # 进行预测
    print("正在进行预测...")
    fake_scores, processing_times = [], []
    predictions = []
    
    # 批量预测以提高速度
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        start_time = time.time()
        batch_scores = predictor.predict_batch(batch_texts)
        end_time = time.time()
        
        # 记录预测结果
        batch_time = end_time - start_time
        fake_scores.extend(batch_scores)
        processing_times.extend([batch_time/len(batch_texts)] * len(batch_texts))
        
        # 使用阈值生成预测标签
        batch_preds = (np.array(batch_scores) >= threshold).astype(int)
        predictions.extend(batch_preds)
        
        if verbose and (i+batch_size) % 100 == 0:
            print(f"已处理 {i+len(batch_texts)}/{len(texts)} 条样本")
    
    # 计算性能指标
    predictions = np.array(predictions)
    fake_scores = np.array(fake_scores)
    
    # 计算分类报告
    classification_rep = classification_report(labels, predictions, target_names=['真实新闻', '虚假新闻'])
    
    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)
    
    # 准确率、精确率、召回率、F1
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    # ROC和AUC
    fpr, tpr, _ = roc_curve(labels, fake_scores)
    roc_auc = auc(fpr, tpr)
    
    # PR曲线和AP
    precision_curve, recall_curve, _ = precision_recall_curve(labels, fake_scores)
    ap = average_precision_score(labels, fake_scores)
    
    # 计算处理时间统计
    avg_time = np.mean(processing_times)
    total_time = sum(processing_times)
    
    # 打印结果
    if verbose:
        print("\n=== 性能评估结果 ===")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"AUC-ROC: {roc_auc:.4f}")
        print(f"平均精度 (AP): {ap:.4f}")
        print(f"平均处理时间: {avg_time*1000:.2f} ms/文本")
        print(f"总处理时间: {total_time:.2f} 秒 (共{len(texts)}条文本)")
        print("\n分类报告:")
        print(classification_rep)
        print("\n混淆矩阵:")
        print(cm)
    
    # 可视化结果
    if save_results:
        # 生成可视化图表
        plot_confusion_matrix(labels, predictions, labels=['真实', '虚假'])
        plot_roc_curve(labels, fake_scores)
        plot_precision_recall_curve(labels, fake_scores)
        plot_score_distribution(labels, fake_scores)
        
        # 生成阈值分析
        best_threshold, best_metrics = generate_threshold_analysis(labels, fake_scores)
        
        # 保存错误分析
        error_analysis_df = analyze_errors(test_data, predictions, fake_scores)
        error_analysis_path = os.path.join(output_path, 'error_analysis.xlsx')
        try:
            error_analysis_df.to_excel(error_analysis_path, index=False)
            print(f"错误分析已保存至 {error_analysis_path}")
        except Exception as e:
            print(f"保存错误分析时出错: {e}")
        
        # 保存详细预测结果
        results_df = test_data.copy()
        results_df['fake_score'] = fake_scores
        results_df['prediction'] = predictions
        results_df['correct'] = (results_df['label'] == results_df['prediction'])
        results_df['processing_time_ms'] = [t * 1000 for t in processing_times]
        
        results_path = os.path.join(output_path, 'prediction_results.xlsx')
        try:
            results_df.to_excel(results_path, index=False)
            print(f"预测结果已保存至 {results_path}")
        except Exception as e:
            print(f"保存预测结果时出错: {e}")
    
    # 返回评估结果字典
    evaluation_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'ap': ap,
        'avg_time_ms': avg_time * 1000,
        'total_time_sec': total_time,
        'sample_count': len(texts),
        'best_threshold': best_threshold,
        'threshold_metrics': best_metrics if save_results else None,
        'confusion_matrix': cm,
        'classification_report': classification_rep,
        'error_analysis': error_analysis_df if save_results else None,
        'full_results': results_df if save_results else None
    }
    
    return evaluation_results

def analyze_errors(data, predictions, scores, output_path=None):
    """分析预测错误
    
    Args:
        data: 测试数据
        predictions: 预测结果
        scores: 预测分数
        output_path: 可选的输出路径
    
    Returns:
        error_analysis_df: 错误分析结果
    """
    # 创建结果DataFrame
    results_df = data.copy()
    results_df['prediction'] = predictions
    results_df['fake_score'] = scores
    results_df['correct'] = (results_df['label'] == results_df['prediction'])
    
    # 分离正确和错误的预测
    correct_df = results_df[results_df['correct']]
    error_df = results_df[~results_df['correct']]
    
    print(f"正确预测: {len(correct_df)} 条 ({len(correct_df)/len(results_df):.2%})")
    print(f"错误预测: {len(error_df)} 条 ({len(error_df)/len(results_df):.2%})")
    
    # 分析错误类型
    false_positives = error_df[(error_df['label'] == 0) & (error_df['prediction'] == 1)]
    false_negatives = error_df[(error_df['label'] == 1) & (error_df['prediction'] == 0)]
    
    print(f"假阳性 (真实新闻被判断为虚假): {len(false_positives)} 条")
    print(f"假阴性 (虚假新闻被判断为真实): {len(false_negatives)} 条")
    
    # 置信度分析
    print("\n错误预测的置信度分析:")
    error_confidence = error_df.apply(
        lambda x: x['fake_score'] if x['prediction'] == 1 else 1 - x['fake_score'], 
        axis=1
    )
    
    confidence_bins = [0, 0.55, 0.7, 0.85, 0.95, 1.0]
    confidence_labels = ['0.5-0.55', '0.55-0.7', '0.7-0.85', '0.85-0.95', '0.95-1.0']
    
    binned_confidence = pd.cut(error_confidence, bins=confidence_bins, labels=confidence_labels)
    confidence_counts = binned_confidence.value_counts().sort_index()
    
    print("错误预测的置信度分布:")
    for label, count in confidence_counts.items():
        print(f"  {label}: {count} 条 ({count/len(error_df):.2%})")
    
    # 创建详细错误分析表格
    error_df['Error_Type'] = error_df.apply(
        lambda x: '假阳性 (FP)' if x['label'] == 0 else '假阴性 (FN)', 
        axis=1
    )
    
    error_df['Confidence'] = error_df.apply(
        lambda x: x['fake_score'] if x['prediction'] == 1 else 1 - x['fake_score'], 
        axis=1
    )
    
    # 计算文本长度
    text_column = 'content' if 'content' in error_df.columns else 'text'
    if text_column in error_df.columns:
        error_df['Text_Length'] = error_df[text_column].apply(len)
    
    # 选择要导出的列
    cols = ['Error_Type', 'Confidence', 'fake_score']
    if 'id' in error_df.columns:
        cols = ['id'] + cols
    if text_column in error_df.columns:
        cols = [text_column] + cols
    if 'label' in error_df.columns:
        cols = ['label'] + cols
    if 'prediction' in error_df.columns:
        cols.append('prediction')
    if 'Text_Length' in error_df.columns:
        cols.append('Text_Length')
    
    error_analysis_df = error_df[cols].sort_values('Confidence', ascending=False)
    
    # 如果提供了输出路径，保存结果
    if output_path:
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            error_analysis_df.to_excel(output_path, index=False)
            print(f"错误分析已保存到: {output_path}")
        except Exception as e:
            print(f"保存错误分析时出错: {e}")
    
    return error_analysis_df

def analyze_text_length_impact(results_df):
    """分析文本长度对预测准确率的影响"""
    print("\n===== 文本长度对预测准确率的影响 =====")
    
    # 添加文本长度列
    text_column = 'content' if 'content' in results_df.columns else 'text'
    if text_column in results_df.columns:
        results_df['Text_Length'] = results_df[text_column].apply(len)
    else:
        print(f"结果中无文本列({text_column})，无法分析长度影响")
        return None
    
    # 设置长度边界
    length_bins = [0, 50, 100, 200, 500, 1000, 2000, float('inf')]
    length_labels = ['0-50', '51-100', '101-200', '201-500', '501-1000', '1001-2000', '2000+']
    
    # 添加长度分组
    results_df['Length_Group'] = pd.cut(results_df['Text_Length'], bins=length_bins, labels=length_labels)
    
    # 按长度分组计算准确率
    length_group_stats = results_df.groupby('Length_Group').agg({
        'correct': ['count', 'mean'],
        'Text_Length': ['mean', 'min', 'max']
    })
    
    # 打印结果
    print("\n长度分组统计:")
    for group, stats in length_group_stats.iterrows():
        count = stats[('correct', 'count')]
        accuracy = stats[('correct', 'mean')]
        avg_length = stats[('Text_Length', 'mean')]
        
        print(f"长度 {group}: {count} 条样本, 平均长度 {avg_length:.1f}, 准确率 {accuracy:.4f}")
    
    # 绘制长度对准确率的影响
    plt.figure(figsize=(10, 6))
    
    # 获取每个组的样本数和准确率
    counts = length_group_stats[('correct', 'count')].values
    accuracies = length_group_stats[('correct', 'mean')].values
    
    # 主图：准确率
    plt.bar(length_labels, accuracies, color='skyblue', alpha=0.7)
    
    # 添加样本数量标签
    for i, (count, acc) in enumerate(zip(counts, accuracies)):
        plt.text(i, acc + 0.02, f'n={count}', ha='center', va='bottom')
    
    # 添加平均精度参考线
    avg_accuracy = results_df['correct'].mean()
    plt.axhline(y=avg_accuracy, color='r', linestyle='--', label=f'平均准确率: {avg_accuracy:.4f}')
    
    plt.xlabel('文本长度', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('文本长度对预测准确率的影响', fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(alpha=0.3)
    plt.legend()
    
    # 保存图表
    os.makedirs('./report', exist_ok=True)
    plt.tight_layout()
    plt.savefig('./report/length_impact.png', dpi=300)
    print("长度影响分析图已保存至 ./report/length_impact.png")
    plt.close()
    
    return length_group_stats

def generate_full_report(model_paths, test_data_path, configs=None, output_dir='./report',
                       thresholds=None, save_results=True, verbose=True):
    """为多个模型生成完整的评估报告
    
    Args:
        model_paths: 模型路径列表或字典 {模型名称: 模型路径}
        test_data_path: 测试数据路径
        configs: 模型配置列表或字典 {模型名称: 配置}
        output_dir: 输出目录
        thresholds: 分类阈值列表或字典 {模型名称: 阈值}
        save_results: 是否保存结果
        verbose: 是否打印详细信息
    """
    # 确保输出目录存在
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
    
    # 标准化输入
    if isinstance(model_paths, dict):
        model_dict = model_paths
    else:
        model_dict = {f"model_{i}": path for i, path in enumerate(model_paths)}
    
    if configs is None:
        configs = {name: None for name in model_dict}
    elif not isinstance(configs, dict):
        configs = {name: cfg for name, cfg in zip(model_dict.keys(), configs)}
    
    if thresholds is None:
        thresholds = {name: 0.5 for name in model_dict}
    elif not isinstance(thresholds, dict):
        thresholds = {name: thr for name, thr in zip(model_dict.keys(), thresholds)}
    
    # 存储所有评估结果
    all_results = {}
    summary_metrics = []
    
    # 评估每个模型
    for model_name, model_path in model_dict.items():
        print(f"\n=== 评估模型: {model_name} ===")
        
        # 为每个模型创建单独的输出目录
        model_output_dir = os.path.join(output_dir, model_name)
        
        # 进行评估
        results = evaluate_model(
            model_path=model_path,
            test_data_path=test_data_path,
            config=configs.get(model_name),
            output_path=model_output_dir,
            threshold=thresholds.get(model_name, 0.5),
            save_results=save_results,
            verbose=verbose
        )
        
        if results is not None:
            all_results[model_name] = results
            
            # 添加到摘要数据
            summary_metrics.append({
                'model_name': model_name,
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1': results['f1'],
                'auc': results['auc'],
                'ap': results['ap'],
                'avg_time_ms': results['avg_time_ms'],
                'best_threshold': results.get('best_threshold', thresholds.get(model_name, 0.5))
            })
    
    # 生成汇总报告
    if save_results和summary_metrics:
        summary_df = pd.DataFrame(summary_metrics)
        
        # 排序并格式化
        summary_df = summary_df.sort_values('f1', ascending=False)
        
        # 保存汇总表格
        summary_path = os.path.join(output_dir, 'models_summary.xlsx')
        try:
            summary_df.to_excel(summary_path, index=False)
            print(f"\n模型评估汇总已保存至 {summary_path}")
        except Exception as e:
            print(f"保存模型评估汇总时出错: {e}")
        
        # 生成汇总图表
        plt.figure(figsize=(12, 8))
        
        # 绘制各指标对比图
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'ap']
        x = np.arange(len(summary_df))
        width = 0.15
        offsets = np.linspace(-(len(metrics)-1)/2*width, (len(metrics)-1)/2*width, len(metrics))
        
        for i, metric in enumerate(metrics):
            plt.bar(x + offsets[i], summary_df[metric], width, label=metric)
        
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14)
        plt.xticks(x, summary_df['model_name'], rotation=45)
        plt.ylim(0, 1.05)
        plt.legend(loc='lower right')
        plt.grid(axis='y', alpha=0.3)
        
        # 保存图表
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, 'model_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"模型对比图已保存至 {comparison_path}")
        plt.close()
        
        # 绘制时间性能对比图
        plt.figure(figsize=(10, 6))
        plt.bar(summary_df['model_name'], summary_df['avg_time_ms'], color='skyblue')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Avg. Processing Time (ms)', fontsize=12)
        plt.title('Model Processing Time Comparison', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # 保存图表
        plt.tight_layout()
        time_comparison_path = os.path.join(output_dir, 'processing_time_comparison.png')
        plt.savefig(time_comparison_path, dpi=300, bbox_inches='tight')
        print(f"处理时间对比图已保存至 {time_comparison_path}")
        plt.close()
    
    return all_results

def main():
    print("===== 模型评估 =====")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='评估假新闻检测模型')
    parser.add_argument('--model', type=str, help='模型路径')
    parser.add_argument('--test', type=str, help='测试数据路径')
    parser.add_argument('--output', type=str, default='./report', help='输出目录路径')
    args = parser.parse_args()
    
    # 使用相对于当前目录(project)的路径
    model_path = args.model if args.model else './models/fake_news_bert_model_final.pth'
    test_file_path = args.test if args.test else './data/test.xls'
    output_dir = args.output
    
    print(f"模型路径: {model_path}")
    print(f"测试文件路径: {test_file_path}")
    print(f"输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 标准模型评估
    print("\n评估标准模型...")
    standard_results = evaluate_model(model_path, test_file_path)
    
    # 生成完整评估报告
    if standard_results and 'full_results' in standard_results:
        print("\n生成完整评估报告...")
        report_path = os.path.join(output_dir, 'evaluation_report.xlsx')
        
        # 修改: 使用字典格式{模型名称: 模型路径}传递参数
        model_dict = {"标准模型": model_path}
        generate_full_report(model_dict, test_file_path, output_dir=output_dir)
        
        print("\n===== 评估完成 =====")
        print(f"所有评估结果和可视化已保存至: {output_dir}")
    else:
        print("未获取到有效的评估结果，无法生成完整报告")

if __name__ == "__main__":
    main()