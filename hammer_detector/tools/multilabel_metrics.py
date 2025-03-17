#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多标签评估工具
用于评估多标签分类任务的性能
"""

import torch
import numpy as np


def get_multi_label(label, image):
    """
    将标签转换为多标签向量
    
    Args:
        label: 标签列表
        image: 图像张量(用于获取设备信息)
    
    Returns:
        multi_label: 多标签向量
        cls_map: 类别映射
    """
    multi_label = torch.zeros([len(label), 4], dtype=torch.long).to(image.device) 
    # 类别映射: [face_swap, face_attribute, text_swap, text_attribute]
    
    # 真实样本都为零向量
    real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
    multi_label[real_label_pos,:] = torch.tensor([0, 0, 0, 0]).to(image.device) 
    
    # 各种篡改类型的映射
    # face_swap (人脸替换)
    pos = np.where(np.array(label) == 'face_swap')[0].tolist() 
    multi_label[pos,:] = torch.tensor([1, 0, 0, 0]).to(image.device) 
    
    # face_attribute (人脸属性修改)
    pos = np.where(np.array(label) == 'face_attribute')[0].tolist()
    multi_label[pos,:] = torch.tensor([0, 1, 0, 0]).to(image.device) 
    
    # text_swap (文本替换)
    pos = np.where(np.array(label) == 'text_swap')[0].tolist()
    multi_label[pos,:] = torch.tensor([0, 0, 1, 0]).to(image.device) 
    
    # text_attribute (文本属性修改)
    pos = np.where(np.array(label) == 'text_attribute')[0].tolist()
    multi_label[pos,:] = torch.tensor([0, 0, 0, 1]).to(image.device) 
    
    # 组合类型
    # face_swap & text_swap
    pos = np.where(np.array(label) == 'face_swap&text_swap')[0].tolist()
    multi_label[pos,:] = torch.tensor([1, 0, 1, 0]).to(image.device) 
    
    # face_swap & text_attribute
    pos = np.where(np.array(label) == 'face_swap&text_attribute')[0].tolist()
    multi_label[pos,:] = torch.tensor([1, 0, 0, 1]).to(image.device) 
    
    # face_attribute & text_swap
    pos = np.where(np.array(label) == 'face_attribute&text_swap')[0].tolist()
    multi_label[pos,:] = torch.tensor([0, 1, 1, 0]).to(image.device) 
    
    # face_attribute & text_attribute
    pos = np.where(np.array(label) == 'face_attribute&text_attribute')[0].tolist()
    multi_label[pos,:] = torch.tensor([0, 1, 0, 1]).to(image.device)
    
    cls_map = {
        0: 'face_swap',
        1: 'face_attribute',
        2: 'text_swap',
        3: 'text_attribute'
    }
    
    return multi_label, cls_map


class AveragePrecisionMeter:
    """
    平均精度计算器
    
    用于计算多标签分类任务的平均精度(AP)
    """
    
    def __init__(self, difficult_examples=False):
        """
        初始化AP计算器
        
        Args:
            difficult_examples: 是否考虑困难样本
        """
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples
    
    def reset(self):
        """重置状态"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
    
    def add(self, output, target):
        """
        添加预测和目标
        
        Args:
            output: 模型输出
            target: 目标标签
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)
        
        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                f'输出张量的维度应为2，但得到了{output.dim()}'
            assert output.size(0) == target.size(0), \
                f'输出和目标的样本数不匹配: {output.size(0)} vs {target.size(0)}'
            assert output.size(1) == target.size(1), \
                f'输出和目标的类别数不匹配: {output.size(1)} vs {target.size(1)}'
        
        # 存储预测和目标
        if self.scores.numel() > 0:
            self.scores = torch.cat((self.scores, output.cpu()), 0)
            self.targets = torch.cat((self.targets, target.cpu()), 0)
        else:
            self.scores = output.cpu()
            self.targets = target.cpu()
    
    def value(self):
        """
        计算AP值
        
        Returns:
            ap: 各类别的平均精度
        """
        if self.scores.numel() == 0:
            return 0
        
        ap = torch.zeros(self.scores.size(1))
        # 对每个类别计算AP
        for k in range(self.scores.size(1)):
            # 提取当前类别的预测和目标
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            
            # 排除困难样本
            if not self.difficult_examples:
                masks = targets >= 0
                scores = scores[masks]
                targets = targets[masks]
            
            # 计算AP
            try:
                ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
            except:
                # 处理计算错误的情况
                ap[k] = 0
        
        return ap
    
    @staticmethod
    def average_precision(output, target, difficult_examples=False):
        """
        计算单个类别的AP
        
        Args:
            output: 预测分数
            target: 目标标签
            difficult_examples: 是否考虑困难样本
            
        Returns:
            precision_at_i: 平均精度
        """
        # 按照分数排序
        sorted, indices = torch.sort(output, dim=0, descending=True)
        
        # 计算精度
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        
        # 处理每个样本
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        
        # 处理没有正样本的情况
        if pos_count == 0:
            return 0.0
        
        # 计算平均精度
        precision_at_i /= pos_count
        
        return precision_at_i
    
    def overall(self):
        """
        计算整体性能指标
        
        Returns:
            OP, OR, OF1, CP, CR, CF1: 各项性能指标
        """
        if self.scores.numel() == 0:
            return 0, 0, 0, 0, 0, 0
        
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        
        # 计算各项指标
        try:
            return self.evaluation(scores, targets)
        except:
            # 处理错误情况
            return 0, 0, 0, 0, 0, 0
    
    def evaluation(self, scores_, targets_):
        """
        评估多标签分类性能
        
        Args:
            scores_: 预测分数
            targets_: 目标标签
            
        Returns:
            OP, OR, OF1, CP, CR, CF1: 各项性能指标
        """
        n, n_class = scores_.shape
        
        # 初始化计数器
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        
        # 对每个类别计算TP, FP, FN
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0  # 忽略-1标签
            Ng[k] = np.sum(targets == 1)  # 真实正例数
            Np[k] = np.sum(scores >= 0)  # 预测为正例数
            Nc[k] = np.sum(targets * (scores >= 0))  # 正确预测的正例数
        
        # 处理Np为0的情况
        Np[Np == 0] = 1  # 防止除以0
        
        # 处理没有正样本或预测为正的情况
        if np.sum(Np) == 0 or np.sum(Nc) == 0:
            return 0, 0, 0, 0, 0, 0
        
        # 计算整体精度(Overall Precision)
        OP = np.sum(Nc) / np.sum(Np)
        
        # 处理没有真实正例的情况
        if np.sum(Ng) == 0:
            return OP, 0, 0, 0, 0, 0
        
        # 计算整体召回率(Overall Recall)
        OR = np.sum(Nc) / np.sum(Ng)
        
        # 计算整体F1分数(Overall F1)
        if OP + OR == 0:
            OF1 = 0
        else:
            OF1 = (2 * OP * OR) / (OP + OR)
        
        # 处理单个类别Ng为0的情况
        Ng[Ng == 0] = 1  # 防止除以0
        
        # 计算类别平均精度(Class-wise Precision)
        CP = np.sum(Nc / Np) / n_class
        
        # 计算类别平均召回率(Class-wise Recall)
        CR = np.sum(Nc / Ng) / n_class
        
        # 计算类别平均F1分数(Class-wise F1)
        if CP + CR == 0:
            CF1 = 0
        else:
            CF1 = (2 * CP * CR) / (CP + CR)
        
        return OP, OR, OF1, CP, CR, CF1 