#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HAMMER 工具函数模块
提供图像和文本预处理以及评估指标记录相关的工具函数
"""

import os
import time
import datetime
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def preprocess_image(image_path, image_size=256):
    """
    预处理图像，转换为模型输入格式
    
    参数:
        image_path: 图像路径
        image_size: 目标图像大小
        
    返回:
        tensor: 预处理后的图像张量 [C, H, W]
    """
    # 图像转换器
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 加载并转换图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    
    return image_tensor


def preprocess_text(text, max_length=77, tokenizer=None):
    """
    预处理文本，转换为模型输入格式
    
    参数:
        text: 输入文本
        max_length: 最大文本长度
        tokenizer: BERT分词器
        
    返回:
        dict: 包含input_ids和attention_mask的字典
    """
    if not text:
        # 如果没有文本，返回空张量
        return {
            'input_ids': torch.zeros((1, max_length), dtype=torch.long),
            'attention_mask': torch.zeros((1, max_length), dtype=torch.long)
        }
    
    # 确保有tokenizer
    if tokenizer is None:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 文本编码
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return encoded


class MetricLogger:
    """
    指标记录器，用于跟踪和记录训练/评估过程中的各种指标
    """
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter
        
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)
            
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.val, meter.avg)
            )
        return self.delimiter.join(loss_str)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
            
    def add_meter(self, name, meter):
        self.meters[name] = meter
        
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if header is not None:
            print(header)
            
        start_time = time.time()
        end = time.time()
        iter_time = AverageMeter()
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        
        log_msg = self.delimiter.join([
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}"
        ])
        
        for obj in iterable:
            yield obj
            iter_time.update(time.time() - end)
            
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                print(log_msg.format(
                    i, len(iterable),
                    eta=eta_string,
                    meters=str(self),
                    time=str(iter_time),
                    data=str(time.time() - end)
                ))
                
            i += 1
            end = time.time()
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("{} Total time: {} ({:.4f} s / it)".format(
            header, total_time_str, total_time / len(iterable)
        ))


class AverageMeter:
    """
    平均值计算器，用于跟踪指标的均值
    """
    def __init__(self, name=None):
        self.name = name
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def synchronize_between_processes(self):
        """
        在多进程环境中同步指标（用于分布式训练）
        """
        if not torch.distributed.is_available():
            return
            
        if not torch.distributed.is_initialized():
            return
            
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        
        t = t.tolist()
        self.sum = t[0]
        self.count = int(t[1])
        self.avg = self.sum / self.count 