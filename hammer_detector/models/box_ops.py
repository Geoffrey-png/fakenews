#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
边界框操作模块
提供边界框坐标转换和IoU计算等功能
"""

import torch


def box_cxcywh_to_xyxy(x):
    """
    将中心点-宽高格式的边界框(cx, cy, w, h)转换为左上-右下格式(x1, y1, x2, y2)
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """
    将左上-右下格式的边界框(x1, y1, x2, y2)转换为中心点-宽高格式(cx, cy, w, h)
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2, test=False):
    """
    计算两组边界框之间的IoU(交并比)
    
    Args:
        boxes1: 第一组边界框，格式为(x1, y1, x2, y2)
        boxes2: 第二组边界框，格式为(x1, y1, x2, y2)
        test: 是否处于测试模式
        
    Returns:
        iou: IoU值
        union: 并集区域
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    if test:
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        union = area1 + area2 - inter
        iou = inter / union
        return iou, union
    else:
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        union = area1[:, None] + area2 - inter
        iou = inter / union
        return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    计算广义IoU，用于更精确的边界框匹配
    """
    # IoU计算
    iou, union = box_iou(boxes1, boxes2)

    # 最小闭包区域
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    # GIoU计算
    giou = iou - (area - union) / area
    return giou


def box_area(boxes):
    """
    计算边界框的面积
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) 