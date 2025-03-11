#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试讯飞星火API解释生成器
"""

import logging
import sys
from explanation_generator import ExplanationGenerator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test")

def test_explanation_generator():
    """测试解释生成器"""
    logger.info("开始测试解释生成器...")
    
    # 初始化解释生成器
    generator = ExplanationGenerator()
    
    # 测试虚假新闻案例
    fake_news = "震惊！某明星深夜现身酒吧，与神秘人密会3小时，知情人士透露对方是外星人"
    prediction = {
        "label": "虚假新闻",
        "confidence": 0.95
    }
    
    logger.info(f"测试新闻文本: {fake_news}")
    logger.info("正在生成解释...")
    
    # 生成解释
    explanation = generator.generate_explanation(fake_news, prediction)
    
    if explanation:
        logger.info(f"生成的解释: \n{explanation}")
        logger.info("测试成功！")
        return True
    else:
        logger.error("解释生成失败！")
        return False

if __name__ == "__main__":
    success = test_explanation_generator()
    sys.exit(0 if success else 1) 