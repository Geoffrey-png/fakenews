#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试讯飞星火API连接
"""

import requests
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def test_spark_api():
    """测试讯飞星火API连接"""
    # API配置
    host = "https://spark-api-open.xf-yun.com/v1/chat/completions"
    api_key = "JXhEvuqgydysdhNpAudL:yUypnQjpWoFJcTgAvvbw"
    model = "4.0Ultra"  # 从文档中获取的正确模型名
    
    # 准备请求数据
    request_data = {
        "model": model,
        "messages": [
            {"role": "user", "content": "你好，请介绍一下你自己"}
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }
    
    # 构建请求头
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    logger.info("开始测试讯飞星火API连接...")
    logger.info(f"API地址: {host}")
    logger.info(f"模型: {model}")
    logger.info(f"请求数据: {json.dumps(request_data, ensure_ascii=False)}")
    logger.info(f"请求头: {headers}")
    
    try:
        # 发送请求
        response = requests.post(
            host,
            headers=headers,
            json=request_data
        )
        
        # 检查响应
        logger.info(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.info("API调用成功！")
            logger.info(f"模型回复: {content[:100]}...")
            return True
        else:
            logger.error(f"API调用失败: {response.status_code} {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"API调用异常: {str(e)}")
        return False

if __name__ == "__main__":
    test_spark_api() 