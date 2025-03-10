#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import json
import time
from pprint import pprint

# 配置API地址
API_URL = "http://localhost:5000"

def test_health_check():
    """测试健康检查端点"""
    print("\n=== 测试健康检查端点 ===")
    response = requests.get(f"{API_URL}/health")
    pprint(response.json())
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    print("健康检查测试通过!")

def test_single_prediction():
    """测试单文本预测端点"""
    print("\n=== 测试单文本预测端点 ===")
    
    # 真实新闻示例
    true_news = "北京冬奥会2022年2月4日开幕，中国代表团获得9金4银2铜的成绩。"
    
    # 假新闻示例
    fake_news = "震惊！某明星深夜现身酒吧，与神秘人密会3小时"
    
    # 测试真实新闻
    print("\n测试真实新闻:")
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": true_news}
    )
    pprint(response.json())
    assert response.status_code == 200
    assert response.json()["success"] == True
    print(f"真实新闻预测结果: {response.json()['prediction']['label']}")
    print(f"置信度: {response.json()['prediction']['confidence']}")
    
    # 测试假新闻
    print("\n测试假新闻:")
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": fake_news}
    )
    pprint(response.json())
    assert response.status_code == 200
    assert response.json()["success"] == True
    print(f"假新闻预测结果: {response.json()['prediction']['label']}")
    print(f"置信度: {response.json()['prediction']['confidence']}")
    
    print("单文本预测测试通过!")

def test_batch_prediction():
    """测试批量预测端点"""
    print("\n=== 测试批量预测端点 ===")
    
    test_texts = [
        "北京冬奥会2022年2月4日开幕，中国代表团获得9金4银2铜的成绩。",
        "震惊！某明星深夜现身酒吧，与神秘人密会3小时",
        "今天天气晴朗，适合户外活动。",
        "中国科学院发布最新研究成果，在量子计算领域取得重大突破。",
        "惊人发现：喝热水能治百病，医学界震惊",
    ]
    
    response = requests.post(
        f"{API_URL}/batch_predict",
        json={"texts": test_texts}
    )
    
    print("批量预测响应状态码:", response.status_code)
    json_response = response.json()
    print(f"处理时间: {json_response.get('processing_time', '未知')}秒")
    print("预测结果:")
    
    for i, (text, result) in enumerate(zip(test_texts, json_response.get("results", []))):
        print(f"\n{i+1}. 文本: {text[:50]}...")
        if result.get("success"):
            pred = result["prediction"]
            print(f"   预测: {pred['label']} (置信度: {pred['confidence'][pred['label']]:.4f})")
        else:
            print(f"   预测失败: {result.get('error', '未知错误')}")
    
    assert response.status_code == 200
    assert json_response["success"] == True
    assert len(json_response["results"]) == len(test_texts)
    
    print("批量预测测试通过!")

def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    # 测试空文本
    print("\n测试空文本:")
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": ""}
    )
    pprint(response.json())
    assert response.status_code == 400
    assert response.json()["success"] == False
    
    # 测试无效JSON
    print("\n测试无效JSON:")
    response = requests.post(
        f"{API_URL}/predict",
        data="这不是JSON",
        headers={"Content-Type": "application/json"}
    )
    pprint(response.json())
    assert response.status_code == 400
    assert response.json()["success"] == False
    
    print("错误处理测试通过!")

def run_all_tests():
    """运行所有测试"""
    print("开始运行API测试...\n")
    start_time = time.time()
    
    try:
        test_health_check()
        test_single_prediction()
        test_batch_prediction()
        test_error_handling()
        
        end_time = time.time()
        print(f"\n所有测试通过! 耗时: {end_time - start_time:.2f}秒")
    except Exception as e:
        print(f"\n测试失败: {e}")
        raise

if __name__ == "__main__":
    run_all_tests() 