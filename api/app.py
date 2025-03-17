#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import os
import json
import torch
import logging
from logging.handlers import RotatingFileHandler
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入预测模型
from api.model_loader import FakeNewsPredictor

# 创建应用实例
app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 配置日志
if not os.path.exists('logs'):
    os.makedirs('logs')

file_handler = RotatingFileHandler('logs/api.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('API服务启动')

# 全局变量
predictor = None

# 加载模型函数
def load_model():
    """加载模型"""
    global predictor
    try:
        app.logger.info('正在加载模型...')
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'fake_news_bert_model_final.pth'))
        predictor = FakeNewsPredictor(model_path)
        app.logger.info('模型加载成功')
    except Exception as e:
        app.logger.error(f'模型加载失败: {str(e)}')
        raise

# 在请求前确保模型已加载
@app.before_request
def ensure_model_loaded():
    global predictor
    if predictor is None:
        try:
            load_model()
        except Exception as e:
            app.logger.error(f'请求处理前模型加载失败: {str(e)}')

# 尝试在应用启动时加载模型
try:
    load_model()
except Exception as e:
    app.logger.error(f'应用启动时模型加载失败: {str(e)}')
    pass  # 让应用继续启动，在第一次请求时再次尝试加载

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time(),
        'model_loaded': predictor is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    接收新闻文本并返回预测结果
    请求格式: { "text": "新闻内容..." }
    """
    global predictor
    
    # 如果模型未加载，尝试加载
    if predictor is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'模型加载失败: {str(e)}'
            }), 500
    
    # 获取请求数据
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({
            'success': False,
            'error': '无效的请求数据，需要JSON格式'
        }), 400
    
    # 获取文本
    text = data.get('text', '')
    if not text:
        return jsonify({
            'success': False,
            'error': '请提供新闻文本'
        }), 400
    
    try:
        # 记录请求
        app.logger.info(f'收到预测请求: {text[:100]}{"..." if len(text) > 100 else ""}')
        
        # 执行预测
        start_time = time.time()
        result = predictor.predict(text)
        end_time = time.time()
        
        # 构建响应
        predicted_class = result['class']
        probabilities = result['probabilities']
        
        response = {
            'success': True,
            'prediction': {
                'label': '真实新闻' if predicted_class == 0 else '虚假新闻',
                'label_id': predicted_class,
                'confidence': {
                    '真实新闻': float(probabilities[0]),
                    '虚假新闻': float(probabilities[1])
                }
            },
            'processing_time': end_time - start_time
        }
        
        # 记录结果
        app.logger.info(f'预测结果: {response["prediction"]["label"]} ' 
                       f'(置信度: {response["prediction"]["confidence"][response["prediction"]["label"]]:.4f})')
        
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f'预测过程中出错: {str(e)}')
        return jsonify({
            'success': False,
            'error': f'预测过程中出错: {str(e)}'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    批量处理多个新闻文本
    请求格式: { "texts": ["新闻1", "新闻2", ...] }
    """
    global predictor
    
    # 如果模型未加载，尝试加载
    if predictor is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'模型加载失败: {str(e)}'
            }), 500
    
    # 获取请求数据
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({
            'success': False,
            'error': '无效的请求数据，需要JSON格式'
        }), 400
    
    # 获取文本列表
    texts = data.get('texts', [])
    if not texts or not isinstance(texts, list):
        return jsonify({
            'success': False,
            'error': '请提供有效的新闻文本列表'
        }), 400
    
    try:
        # 记录请求
        app.logger.info(f'收到批量预测请求: {len(texts)}篇文章')
        
        # 执行预测
        start_time = time.time()
        results = []
        
        for text in texts:
            if not text:
                results.append({
                    'success': False,
                    'error': '空文本'
                })
                continue
                
            result = predictor.predict(text)
            predicted_class = result['class']
            probabilities = result['probabilities']
            
            results.append({
                'success': True,
                'prediction': {
                    'label': '真实新闻' if predicted_class == 0 else '虚假新闻',
                    'label_id': predicted_class,
                    'confidence': {
                        '真实新闻': float(probabilities[0]),
                        '虚假新闻': float(probabilities[1])
                    }
                }
            })
        
        end_time = time.time()
        
        # 构建响应
        response = {
            'success': True,
            'results': results,
            'processing_time': end_time - start_time
        }
        
        # 记录结果
        app.logger.info(f'批量预测完成: {len(texts)}篇文章, 耗时: {end_time - start_time:.2f}秒')
        
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f'批量预测过程中出错: {str(e)}')
        return jsonify({
            'success': False,
            'error': f'批量预测过程中出错: {str(e)}'
        }), 500

if __name__ == '__main__':
    # 生产环境应该使用WSGI服务器（如gunicorn）运行Flask应用
    # 这里只是为了开发和测试方便
    app.logger.info('开始启动Web服务...')
    print("服务器正在启动，请访问http://localhost:5000/health进行测试")
    app.run(host='0.0.0.0', port=5000, debug=False)
    app.logger.info('服务器已关闭') 