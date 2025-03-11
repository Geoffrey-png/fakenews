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

# 导入预测模型和解释生成器
from api.model_loader import FakeNewsPredictor
from api.explanation_generator import ExplanationGenerator

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
explanation_generator = None

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

# 加载解释生成器函数
def load_explanation_generator():
    """加载解释生成器"""
    global explanation_generator
    try:
        app.logger.info('正在初始化解释生成器...')
        explanation_generator = ExplanationGenerator()
        app.logger.info('解释生成器初始化成功')
        # 进行测试调用
        test_result = explanation_generator.generate_explanation(
            "测试新闻文本",
            {"label": "虚假新闻", "confidence": 0.9}
        )
        if test_result:
            app.logger.info(f"解释生成器测试成功: {test_result[:20]}...")
        else:
            app.logger.warning("解释生成器测试返回空结果")
    except Exception as e:
        app.logger.error(f'解释生成器初始化失败: {str(e)}')
        app.logger.error('将继续运行，但无法提供假新闻解释功能')

# 在请求前确保模型已加载
@app.before_request
def ensure_model_loaded():
    global predictor, explanation_generator
    if predictor is None:
        try:
            load_model()
        except Exception as e:
            pass  # 错误会在相应的路由处理中处理
    
    if explanation_generator is None:
        try:
            load_explanation_generator()
        except Exception as e:
            pass  # 解释生成器初始化失败不应阻止API继续运行

# 尝试在应用启动时加载模型
try:
    load_model()
except Exception as e:
    app.logger.warning(f'应用启动时模型加载失败: {str(e)}')
    app.logger.warning('将在第一次请求时尝试再次加载')

# 尝试在应用启动时初始化解释生成器
try:
    load_explanation_generator()
except Exception as e:
    app.logger.warning(f'应用启动时解释生成器初始化失败: {str(e)}')
    app.logger.warning('将在第一次请求时尝试再次初始化')

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
    global predictor, explanation_generator
    
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
        
        # 如果是假新闻且解释生成器可用，则生成解释
        explanation = ""
        if result['class'] == 1 and explanation_generator is not None:  # 1表示假新闻
            try:
                # 构建用于解释生成的预测结果
                pred_for_explanation = {
                    'label': '虚假新闻',
                    'label_id': 1,
                    'confidence': float(result['probabilities'][1])
                }
                explanation = explanation_generator.generate_explanation(text, pred_for_explanation)
                app.logger.info(f'已为假新闻生成解释，长度: {len(explanation)}')
            except Exception as e:
                app.logger.error(f'生成解释失败: {str(e)}')
                # 解释生成失败不影响主要功能
        
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
            'explanation': explanation,  # 添加解释字段
            'process_time': end_time - start_time
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
    global predictor, explanation_generator
    
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
        
        for i, text in enumerate(texts):
            if not text:
                results.append({
                    'success': False,
                    'error': '空文本'
                })
                continue
            
            try:
                # 单文本预测
                text_start_time = time.time()
                result = predictor.predict(text)
                
                # 构建预测结果
                predicted_class = result['class']
                probabilities = result['probabilities']
                
                # 如果是假新闻且解释生成器可用，则生成解释
                explanation = ""
                if predicted_class == 1 and explanation_generator is not None:  # 1表示假新闻
                    try:
                        # 构建用于解释生成的预测结果
                        pred_for_explanation = {
                            'label': '虚假新闻',
                            'label_id': 1,
                            'confidence': float(probabilities[1])
                        }
                        explanation = explanation_generator.generate_explanation(text, pred_for_explanation)
                    except Exception as e:
                        app.logger.error(f'生成第{i+1}条文本解释失败: {str(e)}')
                        # 解释生成失败不影响主要功能
                
                text_end_time = time.time()
                
                # 添加到结果列表
                results.append({
                    'success': True,
                    'prediction': {
                        'label': '真实新闻' if predicted_class == 0 else '虚假新闻',
                        'label_id': predicted_class,
                        'confidence': {
                            '真实新闻': float(probabilities[0]),
                            '虚假新闻': float(probabilities[1])
                        }
                    },
                    'explanation': explanation,  # 添加解释字段
                    'process_time': text_end_time - text_start_time
                })
                
            except Exception as e:
                app.logger.error(f'处理第{i+1}条文本时出错: {str(e)}')
                results.append({
                    'success': False,
                    'error': f'处理文本时出错: {str(e)}'
                })
        
        end_time = time.time()
        
        # 构建响应
        response = {
            'success': True,
            'results': results,
            'total_time': end_time - start_time
        }
        
        app.logger.info(f'批量预测完成，处理了{len(texts)}篇文章，总耗时{end_time - start_time:.2f}秒')
        
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