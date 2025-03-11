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
import platform
import psutil

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
    model_status = "已加载" if predictor is not None else "未加载"
    explanation_status = "已初始化" if explanation_generator is not None else "未初始化"
    
    # 获取基本系统信息
    try:
        memory = psutil.virtual_memory()
        memory_info = {
            "total": f"{memory.total / (1024**3):.2f}GB",
            "available": f"{memory.available / (1024**3):.2f}GB",
            "percent_used": f"{memory.percent}%"
        }
    except:
        memory_info = "无法获取内存信息"
    
    return jsonify({
        'status': 'ok',
        'timestamp': time.time(),
        'model_status': model_status,
        'explanation_status': explanation_status,
        'server_info': {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'memory': memory_info
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """单文本虚假新闻检测API"""
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': '缺少文本参数'}), 400
            
        text = data['text']
        if not text:
            return jsonify({'success': False, 'error': '文本不能为空'}), 400
            
        # 记录请求
        app.logger.info(f'收到预测请求: {text[:100]}...')
        
        # 确保模型已加载
        if predictor is None:
            return jsonify({'success': False, 'error': '模型未加载'}), 503
            
        # 预测
        start_time = time.time()
        raw_prediction = predictor.predict(text)
        end_time = time.time()
        process_time = end_time - start_time
        
        # 转换预测结果格式
        predicted_class = raw_prediction.get('class', 1)  # 默认为虚假新闻
        probabilities = raw_prediction.get('probabilities', [0.5, 0.5])
        
        # 构建前端期望的预测结果格式
        prediction = {
            'label': '真实新闻' if predicted_class == 0 else '虚假新闻',
            'label_id': predicted_class,
            'confidence': {
                '真实新闻': float(probabilities[0]),
                '虚假新闻': float(probabilities[1])
            }
        }
        
        # 记录预测结果
        app.logger.info(f'预测结果: {prediction["label"]} (置信度: {prediction["confidence"][prediction["label"]]:.4f})')
        
        # 构建响应
        response_data = {
            'success': True,
            'text': text,
            'prediction': prediction,
            'process_time': process_time
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f'预测失败: {str(e)}')
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量虚假新闻检测API"""
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'success': False, 'error': '缺少文本数组参数'}), 400
            
        texts = data['texts']
        if not texts or not isinstance(texts, list):
            return jsonify({'success': False, 'error': '文本数组不能为空且必须是数组'}), 400
            
        # 记录请求
        app.logger.info(f'收到批量预测请求: {len(texts)}条文本')
        
        # 确保模型已加载
        if predictor is None:
            return jsonify({'success': False, 'error': '模型未加载'}), 503
            
        # 批量预测
        results = []
        
        for text in texts:
            if not text or not isinstance(text, str):
                results.append({
                    'success': False,
                    'error': '无效的文本'
                })
                continue
                
            try:
                # 预测
                start_time = time.time()
                raw_prediction = predictor.predict(text)
                end_time = time.time()
                process_time = end_time - start_time
                
                # 转换预测结果格式
                predicted_class = raw_prediction.get('class', 1)  # 默认为虚假新闻
                probabilities = raw_prediction.get('probabilities', [0.5, 0.5])
                
                # 构建前端期望的预测结果格式
                prediction = {
                    'label': '真实新闻' if predicted_class == 0 else '虚假新闻',
                    'label_id': predicted_class,
                    'confidence': {
                        '真实新闻': float(probabilities[0]),
                        '虚假新闻': float(probabilities[1])
                    }
                }
                
                # 构建结果
                result = {
                    'success': True,
                    'prediction': prediction,
                    'process_time': process_time
                }
                
                results.append(result)
                
            except Exception as e:
                app.logger.error(f'批量预测单项失败: {str(e)}')
                results.append({
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        app.logger.error(f'批量预测失败: {str(e)}')
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate_explanation', methods=['POST'])
def generate_explanation():
    """生成假新闻解释的API端点"""
    # 确保解释生成器已加载
    if explanation_generator is None:
        return jsonify({
            'success': False,
            'message': '解释生成器未加载，无法提供解释服务'
        }), 503
    
    # 获取请求参数
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': '请求参数为空'
            }), 400
            
        # 提取必要参数
        news_text = data.get('text')
        prediction = data.get('prediction')
        
        if not news_text:
            return jsonify({
                'success': False,
                'message': '缺少新闻文本参数'
            }), 400
            
        if not prediction:
            return jsonify({
                'success': False,
                'message': '缺少预测结果参数'
            }), 400
            
        app.logger.info(f'收到解释生成请求: {news_text[:50]}...')
        
        # 调用解释生成器
        start_time = time.time()
        explanation = explanation_generator.generate_explanation(news_text, prediction)
        end_time = time.time()
        
        app.logger.info(f'解释生成耗时: {end_time - start_time:.2f}秒')
        
        if explanation:
            return jsonify({
                'success': True,
                'explanation': explanation
            })
        else:
            return jsonify({
                'success': False,
                'message': '无法为此新闻生成解释'
            }), 500
            
    except Exception as e:
        app.logger.error(f'解释生成出错: {str(e)}')
        return jsonify({
            'success': False,
            'message': f'服务器错误: {str(e)}'
        }), 500

if __name__ == '__main__':
    # 生产环境应该使用WSGI服务器（如gunicorn）运行Flask应用
    # 这里只是为了开发和测试方便
    app.logger.info('开始启动Web服务...')
    print("服务器正在启动，请访问http://localhost:5000/health进行测试")
    app.run(host='0.0.0.0', port=5000, debug=False)
    app.logger.info('服务器已关闭') 