#!/usr/bin/env python
# -*- coding: utf-8 -*-

print("开始执行API服务器脚本...")

try:
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS
    import time
    import os
    import json
    import torch
    import sys
    import platform
    import psutil
    import argparse
    from werkzeug.utils import secure_filename
    import traceback
    
    print("基本模块导入成功")
    # 关闭标准输出缓冲，确保日志实时显示
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    # 手动刷新标准输出
    print("设置实时日志输出", flush=True)

    # 添加项目根目录到系统路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    print(f"项目根目录: {project_root}")
    print(f"系统路径: {sys.path}")
    
    # 尝试导入Hammer图像检测模块
    try:
        from hammer_detector.detect_image import detect_fake, load_config, build_model, visualize_result
        print("Hammer图像检测模块导入成功")
    except ImportError as e:
        print(f"导入Hammer模块失败: {e}")
        raise

    # 创建应用实例
    app = Flask(__name__)
    # 增强CORS配置
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # 允许所有来源访问所有路由
    print("Flask应用实例创建成功")

    # 配置目录
    RESULTS_FOLDER = os.path.join(project_root, 'results')
    UPLOADS_FOLDER = os.path.join(RESULTS_FOLDER, 'uploads')
    VISUALIZATIONS_FOLDER = os.path.join(RESULTS_FOLDER, 'visualizations')
    app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

    # 确保目录存在
    for folder in [RESULTS_FOLDER, UPLOADS_FOLDER, VISUALIZATIONS_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"创建目录: {folder}")

    # 全局变量
    predictor = None
    explanation_generator = None
    hammer_model = None  # Hammer图像检测模型

    # 导入模型加载器和解释生成器
    try:
        from api.model_loader import FakeNewsPredictor
        from api.explanation_generator import ExplanationGenerator
        print("预测模型和解释生成器导入成功")
    except ImportError as e:
        print(f"导入预测模型失败: {e}")
        print("将继续运行，但文本检测功能不可用")

    # 加载模型函数
    def load_model():
        """加载模型"""
        global predictor
        try:
            print('正在加载模型...')
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'fake_news_bert_model_final.pth'))
            predictor = FakeNewsPredictor(model_path)
            
            # 模型预热，避免首次预测慢
            print('模型预热中...')
            sample_texts = [
                "北京冬奥会2022年2月4日开幕",
                "震惊！某明星深夜现身酒吧",
                "中国科学家在量子计算领域取得重大突破"
            ]
            _ = predictor.predict_batch(sample_texts)
            print('模型预热完成')
            
            print('模型加载成功')
        except Exception as e:
            print(f'模型加载失败: {str(e)}')
            traceback.print_exc()

    # 加载解释生成器函数
    def load_explanation_generator():
        """加载解释生成器"""
        global explanation_generator
        try:
            print('正在初始化解释生成器...')
            explanation_generator = ExplanationGenerator()
            print('解释生成器初始化成功')
            # 进行测试调用
            test_result = explanation_generator.generate_explanation(
                "测试新闻文本",
                {
                    "label": "虚假新闻", 
                    "confidence": {
                        "真实新闻": 0.1,
                        "虚假新闻": 0.9
                    }
                }
            )
            if test_result:
                print(f"解释生成器测试成功: {test_result[:20]}...")
            else:
                print("解释生成器测试返回空结果")
        except Exception as e:
            print(f'解释生成器初始化失败: {str(e)}')
            print('将继续运行，但无法提供假新闻解释功能')

    # 在请求前确保模型已加载
    @app.before_request
    def ensure_model_loaded():
        global predictor, explanation_generator, hammer_model
        if predictor is None and request.path == '/predict':
            try:
                load_model()
            except Exception as e:
                print(f'请求前加载模型失败: {str(e)}')
                
        if explanation_generator is None and request.path == '/generate_explanation':
            try:
                load_explanation_generator()
            except Exception as e:
                print(f'请求前加载解释生成器失败: {str(e)}')

    # 尝试在应用启动时加载模型
    try:
        load_model()
    except Exception as e:
        print(f'应用启动时模型加载失败: {str(e)}')
        print('将在第一次请求时尝试再次加载')

    # 尝试在应用启动时初始化解释生成器
    try:
        load_explanation_generator()
    except Exception as e:
        print(f'应用启动时解释生成器初始化失败: {str(e)}')
        print('将在第一次请求时尝试再次初始化')

    # 健康检查端点
    @app.route('/health', methods=['GET'])
    def health_check():
        """健康检查端点"""
        return jsonify({
            'status': 'ok',
            'timestamp': time.time(),
            'server_info': {
                'platform': platform.platform(),
                'python_version': platform.python_version()
            }
        })

    # 文本检测API端点
    @app.route('/predict', methods=['POST', 'OPTIONS'])
    def predict():
        """单文本虚假新闻检测API"""
        # 处理CORS预检请求
        if request.method == 'OPTIONS':
            return '', 204
            
        try:
            # 获取请求数据
            data = request.get_json()
            if not data or 'text' not in data:
                return jsonify({'success': False, 'error': '缺少文本参数'}), 400
            
            text = data['text']
            if not text:
                return jsonify({'success': False, 'error': '文本不能为空'}), 400
            
            # 记录请求
            print(f'收到预测请求: {text[:100]}...', flush=True)
            
            # 确保模型已加载
            if predictor is None:
                return jsonify({'success': False, 'error': '模型未加载'}), 503
            
            # 预测
            start_time = time.time()
            prediction = predictor.predict(text)  # 现在predict方法已经优化，包含缓存机制
            end_time = time.time()
            process_time = end_time - start_time
            
            # 确保prediction包含label字段
            if 'label' not in prediction:
                # 添加label字段
                prediction['label'] = '真实新闻' if prediction.get('class', 1) == 0 else '虚假新闻'
            
            # 确保confidence字段格式正确
            if 'confidence' not in prediction and 'probabilities' in prediction:
                prediction['confidence'] = {
                    '真实新闻': prediction['probabilities'][0],
                    '虚假新闻': prediction['probabilities'][1]
                }
            
            # 记录预测结果和处理时间
            print(f'预测结果: {prediction["label"]} (置信度: {prediction["probabilities"][1] if "probabilities" in prediction else 0:.4f})', flush=True)
            print(f'处理时间: {process_time:.4f}秒', flush=True)
            
            # 构建响应
            response_data = {
                'success': True,
                'text': text,
                'prediction': prediction,
                'process_time': process_time
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f'预测失败: {str(e)}')
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500

    # 批量文本检测API端点
    @app.route('/batch_predict', methods=['POST', 'OPTIONS'])
    def batch_predict():
        """批量虚假新闻检测API"""
        # 处理CORS预检请求
        if request.method == 'OPTIONS':
            return '', 204
            
        try:
            # 获取请求数据
            data = request.get_json()
            if not data or 'texts' not in data:
                return jsonify({'success': False, 'error': '缺少文本数组参数'}), 400
            
            texts = data['texts']
            if not texts or not isinstance(texts, list):
                return jsonify({'success': False, 'error': '文本数组不能为空且必须是数组'}), 400
            
            # 过滤无效文本
            valid_texts = [text for text in texts if text and isinstance(text, str)]
            if not valid_texts:
                return jsonify({'success': False, 'error': '没有有效的文本'}), 400
            
            # 记录请求
            print(f'收到批量预测请求: {len(valid_texts)}条文本')
            
            # 确保模型已加载
            if predictor is None:
                return jsonify({'success': False, 'error': '模型未加载'}), 503
            
            # 批量预测
            start_time = time.time()
            
            # 使用批处理功能
            batch_predictions = predictor.predict_batch(valid_texts)
            
            end_time = time.time()
            total_process_time = end_time - start_time
            
            # 构建结果
            results = []
            for i, prediction in enumerate(batch_predictions):
                # 构建结果
                result = {
                    'success': True,
                    'prediction': prediction,
                    'process_time': total_process_time / len(valid_texts)  # 平均处理时间
                }
                results.append(result)
            
            print(f'批量预测完成，总耗时: {total_process_time:.4f}秒，平均每条: {total_process_time/len(valid_texts):.4f}秒')
            
            return jsonify({
                'success': True,
                'results': results,
                'total_time': total_process_time
            })
            
        except Exception as e:
            print(f'批量预测失败: {str(e)}')
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500

    # 生成假新闻解释API端点
    @app.route('/generate_explanation', methods=['POST', 'OPTIONS'])
    def generate_explanation():
        """生成假新闻解释的API端点"""
        # 处理CORS预检请求
        if request.method == 'OPTIONS':
            return '', 204
            
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
            
            print(f'收到解释生成请求: {news_text[:50]}...')
            
            # 调用解释生成器
            start_time = time.time()
            explanation = explanation_generator.generate_explanation(news_text, prediction)
            end_time = time.time()
            
            print(f'解释生成耗时: {end_time - start_time:.2f}秒')
            
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
            print(f'解释生成出错: {str(e)}')
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'服务器错误: {str(e)}'
            }), 500

    # 图像检测API接口
    @app.route('/detect/image', methods=['POST'])
    def detect_image():
        """图像检测接口"""
        start_time = time.time()
        
        try:
            # 检查是否有文件上传
            if 'image' not in request.files:
                return jsonify({
                    'error': '没有上传图片',
                    'success': False
                }), 400
                
            file = request.files['image']
            
            # 检查文件名
            if file.filename == '':
                return jsonify({
                    'error': '没有选择图片',
                    'success': False
                }), 400
                
            # 获取相关文本（如果有）
            text = request.form.get('text', '')
            
            # 保存上传的图片
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOADS_FOLDER, filename)
            file.save(file_path)
            print(f"保存上传图片: {file_path}")
            
            # 设置可视化输出路径
            output_path = os.path.join(VISUALIZATIONS_FOLDER, f"vis_{filename}")
            
            # 构建检测参数
            config_path = os.path.join(project_root, 'hammer_detector', 'config.yaml')
            checkpoint_path = os.path.join(project_root, 'hammer_detector', 'weights', 'checkpoint_best.pth')
            
            # 检查文件存在
            if not os.path.exists(config_path):
                return jsonify({
                    'error': f'配置文件不存在: {config_path}',
                    'success': False
                }), 500
                
            if not os.path.exists(checkpoint_path):
                return jsonify({
                    'error': f'模型权重文件不存在: {checkpoint_path}',
                    'success': False
                }), 500
            
            args = argparse.Namespace(
                image=file_path,
                text=text,
                config=config_path,
                checkpoint=checkpoint_path,
                visualize=True,
                output=output_path
            )
            
            # 执行检测
            print("开始执行检测...", flush=True)
            result = detect_fake(args)
            print(f"检测结果: {result}", flush=True)
            
            # 将结果路径转为相对URL
            if 'visualization_path' in result and result['visualization_path']:
                result['visualization_url'] = f'/static/visualizations/{os.path.basename(result["visualization_path"])}'
                
            # 添加成功标志和处理时间
            result['success'] = True
            result['process_time'] = f'{time.time() - start_time:.2f}秒'
            
            return jsonify(result)
            
        except Exception as e:
            print(f"图像检测失败: {e}")
            traceback.print_exc()
            return jsonify({
                'error': f'检测过程出错: {str(e)}',
                'success': False
            }), 500

    # 静态文件服务
    @app.route('/static/visualizations/<path:filename>')
    def serve_visualization(filename):
        """提供可视化结果图像"""
        return send_from_directory(VISUALIZATIONS_FOLDER, filename)

    @app.route('/static/uploads/<path:filename>')
    def serve_upload(filename):
        """提供原始上传图像"""
        return send_from_directory(UPLOADS_FOLDER, filename)

    if __name__ == '__main__':
        # 生产环境应该使用WSGI服务器（如gunicorn）运行Flask应用
        # 但对于开发和测试，直接使用Flask的内置服务器
        print("正在启动API服务器...", flush=True)
        try:
            # 检查并创建必要的目录
            # 检查权重文件夹和文件
            weights_dir = os.path.join(project_root, 'hammer_detector', 'weights')
            if not os.path.exists(weights_dir):
                print(f"创建权重目录: {weights_dir}")
                os.makedirs(weights_dir)
                
            # 检查上传和可视化目录
            uploads_dir = os.path.join(app.config['RESULTS_FOLDER'], 'uploads')
            if not os.path.exists(uploads_dir):
                print(f"创建上传目录: {uploads_dir}")
                os.makedirs(uploads_dir)
                
            vis_dir = os.path.join(app.config['RESULTS_FOLDER'], 'visualizations')
            if not os.path.exists(vis_dir):
                print(f"创建可视化目录: {vis_dir}")
                os.makedirs(vis_dir)
                
            # 启动服务器
            app.run(host='0.0.0.0', port=5000, debug=False)
            print("API服务器已启动!", flush=True)
        except Exception as e:
            print(f"启动服务器时发生错误: {str(e)}", flush=True)
            traceback.print_exc()
except Exception as e:
    print(f"API服务器脚本执行时发生错误: {str(e)}")
    import traceback
    traceback.print_exc() 