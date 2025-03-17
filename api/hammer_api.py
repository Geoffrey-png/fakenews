#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hammer检测器API
提供图像篡改检测服务的简单API
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import time
import argparse
from werkzeug.utils import secure_filename

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入hammer_detector相关模块
try:
    from hammer_detector.detect_image import detect_fake, load_config, build_model
    print("Hammer图像检测模块导入成功")
except ImportError as e:
    print(f"导入Hammer模块失败: {e}")
    sys.exit(1)

# 创建Flask应用
app = Flask(__name__)
# 增强CORS配置
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # 允许所有来源访问所有路由

# 配置目录
RESULTS_FOLDER = os.path.join(project_root, 'results')
UPLOADS_FOLDER = os.path.join(RESULTS_FOLDER, 'uploads')
VISUALIZATIONS_FOLDER = os.path.join(RESULTS_FOLDER, 'visualizations')

# 确保目录存在
for folder in [RESULTS_FOLDER, UPLOADS_FOLDER, VISUALIZATIONS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"创建目录: {folder}")

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time()
    })

@app.route('/detect/image', methods=['POST'])
def detect_image():
    """图像检测端点"""
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
        print("开始执行检测...")
        result = detect_fake(args)
        print(f"检测结果: {result}")
        
        # 将结果路径转为相对URL
        if 'visualization_path' in result and result['visualization_path']:
            result['visualization_url'] = f'/static/visualizations/{os.path.basename(result["visualization_path"])}'
            
        # 添加成功标志和处理时间
        result['success'] = True
        result['process_time'] = f'{time.time() - start_time:.2f}秒'
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        print(f"图像检测失败: {e}")
        traceback.print_exc()
        return jsonify({
            'error': f'检测过程出错: {str(e)}',
            'success': False
        }), 500

@app.route('/static/visualizations/<path:filename>')
def serve_visualization(filename):
    """提供可视化结果图像"""
    return send_from_directory(VISUALIZATIONS_FOLDER, filename)

@app.route('/static/uploads/<path:filename>')
def serve_upload(filename):
    """提供原始上传图像"""
    return send_from_directory(UPLOADS_FOLDER, filename)

if __name__ == '__main__':
    print("启动Hammer检测API服务...")
    # 创建权重目录
    weights_dir = os.path.join(project_root, 'hammer_detector', 'weights')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        print(f"创建权重目录: {weights_dir}")
    
    # 启动服务器
    app.run(host='0.0.0.0', port=5000, debug=False) 