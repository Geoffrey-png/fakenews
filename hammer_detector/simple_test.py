#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HAMMER 简化测试脚本
"""

import os
import sys
import yaml
import torch

# 修复Windows中文编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_config():
    """测试配置加载"""
    print("="*50)
    print("测试配置加载")
    print("="*50)
    
    config_path = os.path.join(current_dir, 'config.yaml')
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("配置加载成功:")
        for key, value in config.items():
            if not isinstance(value, dict):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
        return config
    except Exception as e:
        print(f"配置加载失败: {str(e)}")
        return None

def test_model_import():
    """测试模型导入"""
    print("\n" + "="*50)
    print("测试模型导入")
    print("="*50)
    
    try:
        # 检查Python路径
        print("Python路径:")
        for path in sys.path:
            print(f"  - {path}")
        
        # 检查models目录
        models_dir = os.path.join(current_dir, 'models')
        print(f"\n检查models目录: {models_dir}")
        if os.path.exists(models_dir):
            print("models目录存在")
            print("目录内容:")
            for file in os.listdir(models_dir):
                print(f"  - {file}")
        else:
            print("models目录不存在!")
        
        # 检查__init__.py
        init_file = os.path.join(models_dir, '__init__.py')
        print(f"\n检查models/__init__.py: {init_file}")
        if os.path.exists(init_file):
            print("__init__.py存在")
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print("文件内容:")
                print(content)
        else:
            print("__init__.py不存在!")
        
        # 尝试分步导入
        print("\n尝试分步导入:")
        print("1. 导入models包")
        import models
        print("成功导入models包")
        
        print("2. 从models导入HAMMER")
        from models import HAMMER
        print("成功导入HAMMER")
        
        print("模型导入成功")
        return True
    except Exception as e:
        import traceback
        print(f"模型导入失败: {str(e)}")
        print("\n详细错误信息:")
        traceback.print_exc()
        return False

def test_tools_import():
    """测试工具导入"""
    print("\n" + "="*50)
    print("测试工具导入")
    print("="*50)
    
    try:
        from tools import preprocess_image, preprocess_text
        print("工具导入成功")
        return True
    except Exception as e:
        print(f"工具导入失败: {str(e)}")
        return False

def test_model_init(config):
    """测试模型初始化"""
    print("\n" + "="*50)
    print("测试模型初始化")
    print("="*50)
    
    if config is None:
        print("无配置信息，跳过模型初始化测试")
        return False
    
    try:
        from models import HAMMER
        model = HAMMER(config)
        print("模型初始化成功")
        return model
    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        return False

def test_weight_loading(model):
    """测试权重加载"""
    print("\n" + "="*50)
    print("测试权重加载")
    print("="*50)
    
    if model is False:
        print("模型初始化失败，跳过权重加载测试")
        return False
    
    weight_path = os.path.join(current_dir, 'weights', 'checkpoint_best.pth')
    if not os.path.exists(weight_path):
        print(f"错误: 权重文件不存在: {weight_path}")
        return False
    
    try:
        checkpoint = torch.load(weight_path, map_location='cpu')
        
        # 处理可能的不匹配
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # 加载状态字典
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"权重加载结果: {msg}")
        return True
    except Exception as e:
        print(f"权重加载失败: {str(e)}")
        return False

def main():
    """主函数"""
    try:
        # 测试配置加载
        config = test_config()
        
        # 测试模型导入
        model_import_result = test_model_import()
        
        # 测试工具导入
        tools_import_result = test_tools_import()
        
        # 测试模型初始化
        if model_import_result:
            model = test_model_init(config)
            
            # 测试权重加载
            if model:
                weight_loading_result = test_weight_loading(model)
        
        print("\n" + "="*50)
        print("测试完成")
        print("="*50)
        return 0
    except Exception as e:
        import traceback
        print(f"\n测试过程中发生未捕获的异常: {str(e)}")
        print("\n详细错误信息:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 