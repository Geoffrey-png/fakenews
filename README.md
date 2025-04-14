# 虚假新闻检测系统

![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![Vue](https://img.shields.io/badge/vue-2.7+-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.7+-red.svg)

一个基于深度学习的多模态虚假新闻检测系统，集成了文本分析和图像篡改检测功能，为用户提供全面的新闻真实性分析服务。

[English](README_EN.md) | 中文

## 📋 项目概述

本项目是一个完整的虚假新闻检测解决方案，使用BERT模型进行文本分析，HAMMER模型进行图像篡改检测，并通过友好的Web界面向用户提供服务。系统支持单文本检测、批量文本检测、图像检测和混合检测等多种功能。

该项目旨在通过深度学习和人工智能技术帮助用户识别和分析潜在的虚假新闻内容，提高信息过滤能力，减少虚假信息传播的影响。

### 核心功能

- **单文本检测**：分析单条新闻文本的真实性
- **批量文本检测**：同时处理多条新闻文本
- **图像检测**：识别图像中的各种篡改痕迹
- **混合检测**：结合文本和图像的综合分析
- **虚假新闻解释**：利用大模型API生成可解释性分析结果

## 🏗️ 系统架构

### 前端架构
- **框架**：Vue.js 2.x + Element UI
- **HTTP客户端**：Axios
- **路由**：Vue Router

### 后端架构
- **Web框架**：Flask + Flask-CORS
- **深度学习**：PyTorch + Transformers
- **图像处理**：Pillow
- **NLP处理**：jieba分词
- **部署**：Gunicorn

### 文件结构
```
fake_news_detection/
├── frontend/             # 前端Vue应用
├── api/                  # 后端API服务
├── src/                  # 模型训练和评估代码
├── hammer_detector/      # 图像检测模块
├── models/               # 预训练模型
├── data/                 # 数据集
└── results/              # 结果和可视化
```

## 🚀 安装与部署

### 环境要求
- Python 3.7+
- Node.js 12+
- CUDA 10.1+ (推荐用于GPU加速)

### 后端设置

1. **克隆仓库**
```bash
git clone https://github.com/Geoffrey-png/fakenews.git
cd fakenews
```

2. **安装Python依赖**
```bash
pip install -r requirements.txt
```

3. **安装图像检测模块依赖**
```bash
cd hammer_detector
pip install -r requirements.txt
cd ..
```

4. **下载预训练模型**
```bash
# 下载BERT模型
mkdir -p models/bert_local
# 请从相应的来源下载bert-base-chinese模型

# 下载HAMMER模型权重
mkdir -p hammer_detector/weights
# 请从项目发布页面下载checkpoint_best.pth
```

5. **启动后端服务**
```bash
cd api
bash start_server.sh
# 或在Windows上
python app.py
```

### 前端设置

1. **安装依赖**
```bash
cd frontend
npm install
```

2. **启动开发服务器**
```bash
npm run serve
```

3. **构建生产版本**
```bash
npm run build
```

## 📝 使用指南

### 文本检测

1. 访问系统首页
2. 选择"单文本检测"或"批量检测"功能
3. 输入或粘贴需要检测的新闻文本
4. 点击"开始检测"按钮
5. 查看检测结果和分析解释

### 图像检测

1. 访问系统首页
2. 选择"图像检测"功能
3. 上传需要分析的图像
4. 系统自动进行篡改检测
5. 查看检测结果和篡改区域可视化

## 🔬 技术细节

### 文本检测模型

- 基于BERT的二分类模型
- 使用中文语料预训练和微调
- 准确率超过96%
- 支持缓存优化和批量处理

### 图像检测技术

- 基于HAMMER框架的多模态篡改检测
- 支持人脸、文字、物体篡改的识别
- 提供高精度篡改区域可视化
- 支持多种篡改类型的分类

### API接口

| 接口 | 方法 | 描述 |
|-----|------|-----|
| `/health` | GET | 健康检查接口 |
| `/predict` | POST | 单文本检测接口 |
| `/batch_predict` | POST | 批量文本检测接口 |
| `/detect/image` | POST | 图像检测接口 |
| `/generate_explanation` | POST | 虚假新闻解释生成接口 |

## 📊 示例

### 文本检测结果

![文本检测示例](Figure_1.png)

### 图像检测结果

![图像检测示例](Figure_2.png)

## 👨‍💻 开发者指南

### 添加新功能

1. **前端开发**
   - 在`frontend/src/views/`目录中创建新的页面组件
   - 更新`frontend/src/router/index.js`以添加路由
   - 在`frontend/src/utils/api.js`中添加新的API调用

2. **后端开发**
   - 在`api/app.py`中添加新的API端点
   - 扩展现有模块的功能或创建新模块

### 模型训练

```bash
bash run_train.sh
```

